#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CFG sweep evaluation for DiT-SR (PixArt + Adapter)

- Loads a checkpoint saved by train_4090_auto_v4.py (expects keys: adapter, pixart_trainable).
- Runs deterministic validation and sweeps CFG scales.
- Supports two guidance modes:
    (A) "text+adapter" (matches current validate() in train_4090_auto_v4.py):
        uncond: text dropped + no adapter
        cond:   text kept   + adapter
    (B) "adapter_only" (recommended for SR if you want to remove prompt-induced style/color bias):
        both branches use the SAME text (no drop),
        uncond: no adapter
        cond:   adapter
    In both modes, CFG is an *inference-time* hyperparameter.

Usage example:
    python eval_cfg_sweep.py \
      --ckpt /path/to/last.pth \
      --cfg_scales 0.0 0.5 1.0 1.5 2.0 3.0 4.0 \
      --mode adapter_only \
      --steps 50 \
      --out_dir ./cfg_sweep_out
"""
import os
import argparse
import json
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import your training script as a library (main() is guarded, so it won't start training)
import train_4090_auto_v4 as M


@torch.no_grad()
def load_ckpt_into_models(ckpt_path: str, pixart, adapter):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "adapter" in ckpt:
        adapter.load_state_dict(ckpt["adapter"], strict=True)
    else:
        raise KeyError(f"Checkpoint missing key 'adapter': {ckpt.keys()}")

    if "pixart_trainable" in ckpt:
        curr = pixart.state_dict()
        for k, v in ckpt["pixart_trainable"].items():
            if k in curr:
                curr[k] = v.to(curr[k].dtype)
        pixart.load_state_dict(curr, strict=False)
    else:
        raise KeyError(f"Checkpoint missing key 'pixart_trainable': {ckpt.keys()}")

    epoch = int(ckpt.get("epoch", -1))
    step = int(ckpt.get("step", -1))
    return epoch, step


@torch.no_grad()
def run_val_once(
    pixart,
    adapter,
    vae,
    lpips_fn,
    val_loader,
    y_embed,
    d_info,
    steps: int,
    cfg_scale: float,
    mode: str,
    seed: int,
    out_dir: str,
    save_vis: bool = True,
):
    """
    Returns (psnr_mean, ssim_mean, lpips_mean).
    Also saves a visualization grid for the first batch.
    """
    device = M.DEVICE
    compute_dtype = M.COMPUTE_DTYPE

    scheduler = M.DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        clip_sample=False,
        prediction_type="epsilon",
        set_alpha_to_one=False,
    )
    scheduler.set_timesteps(steps, device=device)

    val_gen = torch.Generator(device=device)
    val_gen.manual_seed(seed)

    psnrs, ssims, lpipss = [], [], []
    vis_done = False

    for batch in tqdm(val_loader, desc=f"Val@{steps} cfg={cfg_scale:g} mode={mode}"):
        hr = batch["hr"].to(device)
        lr = batch["lr"].to(device)

        z_hr = vae.encode(hr).latent_dist.mean * vae.config.scaling_factor
        z_lr = vae.encode(lr).latent_dist.mean * vae.config.scaling_factor

        # Deterministic init: matches your validate()
        if M.USE_LQ_INIT:
            latents = z_lr.clone()
            if M.INIT_NOISE_STD > 0:
                latents = latents + M.INIT_NOISE_STD * M.randn_like_with_generator(latents, val_gen)
        else:
            latents = M.randn_like_with_generator(z_hr, val_gen)

        cond = adapter(z_lr.float())

        for t in scheduler.timesteps:
            t_b = torch.tensor([t], device=device).expand(latents.shape[0])

            with torch.autocast(device_type="cuda", dtype=compute_dtype):
                if mode == "text+adapter":
                    # match validate(): uncond drops text, cond keeps text
                    drop_uncond = torch.ones(latents.shape[0], device=device)
                    drop_cond = torch.zeros(latents.shape[0], device=device)
                elif mode == "adapter_only":
                    # both branches use identical text (no drop) => guidance is only on adapter_cond
                    drop_uncond = torch.zeros(latents.shape[0], device=device)
                    drop_cond = torch.zeros(latents.shape[0], device=device)
                else:
                    raise ValueError(f"Unknown mode={mode}. Use 'text+adapter' or 'adapter_only'.")

                out_uncond = pixart(
                    x=latents.to(compute_dtype),
                    timestep=t_b,
                    y=y_embed,
                    mask=None,
                    data_info=d_info,
                    adapter_cond=None,
                    injection_mode="hybrid",
                    force_drop_ids=drop_uncond,
                )
                out_cond = pixart(
                    x=latents.to(compute_dtype),
                    timestep=t_b,
                    y=y_embed,
                    mask=None,
                    data_info=d_info,
                    adapter_cond=cond,
                    injection_mode="hybrid",
                    force_drop_ids=drop_cond,
                )

                # PixArt may output 8 channels (learned sigma). Keep eps only.
                if out_uncond.shape[1] == 8:
                    out_uncond, _ = out_uncond.chunk(2, dim=1)
                if out_cond.shape[1] == 8:
                    out_cond, _ = out_cond.chunk(2, dim=1)

                out = out_uncond + cfg_scale * (out_cond - out_uncond)

            latents = scheduler.step(out.float(), t, latents.float()).prev_sample

        pred = vae.decode(latents / vae.config.scaling_factor).sample.clamp(-1, 1)

        # Metrics: match your validate() (Y channel, shave=4)
        p01 = (pred + 1) / 2
        h01 = (hr + 1) / 2
        py = M.rgb01_to_y01(p01)[..., 4:-4, 4:-4]
        hy = M.rgb01_to_y01(h01)[..., 4:-4, 4:-4]
        psnrs.append(M.psnr(py, hy, data_range=1.0).item())
        ssims.append(M.ssim(py, hy, data_range=1.0).item())
        lpipss.append(lpips_fn(pred, hr).mean().item())

        if save_vis and (not vis_done):
            os.makedirs(out_dir, exist_ok=True)
            save_path = os.path.join(out_dir, f"vis_steps{steps}_cfg{cfg_scale:g}_{mode}.png")

            lr_np = (lr[0].detach().cpu().float().numpy().transpose(1, 2, 0) + 1) / 2
            hr_np = (hr[0].detach().cpu().float().numpy().transpose(1, 2, 0) + 1) / 2
            pr_np = (pred[0].detach().cpu().float().numpy().transpose(1, 2, 0) + 1) / 2

            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1); plt.imshow(np.clip(lr_np, 0, 1)); plt.title("Input LR"); plt.axis("off")
            plt.subplot(1, 3, 2); plt.imshow(np.clip(hr_np, 0, 1)); plt.title("GT"); plt.axis("off")
            plt.subplot(1, 3, 3); plt.imshow(np.clip(pr_np, 0, 1)); plt.title(f"Pred @{steps} cfg={cfg_scale:g}"); plt.axis("off")
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
            vis_done = True

    return float(np.mean(psnrs)), float(np.mean(ssims)), float(np.mean(lpipss))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Path to last.pth or a best checkpoint")
    ap.add_argument("--cfg_scales", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 3.0, 4.0])
    ap.add_argument("--mode", type=str, default="adapter_only", choices=["text+adapter", "adapter_only"])
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out_dir", type=str, default="./cfg_sweep_out")
    ap.add_argument("--val_mode", type=str, default=None, choices=[None, "train_like", "fixed_bicubic"])
    ap.add_argument("--seed", type=int, default=M.SEED)
    args = ap.parse_args()

    M.seed_everything(args.seed)

    # Build models (same as training)
    pixart = M.PixArtMS_XL_2(input_size=64).to(M.DEVICE)
    base = torch.load(M.PIXART_PATH, map_location="cpu")
    if "state_dict" in base:
        base = base["state_dict"]
    if "pos_embed" in base:
        del base["pos_embed"]
    pixart.load_state_dict(base, strict=False)
    M.apply_lora(pixart, M.LORA_RANK, M.LORA_ALPHA)

    adapter = M.build_adapter("fpn_se", 4, 1152).to(M.DEVICE).float()
    vae = M.AutoencoderKL.from_pretrained(M.VAE_PATH, local_files_only=True).to(M.DEVICE).float().eval()
    vae.enable_slicing()
    lpips_fn = M.lpips.LPIPS(net="vgg").to(M.DEVICE).eval()

    # Load checkpoint
    epoch, step = load_ckpt_into_models(args.ckpt, pixart, adapter)
    pixart.eval(); adapter.eval()

    # Text embedding (same file as training)
    y_embed = torch.load(M.T5_EMBED_PATH, map_location="cpu")["prompt_embeds"].unsqueeze(1).to(M.DEVICE)
    d_info = {
        "img_hw": torch.tensor([[512.0, 512.0]], device=M.DEVICE),
        "aspect_ratio": torch.tensor([1.0], device=M.DEVICE),
    }

    # Build validation set
    val_mode = args.val_mode or M.VAL_MODE
    if val_mode == "train_like":
        val_ds = M.DF2K_Val_Degraded_Dataset(M.VAL_HR_DIR, crop_size=512, seed=args.seed, deg_mode=M.TRAIN_DEG_MODE)
    else:
        # fixed bicubic: uses VAL_LR_DIR if exists, else bicubic down/up
        val_ds = M.DF2K_Val_Fixed_Dataset(M.VAL_HR_DIR, lr_root=M.VAL_LR_DIR, crop_size=512)
    val_loader = M.DataLoader(val_ds, batch_size=1, shuffle=False)

    os.makedirs(args.out_dir, exist_ok=True)
    results = {
        "ckpt": args.ckpt,
        "epoch": epoch,
        "step": step,
        "steps": args.steps,
        "mode": args.mode,
        "seed": args.seed,
        "cfg": {},
    }

    for cfg in args.cfg_scales:
        ps, ss, lp = run_val_once(
            pixart=pixart,
            adapter=adapter,
            vae=vae,
            lpips_fn=lpips_fn,
            val_loader=val_loader,
            y_embed=y_embed,
            d_info=d_info,
            steps=args.steps,
            cfg_scale=float(cfg),
            mode=args.mode,
            seed=args.seed,
            out_dir=args.out_dir,
            save_vis=True,
        )
        results["cfg"][str(cfg)] = {"psnr": ps, "ssim": ss, "lpips": lp}
        print(f"[cfg={cfg:g} mode={args.mode}] PSNR={ps:.2f} SSIM={ss:.4f} LPIPS={lp:.4f}")

    with open(os.path.join(args.out_dir, "cfg_sweep_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("✅ Done. Results saved to cfg_sweep_results.json")


if __name__ == "__main__":
    main()
