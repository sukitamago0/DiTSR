#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import csv
import argparse
import importlib
from pathlib import Path

import torch
from torchvision.transforms import functional as TF
from PIL import Image
from tqdm import tqdm

from diffusers import AutoencoderKL, DDIMScheduler
import lpips


def _load_png_as_chw01(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return TF.to_tensor(img)


def _save_png_bchw01(x_bchw01: torch.Tensor, path: str) -> None:
    x = (x_bchw01[0].clamp(0.0, 1.0) * 255.0 + 0.5).to(torch.uint8)
    img = x.permute(1, 2, 0).cpu().numpy()
    Image.fromarray(img).save(path)


def _psnr_rgb(p01: torch.Tensor, h01: torch.Tensor, crop: int) -> float:
    if crop > 0:
        p01 = p01[..., crop:-crop, crop:-crop]
        h01 = h01[..., crop:-crop, crop:-crop]
    mse = torch.mean((p01 - h01) ** 2)
    return float(10.0 * torch.log10(1.0 / (mse + 1e-10)))


def load_ckpt_into_models(ckpt_path: str, pixart, adapter):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "adapter" not in ckpt or "pixart_trainable" not in ckpt:
        raise KeyError(f"Checkpoint must contain 'adapter' and 'pixart_trainable'. keys={ckpt.keys()}")
    adapter.load_state_dict(ckpt["adapter"], strict=True)
    curr = pixart.state_dict()
    for k, v in ckpt["pixart_trainable"].items():
        if k in curr:
            curr[k] = v.to(curr[k].dtype)
    pixart.load_state_dict(curr, strict=False)
    return int(ckpt.get("epoch", -1)), int(ckpt.get("step", -1))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_module", type=str, default="train_4090_auto_v4")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--pack_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--adapter_type", type=str, default="fpn_se", choices=["fpn_se", "fpn_hf"])
    ap.add_argument("--cfg", type=float, default=3.5)
    ap.add_argument("--mode", type=str, default="text+adapter", choices=["text+adapter", "adapter_only"])
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--crop_border", type=int, default=4)
    ap.add_argument("--save_preds", action="store_true")
    ap.add_argument("--vae_encode_mode", type=str, default="mean", choices=["mean", "sample"])
    args = ap.parse_args()

    M = importlib.import_module(args.train_module)
    if args.seed is None:
        args.seed = int(getattr(M, "SEED", 3407))

    pack = Path(args.pack_dir)
    gt_dir = pack / "gt512"
    lq_dir = pack / "lq512"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_preds:
        (out_dir / "preds").mkdir(parents=True, exist_ok=True)

    device = torch.device(getattr(M, "DEVICE", "cuda"))

    pixart = M.PixArtMS_XL_2(input_size=64).to(device)
    base = torch.load(M.PIXART_PATH, map_location="cpu")
    if "state_dict" in base:
        base = base["state_dict"]
    if "pos_embed" in base:
        del base["pos_embed"]
    pixart.load_state_dict(base, strict=False)
    M.apply_lora(pixart, M.LORA_RANK, M.LORA_ALPHA)

    adapter = M.build_adapter(args.adapter_type, 4, 1152).to(device).float()
    vae = AutoencoderKL.from_pretrained(M.VAE_PATH, local_files_only=True).to(device).float().eval()
    vae.enable_slicing()
    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()

    epoch, step = load_ckpt_into_models(args.ckpt, pixart, adapter)
    pixart.eval(); adapter.eval()

    y_embed = torch.load(M.T5_EMBED_PATH, map_location="cpu")["prompt_embeds"].unsqueeze(1).to(device)
    d_info = {
        "img_hw": torch.tensor([[512.0, 512.0]], device=device),
        "aspect_ratio": torch.tensor([1.0], device=device),
    }

    scheduler = DDIMScheduler(
        num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02, beta_schedule="linear",
        clip_sample=False, set_alpha_to_one=False, steps_offset=1
    )
    scheduler.set_timesteps(args.steps, device=device)

    gen = torch.Generator(device=device).manual_seed(args.seed)

    names = sorted([p.stem for p in gt_dir.glob("*.png")])
    rows = []
    psnr_y_all, ssim_y_all, lpips_all, psnr_rgb_all = [], [], [], []
    vis_done = False

    for name in tqdm(names, desc="DiTSR inference"):
        gt01 = _load_png_as_chw01(str(gt_dir / f"{name}.png")).unsqueeze(0).to(device)
        lq01 = _load_png_as_chw01(str(lq_dir / f"{name}.png")).unsqueeze(0).to(device)
        gt = (gt01 * 2.0 - 1.0).clamp(-1.0, 1.0)
        lr = (lq01 * 2.0 - 1.0).clamp(-1.0, 1.0)

        with torch.autocast(device_type="cuda", dtype=M.COMPUTE_DTYPE):
            if args.vae_encode_mode == "sample":
                z_hr = vae.encode(gt).latent_dist.sample(generator=gen) * vae.config.scaling_factor
                z_lr = vae.encode(lr).latent_dist.sample(generator=gen) * vae.config.scaling_factor
            else:
                z_hr = vae.encode(gt).latent_dist.mean * vae.config.scaling_factor
                z_lr = vae.encode(lr).latent_dist.mean * vae.config.scaling_factor

        if getattr(M, "USE_LQ_INIT", False):
            latents = z_lr.clone()
            init_std = float(getattr(M, "INIT_NOISE_STD", 0.0))
            if init_std > 0:
                latents = latents + init_std * M.randn_like_with_generator(latents, gen)
        else:
            latents = M.randn_like_with_generator(z_hr, gen)

        cond = adapter(z_lr.float())

        for t in scheduler.timesteps:
            t_b = torch.tensor([t], device=device).expand(latents.shape[0])
            with torch.autocast(device_type="cuda", dtype=M.COMPUTE_DTYPE):
                if args.mode == "text+adapter":
                    drop_uncond = torch.ones(latents.shape[0], device=device)
                    drop_cond = torch.zeros(latents.shape[0], device=device)
                else:
                    drop_uncond = torch.zeros(latents.shape[0], device=device)
                    drop_cond = torch.zeros(latents.shape[0], device=device)

                out_uncond = pixart(
                    x=latents.to(M.COMPUTE_DTYPE), timestep=t_b, y=y_embed,
                    mask=None, data_info=d_info, adapter_cond=None,
                    injection_mode="hybrid", force_drop_ids=drop_uncond
                )
                out_cond = pixart(
                    x=latents.to(M.COMPUTE_DTYPE), timestep=t_b, y=y_embed,
                    mask=None, data_info=d_info, adapter_cond=cond,
                    injection_mode="hybrid", force_drop_ids=drop_cond
                )
                if out_uncond.shape[1] == 8: out_uncond, _ = out_uncond.chunk(2, dim=1)
                if out_cond.shape[1] == 8: out_cond, _ = out_cond.chunk(2, dim=1)
                out = out_uncond + float(args.cfg) * (out_cond - out_uncond)
            latents = scheduler.step(out.float(), int(t), latents.float()).prev_sample

        pred = vae.decode(latents / vae.config.scaling_factor).sample.clamp(-1.0, 1.0)
        pred01 = (pred + 1.0) * 0.5

        py = M.rgb01_to_y01(pred01)
        hy = M.rgb01_to_y01(gt01)
        if args.crop_border > 0:
            py = py[..., args.crop_border:-args.crop_border, args.crop_border:-args.crop_border]
            hy = hy[..., args.crop_border:-args.crop_border, args.crop_border:-args.crop_border]

        if hasattr(M, "psnr") and hasattr(M, "ssim"):
            psnr_y = float(M.psnr(py, hy, data_range=1.0).item())
            ssim_y = float(M.ssim(py, hy, data_range=1.0).item())
        else:
            mse = torch.mean((py - hy) ** 2)
            psnr_y = float(10.0 * torch.log10(1.0 / (mse + 1e-10)))
            ssim_y = None

        psnr_rgb = _psnr_rgb(pred01, gt01, crop=args.crop_border)
        lp = float(lpips_fn(pred, gt).mean().item())

        psnr_y_all.append(psnr_y)
        if ssim_y is not None: ssim_y_all.append(ssim_y)
        psnr_rgb_all.append(psnr_rgb)
        lpips_all.append(lp)

        rows.append({"name": name, "psnr_y": psnr_y, "ssim_y": ssim_y if ssim_y is not None else "",
                     "psnr_rgb": psnr_rgb, "lpips_vgg": lp})

        if args.save_preds:
            _save_png_bchw01(pred01, str(out_dir / "preds" / f"{name}.png"))

        if not vis_done:
            vis_done = True
            grid = torch.cat([lq01[0].clamp(0,1), pred01[0].clamp(0,1), gt01[0].clamp(0,1)], dim=2)
            _save_png_bchw01(grid.unsqueeze(0), str(out_dir / "vis_first.png"))

    summary = {
        "train_module": args.train_module,
        "ckpt": args.ckpt,
        "epoch": epoch,
        "step": step,
        "pack_dir": str(pack),
        "mode": args.mode,
        "cfg": args.cfg,
        "steps": args.steps,
        "seed": args.seed,
        "vae_encode_mode": args.vae_encode_mode,
        "n": len(names),
        "psnr_y_mean": float(sum(psnr_y_all)/len(psnr_y_all)),
        "psnr_rgb_mean": float(sum(psnr_rgb_all)/len(psnr_rgb_all)),
        "lpips_mean": float(sum(lpips_all)/len(lpips_all)),
    }
    if len(ssim_y_all) > 0:
        summary["ssim_y_mean"] = float(sum(ssim_y_all)/len(ssim_y_all))

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(out_dir / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
