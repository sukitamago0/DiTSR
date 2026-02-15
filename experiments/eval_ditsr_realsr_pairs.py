#!/usr/bin/env python3
"""Evaluate DiTSR checkpoint on a small list of real-SR pairs.

- Loads model components from a training script (by file path), e.g.
  experiments/train_4090_auto_v9_gan_edge_gtmask.py
- Runs inference for user-provided LR/HR image pairs.
- Reports PSNR, SSIM, LPIPS for:
  1) DiTSR prediction
  2) Bicubic-upsample baseline

Note:
- DiTSR v8/v9 scripts are trained around 512x512 crops, so default behavior resizes
  LR/HR to 512 for inference/metrics (`--eval-size 512`).
"""

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

import lpips
from diffusers import AutoencoderKL, DDIMScheduler
from torchmetrics.functional import structural_similarity_index_measure as tm_ssim


def load_train_module(script_path: str):
    script_path = str(Path(script_path).resolve())
    spec = importlib.util.spec_from_file_location("ditsr_train_mod", script_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def load_png_chw01(path: str) -> torch.Tensor:
    return TF.to_tensor(Image.open(path).convert("RGB"))


def resize_bchw(x: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False)


def psnr01(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a - b) ** 2).clamp_min(1e-12)
    return float((10.0 * torch.log10(1.0 / mse)).item())


def rgb01_to_y01(rgb01: torch.Tensor) -> torch.Tensor:
    r, g, b = rgb01[:, 0:1], rgb01[:, 1:2], rgb01[:, 2:3]
    return (16.0 + 65.481 * r + 128.553 * g + 24.966 * b) / 255.0


def load_ckpt_into_models(ckpt_path: str, pixart, adapter):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "adapter" not in ckpt or "pixart_trainable" not in ckpt:
        raise KeyError(f"checkpoint missing keys; found={list(ckpt.keys())}")
    adapter.load_state_dict(ckpt["adapter"], strict=False)
    curr = pixart.state_dict()
    for k, v in ckpt["pixart_trainable"].items():
        if k in curr:
            curr[k] = v.to(curr[k].dtype)
    pixart.load_state_dict(curr, strict=False)
    return ckpt


@torch.no_grad()
def infer_one(mod, pixart, adapter, vae, y_embed, d_info, lr_m11, steps: int, cfg_scale: float):
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        clip_sample=False,
        prediction_type="v_prediction",
        set_alpha_to_one=False,
    )
    scheduler.set_timesteps(steps, device=lr_m11.device)

    z_lr = vae.encode(lr_m11).latent_dist.mean * vae.config.scaling_factor

    if getattr(mod, "USE_LQ_INIT", True):
        latents, run_timesteps = mod.get_lq_init_latents(
            z_lr.to(mod.COMPUTE_DTYPE), scheduler, steps,
            torch.Generator(device=lr_m11.device).manual_seed(int(getattr(mod, "SEED", 3407))),
            float(getattr(mod, "LQ_INIT_STRENGTH", 0.1)), mod.COMPUTE_DTYPE,
        )
    else:
        latents = torch.randn_like(z_lr)
        run_timesteps = scheduler.timesteps

    cond = adapter(z_lr.float())
    aug_level = torch.zeros((latents.shape[0],), device=lr_m11.device, dtype=mod.COMPUTE_DTYPE)

    for t in run_timesteps:
        t_b = torch.tensor([t], device=lr_m11.device).expand(latents.shape[0])
        with torch.autocast(device_type="cuda", dtype=mod.COMPUTE_DTYPE):
            drop_uncond = torch.ones(latents.shape[0], device=lr_m11.device)
            drop_cond = torch.ones(latents.shape[0], device=lr_m11.device)
            model_in = torch.cat([latents.to(mod.COMPUTE_DTYPE), z_lr.to(mod.COMPUTE_DTYPE)], dim=1)
            out_uncond = pixart(
                x=model_in, timestep=t_b, y=y_embed, aug_level=aug_level,
                mask=None, data_info=d_info, adapter_cond=None,
                injection_mode="hybrid", force_drop_ids=drop_uncond,
            )
            out_cond = pixart(
                x=model_in, timestep=t_b, y=y_embed, aug_level=aug_level,
                mask=None, data_info=d_info, adapter_cond=cond,
                injection_mode="hybrid", force_drop_ids=drop_cond,
            )
            if out_uncond.shape[1] == 8:
                out_uncond, _ = out_uncond.chunk(2, dim=1)
            if out_cond.shape[1] == 8:
                out_cond, _ = out_cond.chunk(2, dim=1)
            out = out_uncond + cfg_scale * (out_cond - out_uncond)
        latents = scheduler.step(out.float(), t, latents.float()).prev_sample

    pred_m11 = vae.decode(latents / vae.config.scaling_factor).sample.clamp(-1, 1)
    return pred_m11


def parse_pairs(arg: str):
    if arg:
        return json.loads(arg)
    return [
        {
            "name": "DRealSR_Canon_10",
            "hr": "/data/DRealSR/HR/Canon_10_x4.png",
            "lr": "/data/DRealSR/LR/Canon_10_x1.png",
        },
        {
            "name": "RealSR_Canon_001",
            "hr": "/data/RealSR/Canon/Test/4/Canon_001_HR.png",
            "lr": "/data/RealSR/Canon/Test/4/Canon_001_LR4.png",
        },
        {
            "name": "RealSR_Nikon_001",
            "hr": "/data/RealSR/Nikon/Test/4/Nikon_001_HR.png",
            "lr": "/data/RealSR/Nikon/Test/4/Nikon_001_LR4.png",
        },
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-script", default="experiments/train_4090_auto_v9_gan_edge_gtmask.py")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--eval-size", type=int, default=512, help="Resize LR/HR to this square size before inference/metrics")
    ap.add_argument("--crop-border", type=int, default=4)
    ap.add_argument("--pairs-json", default="", help="JSON list of {name,hr,lr}; default uses the 3 pairs from request")
    ap.add_argument("--out-csv", default="experiments_results/realsr_eval/results.csv")
    ap.add_argument("--out-json", default="experiments_results/realsr_eval/summary.json")
    ap.add_argument("--save-preds-dir", default="experiments_results/realsr_eval/preds")
    args = ap.parse_args()

    mod = load_train_module(args.train_script)
    device = torch.device(getattr(mod, "DEVICE", "cuda"))

    pixart = mod.PixArtMSV8_XL_2(
        input_size=64, in_channels=8,
        sparse_inject_ratio=mod.SPARSE_INJECT_RATIO,
        injection_cutoff_layer=mod.INJECTION_CUTOFF_LAYER,
        injection_strategy=mod.INJECTION_STRATEGY,
    ).to(device)

    base = torch.load(mod.PIXART_PATH, map_location="cpu")
    if "state_dict" in base:
        base = base["state_dict"]
    if "pos_embed" in base:
        del base["pos_embed"]
    if "x_embedder.proj.weight" in base and base["x_embedder.proj.weight"].shape[1] == 4:
        w4 = base["x_embedder.proj.weight"]
        w8 = torch.zeros((w4.shape[0], 8, w4.shape[2], w4.shape[3]), dtype=w4.dtype)
        w8[:, :4] = w4
        base["x_embedder.proj.weight"] = w8
    pixart.load_pretrained_weights_with_zero_init(base)
    mod.apply_lora(pixart, mod.LORA_RANK, mod.LORA_ALPHA)

    adapter = mod.build_adapter_v7(
        in_channels=4,
        hidden_size=1152,
        injection_layers_map=getattr(pixart, "injection_layers", None),
    ).to(device).float()

    vae = AutoencoderKL.from_pretrained(mod.VAE_PATH, local_files_only=True).to(device).float().eval()
    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    load_ckpt_into_models(args.ckpt, pixart, adapter)
    pixart.eval(); adapter.eval()

    y_embed = torch.load(mod.T5_EMBED_PATH, map_location="cpu")["prompt_embeds"].unsqueeze(1).to(device)
    d_info = {
        "img_hw": torch.tensor([[float(args.eval_size), float(args.eval_size)]], device=device),
        "aspect_ratio": torch.tensor([1.0], device=device),
    }

    out_pred_dir = Path(args.save_preds_dir)
    out_pred_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    pairs = parse_pairs(args.pairs_json)
    cb = int(args.crop_border)

    for item in pairs:
        name, hr_p, lr_p = item["name"], item["hr"], item["lr"]
        hr = load_png_chw01(hr_p).unsqueeze(0).to(device)
        lr = load_png_chw01(lr_p).unsqueeze(0).to(device)
        hr = resize_bchw(hr, args.eval_size)
        lr = resize_bchw(lr, args.eval_size)

        hr_m11 = (hr * 2 - 1).clamp(-1, 1)
        lr_m11 = (lr * 2 - 1).clamp(-1, 1)

        pred_m11 = infer_one(mod, pixart, adapter, vae, y_embed, d_info, lr_m11, args.steps, args.cfg)
        pred01 = (pred_m11 + 1) * 0.5

        bic01 = resize_bchw(lr, args.eval_size)

        if cb > 0:
            pr_c = pred01[..., cb:-cb, cb:-cb]
            bi_c = bic01[..., cb:-cb, cb:-cb]
            hr_c = hr[..., cb:-cb, cb:-cb]
        else:
            pr_c, bi_c, hr_c = pred01, bic01, hr

        py, hy = rgb01_to_y01(pr_c), rgb01_to_y01(hr_c)
        by = rgb01_to_y01(bi_c)

        lp_pred = float(lpips_fn((pred01 * 2 - 1).clamp(-1, 1), (hr * 2 - 1).clamp(-1, 1)).mean().item())
        lp_bic = float(lpips_fn((bic01 * 2 - 1).clamp(-1, 1), (hr * 2 - 1).clamp(-1, 1)).mean().item())

        row = {
            "name": name,
            "psnr_rgb_pred": psnr01(pr_c, hr_c),
            "ssim_y_pred": float(tm_ssim(py, hy, data_range=1.0).item()),
            "lpips_pred": lp_pred,
            "psnr_rgb_bic": psnr01(bi_c, hr_c),
            "ssim_y_bic": float(tm_ssim(by, hy, data_range=1.0).item()),
            "lpips_bic": lp_bic,
        }
        rows.append(row)

        save_path = out_pred_dir / f"{name}.png"
        img = (pred01[0].clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        Image.fromarray(img).save(save_path)

    summary = {
        "n": len(rows),
        "eval_size": args.eval_size,
        "crop_border": cb,
        "steps": args.steps,
        "cfg": args.cfg,
        "mean": {
            k: float(np.mean([r[k] for r in rows])) for k in rows[0].keys() if k != "name"
        } if rows else {},
        "rows": rows,
    }

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["name"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary["mean"], indent=2, ensure_ascii=False))
    print(f"✅ Saved CSV: {out_csv}")
    print(f"✅ Saved JSON: {out_json}")
    print(f"✅ Saved preds: {out_pred_dir}")


if __name__ == "__main__":
    main()
