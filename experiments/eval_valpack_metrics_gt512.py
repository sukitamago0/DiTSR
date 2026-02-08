#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate metrics on a valpack with fixed GT directory = gt512.

Pack layout expected:
  pack_dir/
    gt512/ 000000.png ...
    lq512/ 000000.png ...   (same-res degraded input used by v4)
    lq128/ 000000.png ...   (true x4 LR)

Methods:
  --method identity : compare lq512 vs gt512 (difficulty of degradation)
  --method bicubic  : upsample lq128->512 bicubic, compare vs gt512 (classic baseline)
  --method folder   : compare pred_dir/{stem}.png vs gt512/{stem}.png
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F


IMG_EXT = ".png"


def rgb01_to_y01(rgb01_bchw: torch.Tensor) -> torch.Tensor:
    # match train_4090_auto_v4.py
    r, g, b = rgb01_bchw[:, 0:1], rgb01_bchw[:, 1:2], rgb01_bchw[:, 2:3]
    return (16.0 + 65.481 * r + 128.553 * g + 24.966 * b) / 255.0


def load_png_chw01(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.array(img, copy=True)  # writable
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return t


def resize_chw(x: torch.Tensor, size_hw=(512, 512)) -> torch.Tensor:
    return F.interpolate(x.unsqueeze(0), size=size_hw, mode="bicubic", align_corners=False).squeeze(0)


def psnr01(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a - b) ** 2).clamp_min(1e-12)
    return float(10.0 * torch.log10(1.0 / mse).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack_dir", type=str, required=True)
    ap.add_argument("--method", type=str, required=True, choices=["identity", "bicubic", "folder"])
    ap.add_argument("--pred_dir", type=str, default=None)
    ap.add_argument("--max_n", type=int, default=50)
    ap.add_argument("--crop_border", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out_json", type=str, default=None)
    args = ap.parse_args()

    pack = Path(args.pack_dir)
    gt_dir = pack / "gt512"
    lq512_dir = pack / "lq512"
    lq128_dir = pack / "lq128"

    if not gt_dir.is_dir():
        raise FileNotFoundError(f"gt512 dir not found: {gt_dir}")
    if args.method == "identity" and not lq512_dir.is_dir():
        raise FileNotFoundError(f"lq512 dir not found: {lq512_dir}")
    if args.method == "bicubic" and not lq128_dir.is_dir():
        raise FileNotFoundError(f"lq128 dir not found: {lq128_dir}")
    if args.method == "folder":
        if args.pred_dir is None:
            raise ValueError("--pred_dir required for method=folder")
        if not Path(args.pred_dir).is_dir():
            raise FileNotFoundError(f"pred_dir not found: {args.pred_dir}")

    gt_files = sorted([p for p in gt_dir.iterdir() if p.suffix.lower() == IMG_EXT])
    stems = [p.stem for p in gt_files][: args.max_n]

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # LPIPS (optional but very useful)
    import lpips
    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    cb = int(args.crop_border)
    psnr_y_list, psnr_rgb_list, ssim_y_list, lpips_list = [], [], [], []
    missing = []

    # SSIM: use torchmetrics if available
    try:
        from torchmetrics.functional import structural_similarity_index_measure as tm_ssim
        have_ssim = True
    except Exception:
        have_ssim = False
        tm_ssim = None

    for stem in stems:
        gt01 = load_png_chw01(str(gt_dir / f"{stem}.png"))
        if gt01.shape[-2:] != (512, 512):
            gt01 = resize_chw(gt01, (512, 512))

        if args.method == "identity":
            pr01 = load_png_chw01(str(lq512_dir / f"{stem}.png"))
            if pr01.shape[-2:] != (512, 512):
                pr01 = resize_chw(pr01, (512, 512))
        elif args.method == "bicubic":
            lr01 = load_png_chw01(str(lq128_dir / f"{stem}.png"))
            pr01 = resize_chw(lr01, (512, 512))
        else:  # folder
            p = Path(args.pred_dir) / f"{stem}.png"
            if not p.is_file():
                missing.append(str(p))
                continue
            pr01 = load_png_chw01(str(p))
            if pr01.shape[-2:] != (512, 512):
                pr01 = resize_chw(pr01, (512, 512))

        gt_b = gt01.unsqueeze(0).to(device)
        pr_b = pr01.unsqueeze(0).to(device)

        if cb > 0:
            gt_c = gt_b[..., cb:-cb, cb:-cb]
            pr_c = pr_b[..., cb:-cb, cb:-cb]
        else:
            gt_c, pr_c = gt_b, pr_b

        # PSNR RGB
        psnr_rgb_list.append(psnr01(pr_c, gt_c))

        # PSNR/SSIM on Y
        gy = rgb01_to_y01(gt_c)
        py = rgb01_to_y01(pr_c)
        psnr_y_list.append(psnr01(py, gy))
        if have_ssim:
            ssim_y_list.append(float(tm_ssim(py, gy, data_range=1.0).item()))

        # LPIPS (expects [-1,1])
        pr_m11 = (pr_b * 2.0 - 1.0).clamp(-1, 1)
        gt_m11 = (gt_b * 2.0 - 1.0).clamp(-1, 1)
        lpips_list.append(float(lpips_fn(pr_m11, gt_m11).mean().item()))

    summary = {
        "method": args.method,
        "pack_dir": args.pack_dir,
        "gt_dir": str(gt_dir),
        "lq512_dir": str(lq512_dir) if lq512_dir.is_dir() else None,
        "lq128_dir": str(lq128_dir) if lq128_dir.is_dir() else None,
        "pred_dir": args.pred_dir,
        "crop_border": cb,
        "n": len(psnr_y_list),
        "psnr_y_mean": float(np.mean(psnr_y_list)) if psnr_y_list else None,
        "psnr_rgb_mean": float(np.mean(psnr_rgb_list)) if psnr_rgb_list else None,
        "ssim_y_mean": float(np.mean(ssim_y_list)) if ssim_y_list else None,
        "lpips_mean": float(np.mean(lpips_list)) if lpips_list else None,
        "missing": missing[:50] if missing else None,
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"✅ Saved: {args.out_json}")


if __name__ == "__main__":
    main()
