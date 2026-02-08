#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export a deterministic validation "pack" for fair baseline comparisons.

Exports:
    gt512/   : GT 512x512
    lq128/   : true x4 LR (128x128) AFTER the same degradation pipeline
    lq512/   : upsampled LR used by your DiT-SR (512x512), matches DegradationPipeline output
"""

import os
import json
import argparse
from pathlib import Path

import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
from tqdm import tqdm

import train_4090_auto_v4 as M


def _to_uint8_chw01(x_chw01: torch.Tensor) -> torch.Tensor:
    x = (x_chw01.clamp(0.0, 1.0) * 255.0 + 0.5).to(torch.uint8)
    return x


def _save_png_chw01(x_chw01: torch.Tensor, path: str) -> None:
    x_u8 = _to_uint8_chw01(x_chw01)
    img = x_u8.permute(1, 2, 0).cpu().numpy()
    Image.fromarray(img).save(path)


def _replay_to_lr_small(pipeline: M.DegradationPipeline, hr_tensor_m11: torch.Tensor, meta: dict) -> torch.Tensor:
    img = (hr_tensor_m11 + 1.0) * 0.5  # [-1,1] -> [0,1]

    def build_stage(stage: int):
        prefix = f"stage{stage}_"
        blur_applied = bool(int(meta[prefix + "blur_applied"]))
        k_size = int(meta[prefix + "k_size"])
        sigma = float(meta[prefix + "sigma"])
        sigma_x = float(meta[prefix + "sigma_x"])
        sigma_y = float(meta[prefix + "sigma_y"])
        theta = float(meta[prefix + "theta"])
        resize_scale = float(meta[prefix + "resize_scale"])
        resize_interp_idx = int(meta[prefix + "resize_interp"])
        resize_interp = M.RESIZE_INTERP_MODES[resize_interp_idx]
        noise_std = float(meta[prefix + "noise_std"])
        jpeg_quality = int(meta[prefix + "jpeg_quality"])
        noise = meta.get(prefix + "noise", None)
        if isinstance(noise, torch.Tensor):
            noise = noise.to(img.device, dtype=img.dtype)
        return dict(
            blur_applied=blur_applied,
            k_size=k_size,
            sigma=sigma,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            theta=theta,
            resize_scale=resize_scale,
            resize_interp=resize_interp,
            noise_std=noise_std,
            jpeg_quality=jpeg_quality,
            noise=noise,
        )

    def apply_ops(img_in: torch.Tensor, ops: list[str], params: dict) -> torch.Tensor:
        out = img_in
        for op in ops:
            if op == "blur" and params["blur_applied"]:
                if params["sigma_x"] > 0 and params["sigma_y"] > 0:
                    out = pipeline._apply_aniso_blur(out, params["k_size"], params["sigma_x"], params["sigma_y"], params["theta"])
                else:
                    out = TF.gaussian_blur(out, params["k_size"], [params["sigma"], params["sigma"]])
            elif op == "resize":
                mid_h = max(1, int(pipeline.crop_size * params["resize_scale"]))
                mid_w = max(1, int(pipeline.crop_size * params["resize_scale"]))
                out = TF.resize(out, [mid_h, mid_w], interpolation=params["resize_interp"], antialias=True)
            elif op == "noise":
                if params["noise_std"] > 0:
                    noise = params.get("noise", None)
                    if noise is None:
                        noise = torch.zeros_like(out)
                    out = (out + noise * params["noise_std"]).clamp(0.0, 1.0)
            elif op == "jpeg":
                out = pipeline._apply_jpeg(out, params["jpeg_quality"])
        return out

    ops_stage1 = str(meta["ops_stage1"]).split(",") if str(meta["ops_stage1"]) else []
    ops_stage2 = str(meta["ops_stage2"]).split(",") if str(meta["ops_stage2"]) else []
    use_two_stage = bool(int(meta["use_two_stage"]))

    stage1 = build_stage(1)
    out = apply_ops(img, ops_stage1, stage1)
    if use_two_stage:
        stage2 = build_stage(2)
        out = apply_ops(out, ops_stage2, stage2)

    down_h = int(meta["down_h"])
    down_w = int(meta["down_w"])
    lr_small = TF.resize(out, [down_h, down_w], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
    return lr_small


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--val_hr_dir", type=str, default=M.VAL_HR_DIR)
    ap.add_argument("--crop_size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=M.SEED)
    ap.add_argument("--max_samples", type=int, default=50)
    ap.add_argument("--deg_mode", type=str, default=M.TRAIN_DEG_MODE, choices=["bicubic", "highorder"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    gt_dir = out_dir / "gt512"
    lq128_dir = out_dir / "lq128"
    lq512_dir = out_dir / "lq512"
    for d in [gt_dir, lq128_dir, lq512_dir]:
        d.mkdir(parents=True, exist_ok=True)

    hr_paths = sorted(Path(args.val_hr_dir).glob("*.png"))
    if len(hr_paths) == 0:
        raise FileNotFoundError(f"No PNG found in {args.val_hr_dir}")

    norm = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    to_tensor = transforms.ToTensor()
    pipeline = M.DegradationPipeline(args.crop_size)

    manifest = {
        "val_hr_dir": args.val_hr_dir,
        "crop_size": args.crop_size,
        "seed": args.seed,
        "deg_mode": args.deg_mode,
        "items": [],
    }

    n = min(args.max_samples, len(hr_paths))
    for idx in tqdm(range(n), desc="Exporting val pack"):
        hr_path = hr_paths[idx]
        hr_pil = Image.open(str(hr_path)).convert("RGB")
        hr_crop = TF.center_crop(hr_pil, (args.crop_size, args.crop_size))
        hr = norm(to_tensor(hr_crop))  # [-1,1], [3,512,512]

        if args.deg_mode == "bicubic":
            hr01 = (hr + 1.0) * 0.5
            lr_small = TF.resize(hr01, (args.crop_size // 4, args.crop_size // 4),
                                 interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
            lr_up = TF.resize(lr_small, (args.crop_size, args.crop_size),
                              interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
            lr_up = (lr_up * 2.0 - 1.0).clamp(-1.0, 1.0)
        else:
            gen = torch.Generator().manual_seed(int(args.seed) + int(idx))
            lr_up, meta = pipeline(hr, return_meta=True, meta=None, generator=gen)  # [-1,1]
            lr_small = _replay_to_lr_small(pipeline, hr, meta)  # [0,1] @128

        name = f"{idx:06d}"
        _save_png_chw01((hr + 1.0) * 0.5, str(gt_dir / f"{name}.png"))
        _save_png_chw01(lr_small, str(lq128_dir / f"{name}.png"))
        _save_png_chw01((lr_up + 1.0) * 0.5, str(lq512_dir / f"{name}.png"))
        manifest["items"].append({"name": name, "hr_path": str(hr_path)})

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Val pack saved to: {out_dir}")


if __name__ == "__main__":
    main()
