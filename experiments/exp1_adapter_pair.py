#!/usr/bin/env python3
"""
Experiment 1 (minimal): paired adapter comparison for the same inputs.

Goal
----
For each latent sample, run control_mult=0 and control_mult=1 under the same
lr_sde start, then:
  - save a paired visualization (LR / GT / Pred@0 / Pred@1)
  - record per-sample metrics and deltas
This directly tests whether the adapter changes results without large sweeps.
"""
import argparse
import glob
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

try:
    from diffusers import AutoencoderKL, DPMSolverMultistepScheduler
except ImportError as exc:
    raise ImportError(
        "Missing dependency: diffusers. Install it in your env, e.g. "
        "`pip install diffusers` or ensure your conda env includes it."
    ) from exc

torch.backends.cudnn.enabled = False

try:
    from torchmetrics.functional import peak_signal_noise_ratio as psnr
    from torchmetrics.functional import structural_similarity_index_measure as ssim
    import lpips
    USE_METRICS = True
except ImportError:
    USE_METRICS = False

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from diffusion.model.nets.PixArtMS import PixArtMS_XL_2
from diffusion.model.nets.adapter import build_adapter


def stable_int_hash(s: str, mod: int = 2**32) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h, 16) % mod


def rgb01_to_y01(rgb01: torch.Tensor) -> torch.Tensor:
    r = rgb01[:, 0:1]
    g = rgb01[:, 1:2]
    b = rgb01[:, 2:3]
    y = (16.0 + 65.481 * r + 128.553 * g + 24.966 * b) / 255.0
    return y.clamp(0.0, 1.0)


def shave_border(x: torch.Tensor, shave: int) -> torch.Tensor:
    if shave <= 0:
        return x
    return x[..., shave:-shave, shave:-shave]


def load_inject_state_dict(pixart, inject_sd: Dict[str, torch.Tensor]) -> None:
    sd = pixart.state_dict()
    for k, v in inject_sd.items():
        if k in sd:
            sd[k] = v.to(sd[k].dtype)
    pixart.load_state_dict(sd, strict=False)


def cast_inject_modules_fp32(pixart) -> None:
    if hasattr(pixart, "injection_scales"):
        for s in pixart.injection_scales:
            s.data = s.data.float()
    if hasattr(pixart, "adapter_proj"):
        pixart.adapter_proj = pixart.adapter_proj.to(torch.float32)
    if hasattr(pixart, "adapter_norm"):
        pixart.adapter_norm = pixart.adapter_norm.to(torch.float32)
    if hasattr(pixart, "cross_attn_scale"):
        pixart.cross_attn_scale.data = pixart.cross_attn_scale.data.float()
    if hasattr(pixart, "input_adapter_ln"):
        pixart.input_adapter_ln = pixart.input_adapter_ln.to(torch.float32)
    if hasattr(pixart, "input_adaln"):
        pixart.input_adaln = pixart.input_adaln.to(torch.float32)


def build_text_cond(t5_path: str, device: str, dtype_pixart: torch.dtype):
    y_embed = torch.load(t5_path, map_location="cpu")["prompt_embeds"].unsqueeze(1)
    y_embed = y_embed.to(device).to(dtype_pixart)
    data_info = {
        "img_hw": torch.tensor([[512.0, 512.0]], device=device, dtype=dtype_pixart),
        "aspect_ratio": torch.tensor([1.0], device=device, dtype=dtype_pixart),
    }
    return y_embed, data_info


@dataclass
class SampleConfig:
    num_steps: int
    sde_strength: float
    fixed_noise_seed: int
    injection_mode: str
    control_mult: float


@torch.no_grad()
def sample_sr_from_latent(
    pixart,
    adapter,
    vae,
    vae_cpu,
    lr_latent: torch.Tensor,
    y_embed,
    data_info,
    cfg: SampleConfig,
    use_amp: bool,
    dtype_pixart: torch.dtype,
    device: str,
    decode_on_cpu: bool,
) -> torch.Tensor:
    scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000, solver_order=2)
    scheduler.set_timesteps(cfg.num_steps, device=device)

    start_t_val = int(1000 * cfg.sde_strength)
    timesteps = scheduler.timesteps
    idxs = (timesteps <= start_t_val).nonzero(as_tuple=True)[0]
    start_idx = int(idxs[0].item()) if len(idxs) > 0 else 0
    run_ts = timesteps[start_idx:]
    g = torch.Generator(device=device).manual_seed(int(cfg.fixed_noise_seed))
    noise = torch.randn(
        lr_latent.shape,
        generator=g,
        device=device,
        dtype=lr_latent.dtype,
    )
    t_start = torch.tensor([start_t_val], device=device).long()
    latents = scheduler.add_noise(lr_latent, noise, t_start)

    adapter_cond = None
    if cfg.control_mult != 0.0:
        with torch.cuda.amp.autocast(enabled=False):
            cond = adapter(lr_latent.float().to(device))
            if isinstance(cond, list):
                cond = [c * cfg.control_mult for c in cond]
            else:
                cond = cond * cfg.control_mult
            adapter_cond = cond

    for t in run_ts:
        t_tensor = t.unsqueeze(0).to(device)
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=dtype_pixart):
            out = pixart(
                latents.to(dtype_pixart),
                t_tensor,
                y_embed,
                data_info=data_info,
                adapter_cond=adapter_cond,
                injection_mode=cfg.injection_mode,
            )
            if out.shape[1] == 8:
                out, _ = out.chunk(2, dim=1)
        latents = scheduler.step(out.float(), t, latents.float()).prev_sample

    if decode_on_cpu:
        if vae_cpu is None:
            raise ValueError("vae_cpu must be provided when decode_on_cpu is True")
        latents_cpu = latents.detach().cpu()
        pred_img = vae_cpu.decode(latents_cpu / vae_cpu.config.scaling_factor).sample
    else:
        pred_img = vae.decode(latents / vae.config.scaling_factor).sample
    pred_img_01 = torch.clamp((pred_img + 1.0) / 2.0, 0.0, 1.0)
    return pred_img_01


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--latent_dir", type=str, required=True)
    ap.add_argument("--pixart_path", type=str, default=os.path.join(PROJECT_ROOT, "output", "pretrained_models", "PixArt-XL-2-512x512.pth"))
    ap.add_argument("--vae_path", type=str, default=os.path.join(PROJECT_ROOT, "output", "pretrained_models", "sd-vae-ft-ema"))
    ap.add_argument("--t5_path", type=str, default=os.path.join(PROJECT_ROOT, "output", "quality_embed.pth"))
    ap.add_argument("--adapter_type", type=str, default="fpn_se", choices=["fpn", "fpn_se"])
    ap.add_argument("--num_val_images", type=int, default=20)
    ap.add_argument("--num_steps", type=int, default=20)
    ap.add_argument("--fixed_noise_seed", type=int, default=42)
    ap.add_argument("--sde_strength", type=float, default=0.35)
    ap.add_argument("--injection_mode", type=str, default="hybrid", choices=["input", "cross_attn", "hybrid"])
    ap.add_argument("--metric_y", action="store_true")
    ap.add_argument("--shave", type=int, default=4)
    ap.add_argument("--max_vis", type=int, default=3)
    ap.add_argument("--no_lpips", action="store_true")
    ap.add_argument("--gt_decode_cpu", action="store_true")
    ap.add_argument("--pred_decode_cpu", action="store_true")
    ap.add_argument("--out_dir", type=str, default=os.path.join(PROJECT_ROOT, "experiments_results", "exp1_adapter_pair"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_pixart = torch.float16
    use_amp = (device == "cuda")

    pixart = PixArtMS_XL_2(input_size=64).to(device).to(dtype_pixart).eval()
    base = torch.load(args.pixart_path, map_location="cpu")
    if "state_dict" in base:
        base = base["state_dict"]
    if "pos_embed" in base:
        del base["pos_embed"]
    pixart.load_state_dict(base, strict=False)

    adapter = build_adapter(args.adapter_type, in_channels=4, hidden_size=1152).to(device).eval()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if "adapter" not in ckpt or "pixart_inject" not in ckpt:
        raise KeyError("ckpt must contain keys: adapter, pixart_inject")
    adapter.load_state_dict(ckpt["adapter"], strict=True)
    load_inject_state_dict(pixart, ckpt["pixart_inject"])
    cast_inject_modules_fp32(pixart)

    vae = AutoencoderKL.from_pretrained(args.vae_path, local_files_only=True).to(device).float().eval()
    vae.enable_slicing()
    vae_cpu = None
    if args.gt_decode_cpu or args.pred_decode_cpu:
        vae_cpu = AutoencoderKL.from_pretrained(args.vae_path, local_files_only=True).to("cpu").float().eval()
        vae_cpu.enable_slicing()

    y_embed, data_info = build_text_cond(args.t5_path, device, dtype_pixart)

    lpips_fn = None
    if USE_METRICS and (not args.no_lpips):
        lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
        for p in lpips_fn.parameters():
            p.requires_grad = False

    lat_paths = sorted(glob.glob(os.path.join(args.latent_dir, "*.pt")))
    if len(lat_paths) == 0:
        raise FileNotFoundError(f"No latent .pt files in {args.latent_dir}")
    paths = lat_paths[: min(len(lat_paths), args.num_val_images)]

    records = []
    pbar = tqdm(paths, desc="[exp1] paired adapter", dynamic_ncols=True)
    for idx, p in enumerate(pbar):
        seed_i = (stable_int_hash(p) + 12345) & 0xFFFFFFFF
        d = torch.load(p, map_location="cpu")
        if "hr_latent" not in d or "lr_latent" not in d:
            raise KeyError(f"latent file missing hr_latent/lr_latent: {p}")
        hr_latent = d["hr_latent"].unsqueeze(0).to(device).float()
        lr_latent = d["lr_latent"].unsqueeze(0).to(device).float()

        if args.gt_decode_cpu:
            if vae_cpu is None:
                raise ValueError("vae_cpu was not initialized for gt_decode_cpu")
            hr_latent_cpu = hr_latent.detach().cpu()
            gt_img = vae_cpu.decode(hr_latent_cpu / vae_cpu.config.scaling_factor).sample
        else:
            gt_img = vae.decode(hr_latent / vae.config.scaling_factor).sample
        gt01 = torch.clamp((gt_img + 1.0) / 2.0, 0.0, 1.0)

        cfg0 = SampleConfig(
            num_steps=args.num_steps,
            sde_strength=args.sde_strength,
            fixed_noise_seed=seed_i,
            injection_mode=args.injection_mode,
            control_mult=0.0,
        )
        cfg1 = SampleConfig(
            num_steps=args.num_steps,
            sde_strength=args.sde_strength,
            fixed_noise_seed=seed_i,
            injection_mode=args.injection_mode,
            control_mult=1.0,
        )
        pred0 = sample_sr_from_latent(
            pixart=pixart,
            adapter=adapter,
            vae=vae,
            vae_cpu=vae_cpu,
            lr_latent=lr_latent,
            y_embed=y_embed,
            data_info=data_info,
            cfg=cfg0,
            use_amp=use_amp,
            dtype_pixart=dtype_pixart,
            device=device,
            decode_on_cpu=args.pred_decode_cpu,
        )
        pred1 = sample_sr_from_latent(
            pixart=pixart,
            adapter=adapter,
            vae=vae,
            vae_cpu=vae_cpu,
            lr_latent=lr_latent,
            y_embed=y_embed,
            data_info=data_info,
            cfg=cfg1,
            use_amp=use_amp,
            dtype_pixart=dtype_pixart,
            device=device,
            decode_on_cpu=args.pred_decode_cpu,
        )

        if args.pred_decode_cpu:
            gt01 = gt01.detach().cpu()
        if gt01.device != pred0.device:
            if gt01.device.type == "cpu":
                pred0 = pred0.detach().cpu()
                pred1 = pred1.detach().cpu()
            else:
                gt01 = gt01.to(pred0.device)

        row = {
            "path": p,
            "seed": int(seed_i),
            "sde_strength": args.sde_strength,
            "injection_mode": args.injection_mode,
            "num_steps": args.num_steps,
        }

        if USE_METRICS:
            if args.metric_y:
                py0 = shave_border(rgb01_to_y01(pred0), args.shave)
                py1 = shave_border(rgb01_to_y01(pred1), args.shave)
                gy = shave_border(rgb01_to_y01(gt01), args.shave)
                row["psnr_0"] = float(psnr(py0, gy, data_range=1.0).item())
                row["psnr_1"] = float(psnr(py1, gy, data_range=1.0).item())
                row["ssim_0"] = float(ssim(py0, gy, data_range=1.0).item())
                row["ssim_1"] = float(ssim(py1, gy, data_range=1.0).item())
            else:
                row["psnr_0"] = float(psnr(pred0, gt01, data_range=1.0).item())
                row["psnr_1"] = float(psnr(pred1, gt01, data_range=1.0).item())
                row["ssim_0"] = float(ssim(pred0, gt01, data_range=1.0).item())
                row["ssim_1"] = float(ssim(pred1, gt01, data_range=1.0).item())
            row["psnr_delta"] = row["psnr_1"] - row["psnr_0"]
            row["ssim_delta"] = row["ssim_1"] - row["ssim_0"]

            if lpips_fn is not None:
                pred0_norm = pred0 * 2.0 - 1.0
                pred1_norm = pred1 * 2.0 - 1.0
                gt_norm = gt01 * 2.0 - 1.0
                row["lpips_0"] = float(lpips_fn(pred0_norm, gt_norm).item())
                row["lpips_1"] = float(lpips_fn(pred1_norm, gt_norm).item())
                row["lpips_delta"] = row["lpips_1"] - row["lpips_0"]

        records.append(row)

        if args.max_vis > 0 and idx < args.max_vis:
            if args.pred_decode_cpu:
                if vae_cpu is None:
                    raise ValueError("vae_cpu was not initialized for pred_decode_cpu")
                lr_latent_cpu = lr_latent.detach().cpu()
                lr_img = vae_cpu.decode(lr_latent_cpu / vae_cpu.config.scaling_factor).sample
            else:
                lr_img = vae.decode(lr_latent / vae.config.scaling_factor).sample
            lr_vis = torch.clamp((lr_img + 1.0) / 2.0, 0.0, 1.0)
            lr01 = lr_vis[0].detach().permute(1, 2, 0).cpu().numpy()
            gt_vis = gt01[0].detach().permute(1, 2, 0).cpu().numpy()
            p0 = pred0[0].detach().permute(1, 2, 0).cpu().numpy()
            p1 = pred1[0].detach().permute(1, 2, 0).cpu().numpy()
            import matplotlib.pyplot as plt

            save_path = os.path.join(args.out_dir, f"pair_idx{idx:03d}.png")
            plt.figure(figsize=(16, 4))
            plt.subplot(1, 4, 1); plt.imshow(lr01); plt.title("LR"); plt.axis("off")
            plt.subplot(1, 4, 2); plt.imshow(gt_vis); plt.title("GT"); plt.axis("off")
            plt.subplot(1, 4, 3); plt.imshow(p0); plt.title("Pred ctrl=0"); plt.axis("off")
            plt.subplot(1, 4, 4); plt.imshow(p1); plt.title("Pred ctrl=1"); plt.axis("off")
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()

    detail_path = os.path.join(args.out_dir, "detail.jsonl")
    with open(detail_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def avg(key: str) -> float:
        vals = [r[key] for r in records if key in r]
        if not vals:
            return float("nan")
        return float(sum(vals) / len(vals))

    summary = {
        "psnr_0": avg("psnr_0"),
        "psnr_1": avg("psnr_1"),
        "psnr_delta": avg("psnr_delta"),
        "ssim_0": avg("ssim_0"),
        "ssim_1": avg("ssim_1"),
        "ssim_delta": avg("ssim_delta"),
        "lpips_0": avg("lpips_0"),
        "lpips_1": avg("lpips_1"),
        "lpips_delta": avg("lpips_delta"),
        "num": len(records),
    }
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "ckpt": args.ckpt,
                "latent_dir": args.latent_dir,
                "num_val_images": args.num_val_images,
                "num_steps": args.num_steps,
                "fixed_noise_seed": args.fixed_noise_seed,
                "sde_strength": args.sde_strength,
                "injection_mode": args.injection_mode,
                "metric_y": bool(args.metric_y),
                "shave": int(args.shave),
                "summary": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"✅ Done. Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
