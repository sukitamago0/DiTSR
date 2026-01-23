#!/usr/bin/env python3
"""
Diag script for current AdaLN SR setup.

What this script does
---------------------
1) Loads the current PixArt+Adapter+injection checkpoint and runs SR sampling.
2) Sweeps:
   - start mode: pure noise vs LR-latent SDE start
   - sde_strength
   - injection_mode
   - control_mult (adapter strength)
3) Reports:
   - PSNR/SSIM/LPIPS (optional, if torchmetrics/lpips installed)
   - color_mean_abs + per-channel color bias
   - laplacian_energy as a simple sharpness proxy
4) Saves per-image JSONL logs and a summary JSON, plus optional preview grids.

Why this helps (how to interpret)
---------------------------------
Use the sweep results to isolate weak points:
1) start_mode
   - "pure": starts from pure noise (may cause color drift but can add texture).
   - "lr_sde": starts from LR latent + noise (better color alignment, sometimes less detail).
   If lr_sde beats pure on color_mean_abs and PSNR, your drift is likely from pure noise start.
2) sde_strength
   - Higher values: more noise injection => more freedom, risk of color/structure drift.
   - Lower values: tighter to LR, but weaker detail enhancement.
3) injection_mode
   - "input": AdaLN/FiLM only.
   - "cross_attn": visual tokens only.
   - "hybrid": both. If hybrid is worse, one path may be harming stability.
4) control_mult
   - 0 disables adapter (baseline).
   - 0.5/1/1.5 shows if adapter is too weak or too strong.
   If control_mult=0 is similar to 1.0, conditioning is not effective.

Outputs (files and fields)
--------------------------
1) detail_*.jsonl (per-image):
   - psnr/ssim/lpips (if enabled)
   - color_mean_abs + color_r_abs/color_g_abs/color_b_abs (mean RGB bias vs GT)
   - laplacian_energy (higher usually means sharper edges)
2) summary.json:
   - averages per sweep setting to compare trends quickly
3) vis_*.png (optional):
   - side-by-side (LR, GT, Pred) for the first N images per setting

Usage (minimal)
--------------
python experiments/diag_eval_adaln_sr.py \
  --ckpt /path/to/epochXXX_psnrYY_lpZZ.pth \
  --val_hr_dir /path/to/DIV2K_valid_HR

Usage (recommended sweep)
-------------------------
python experiments/diag_eval_adaln_sr.py \
  --ckpt /path/to/epochXXX_psnrYY_lpZZ.pth \
  --val_hr_dir /path/to/DIV2K_valid_HR \
  --start_modes pure,lr_sde \
  --sde_strengths 0.35,0.45,0.55 \
  --injection_modes input,cross_attn,hybrid \
  --control_mults 0,0.5,1,1.5 \
  --num_val_images 50 \
  --metric_y --shave 4 \
  --max_vis 3
"""
import argparse
import glob
import hashlib
import io
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from PIL import Image
from tqdm import tqdm

from diffusers import AutoencoderKL, DPMSolverMultistepScheduler

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


def pil_to_tensor_norm01(pil: Image.Image) -> torch.Tensor:
    arr = np.asarray(pil, dtype=np.uint8).copy()
    x = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return x


def norm01_to_norm11(x01: torch.Tensor) -> torch.Tensor:
    return x01 * 2.0 - 1.0


def transforms_to_pil(x01: torch.Tensor) -> Image.Image:
    x = (x01.clamp(0, 1) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(x)


def center_crop(pil: Image.Image, size: int = 512) -> Image.Image:
    w, h = pil.size
    if w < size or h < size:
        pil = pil.resize((max(size, w), max(size, h)), resample=Image.BICUBIC)
        w, h = pil.size
    left = (w - size) // 2
    top = (h - size) // 2
    return pil.crop((left, top, left + size, top + size))


def _jpeg_compress_tensor(x11: torch.Tensor, quality: int) -> torch.Tensor:
    x = x11.clamp(-1, 1)
    x01 = (x + 1.0) / 2.0
    pil = transforms_to_pil(x01.cpu())
    buffer = io.BytesIO()
    pil.save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    pil2 = Image.open(buffer).convert("RGB")
    x01b = pil_to_tensor_norm01(pil2)
    return norm01_to_norm11(x01b)


def degrade_hr_to_lr_tensor(
    hr11_cpu: torch.Tensor,
    mode: str,
    rng: random.Random,
    torch_gen: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    hr11_cpu: [3,512,512] in [-1,1] on CPU
    return  : [3,512,512] in [-1,1] on CPU
    """
    if mode == "bicubic":
        hr = hr11_cpu.unsqueeze(0)
        lr_small = F.interpolate(hr, scale_factor=0.25, mode="bicubic", align_corners=False)
        lr = F.interpolate(lr_small, size=(512, 512), mode="bicubic", align_corners=False)
        return lr.squeeze(0)

    blur_k = rng.choice([3, 5, 7])
    blur_sigma = rng.uniform(0.2, 1.2)
    hr = hr11_cpu.unsqueeze(0)
    hr_blur = TF.gaussian_blur(hr, (blur_k, blur_k), [blur_sigma, blur_sigma])
    lr_small = F.interpolate(hr_blur, scale_factor=0.25, mode="bicubic", align_corners=False)

    noise_std = rng.uniform(0.0, 0.02)
    if noise_std > 0:
        if torch_gen is None:
            eps = torch.randn_like(lr_small)
        else:
            eps = torch.randn(
                lr_small.shape,
                generator=torch_gen,
                device=lr_small.device,
                dtype=lr_small.dtype,
            )
        lr_small = (lr_small + eps * noise_std).clamp(-1, 1)

    jpeg_q = rng.randint(30, 95)
    lr_small_cpu = _jpeg_compress_tensor(lr_small.squeeze(0).cpu(), jpeg_q).unsqueeze(0)
    lr = F.interpolate(lr_small_cpu, size=(512, 512), mode="bicubic", align_corners=False)
    return lr.squeeze(0)


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


def _color_bias(pred01: torch.Tensor, gt01: torch.Tensor) -> Dict[str, float]:
    pred_mean = pred01.mean(dim=[0, 2, 3])
    gt_mean = gt01.mean(dim=[0, 2, 3])
    diff = (pred_mean - gt_mean).abs()
    return {
        "color_mean_abs": float(diff.mean().item()),
        "color_r_abs": float(diff[0].item()),
        "color_g_abs": float(diff[1].item()),
        "color_b_abs": float(diff[2].item()),
    }


def _laplacian_energy(x01: torch.Tensor) -> float:
    # simple sharpness proxy on Y channel
    y = rgb01_to_y01(x01)
    k = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], device=y.device, dtype=y.dtype)
    k = k.view(1, 1, 3, 3)
    edge = F.conv2d(y, k, padding=1)
    return float(edge.abs().mean().item())


@dataclass
class SampleConfig:
    num_steps: int
    sde_strength: float
    fixed_noise_seed: int
    start_mode: str
    injection_mode: str
    control_mult: float


@torch.no_grad()
def sample_sr(
    pixart,
    adapter,
    vae,
    vae_cpu,
    lr_img_11: torch.Tensor,
    y_embed,
    data_info,
    cfg: SampleConfig,
    use_amp: bool,
    dtype_pixart: torch.dtype,
    device: str,
    decode_on_cpu: bool,
    encode_on_cpu: bool,
) -> torch.Tensor:
    if encode_on_cpu:
        if vae_cpu is None:
            raise ValueError("vae_cpu must be provided when encode_on_cpu is True")
        lr_latent = (
            vae_cpu.encode(lr_img_11.detach().cpu()).latent_dist.sample()
            * vae_cpu.config.scaling_factor
        )
        lr_latent = lr_latent.to(device)
    else:
        lr_latent = vae.encode(lr_img_11).latent_dist.sample() * vae.config.scaling_factor

    scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000, solver_order=2)
    scheduler.set_timesteps(cfg.num_steps, device=device)

    if cfg.start_mode == "pure":
        g = torch.Generator(device=device).manual_seed(int(cfg.fixed_noise_seed))
        latents = torch.randn(lr_latent.shape, generator=g, device=device, dtype=lr_latent.dtype)
        run_ts = list(scheduler.timesteps)
    elif cfg.start_mode == "lr_sde":
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
    else:
        raise ValueError(f"Unknown start_mode={cfg.start_mode}")

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

    if cfg.start_mode == "pure":
        g = torch.Generator(device=device).manual_seed(int(cfg.fixed_noise_seed))
        latents = torch.randn(lr_latent.shape, generator=g, device=device, dtype=lr_latent.dtype)
        run_ts = list(scheduler.timesteps)
    elif cfg.start_mode == "lr_sde":
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
    else:
        raise ValueError(f"Unknown start_mode={cfg.start_mode}")

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
    ap.add_argument("--ckpt", type=str, required=True, help="epochXXX_*.pth or last_full_state.pth")
    ap.add_argument("--val_hr_dir", type=str, default=os.path.join(PROJECT_ROOT, "dataset", "DIV2K_valid_HR"))
    ap.add_argument("--latent_dir", type=str, default=None, help="use offline HR/LR latents (.pt) instead of images")
    ap.add_argument("--pixart_path", type=str, default=os.path.join(PROJECT_ROOT, "output", "pretrained_models", "PixArt-XL-2-512x512.pth"))
    ap.add_argument("--vae_path", type=str, default=os.path.join(PROJECT_ROOT, "output", "pretrained_models", "sd-vae-ft-ema"))
    ap.add_argument("--t5_path", type=str, default=os.path.join(PROJECT_ROOT, "output", "quality_embed.pth"))
    ap.add_argument("--adapter_type", type=str, default="fpn_se", choices=["fpn", "fpn_se"])

    ap.add_argument("--val_degrade_mode", type=str, default="realistic", choices=["realistic", "bicubic"])
    ap.add_argument("--num_val_images", type=int, default=20)
    ap.add_argument("--start_modes", type=str, default="pure,lr_sde")
    ap.add_argument("--injection_modes", type=str, default="hybrid")
    ap.add_argument("--control_mults", type=str, default="1.0")
    ap.add_argument("--sde_strengths", type=str, default="0.35,0.45,0.55")
    ap.add_argument("--num_steps", type=int, default=20)
    ap.add_argument("--fixed_noise_seed", type=int, default=42)

    ap.add_argument("--metric_y", action="store_true")
    ap.add_argument("--shave", type=int, default=4)
    ap.add_argument("--max_vis", type=int, default=3)
    ap.add_argument("--no_lpips", action="store_true", help="disable LPIPS to save VRAM")
    ap.add_argument("--gt_decode_cpu", action="store_true", help="decode GT on CPU to save VRAM")
    ap.add_argument("--pred_decode_cpu", action="store_true", help="decode prediction on CPU to save VRAM")
    ap.add_argument("--encode_cpu", action="store_true", help="encode HR/LR on CPU to save VRAM")
    ap.add_argument("--out_dir", type=str, default=os.path.join(PROJECT_ROOT, "experiments_results", "diag_adaln_sr"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_pixart = torch.float16
    use_amp = (device == "cuda")

    # Load base model
    pixart = PixArtMS_XL_2(input_size=64).to(device).to(dtype_pixart).eval()
    base = torch.load(args.pixart_path, map_location="cpu")
    if "state_dict" in base:
        base = base["state_dict"]
    if "pos_embed" in base:
        del base["pos_embed"]
    pixart.load_state_dict(base, strict=False)

    # Load adapter
    adapter = build_adapter(args.adapter_type, in_channels=4, hidden_size=1152).to(device).eval()

    # Load ckpt (adapter + pixart_inject)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if "adapter" not in ckpt or "pixart_inject" not in ckpt:
        raise KeyError("ckpt must contain keys: adapter, pixart_inject")
    adapter.load_state_dict(ckpt["adapter"], strict=True)
    load_inject_state_dict(pixart, ckpt["pixart_inject"])
    cast_inject_modules_fp32(pixart)

    # VAE
    vae = AutoencoderKL.from_pretrained(args.vae_path, local_files_only=True).to(device).float().eval()
    vae.enable_slicing()
    vae_cpu = None
    if args.gt_decode_cpu or args.pred_decode_cpu or args.encode_cpu:
        vae_cpu = AutoencoderKL.from_pretrained(args.vae_path, local_files_only=True).to("cpu").float().eval()
        vae_cpu.enable_slicing()

    # Text cond
    y_embed, data_info = build_text_cond(args.t5_path, device, dtype_pixart)

    # Metrics
    lpips_fn = None
    if USE_METRICS and (not args.no_lpips):
        lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
        for p in lpips_fn.parameters():
            p.requires_grad = False

    # collect val images
    if args.latent_dir:
        lat_paths = sorted(glob.glob(os.path.join(args.latent_dir, "*.pt")))
        if len(lat_paths) == 0:
            raise FileNotFoundError(f"No latent .pt files in {args.latent_dir}")
        paths = lat_paths[: min(len(lat_paths), args.num_val_images)]
        data_mode = "latent"
    else:
        exts = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"]
        paths = []
        for e in exts:
            paths += glob.glob(os.path.join(args.val_hr_dir, e))
        paths = sorted(list(set(paths)))
        if len(paths) == 0:
            raise FileNotFoundError(f"No images in {args.val_hr_dir}")
        paths = paths[: min(len(paths), args.num_val_images)]
        data_mode = "image"

    start_modes = [s.strip() for s in args.start_modes.split(",") if s.strip()]
    injection_modes = [s.strip() for s in args.injection_modes.split(",") if s.strip()]
    control_mults = [float(s) for s in args.control_mults.split(",") if s.strip()]
    sde_strengths = [float(s) for s in args.sde_strengths.split(",") if s.strip()]

    summary = []

    for start_mode in start_modes:
        for injection_mode in injection_modes:
            for control_mult in control_mults:
                for sde_strength in sde_strengths:
                    cfg = SampleConfig(
                        num_steps=args.num_steps,
                        sde_strength=sde_strength,
                        fixed_noise_seed=args.fixed_noise_seed,
                        start_mode=start_mode,
                        injection_mode=injection_mode,
                        control_mult=control_mult,
                    )
                    records = []
                    pbar = tqdm(paths, desc=f"[diag] {cfg}", dynamic_ncols=True)
                    for idx, p in enumerate(pbar):
                        seed_i = (stable_int_hash(p) + 12345) & 0xFFFFFFFF
                        if data_mode == "latent":
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

                            pred01 = sample_sr_from_latent(
                                pixart=pixart,
                                adapter=adapter,
                                vae=vae,
                                vae_cpu=vae_cpu,
                                lr_latent=lr_latent,
                                y_embed=y_embed,
                                data_info=data_info,
                                cfg=cfg,
                                use_amp=use_amp,
                                dtype_pixart=dtype_pixart,
                                device=device,
                                decode_on_cpu=args.pred_decode_cpu,
                            )
                            if args.pred_decode_cpu:
                                gt01 = gt01.detach().cpu()
                            if gt01.device != pred01.device:
                                if gt01.device.type == "cpu":
                                    pred01 = pred01.detach().cpu()
                                else:
                                    gt01 = gt01.to(pred01.device)

                            lr_vis = None
                            if args.max_vis > 0 and idx < args.max_vis:
                                if args.pred_decode_cpu:
                                    lr_latent_cpu = lr_latent.detach().cpu()
                                    lr_img = vae_cpu.decode(lr_latent_cpu / vae_cpu.config.scaling_factor).sample
                                else:
                                    lr_img = vae.decode(lr_latent / vae.config.scaling_factor).sample
                                lr_vis = torch.clamp((lr_img + 1.0) / 2.0, 0.0, 1.0)
                        else:
                            pil = Image.open(p).convert("RGB")
                            pil = center_crop(pil, 512)
                            hr01 = pil_to_tensor_norm01(pil)
                            hr11 = norm01_to_norm11(hr01).unsqueeze(0)

                            rng = random.Random(seed_i)
                            torch_gen = torch.Generator(device="cpu").manual_seed(int(seed_i))

                            lr11 = degrade_hr_to_lr_tensor(
                                hr11.squeeze(0).cpu(),
                                args.val_degrade_mode,
                                rng,
                                torch_gen=torch_gen,
                            ).unsqueeze(0).to(device).float()

                            if args.encode_cpu:
                                if vae_cpu is None:
                                    raise ValueError("vae_cpu was not initialized for encode_cpu")
                                hr_latent = (
                                    vae_cpu.encode(hr11.cpu().float()).latent_dist.sample()
                                    * vae_cpu.config.scaling_factor
                                )
                                hr_latent = hr_latent.to(device)
                            else:
                                hr_latent = vae.encode(hr11.to(device).float()).latent_dist.sample() * vae.config.scaling_factor
                            if args.gt_decode_cpu:
                                if vae_cpu is None:
                                    raise ValueError("vae_cpu was not initialized for gt_decode_cpu")
                                hr_latent_cpu = hr_latent.detach().cpu()
                                gt_img = vae_cpu.decode(hr_latent_cpu / vae_cpu.config.scaling_factor).sample
                            else:
                                gt_img = vae.decode(hr_latent / vae.config.scaling_factor).sample
                            gt01 = torch.clamp((gt_img + 1.0) / 2.0, 0.0, 1.0)

                            pred01 = sample_sr(
                                pixart=pixart,
                                adapter=adapter,
                                vae=vae,
                                vae_cpu=vae_cpu,
                                lr_img_11=lr11,
                                y_embed=y_embed,
                                data_info=data_info,
                                cfg=cfg,
                                use_amp=use_amp,
                                dtype_pixart=dtype_pixart,
                                device=device,
                                decode_on_cpu=args.pred_decode_cpu,
                                encode_on_cpu=args.encode_cpu,
                            )
                            if args.pred_decode_cpu:
                                gt01 = gt01.detach().cpu()
                            if gt01.device != pred01.device:
                                if gt01.device.type == "cpu":
                                    pred01 = pred01.detach().cpu()
                                else:
                                    gt01 = gt01.to(pred01.device)

                            lr_vis = torch.clamp((lr11 + 1.0) / 2.0, 0.0, 1.0)

                        row = {
                            "path": p,
                            "seed": int(seed_i),
                            "data_mode": data_mode,
                            "start_mode": start_mode,
                            "injection_mode": injection_mode,
                            "control_mult": control_mult,
                            "sde_strength": sde_strength,
                        }

                        if USE_METRICS:
                            if args.metric_y:
                                py = shave_border(rgb01_to_y01(pred01), args.shave)
                                gy = shave_border(rgb01_to_y01(gt01), args.shave)
                                row["psnr"] = float(psnr(py, gy, data_range=1.0).item())
                                row["ssim"] = float(ssim(py, gy, data_range=1.0).item())
                            else:
                                row["psnr"] = float(psnr(pred01, gt01, data_range=1.0).item())
                                row["ssim"] = float(ssim(pred01, gt01, data_range=1.0).item())

                            if lpips_fn is not None:
                                pred_norm = pred01 * 2.0 - 1.0
                                gt_norm = gt01 * 2.0 - 1.0
                                row["lpips"] = float(lpips_fn(pred_norm, gt_norm).item())

                        row.update(_color_bias(pred01, gt01))
                        row["laplacian_energy"] = _laplacian_energy(pred01)
                        records.append(row)

                        if args.max_vis > 0 and idx < args.max_vis:
                            lr01 = lr_vis[0].detach().permute(1, 2, 0).cpu().numpy()
                            gt_vis = gt01[0].detach().permute(1, 2, 0).cpu().numpy()
                            pred_vis = pred01[0].detach().permute(1, 2, 0).cpu().numpy()
                            import matplotlib.pyplot as plt

                            save_path = os.path.join(
                                args.out_dir,
                                f"vis_{start_mode}_inj{injection_mode}_m{control_mult}_sde{sde_strength}_idx{idx:03d}.png",
                            )
                            plt.figure(figsize=(12, 4))
                            plt.subplot(1, 3, 1); plt.imshow(lr01); plt.title("LR"); plt.axis("off")
                            plt.subplot(1, 3, 2); plt.imshow(gt_vis); plt.title("GT"); plt.axis("off")
                            plt.subplot(1, 3, 3); plt.imshow(pred_vis); plt.title("Pred"); plt.axis("off")
                            plt.tight_layout()
                            plt.savefig(save_path, bbox_inches="tight")
                            plt.close()

                    detail_path = os.path.join(
                        args.out_dir,
                        f"detail_{start_mode}_inj{injection_mode}_m{control_mult}_sde{sde_strength}.jsonl",
                    )
                    with open(detail_path, "w", encoding="utf-8") as f:
                        for r in records:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")

                    def avg(key: str) -> float:
                        vals = [r[key] for r in records if key in r]
                        if not vals:
                            return float("nan")
                        return float(sum(vals) / len(vals))

                    summary.append(
                        {
                            "start_mode": start_mode,
                            "injection_mode": injection_mode,
                            "control_mult": control_mult,
                            "sde_strength": sde_strength,
                            "num": len(records),
                            "psnr": avg("psnr"),
                            "ssim": avg("ssim"),
                            "lpips": avg("lpips"),
                            "color_mean_abs": avg("color_mean_abs"),
                            "laplacian_energy": avg("laplacian_energy"),
                        }
                    )

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "ckpt": args.ckpt,
                "val_hr_dir": args.val_hr_dir,
                "latent_dir": args.latent_dir,
                "data_mode": data_mode,
                "val_degrade_mode": args.val_degrade_mode,
                "num_val_images": args.num_val_images,
                "start_modes": start_modes,
                "injection_modes": injection_modes,
                "control_mults": control_mults,
                "sde_strengths": sde_strengths,
                "num_steps": args.num_steps,
                "fixed_noise_seed": args.fixed_noise_seed,
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
