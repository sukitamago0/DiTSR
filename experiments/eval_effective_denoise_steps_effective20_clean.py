
# experiments/eval_effective_denoise_steps_effective20_clean.py
# Purpose:
#   Evaluate an existing checkpoint with a fixed "effective denoise steps" budget.
#   If infer_strength=0.5 and effective_denoise_steps=20, we set total_steps=40 so that
#   init_steps=int(total_steps*strength)=20, and the remaining denoise steps are 20.
#
# Baseline alignment:
#   This script reuses the SAME validation logic as experiments/train_full_mse_lora_aligned_strength05_fixdevice.py:
#   - same HR->LR degradation
#   - same VAE encode/decode pipeline
#   - same aligned betas via diffusion.IDDPM.betas -> diffusers DDIMScheduler(trained_betas=...)
#   - same adapter_cond inference (autocast disabled => fp32)
#   - same PixArt injection fp32 constraints (LN/proj/scales)
#
# Hardware:
#   For 8GB GPUs, use --vae_on_cpu --lpips_on_cpu to avoid OOM.

import os
import sys
import glob
import io
import math
import random
import hashlib
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import numpy as np
from torchvision.transforms import functional as TF

from diffusers import AutoencoderKL, DDIMScheduler

# -------------------------
# 0) Environment hard constraint (same as training)
# -------------------------
torch.backends.cudnn.enabled = False

# -------------------------
# 1) Metrics (same style as training)
# -------------------------
try:
    from torchmetrics.functional import peak_signal_noise_ratio as tm_psnr
    from torchmetrics.functional import structural_similarity_index_measure as tm_ssim
    import lpips as lpips_lib
    USE_METRICS = True
    print("✅ Metrics libraries loaded (PSNR, SSIM, LPIPS).")
except Exception:
    USE_METRICS = False
    lpips_lib = None
    print("⚠️ Metrics missing. Install: pip install torchmetrics lpips")

# -------------------------
# 2) Paths (match training defaults)
# -------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

PIXART_PATH = os.path.join(PROJECT_ROOT, "output", "pretrained_models", "PixArt-XL-2-512x512.pth")
VAE_PATH    = os.path.join(PROJECT_ROOT, "output", "pretrained_models", "sd-vae-ft-ema")
T5_EMBED_PATH = os.path.join(PROJECT_ROOT, "output", "quality_embed.pth")

# -------------------------
# 3) Import repo modules (MUST match training repo)
# -------------------------
try:
    from diffusion.model.nets.PixArtMS import PixArtMS_XL_2
    from diffusion.model.nets.adapter import build_adapter
    from diffusion import IDDPM
except Exception as e:
    raise ImportError(f"❌ Import failed. This script must run inside PixArt-alpha repo. err={e}") from e

# -------------------------
# 4) Utilities copied from training baseline
# -------------------------
def stable_int_hash(s: str, mod: int = 2**32) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h, 16) % mod

def pil_to_tensor_norm01(pil: Image.Image) -> torch.Tensor:
    # Make array writable to avoid the "not writable" warning
    arr = np.asarray(pil, dtype=np.uint8).copy()  # [H,W,3]
    x = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return x

def norm01_to_norm11(x01: torch.Tensor) -> torch.Tensor:
    return x01 * 2.0 - 1.0

def _jpeg_compress_tensor(x11: torch.Tensor, quality: int) -> torch.Tensor:
    img = x11.clamp(-1.0, 1.0)
    img = (img + 1.0) / 2.0
    img = TF.to_pil_image(img)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    img = Image.open(buffer).convert("RGB")
    img = TF.to_tensor(img)
    img = (img * 2.0) - 1.0
    return img

def center_crop(pil: Image.Image, size: int = 512) -> Image.Image:
    w, h = pil.size
    if w < size or h < size:
        pil = pil.resize((max(size, w), max(size, h)), resample=Image.BICUBIC)
        w, h = pil.size
    left = (w - size) // 2
    top  = (h - size) // 2
    return pil.crop((left, top, left + size, top + size))

def degrade_hr_to_lr_tensor(
    hr11: torch.Tensor,
    mode: str,
    rng: random.Random,
    torch_gen: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    hr11: [3,512,512] in [-1,1] (cpu)
    return lr11: [3,512,512] in [-1,1] (cpu)
    """
    if mode == "bicubic":
        hr = hr11.unsqueeze(0)
        lr_small = F.interpolate(hr, scale_factor=0.25, mode="bicubic", align_corners=False)
        lr = F.interpolate(lr_small, size=(512,512), mode="bicubic", align_corners=False)
        return lr.squeeze(0)

    blur_k = rng.choice([3, 5, 7])
    blur_sigma = rng.uniform(0.2, 1.2)

    hr = hr11.unsqueeze(0)
    hr_blur = TF.gaussian_blur(hr.squeeze(0), blur_k, [blur_sigma, blur_sigma]).unsqueeze(0)

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
    lr_small_cpu = lr_small.squeeze(0).cpu()
    lr_small_cpu = _jpeg_compress_tensor(lr_small_cpu, jpeg_q).unsqueeze(0)

    lr = F.interpolate(lr_small_cpu, size=(512, 512), mode="bicubic", align_corners=False)
    return lr.squeeze(0)

def rgb01_to_y01(rgb01: torch.Tensor) -> torch.Tensor:
    # rgb01: [1,3,H,W] in [0,1]
    r = rgb01[:, 0:1]
    g = rgb01[:, 1:2]
    b = rgb01[:, 2:3]
    y = (16.0 + 65.481 * r + 128.553 * g + 24.966 * b) / 255.0
    return y.clamp(0.0, 1.0)

def shave_border(x: torch.Tensor, shave: int) -> torch.Tensor:
    if shave <= 0:
        return x
    return x[..., shave:-shave, shave:-shave]

def build_text_cond(device: str, dtype_pixart: torch.dtype):
    y_embed = torch.load(T5_EMBED_PATH, map_location="cpu")["prompt_embeds"].unsqueeze(1)
    y_embed = y_embed.to(device).to(dtype_pixart)
    data_info = {
        "img_hw": torch.tensor([[512.,512.]], device=device, dtype=dtype_pixart),
        "aspect_ratio": torch.tensor([1.], device=device, dtype=dtype_pixart),
    }
    return y_embed, data_info

# -------------------------
# 5) Checkpoint helpers (aligned with training)
# -------------------------
def extract_inject_state_dict(pixart) -> Dict[str, torch.Tensor]:
    sd = pixart.state_dict()
    keep = {}
    for k,v in sd.items():
        if (
            k.startswith("injection_scales")
            or k.startswith("adapter_proj")
            or k.startswith("adapter_norm")
            or k.startswith("cross_attn_scale")
            or k.startswith("input_adapter_ln")
            or k.startswith("input_adaln")
        ):
            keep[k] = v.detach().cpu()
    return keep

def load_inject_state_dict(pixart, inject_sd: Dict[str, torch.Tensor]):
    sd = pixart.state_dict()
    for k, v in inject_sd.items():
        if k in sd:
            sd[k] = v.to(sd[k].dtype)
    pixart.load_state_dict(sd, strict=False)

def enforce_fp32_injection_modules(pixart):
    # Mirror the training constraint: these modules MUST stay fp32 to avoid LN dtype mismatch.
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

# -------------------------
# 6) Schedule alignment (same as training baseline)
# -------------------------
def _get_training_betas(diffusion) -> np.ndarray:
    betas = getattr(diffusion, "betas", None)
    if betas is None:
        raise AttributeError("diffusion has no attribute `betas` (required for aligned scheduler).")
    if isinstance(betas, torch.Tensor):
        betas = betas.detach().cpu().numpy()
    else:
        betas = np.asarray(betas)
    betas = betas.astype(np.float64)
    return betas

def build_val_scheduler(diffusion, num_infer_steps: int, device: str) -> DDIMScheduler:
    trained_betas = _get_training_betas(diffusion)
    scheduler = DDIMScheduler(
        num_train_timesteps=int(trained_betas.shape[0]),
        trained_betas=trained_betas,
        clip_sample=False,
        set_alpha_to_one=False,
        prediction_type="epsilon",
    )
    scheduler.set_timesteps(int(num_infer_steps), device=device)
    return scheduler

def prepare_img2img_latents(
    scheduler: DDIMScheduler,
    lr_latent: torch.Tensor,
    strength: float,
    noise_seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    lr_latent: [B,4,64,64], acts as the coarse init (already aligned shape).
    strength in [0,1]: higher => more noise => more freedom.
    Returns: (latents, timesteps_to_run)
    """
    strength = float(strength)
    strength = max(0.0, min(1.0, strength))

    num_steps = len(scheduler.timesteps)
    init_steps = int(num_steps * strength)
    init_steps = min(max(init_steps, 0), num_steps)

    # timesteps are descending
    timesteps_to_run = scheduler.timesteps[init_steps:]

    g = torch.Generator(device=lr_latent.device).manual_seed(int(noise_seed))
    noise = torch.randn(lr_latent.shape, generator=g, device=lr_latent.device, dtype=lr_latent.dtype)

    # add_noise expects timestep as (B,)
    if init_steps == 0:
        latents = lr_latent
    else:
        t0 = scheduler.timesteps[init_steps - 1].expand(lr_latent.shape[0]).to(lr_latent.device)
        latents = scheduler.add_noise(lr_latent, noise, t0)

    return latents, timesteps_to_run

# -------------------------
# 7) Validation dataset (same as training baseline)
# -------------------------
class ValImageDataset(Dataset):
    def __init__(self, hr_dir: str, max_files: Optional[int] = None):
        exts = ["*.png","*.jpg","*.jpeg","*.PNG","*.JPG"]
        paths = []
        for e in exts:
            paths += glob.glob(os.path.join(hr_dir, e))
        self.paths = sorted(list(set(paths)))
        if max_files is not None:
            self.paths = self.paths[:max_files]
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No HR images found in: {hr_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        pil = Image.open(p).convert("RGB")
        pil = center_crop(pil, 512)
        hr01 = pil_to_tensor_norm01(pil)
        hr11 = norm01_to_norm11(hr01)
        return {"hr_img_11": hr11, "path": p}

# -------------------------
# 8) Main eval
# -------------------------
@torch.no_grad()
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None, choices=["cuda","cpu"])
    parser.add_argument("--ckpt", type=str, required=True, help="path to last_full_state.pth or epochXXX_*.pth")
    parser.add_argument("--val_hr_dir", type=str, required=True)
    parser.add_argument("--val_degrade_mode", type=str, default="realistic", choices=["realistic","bicubic"])
    parser.add_argument("--max_val", type=int, default=1)
    parser.add_argument("--infer_strength", type=float, default=0.5)
    parser.add_argument("--effective_denoise_steps", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--metric_y", action="store_true")
    parser.add_argument("--metric_shave", type=int, default=4)
    parser.add_argument("--vae_on_cpu", action="store_true")
    parser.add_argument("--lpips_on_cpu", action="store_true")
    parser.add_argument("--no_lpips", action="store_true")
    parser.add_argument("--num_vis", type=int, default=1)
    parser.add_argument("--out_dir", type=str, default=os.path.join(PROJECT_ROOT, "experiments_results", "eval_effective_steps"))
    args = parser.parse_args()

    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device == "cuda")
    dtype_pixart = torch.float16 if device == "cuda" else torch.float32

    strength = float(args.infer_strength)
    eff_steps = int(args.effective_denoise_steps)
    # total_steps so that remaining denoise steps ≈ eff_steps
    total_steps = int(math.ceil(eff_steps / max(strength, 1e-8)))
    total_steps = max(total_steps, eff_steps)

    print(f"[Env] device={device} cudnn.enabled={torch.backends.cudnn.enabled}")
    print(f"[EvalCfg] strength={strength} effective_steps={eff_steps} -> total_steps={total_steps}")
    print(f"[MemCfg] vae_on_cpu={bool(args.vae_on_cpu)} lpips_on_cpu={bool(args.lpips_on_cpu)} no_lpips={bool(args.no_lpips)}")

    os.makedirs(args.out_dir, exist_ok=True)
    vis_dir = os.path.join(args.out_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    # Load ckpt
    ckpt = torch.load(args.ckpt, map_location="cpu")
    # Training baseline saves as last_full_state with these keys
    if ("adapter" not in ckpt) or ("pixart_inject" not in ckpt):
        raise KeyError("Checkpoint must contain keys: 'adapter' and 'pixart_inject' (expected last_full_state.pth).")

    # Detect adapter type from state dict (avoid fpn vs fpn_se mismatch)
    adapter_sd = ckpt["adapter"]
    has_se = any(k.startswith("se0.") or k.startswith("se1.") or k.startswith("se2.") or k.startswith("se3.") for k in adapter_sd.keys())
    adapter_type = "fpn_se" if has_se else "fpn"
    print(f"[Adapter] detected adapter_type={adapter_type} (has_se={has_se})")

    # Build models
    print("Loading PixArt...")
    pixart = PixArtMS_XL_2(input_size=64).to(device).to(dtype_pixart).eval()
    base = torch.load(PIXART_PATH, map_location="cpu")
    if "state_dict" in base:
        base = base["state_dict"]
    if "pos_embed" in base:
        del base["pos_embed"]
    pixart.load_state_dict(base, strict=False)

    # Load inject weights from ckpt
    load_inject_state_dict(pixart, ckpt["pixart_inject"])
    enforce_fp32_injection_modules(pixart)

    print("Loading Adapter...")
    adapter = build_adapter(adapter_type, in_channels=4, hidden_size=1152).to(device).eval()
    adapter.load_state_dict(adapter_sd, strict=True)

    # VAE / LPIPS placement
    vae_device = "cpu" if args.vae_on_cpu else device
    vae = AutoencoderKL.from_pretrained(VAE_PATH, local_files_only=True).to(vae_device).float().eval()
    vae.enable_slicing()

    lpips_fn = None
    if USE_METRICS and (not args.no_lpips):
        lpips_device = "cpu" if args.lpips_on_cpu else device
        lpips_fn = lpips_lib.LPIPS(net="vgg").to(lpips_device).eval()
        for p in lpips_fn.parameters():
            p.requires_grad = False

    # Text condition
    y_embed, data_info = build_text_cond(device=device, dtype_pixart=dtype_pixart)

    # Scheduler aligned to training betas
    diffusion = IDDPM(str(1000))
    scheduler = build_val_scheduler(diffusion, num_infer_steps=total_steps, device=device)

    # Data
    val_ds = ValImageDataset(args.val_hr_dir, max_files=int(args.max_val) if args.max_val > 0 else None)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=int(args.num_workers), pin_memory=(device=="cuda"))

    psnr_list, ssim_list, lpips_list = [], [], []
    vis_done = 0

    pbar = tqdm(val_loader, desc="Val", dynamic_ncols=True)
    for it, batch in enumerate(pbar):
        hr_img_11_cpu = batch["hr_img_11"][0].float().cpu()  # [3,512,512] CPU
        path = batch["path"][0]

        seed_i = (stable_int_hash(path) + 12345) & 0xFFFFFFFF
        rng = random.Random(seed_i)
        torch_gen = torch.Generator(device="cpu").manual_seed(seed_i)

        lr_img_11_cpu = degrade_hr_to_lr_tensor(hr_img_11_cpu, args.val_degrade_mode, rng, torch_gen=torch_gen)  # CPU
        # Encode HR/LR with VAE
        if vae_device == "cpu":
            hr_in = hr_img_11_cpu.unsqueeze(0).to("cpu")
            lr_in = lr_img_11_cpu.unsqueeze(0).to("cpu")
            hr_latent = vae.encode(hr_in).latent_dist.sample() * vae.config.scaling_factor
            lr_latent = vae.encode(lr_in).latent_dist.sample() * vae.config.scaling_factor
            hr_latent = hr_latent.to(device).to(torch.float32)
            lr_latent = lr_latent.to(device).to(torch.float32)
        else:
            hr_in = hr_img_11_cpu.unsqueeze(0).to(device)
            lr_in = lr_img_11_cpu.unsqueeze(0).to(device)
            hr_latent = vae.encode(hr_in).latent_dist.sample() * vae.config.scaling_factor
            lr_latent = vae.encode(lr_in).latent_dist.sample() * vae.config.scaling_factor
            hr_latent = hr_latent.to(torch.float32)
            lr_latent = lr_latent.to(torch.float32)

        # Adapter cond in fp32, autocast disabled
        with torch.cuda.amp.autocast(enabled=False):
            cond = adapter(lr_latent.float())

        # Prepare img2img start (same as training)
        latents, timesteps_to_run = prepare_img2img_latents(
            scheduler=scheduler,
            lr_latent=lr_latent,
            strength=strength,
            noise_seed=42,
        )

        # Denoise
        for t in timesteps_to_run:
            t_tensor = t.unsqueeze(0).to(device)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=dtype_pixart):
                out = pixart(
                    latents.to(dtype_pixart),
                    t_tensor,
                    y_embed,
                    data_info=data_info,
                    adapter_cond=cond,
                    injection_mode="hybrid",
                )
                if out.shape[1] == 8:
                    out, _ = out.chunk(2, dim=1)
            latents = scheduler.step(out.float(), t, latents.float()).prev_sample

        # Decode pred / gt
        if vae_device == "cpu":
            pred = vae.decode(latents.detach().cpu() / vae.config.scaling_factor).sample
            gt   = vae.decode(hr_latent.detach().cpu() / vae.config.scaling_factor).sample
            lr_img_11 = lr_img_11_cpu.unsqueeze(0)
        else:
            pred = vae.decode(latents / vae.config.scaling_factor).sample.detach().cpu()
            gt   = vae.decode(hr_latent / vae.config.scaling_factor).sample.detach().cpu()
            lr_img_11 = lr_img_11_cpu.unsqueeze(0)

        pred01 = torch.clamp((pred + 1.0) / 2.0, 0.0, 1.0)
        gt01   = torch.clamp((gt   + 1.0) / 2.0, 0.0, 1.0)
        lr01   = torch.clamp((lr_img_11 + 1.0) / 2.0, 0.0, 1.0)

        # Metrics on CPU
        if USE_METRICS:
            if args.metric_y:
                pred_y = shave_border(rgb01_to_y01(pred01), int(args.metric_shave))
                gt_y   = shave_border(rgb01_to_y01(gt01),   int(args.metric_shave))
                p = float(tm_psnr(pred_y, gt_y, data_range=1.0).item())
                s = float(tm_ssim(pred_y, gt_y, data_range=1.0).item())
            else:
                p = float(tm_psnr(pred01, gt01, data_range=1.0).item())
                s = float(tm_ssim(pred01, gt01, data_range=1.0).item())
            psnr_list.append(p)
            ssim_list.append(s)

            if (lpips_fn is not None):
                pred_norm = pred01 * 2.0 - 1.0
                gt_norm   = gt01   * 2.0 - 1.0
                # lpips on selected device
                lpips_dev = next(lpips_fn.parameters()).device
                l = float(lpips_fn(pred_norm.to(lpips_dev), gt_norm.to(lpips_dev)).detach().cpu().item())
                lpips_list.append(l)

        # Visualization
        if vis_done < int(args.num_vis):
            save_path = os.path.join(vis_dir, f"it{it:04d}.png")
            lr_np = lr01[0].permute(1,2,0).numpy()
            gt_np = gt01[0].permute(1,2,0).numpy()
            pr_np = pred01[0].permute(1,2,0).numpy()
            plt.figure(figsize=(12,4))
            plt.subplot(1,3,1); plt.imshow(lr_np); plt.title("LR"); plt.axis("off")
            plt.subplot(1,3,2); plt.imshow(pr_np); plt.title("Pred"); plt.axis("off")
            plt.subplot(1,3,3); plt.imshow(gt_np); plt.title("GT"); plt.axis("off")
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
            vis_done += 1

        if USE_METRICS and len(psnr_list) > 0:
            pbar.set_postfix({
                "PSNR": f"{sum(psnr_list)/len(psnr_list):.2f}",
                "SSIM": f"{sum(ssim_list)/len(ssim_list):.4f}",
                "LPIPS": f"{(sum(lpips_list)/len(lpips_list)):.4f}" if len(lpips_list)>0 else "NA"
            })

    avg_psnr = sum(psnr_list)/len(psnr_list) if len(psnr_list)>0 else float("nan")
    avg_ssim = sum(ssim_list)/len(ssim_list) if len(ssim_list)>0 else float("nan")
    avg_lp   = sum(lpips_list)/len(lpips_list) if len(lpips_list)>0 else float("nan")
    print(f"[VAL] PSNR={avg_psnr:.2f} SSIM={avg_ssim:.4f} LPIPS={avg_lp if math.isfinite(avg_lp) else float('nan'):.4f}")
    print(f"[OUT] vis_dir={vis_dir}")

if __name__ == "__main__":
    main()
