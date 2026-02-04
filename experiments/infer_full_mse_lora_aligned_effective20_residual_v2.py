import argparse
import glob
import hashlib
import io
import math
import os
import random
import sys
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from tqdm import tqdm

# -------------------------
# 0) Environment constraints
# -------------------------
torch.backends.cudnn.enabled = False

# -------------------------
# 1) Metrics
# -------------------------
try:
    from torchmetrics.functional import peak_signal_noise_ratio as psnr
    from torchmetrics.functional import structural_similarity_index_measure as ssim
    import lpips
    USE_METRICS = True
    print("✅ Metrics libraries loaded (PSNR, SSIM, LPIPS).")
except ImportError:
    USE_METRICS = False
    print("⚠️ Metrics missing. Install: pip install torchmetrics lpips")

# -------------------------
# 2) Paths & defaults (align with training script)
# -------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

VAL_HR_DIR = os.path.join(PROJECT_ROOT, "dataset", "DIV2K_valid_HR")
PIXART_PATH = os.path.join(PROJECT_ROOT, "output", "pretrained_models", "PixArt-XL-2-512x512.pth")
VAE_PATH = os.path.join(PROJECT_ROOT, "output", "pretrained_models", "sd-vae-ft-ema")
T5_EMBED_PATH = os.path.join(PROJECT_ROOT, "output", "quality_embed.pth")

OUT_DIR = os.path.join(PROJECT_ROOT, "experiments_results", "infer_full_mse_lora_aligned_effective20_residual_v2")
VIS_DIR = os.path.join(OUT_DIR, "vis")
os.makedirs(VIS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE_PIXART = torch.float16 if DEVICE == "cuda" else torch.float32
USE_AMP = DEVICE == "cuda"

VAL_DEGRADE_MODE = "realistic"
FIXED_NOISE_SEED = 42

METRIC_Y_CHANNEL = True
METRIC_SHAVE_BORDER = 4
RESIDUAL_MODE = True

# -------------------------
# 3) Import your model
# -------------------------
from diffusion.model.nets.PixArtMS import PixArtMS_XL_2
from diffusion.model.nets.adapter import build_adapter
from diffusion import IDDPM


# -------------------------
# 4) Validation dataset (HR -> degrade -> encode)
# -------------------------
def pil_to_tensor_norm01(pil: Image.Image) -> torch.Tensor:
    arr = np.asarray(pil, dtype=np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0


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
    return (img * 2.0) - 1.0


def center_crop(pil: Image.Image, size: int = 512) -> Image.Image:
    w, h = pil.size
    if w < size or h < size:
        pil = pil.resize((max(size, w), max(size, h)), resample=Image.BICUBIC)
        w, h = pil.size
    left = (w - size) // 2
    top = (h - size) // 2
    return pil.crop((left, top, left + size, top + size))


def degrade_hr_to_lr_tensor(
    hr11: torch.Tensor,
    mode: str,
    rng: random.Random,
    torch_gen: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if mode == "bicubic":
        hr = hr11.unsqueeze(0)
        lr_small = F.interpolate(hr, scale_factor=0.25, mode="bicubic", align_corners=False)
        lr = F.interpolate(lr_small, size=(512, 512), mode="bicubic", align_corners=False)
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


class ValImageDataset(Dataset):
    def __init__(self, hr_dir: str, max_files: Optional[int] = None):
        exts = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"]
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


def stable_int_hash(s: str, mod: int = 2**32) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % mod


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


# -------------------------
# 5) Scheduler helpers
# -------------------------
def _get_training_betas(diffusion) -> np.ndarray:
    betas = getattr(diffusion, "betas", None)
    if betas is None:
        raise AttributeError("diffusion has no attribute `betas` (required for aligned scheduler).")
    if isinstance(betas, torch.Tensor):
        betas = betas.detach().cpu().numpy()
    return np.asarray(betas).astype(np.float64)


def build_val_scheduler(diffusion, num_infer_steps: int, device: str) -> DDIMScheduler:
    trained_betas = _get_training_betas(diffusion)
    scheduler = DDIMScheduler(
        num_train_timesteps=int(trained_betas.shape[0]),
        trained_betas=trained_betas,
        clip_sample=False,
        prediction_type="epsilon",
        set_alpha_to_one=False,
    )
    scheduler.set_timesteps(int(num_infer_steps), device=device)
    return scheduler


def prepare_img2img_latents(
    scheduler: DDIMScheduler,
    lr_latent: torch.Tensor,
    strength: float,
    noise_seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    strength = float(strength)
    strength = max(0.0, min(1.0, strength))

    num_steps = int(scheduler.num_inference_steps)
    init_steps = int(num_steps * strength)
    init_steps = max(1, min(num_steps, init_steps))
    t_start_idx = num_steps - init_steps
    timesteps = scheduler.timesteps[t_start_idx:]

    g = torch.Generator(device=lr_latent.device).manual_seed(int(noise_seed))
    noise = torch.randn(lr_latent.shape, generator=g, device=lr_latent.device, dtype=lr_latent.dtype)

    t_start = timesteps[0]
    if not torch.is_tensor(t_start):
        t_start = torch.tensor(t_start, device=lr_latent.device, dtype=torch.long)
    t_batch = t_start.expand(lr_latent.shape[0])

    latents = scheduler.add_noise(lr_latent, noise, t_batch)
    return latents, timesteps


def build_text_cond(device: str, dtype: torch.dtype):
    y_embed = torch.load(T5_EMBED_PATH, map_location="cpu")["prompt_embeds"].unsqueeze(1)
    y_embed = y_embed.to(device).to(dtype)
    data_info = {
        "img_hw": torch.tensor([[512.0, 512.0]], device=device, dtype=dtype),
        "aspect_ratio": torch.tensor([1.0], device=device, dtype=dtype),
    }
    return y_embed, data_info


@torch.no_grad()
def validate_epoch(
    pixart,
    adapter,
    vae,
    val_loader,
    y_embed,
    data_info,
    diffusion,
    infer_strength: float,
    num_infer_steps: int,
    lpips_fn=None,
    max_vis: int = 1,
):
    pixart.eval()
    adapter.eval()

    scheduler = build_val_scheduler(diffusion, num_infer_steps, DEVICE)

    psnr_list, ssim_list, lpips_list = [], [], []
    vis_done = 0

    pbar = tqdm(val_loader, desc="Validate", dynamic_ncols=True)
    for it, batch in enumerate(pbar):
        hr_img_11 = batch["hr_img_11"].to(DEVICE).float()
        bsz = hr_img_11.shape[0]

        for bi in range(bsz):
            item_hr = hr_img_11[bi : bi + 1]
            path = batch["path"][bi]

            seed_i = (stable_int_hash(path) + 12345) & 0xFFFFFFFF
            rng = random.Random(seed_i)
            torch_gen = torch.Generator(device="cpu").manual_seed(seed_i)

            lr_img_11 = degrade_hr_to_lr_tensor(
                item_hr.squeeze(0).detach().cpu(),
                VAL_DEGRADE_MODE,
                rng,
                torch_gen=torch_gen,
            ).unsqueeze(0).to(DEVICE).float()

            hr_latent = vae.encode(item_hr).latent_dist.sample() * vae.config.scaling_factor
            lr_latent = vae.encode(lr_img_11).latent_dist.sample() * vae.config.scaling_factor

            with torch.cuda.amp.autocast(enabled=False):
                cond = adapter(lr_latent.float())

            latents, run_ts = prepare_img2img_latents(
                scheduler,
                torch.zeros_like(lr_latent).to(dtype=torch.float32),
                strength=infer_strength,
                noise_seed=FIXED_NOISE_SEED,
            )

            for t in run_ts:
                t_tensor = t.unsqueeze(0).to(DEVICE)
                with torch.cuda.amp.autocast(enabled=USE_AMP, dtype=DTYPE_PIXART):
                    out = pixart(
                        latents.to(DTYPE_PIXART),
                        t_tensor,
                        y_embed,
                        data_info=data_info,
                        adapter_cond=cond,
                        injection_mode="hybrid",
                    )
                    if out.shape[1] == 8:
                        out, _ = out.chunk(2, dim=1)
                latents = scheduler.step(out.float(), t, latents.float()).prev_sample

            pred_latent = latents + lr_latent
            pred_img = vae.decode(pred_latent / vae.config.scaling_factor).sample
            pred_img_01 = torch.clamp((pred_img + 1.0) / 2.0, 0.0, 1.0)

            gt_img = vae.decode(hr_latent / vae.config.scaling_factor).sample
            gt_img_01 = torch.clamp((gt_img + 1.0) / 2.0, 0.0, 1.0)

            if USE_METRICS:
                if METRIC_Y_CHANNEL:
                    pred_y = rgb01_to_y01(pred_img_01)
                    gt_y = rgb01_to_y01(gt_img_01)
                    pred_y = shave_border(pred_y, METRIC_SHAVE_BORDER)
                    gt_y = shave_border(gt_y, METRIC_SHAVE_BORDER)
                    p = psnr(pred_y, gt_y, data_range=1.0).item()
                    s = ssim(pred_y, gt_y, data_range=1.0).item()
                else:
                    p = psnr(pred_img_01, gt_img_01, data_range=1.0).item()
                    s = ssim(pred_img_01, gt_img_01, data_range=1.0).item()

                psnr_list.append(p)
                ssim_list.append(s)

                if lpips_fn is not None:
                    pred_norm = pred_img_01 * 2.0 - 1.0
                    gt_norm = gt_img_01 * 2.0 - 1.0
                    l = lpips_fn(pred_norm, gt_norm).item()
                    lpips_list.append(l)

            if vis_done < max_vis:
                save_path = os.path.join(VIS_DIR, f"val_it{it:04d}.png")
                lr_np = torch.clamp((lr_img_11 + 1.0) / 2.0, 0.0, 1.0)[0].permute(1, 2, 0).cpu().numpy()
                gt_np = gt_img_01[0].permute(1, 2, 0).cpu().numpy()
                pr_np = pred_img_01[0].permute(1, 2, 0).cpu().numpy()

                import matplotlib.pyplot as plt

                plt.figure(figsize=(12, 4))
                plt.subplot(1, 3, 1)
                plt.imshow(lr_np)
                plt.title("LR")
                plt.axis("off")
                plt.subplot(1, 3, 2)
                plt.imshow(gt_np)
                plt.title("GT")
                plt.axis("off")
                plt.subplot(1, 3, 3)
                plt.imshow(pr_np)
                plt.title("Pred")
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(save_path, bbox_inches="tight")
                plt.close()
                vis_done += 1

        if USE_METRICS and psnr_list:
            pbar.set_postfix(
                {
                    "PSNR": f"{sum(psnr_list) / len(psnr_list):.2f}",
                    "SSIM": f"{sum(ssim_list) / len(ssim_list):.4f}",
                    "LPIPS": f"{(sum(lpips_list) / len(lpips_list)):.4f}" if lpips_list else "NA",
                }
            )

    avg_psnr = sum(psnr_list) / len(psnr_list) if psnr_list else float("nan")
    avg_ssim = sum(ssim_list) / len(ssim_list) if ssim_list else float("nan")
    avg_lp = sum(lpips_list) / len(lpips_list) if lpips_list else float("nan")
    return avg_psnr, avg_ssim, avg_lp


def load_inject_state_dict(pixart, inject_sd):
    sd = pixart.state_dict()
    for k, v in inject_sd.items():
        if k in sd:
            sd[k] = v.to(sd[k].dtype)
    pixart.load_state_dict(sd, strict=False)


def align_inference_dtypes(pixart):
    if hasattr(pixart, "injection_scales"):
        for s in pixart.injection_scales:
            s.data = s.data.float()
    if hasattr(pixart, "cross_attn_scale"):
        pixart.cross_attn_scale.data = pixart.cross_attn_scale.data.float()
    if hasattr(pixart, "adapter_proj"):
        pixart.adapter_proj = pixart.adapter_proj.to(torch.float32)
    if hasattr(pixart, "adapter_norm"):
        pixart.adapter_norm = pixart.adapter_norm.to(torch.float32)
    if hasattr(pixart, "input_adapter_ln"):
        pixart.input_adapter_ln = pixart.input_adapter_ln.to(torch.float32)
    if hasattr(pixart, "input_adaln"):
        pixart.input_adaln = pixart.input_adaln.to(torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="path to adapter+pixart_inject checkpoint")
    parser.add_argument("--adapter_type", type=str, default="fpn_hf", choices=["fpn", "fpn_se", "fpn_hf"])
    parser.add_argument("--val_dir", type=str, default=VAL_HR_DIR)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_infer_steps", type=int, default=40)
    parser.add_argument("--infer_strength", type=float, default=0.5)
    parser.add_argument("--max_vis", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"DEVICE={DEVICE} | AMP={USE_AMP} | cudnn.enabled={torch.backends.cudnn.enabled}")
    print(f"[Config] num_infer_steps={args.num_infer_steps} | infer_strength={args.infer_strength}")
    print(f"[Config] residual_mode={RESIDUAL_MODE}")

    val_ds = ValImageDataset(args.val_dir, max_files=args.max_samples)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=(DEVICE == "cuda"))

    print("Loading PixArt...")
    pixart = PixArtMS_XL_2(input_size=64).to(DEVICE).to(DTYPE_PIXART).eval()
    ckpt = torch.load(PIXART_PATH, map_location="cpu")
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if "pos_embed" in ckpt:
        del ckpt["pos_embed"]
    pixart.load_state_dict(ckpt, strict=False)

    print("Loading Adapter...")
    adapter = build_adapter(args.adapter_type, in_channels=4, hidden_size=1152).to(DEVICE).eval()

    print(f"Loading checkpoint: {args.ckpt}")
    payload = torch.load(args.ckpt, map_location="cpu")
    if "adapter" not in payload or "pixart_inject" not in payload:
        raise KeyError("Checkpoint missing adapter/pixart_inject keys.")
    adapter.load_state_dict(payload["adapter"], strict=True)
    load_inject_state_dict(pixart, payload["pixart_inject"])
    align_inference_dtypes(pixart)

    diffusion = IDDPM(str(1000))
    vae = AutoencoderKL.from_pretrained(VAE_PATH, local_files_only=True).to(DEVICE).float().eval()
    vae.enable_slicing()

    y_embed, data_info = build_text_cond(DEVICE, DTYPE_PIXART)

    lpips_fn = None
    if USE_METRICS:
        lpips_fn = lpips.LPIPS(net="vgg").to(DEVICE).eval()
        for p in lpips_fn.parameters():
            p.requires_grad = False

    vpsnr, vssim, vlp = validate_epoch(
        pixart,
        adapter,
        vae,
        val_loader,
        y_embed,
        data_info,
        diffusion,
        infer_strength=args.infer_strength,
        num_infer_steps=args.num_infer_steps,
        lpips_fn=lpips_fn,
        max_vis=args.max_vis,
    )
    print(f"[VAL] PSNR={vpsnr:.2f} SSIM={vssim:.4f} LPIPS={vlp if math.isfinite(vlp) else float('nan'):.4f}")


if __name__ == "__main__":
    main()
