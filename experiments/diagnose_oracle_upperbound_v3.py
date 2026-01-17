import os
import sys
import json
import argparse
import hashlib
import random
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

# =========================================================
# Project root & sys.path (match experiments/train_full_mse*.py)
# =========================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

# 与训练脚本保持一致：禁用 cudnn (避免非确定性/某些 shape 的 cudnn bug)
torch.backends.cudnn.enabled = False

# =========================================================
# Paths (match experiments/train_full_mse*.py defaults)
# =========================================================
PIXART_PATH = os.path.join(PROJECT_ROOT, "output/pretrained_models/PixArt-XL-2-512x512.pth")
VAE_PATH = os.path.join(PROJECT_ROOT, "output/pretrained_models/sd-vae-ft-ema")
T5_EMBED_PATH = os.path.join(PROJECT_ROOT, "output/quality_embed.pth")

# =========================================================
# Optional metrics
# =========================================================
try:
    from skimage.metrics import peak_signal_noise_ratio as _psnr
    from skimage.metrics import structural_similarity as _ssim
except Exception:
    _psnr = None
    _ssim = None

try:
    import lpips as _lpips
except Exception:
    _lpips = None

# =========================================================
# Helpers (copied from train_full_mse.py to keep behavior identical)
# =========================================================

def pil_to_tensor_norm01(pil_rgb: Image.Image) -> torch.Tensor:
    arr = np.asarray(pil_rgb, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def center_crop(pil_img: Image.Image, size: int) -> Image.Image:
    w, h = pil_img.size
    if min(w, h) < size:
        scale = size / min(w, h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        pil_img = pil_img.resize((new_w, new_h), Image.BICUBIC)
        w, h = pil_img.size
    left = (w - size) // 2
    top = (h - size) // 2
    return pil_img.crop((left, top, left + size, top + size))


def _jpeg_compress_tensor(x01: torch.Tensor, quality: int) -> torch.Tensor:
    from io import BytesIO

    x = (x01.clamp(0, 1) * 255.0).round().to(torch.uint8)
    x = x.permute(1, 2, 0).cpu().numpy()
    img = Image.fromarray(x)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    img2 = Image.open(buf).convert("RGB")
    arr = np.asarray(img2, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _gaussian_blur_tensor(x01: torch.Tensor, sigma: float) -> torch.Tensor:
    k = int(max(3, round(sigma * 6)))
    if k % 2 == 0:
        k += 1
    coords = torch.arange(k) - k // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    g2d = g[:, None] * g[None, :]
    g2d = g2d.to(x01.device, dtype=x01.dtype)
    g2d = g2d[None, None, :, :]
    x = x01.unsqueeze(0)
    x = F.pad(x, (k // 2, k // 2, k // 2, k // 2), mode="reflect")
    out = torch.zeros_like(x)
    for c in range(3):
        out[:, c : c + 1] = F.conv2d(x[:, c : c + 1], g2d)
    return out.squeeze(0)


def degrade_hr_to_lr_tensor(
    hr11: torch.Tensor,
    mode: str,
    rng_py: random.Random,
    torch_gen: torch.Generator,
    target_size: int = 128,
) -> torch.Tensor:
    hr01 = (hr11 + 1) / 2
    if mode == "bicubic":
        lr = F.interpolate(hr01.unsqueeze(0), size=(target_size, target_size), mode="bicubic", align_corners=False)
        return lr.squeeze(0).clamp(0, 1)

    # realistic degradation: downsample + blur + jpeg + noise
    lr = F.interpolate(hr01.unsqueeze(0), size=(target_size, target_size), mode="bicubic", align_corners=False).squeeze(0)

    if rng_py.random() < 0.8:
        sigma = rng_py.uniform(0.2, 1.5)
        lr = _gaussian_blur_tensor(lr, sigma)

    if rng_py.random() < 0.8:
        q = rng_py.randint(30, 95)
        lr = _jpeg_compress_tensor(lr, q)

    if rng_py.random() < 0.7:
        noise_std = rng_py.uniform(0.0, 0.03)
        noise = torch.randn(lr.shape, generator=torch_gen) * noise_std
        lr = (lr + noise).clamp(0, 1)

    return lr


def stable_int_hash(s: str) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def rgb01_to_y01(rgb01: np.ndarray) -> np.ndarray:
    # ITU-R BT.601
    r = rgb01[..., 0]
    g = rgb01[..., 1]
    b = rgb01[..., 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)


def shave_border(img: np.ndarray, shave: int) -> np.ndarray:
    if shave <= 0:
        return img
    return img[shave:-shave, shave:-shave]


def compute_metrics(
    pred01: np.ndarray,
    gt01: np.ndarray,
    metric_y: bool,
    shave: int,
    lpips_fn: Optional[Any],
) -> Tuple[float, float, Optional[float]]:
    pred01 = np.clip(pred01, 0.0, 1.0)
    gt01 = np.clip(gt01, 0.0, 1.0)

    pred01 = shave_border(pred01, shave)
    gt01 = shave_border(gt01, shave)

    if metric_y:
        pred_m = rgb01_to_y01(pred01)
        gt_m = rgb01_to_y01(gt01)
        data_range = 1.0
        psnr = float(_psnr(gt_m, pred_m, data_range=data_range)) if _psnr else float("nan")
        ssim = float(_ssim(gt_m, pred_m, data_range=data_range)) if _ssim else float("nan")
    else:
        data_range = 1.0
        psnr = float(_psnr(gt01, pred01, data_range=data_range)) if _psnr else float("nan")
        ssim = float(_ssim(gt01, pred01, channel_axis=2, data_range=data_range)) if _ssim else float("nan")

    lp = None
    if lpips_fn is not None:
        # lpips expects [-1,1]
        t_pred = torch.from_numpy(pred01).permute(2, 0, 1).unsqueeze(0) * 2 - 1
        t_gt = torch.from_numpy(gt01).permute(2, 0, 1).unsqueeze(0) * 2 - 1
        with torch.no_grad():
            lp = float(lpips_fn(t_pred.to(lpips_fn.net[0].weight.device), t_gt.to(lpips_fn.net[0].weight.device)).item())

    return psnr, ssim, lp


def load_inject_state_dict(pixart: torch.nn.Module, inject_sd: Dict[str, torch.Tensor]) -> None:
    # train_full_mse.py 的做法：只加载注入相关 key
    model_sd = pixart.state_dict()
    filtered = {k: v for k, v in inject_sd.items() if k in model_sd}
    pixart.load_state_dict(filtered, strict=False)


def force_fp32_injection_modules(pixart: torch.nn.Module) -> None:
    # 复用 train_full_mse_adaln.py 的 dtype 约定：注入相关模块保持 fp32
    if hasattr(pixart, "injection_scales"):
        for p in pixart.injection_scales:
            p.data = p.data.float()
    if hasattr(pixart, "cross_attn_scale"):
        pixart.cross_attn_scale.data = pixart.cross_attn_scale.data.float()

    for name in ["input_adapter_ln", "adapter_proj", "adapter_norm"]:
        if hasattr(pixart, name):
            getattr(pixart, name).to(dtype=torch.float32)

    # AdaLN/FILM 版本新增的模块
    if hasattr(pixart, "input_adaln"):
        for m in pixart.input_adaln:
            m.to(dtype=torch.float32)


def load_text_cond(device: torch.device) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # === 关键：quality_embed.pth 是 dict，不是 Tensor ===
    bundle = torch.load(T5_EMBED_PATH, map_location="cpu")
    if isinstance(bundle, dict) and "prompt_embeds" in bundle:
        prompt_embeds = bundle["prompt_embeds"]
    elif torch.is_tensor(bundle):
        prompt_embeds = bundle
    else:
        raise ValueError(f"Unexpected T5 embed format: {type(bundle)} keys={list(bundle.keys()) if isinstance(bundle, dict) else None}")

    # 训练脚本：unsqueeze(1)
    y_embed = prompt_embeds.unsqueeze(1).to(device)

    img_hw = torch.tensor([[512.0, 512.0]], device=device)
    aspect_ratio = torch.tensor([[1.0]], device=device)
    data_info = {"img_hw": img_hw, "aspect_ratio": aspect_ratio}

    return y_embed, data_info


def encode_latent(vae, img01_bchw: torch.Tensor) -> torch.Tensor:
    # img01 -> img11
    img11 = img01_bchw * 2 - 1
    posterior = vae.encode(img11).latent_dist
    lat = posterior.sample() * vae.config.scaling_factor
    return lat


def decode_latent(vae, lat: torch.Tensor) -> torch.Tensor:
    img11 = vae.decode(lat / vae.config.scaling_factor).sample
    img01 = (img11 + 1) / 2
    return img01.clamp(0, 1)


@torch.no_grad()
def sample_sr_dpm(
    pixart,
    adapter,
    lr_latent: torch.Tensor,
    y_embed: torch.Tensor,
    data_info: Dict[str, torch.Tensor],
    cond_latent_for_adapter: torch.Tensor,
    *,
    num_steps: int,
    sde_strength: float,
    fixed_noise_seed: int,
    dtype_pixart: torch.dtype,
    use_amp: bool,
    injection_mode: str,
) -> torch.Tensor:
    from diffusers import DPMSolverMultistepScheduler

    scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    scheduler.set_timesteps(num_steps, device=lr_latent.device)

    start_t_idx = int(len(scheduler.timesteps) * sde_strength)
    start_t_idx = min(max(start_t_idx, 0), len(scheduler.timesteps) - 1)
    start_t_val = scheduler.timesteps[start_t_idx]

    g = torch.Generator(device=lr_latent.device)
    g.manual_seed(int(fixed_noise_seed))
    # [FIX] 你的 torch 版本对 randn_like(generator=...) 不支持（会报 unexpected keyword 'generator'）。
    # 参照训练脚本：用 torch.randn(shape, generator=..., device=..., dtype=...) 生成可复现噪声。
    noise = torch.randn(
        lr_latent.shape,
        generator=g,
        device=lr_latent.device,
        dtype=lr_latent.dtype,
    )
    latents = scheduler.add_noise(lr_latent, noise, start_t_val)

    # === 条件 ===
    with torch.cuda.amp.autocast(enabled=False):
        cond = adapter(cond_latent_for_adapter.float())

    run_ts = [t for t in scheduler.timesteps if t <= start_t_val]

    for t in run_ts:
        t_in = torch.tensor([t], device=lr_latent.device).long()
        with torch.cuda.amp.autocast(enabled=use_amp):
            out = pixart(
                latents.to(dtype_pixart),
                t_in,
                y_embed,
                data_info=data_info,
                adapter_cond=cond,
                injection_mode=injection_mode,
            )

        latents = scheduler.step(out.float(), t, latents.float()).prev_sample

    return latents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="last_full_state.pth")
    parser.add_argument("--val_hr_dir", type=str, required=True)
    parser.add_argument("--val_degrade_mode", type=str, default="realistic", choices=["realistic", "bicubic"])  # keep consistent
    parser.add_argument("--num_val_images", type=int, default=50)
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--sde_strength", type=float, default=0.45)
    parser.add_argument("--fixed_noise_seed", type=int, default=42)
    parser.add_argument("--metric_y", action="store_true")
    parser.add_argument("--shave", type=int, default=4)
    parser.add_argument("--injection_mode", type=str, default="hybrid", choices=["input", "cross_attn", "hybrid"])
    parser.add_argument("--dtype_pixart", type=str, default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--vae_fp16", action="store_true")
    parser.add_argument("--no_lpips", action="store_true")
    parser.add_argument("--out_json", type=str, default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (args.dtype_pixart == "fp16")
    dtype_pixart = torch.float16 if args.dtype_pixart == "fp16" else torch.float32

    # Metrics
    lpips_fn = None
    if (not args.no_lpips) and (_lpips is not None) and (device.type == "cuda"):
        lpips_fn = _lpips.LPIPS(net="vgg").to(device)
        lpips_fn.eval()

    # -------------------------------------------------
    # Load PixArt base + inject ckpt (exactly like training)
    # -------------------------------------------------
    print("DEVICE=%s | AMP=%s | cudnn.enabled=%s" % (str(device), str(use_amp), str(torch.backends.cudnn.enabled)))
    print("Loading PixArt base weights...")

    # import here to ensure sys.path is set
    from diffusion.model.nets.PixArtMS import PixArtMS_XL_2
    from diffusion.model.nets.adapter import MultiLevelAdapter
    from diffusers import AutoencoderKL

    pixart = PixArtMS_XL_2(input_size=64, in_channels=4, model_max_length=120).to(device)

    base = torch.load(PIXART_PATH, map_location="cpu")
    if isinstance(base, dict) and "state_dict" in base:
        base = base["state_dict"]
    pixart.load_state_dict(base, strict=False)

    pixart = pixart.to(device=device, dtype=dtype_pixart)

    adapter = MultiLevelAdapter().to(device=device, dtype=torch.float32)

    # keep injection-related modules in fp32
    force_fp32_injection_modules(pixart)

    print(f"Loading finetuned ckpt: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu")

    if not isinstance(ckpt, dict):
        raise ValueError("Unexpected ckpt format; expected dict")

    if "pixart_inject" not in ckpt or "adapter" not in ckpt:
        raise KeyError(f"ckpt keys={list(ckpt.keys())} (expected pixart_inject + adapter like train_full_mse)")

    load_inject_state_dict(pixart, ckpt["pixart_inject"])
    adapter.load_state_dict(ckpt["adapter"], strict=True)

    pixart.eval()
    adapter.eval()

    # -------------------------------------------------
    # Load VAE (local_files_only=True) like training
    # -------------------------------------------------
    print("Loading VAE (local_files_only=True)...")
    vae = AutoencoderKL.from_pretrained(VAE_PATH, local_files_only=True).to(device)
    vae.enable_slicing()
    if args.vae_fp16 and device.type == "cuda":
        vae = vae.to(dtype=torch.float16)
    else:
        vae = vae.to(dtype=torch.float32)
    vae.eval()

    # -------------------------------------------------
    # Text embed + data_info (match train script)
    # -------------------------------------------------
    y_embed, data_info = load_text_cond(device)

    # -------------------------------------------------
    # Collect images
    # -------------------------------------------------
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    paths = [os.path.join(args.val_hr_dir, f) for f in sorted(os.listdir(args.val_hr_dir)) if f.lower().endswith(exts)]
    paths = paths[: max(0, int(args.num_val_images))]

    if len(paths) == 0:
        raise ValueError(f"No images found in {args.val_hr_dir}")

    # -------------------------------------------------
    # Eval loop: baseline (lr_cond) vs oracle (hr_cond)
    # -------------------------------------------------
    sums = {
        "baseline": {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "n": 0},
        "oracle_hrcond": {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "n": 0},
    }

    for p in paths:
        img_id = os.path.basename(p)

        # HR image to [0,1]
        pil = Image.open(p).convert("RGB")
        pil = center_crop(pil, 512)
        hr01 = pil_to_tensor_norm01(pil).unsqueeze(0)  # [1,3,512,512]

        # per-image deterministic degradation RNG (match training)
        rng_py = random.Random(stable_int_hash(img_id))
        torch_gen = torch.Generator(device="cpu")
        torch_gen.manual_seed(stable_int_hash(img_id))

        # degrade (cpu) -> move to device
        lr01 = degrade_hr_to_lr_tensor(hr01[0] * 2 - 1, mode=args.val_degrade_mode, rng_py=rng_py, torch_gen=torch_gen, target_size=128)
        lr01 = lr01.unsqueeze(0)

        # encode latents
        hr_latent = encode_latent(vae, hr01.to(device=device, dtype=vae.dtype))
        lr_latent = encode_latent(vae, lr01.to(device=device, dtype=vae.dtype))

        # run sampling twice
        pred_latent_baseline = sample_sr_dpm(
            pixart,
            adapter,
            lr_latent,
            y_embed,
            data_info,
            cond_latent_for_adapter=lr_latent,
            num_steps=args.num_steps,
            sde_strength=args.sde_strength,
            fixed_noise_seed=args.fixed_noise_seed,
            dtype_pixart=dtype_pixart,
            use_amp=use_amp,
            injection_mode=args.injection_mode,
        )

        pred_latent_oracle = sample_sr_dpm(
            pixart,
            adapter,
            lr_latent,
            y_embed,
            data_info,
            cond_latent_for_adapter=hr_latent,
            num_steps=args.num_steps,
            sde_strength=args.sde_strength,
            fixed_noise_seed=args.fixed_noise_seed,
            dtype_pixart=dtype_pixart,
            use_amp=use_amp,
            injection_mode=args.injection_mode,
        )

        # decode
        pred01_baseline = decode_latent(vae, pred_latent_baseline).to(torch.float32)[0]
        pred01_oracle = decode_latent(vae, pred_latent_oracle).to(torch.float32)[0]
        gt01 = decode_latent(vae, hr_latent).to(torch.float32)[0]

        pred_np_base = pred01_baseline.permute(1, 2, 0).cpu().numpy()
        pred_np_oracle = pred01_oracle.permute(1, 2, 0).cpu().numpy()
        gt_np = gt01.permute(1, 2, 0).cpu().numpy()

        p_psnr, p_ssim, p_lp = compute_metrics(pred_np_base, gt_np, args.metric_y, args.shave, lpips_fn)
        o_psnr, o_ssim, o_lp = compute_metrics(pred_np_oracle, gt_np, args.metric_y, args.shave, lpips_fn)

        sums["baseline"]["psnr"] += float(p_psnr)
        sums["baseline"]["ssim"] += float(p_ssim)
        sums["baseline"]["lpips"] += float(0.0 if p_lp is None else p_lp)
        sums["baseline"]["n"] += 1

        sums["oracle_hrcond"]["psnr"] += float(o_psnr)
        sums["oracle_hrcond"]["ssim"] += float(o_ssim)
        sums["oracle_hrcond"]["lpips"] += float(0.0 if o_lp is None else o_lp)
        sums["oracle_hrcond"]["n"] += 1

        if sums["baseline"]["n"] % 10 == 0:
            b_n = sums["baseline"]["n"]
            b_psnr = sums["baseline"]["psnr"] / b_n
            b_ssim = sums["baseline"]["ssim"] / b_n
            o_psnr_m = sums["oracle_hrcond"]["psnr"] / b_n
            o_ssim_m = sums["oracle_hrcond"]["ssim"] / b_n
            print(f"[{b_n}/{len(paths)}] baseline PSNR={b_psnr:.2f} SSIM={b_ssim:.4f} | oracle(hr_cond) PSNR={o_psnr_m:.2f} SSIM={o_ssim_m:.4f}")

        # keep VRAM steady
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def _avg(block: Dict[str, Any]) -> Dict[str, float]:
        n = max(1, int(block["n"]))
        return {
            "n": int(block["n"]),
            "psnr": float(block["psnr"] / n),
            "ssim": float(block["ssim"] / n),
            "lpips": float(block["lpips"] / n) if (not args.no_lpips) else float("nan"),
        }

    baseline = _avg(sums["baseline"])
    oracle = _avg(sums["oracle_hrcond"])

    out = {
        "meta": {
            "ckpt": args.ckpt,
            "val_hr_dir": args.val_hr_dir,
            "val_degrade_mode": args.val_degrade_mode,
            "num_val_images": len(paths),
            "num_steps": args.num_steps,
            "sde_strength": args.sde_strength,
            "fixed_noise_seed": args.fixed_noise_seed,
            "metric_y": bool(args.metric_y),
            "shave": int(args.shave),
            "injection_mode": args.injection_mode,
            "dtype_pixart": str(dtype_pixart),
            "vae_fp16": bool(args.vae_fp16),
        },
        "baseline_lrcond": baseline,
        "oracle_hrcond": oracle,
        "delta_oracle_minus_base": {
            "psnr": float(oracle["psnr"] - baseline["psnr"]),
            "ssim": float(oracle["ssim"] - baseline["ssim"]),
            "lpips": float(oracle["lpips"] - baseline["lpips"]) if (not args.no_lpips) else float("nan"),
        },
    }

    print("\n=== ORACLE RESULT (avg) ===")
    print(json.dumps(out, indent=2))

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"Saved to: {args.out_json}")


if __name__ == "__main__":
    main()
