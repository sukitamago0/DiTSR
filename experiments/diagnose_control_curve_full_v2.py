# experiments/diagnose_control_curve_full_v2.py
import os
import sys
import io
import glob
import json
import math
import random
import hashlib
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
from PIL import Image
from tqdm import tqdm

from diffusers import AutoencoderKL, DPMSolverMultistepScheduler

# -------------------------
# 0) 环境硬约束：与 train_full_mse.py 一致
# -------------------------
torch.backends.cudnn.enabled = False

# -------------------------
# 1) Metrics（可选）
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
# 2) 路径：严格参照训练脚本
# -------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

PIXART_PATH = os.path.join(PROJECT_ROOT, "output", "pretrained_models", "PixArt-XL-2-512x512.pth")
VAE_PATH    = os.path.join(PROJECT_ROOT, "output", "pretrained_models", "sd-vae-ft-ema")
T5_EMBED_PATH = os.path.join(PROJECT_ROOT, "output", "quality_embed.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE_PIXART = torch.float16
USE_AMP = (DEVICE == "cuda")


# -------------------------
# 3) 导入你的模型（必须通过 PROJECT_ROOT 找到 diffusion）
# -------------------------
try:
    from diffusion.model.nets.PixArtMS import PixArtMS_XL_2
    from diffusion.model.nets.adapter import MultiLevelAdapter
except ImportError as e:
    print(f"❌ Import failed: {e}")
    raise


# -------------------------
# 4) 与训练脚本一致的稳定 hash
# -------------------------
def stable_int_hash(s: str, mod: int = 2**32) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h, 16) % mod


# -------------------------
# 5) 图像工具：严格对齐训练脚本，并修复 numpy not writable 警告
# -------------------------
def pil_to_tensor_norm01(pil: Image.Image) -> torch.Tensor:
    # 关键：copy() 避免 “NumPy array is not writable” 警告
    arr = np.asarray(pil, dtype=np.uint8).copy()  # [H,W,3], writable
    x = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return x

def norm01_to_norm11(x01: torch.Tensor) -> torch.Tensor:
    return x01 * 2.0 - 1.0

def transforms_to_pil(x01: torch.Tensor) -> Image.Image:
    x = (x01.clamp(0, 1) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(x)

def _jpeg_compress_tensor(x11: torch.Tensor, quality: int) -> torch.Tensor:
    x = x11.clamp(-1, 1)
    x01 = (x + 1.0) / 2.0
    x01 = x01.cpu()
    pil = transforms_to_pil(x01)
    buffer = io.BytesIO()
    pil.save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    pil2 = Image.open(buffer).convert("RGB")
    x01b = pil_to_tensor_norm01(pil2)
    return norm01_to_norm11(x01b)

def center_crop(pil: Image.Image, size: int = 512) -> Image.Image:
    w, h = pil.size
    if w < size or h < size:
        pil = pil.resize((max(size, w), max(size, h)), resample=Image.BICUBIC)
        w, h = pil.size
    left = (w - size) // 2
    top  = (h - size) // 2
    return pil.crop((left, top, left + size, top + size))

def gaussian_kernel2d(k: int, sigma: float, device, dtype):
    ax = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel

def depthwise_conv2d(x: torch.Tensor, kernel2d: torch.Tensor) -> torch.Tensor:
    k = kernel2d.shape[0]
    w = kernel2d.view(1, 1, k, k).repeat(3, 1, 1, 1)
    return F.conv2d(x, w, padding=k // 2, groups=3)

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
        lr = F.interpolate(lr_small, size=(512, 512), mode="bicubic", align_corners=False)
        return lr.squeeze(0)

    blur_k = rng.choice([3, 5, 7])
    blur_sigma = rng.uniform(0.2, 1.2)

    hr = hr11.unsqueeze(0)
    kernel = gaussian_kernel2d(blur_k, blur_sigma, device=hr.device, dtype=hr.dtype)
    hr_blur = depthwise_conv2d(hr, kernel)

    lr_small = F.interpolate(hr_blur, scale_factor=0.25, mode="bicubic", align_corners=False)

    noise_std = rng.uniform(0.0, 0.02)
    if noise_std > 0:
        if torch_gen is None:
            eps = torch.randn(lr_small.shape, device=lr_small.device, dtype=lr_small.dtype)
        else:
            eps = torch.randn(lr_small.shape, generator=torch_gen, device=lr_small.device, dtype=lr_small.dtype)
        lr_small = (lr_small + eps * noise_std).clamp(-1, 1)

    jpeg_q = rng.randint(30, 95)
    lr_small_cpu = lr_small.squeeze(0).cpu()
    lr_small_cpu = _jpeg_compress_tensor(lr_small_cpu, jpeg_q).unsqueeze(0)

    lr = F.interpolate(lr_small_cpu, size=(512, 512), mode="bicubic", align_corners=False)
    return lr.squeeze(0)


# -------------------------
# 6) Val Dataset
# -------------------------
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


# -------------------------
# 7) Metrics：Y 通道 + shave（同训练脚本）
# -------------------------
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
# 8) 文本条件：同训练脚本
# -------------------------
def build_text_cond(device: str, dtype_pixart: torch.dtype):
    y_embed = torch.load(T5_EMBED_PATH, map_location="cpu")["prompt_embeds"].unsqueeze(1)
    y_embed = y_embed.to(device).to(dtype_pixart)
    data_info = {
        "img_hw": torch.tensor([[512., 512.]], device=device, dtype=dtype_pixart),
        "aspect_ratio": torch.tensor([1.], device=device, dtype=dtype_pixart),
    }
    return y_embed, data_info


# -------------------------
# 9) 注入模块必须保持 FP32（避免 LN dtype mismatch）
# -------------------------
def ensure_inject_modules_fp32(pixart):
    # injection scales
    if hasattr(pixart, "injection_scales"):
        for s in pixart.injection_scales:
            s.data = s.data.float()
    # cross scale
    if hasattr(pixart, "cross_attn_scale"):
        pixart.cross_attn_scale.data = pixart.cross_attn_scale.data.float()
    # proj/norm/ln
    if hasattr(pixart, "adapter_proj"):
        pixart.adapter_proj = pixart.adapter_proj.to(torch.float32)
    if hasattr(pixart, "adapter_norm"):
        pixart.adapter_norm = pixart.adapter_norm.to(torch.float32)
    if hasattr(pixart, "input_adapter_ln"):
        pixart.input_adapter_ln = pixart.input_adapter_ln.to(torch.float32)


def extract_inject_state_dict(pixart) -> Dict[str, torch.Tensor]:
    sd = pixart.state_dict()
    keep = {}
    for k, v in sd.items():
        if (
            k.startswith("injection_scales")
            or k.startswith("adapter_proj")
            or k.startswith("adapter_norm")
            or k.startswith("cross_attn_scale")
            or k.startswith("input_adapter_ln")
        ):
            keep[k] = v.detach().cpu()
    return keep

def load_inject_state_dict(pixart, inject_sd: Dict[str, torch.Tensor]):
    sd = pixart.state_dict()
    for k, v in inject_sd.items():
        if k in sd:
            sd[k] = v.to(sd[k].dtype)
    pixart.load_state_dict(sd, strict=False)


# -------------------------
# 10) 控制倍率：只改参数值，跑完恢复（不污染模型）
# -------------------------
def get_scale_snapshot(pixart) -> Dict[str, object]:
    inj = []
    if hasattr(pixart, "injection_scales"):
        inj = [float(s.detach().float().cpu().item()) for s in pixart.injection_scales]
    cross = float(pixart.cross_attn_scale.detach().float().cpu().item()) if hasattr(pixart, "cross_attn_scale") else 0.0
    return {"injection_scales": inj, "cross_attn_scale": cross}

def apply_control_multiplier(pixart, mult: float, target: str):
    """
    target:
      - both  : inj *= mult, cross *= mult
      - input : inj *= mult, cross unchanged
      - cross : cross *= mult, inj unchanged
    """
    assert target in ["both", "input", "cross"]
    orig_inj = [s.data.clone() for s in pixart.injection_scales]
    orig_cross = pixart.cross_attn_scale.data.clone()

    with torch.no_grad():
        if target in ["both", "input"]:
            for i, s in enumerate(pixart.injection_scales):
                s.data = orig_inj[i] * float(mult)
        if target == "cross":
            # inj unchanged
            for i, s in enumerate(pixart.injection_scales):
                s.data = orig_inj[i]
        if target in ["both", "cross"]:
            pixart.cross_attn_scale.data = orig_cross * float(mult)
        if target == "input":
            pixart.cross_attn_scale.data = orig_cross

    def restore():
        with torch.no_grad():
            for i, s in enumerate(pixart.injection_scales):
                s.data = orig_inj[i]
            pixart.cross_attn_scale.data = orig_cross

    return restore


# -------------------------
# 11) 单样本采样：严格对齐 validate_epoch 的采样流程
# -------------------------
@torch.no_grad()
def sample_one(
    hr_img_11: torch.Tensor,   # [1,3,512,512] float32 on DEVICE
    path: str,
    pixart,
    adapter,
    vae,
    y_embed,
    data_info,
    degrade_mode: str,
    num_steps: int,
    sde_strength: float,
    fixed_noise_seed: int,
    injection_mode: str,
):
    # 每张图固定退化随机性
    seed_i = (stable_int_hash(path) + 12345) & 0xFFFFFFFF
    rng = random.Random(seed_i)
    torch_gen = torch.Generator(device="cpu").manual_seed(seed_i)

    # 退化：CPU（与训练一致）
    lr_img_11 = degrade_hr_to_lr_tensor(
        hr_img_11.squeeze(0).detach().cpu(),
        degrade_mode,
        rng,
        torch_gen=torch_gen,
    ).unsqueeze(0).to(DEVICE).float()

    # VAE encode：fp32（与训练一致）
    hr_latent = vae.encode(hr_img_11).latent_dist.sample() * vae.config.scaling_factor
    lr_latent = vae.encode(lr_img_11).latent_dist.sample() * vae.config.scaling_factor

    scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000, solver_order=2)
    scheduler.set_timesteps(num_steps, device=DEVICE)

    start_t_val = int(1000 * float(sde_strength))
    run_ts = [t for t in scheduler.timesteps if int(t.item()) <= start_t_val]

    g = torch.Generator(device=DEVICE).manual_seed(int(fixed_noise_seed))
    latents = lr_latent.to(DEVICE)
    noise = torch.randn(latents.shape, generator=g, device=DEVICE, dtype=latents.dtype)
    t_start = torch.tensor([start_t_val], device=DEVICE).long()
    latents = scheduler.add_noise(latents, noise, t_start)

    # adapter：fp32 + autocast disabled（与训练一致）
    with torch.cuda.amp.autocast(enabled=False):
        cond = adapter(lr_latent.float())

    for t in run_ts:
        t_tensor = t.unsqueeze(0).to(DEVICE)
        with torch.cuda.amp.autocast(enabled=USE_AMP, dtype=DTYPE_PIXART):
            out = pixart(
                latents.to(DTYPE_PIXART),
                t_tensor,
                y_embed,
                data_info=data_info,
                adapter_cond=cond,
                injection_mode=injection_mode,
            )
            if out.shape[1] == 8:
                out, _ = out.chunk(2, dim=1)

        latents = scheduler.step(out.float(), t, latents.float()).prev_sample

    pred_img = vae.decode(latents / vae.config.scaling_factor).sample
    pred01 = torch.clamp((pred_img + 1.0) / 2.0, 0.0, 1.0)

    gt_img = vae.decode(hr_latent / vae.config.scaling_factor).sample
    gt01 = torch.clamp((gt_img + 1.0) / 2.0, 0.0, 1.0)

    return pred01, gt01


def compute_metrics(pred01: torch.Tensor, gt01: torch.Tensor, metric_y: bool, shave: int, lpips_fn=None):
    if not USE_METRICS:
        return float("nan"), float("nan"), float("nan")

    if metric_y:
        pred_y = rgb01_to_y01(pred01)
        gt_y   = rgb01_to_y01(gt01)
        pred_y = shave_border(pred_y, shave)
        gt_y   = shave_border(gt_y, shave)
        p = psnr(pred_y, gt_y, data_range=1.0).item()
        s = ssim(pred_y, gt_y, data_range=1.0).item()
    else:
        p = psnr(pred01, gt01, data_range=1.0).item()
        s = ssim(pred01, gt01, data_range=1.0).item()

    l = float("nan")
    if lpips_fn is not None:
        pred_norm = pred01 * 2.0 - 1.0
        gt_norm   = gt01 * 2.0 - 1.0
        l = float(lpips_fn(pred_norm, gt_norm).item())

    return p, s, l


# -------------------------
# 12) 主流程：扫 control_mult 曲线
# -------------------------
def parse_csv_floats(s: str) -> List[float]:
    items = [x.strip() for x in s.split(",") if x.strip() != ""]
    return [float(x) for x in items]

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--val_hr_dir", type=str, required=True)
    parser.add_argument("--val_degrade_mode", type=str, default="realistic", choices=["realistic", "bicubic"])
    parser.add_argument("--num_val_images", type=int, default=100, help="-1 means all")
    parser.add_argument("--val_repeat", type=int, default=1)

    parser.add_argument("--control_target", type=str, default="both", choices=["both", "input", "cross"])
    parser.add_argument("--control_mults", type=str, default="0.0,0.5,1.0,1.5,2.0,3.0")

    parser.add_argument("--injection_mode", type=str, default="hybrid", choices=["hybrid", "input", "cross_attn"])

    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--sde_strength", type=float, default=0.45)
    parser.add_argument("--fixed_noise_seed", type=int, default=42)

    parser.add_argument("--metric_y", action="store_true")
    parser.add_argument("--shave", type=int, default=4)

    parser.add_argument("--out_json", type=str, required=True)
    args = parser.parse_args()

    ckpt_path = os.path.abspath(args.ckpt)
    val_hr_dir = os.path.abspath(os.path.join(PROJECT_ROOT, args.val_hr_dir)) if not os.path.isabs(args.val_hr_dir) else args.val_hr_dir
    out_json = os.path.abspath(os.path.join(PROJECT_ROOT, args.out_json)) if not os.path.isabs(args.out_json) else args.out_json
    os.makedirs(os.path.dirname(out_json), exist_ok=True)

    control_mults = parse_csv_floats(args.control_mults)

    print(f"DEVICE={DEVICE} | AMP={USE_AMP} | cudnn.enabled={torch.backends.cudnn.enabled}")
    print("Loading PixArt base weights...")
    pixart = PixArtMS_XL_2(input_size=64).to(DEVICE).to(DTYPE_PIXART).eval()
    ckpt_base = torch.load(PIXART_PATH, map_location="cpu")
    if "state_dict" in ckpt_base:
        ckpt_base = ckpt_base["state_dict"]
    if "pos_embed" in ckpt_base:
        del ckpt_base["pos_embed"]
    pixart.load_state_dict(ckpt_base, strict=False)

    print("Loading Adapter...")
    adapter = MultiLevelAdapter(in_channels=4, hidden_size=1152).to(DEVICE).eval()

    print(f"Loading finetuned ckpt: {ckpt_path}")
    fin = torch.load(ckpt_path, map_location="cpu")
    if "adapter" in fin:
        adapter.load_state_dict(fin["adapter"], strict=True)
    else:
        raise KeyError("ckpt missing key: adapter")
    if "pixart_inject" in fin:
        load_inject_state_dict(pixart, fin["pixart_inject"])
    else:
        print("⚠️ ckpt has no pixart_inject; continue with base inject params.")

    # 关键：保证注入相关模块 fp32（避免 LN dtype mismatch）
    ensure_inject_modules_fp32(pixart)

    print("Loading VAE (local_files_only=True)...")
    vae = AutoencoderKL.from_pretrained(VAE_PATH, local_files_only=True).to(DEVICE).float().eval()
    vae.enable_slicing()

    y_embed, data_info = build_text_cond(DEVICE, DTYPE_PIXART)

    lpips_fn = None
    if USE_METRICS:
        lpips_fn = lpips.LPIPS(net="vgg").to(DEVICE).eval()
        for p in lpips_fn.parameters():
            p.requires_grad = False

    # dataset
    max_files = None if args.num_val_images == -1 else int(args.num_val_images)
    ds = ValImageDataset(val_hr_dir, max_files=max_files)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=(DEVICE == "cuda"))

    meta = {
        "ckpt": ckpt_path,
        "val_hr_dir": val_hr_dir,
        "val_degrade_mode": args.val_degrade_mode,
        "num_val_images": (len(ds) if max_files is None else max_files),
        "val_repeat": int(args.val_repeat),
        "control_mults": control_mults,
        "control_target": args.control_target,
        "num_steps": int(args.num_steps),
        "sde_strength": float(args.sde_strength),
        "fixed_noise_seed": int(args.fixed_noise_seed),
        "metric_y": bool(args.metric_y),
        "shave": int(args.shave),
        "injection_mode": args.injection_mode,
        "device": DEVICE,
        "dtype_pixart": str(DTYPE_PIXART),
        "use_amp": bool(USE_AMP),
    }

    results = {"meta": meta, "curve": []}

    base_snapshot = get_scale_snapshot(pixart)

    for m in control_mults:
        # 每个 m 之前恢复到 base，再应用倍率（避免累积漂移）
        restore = apply_control_multiplier(pixart, 1.0, "both")
        restore()

        restore2 = apply_control_multiplier(pixart, float(m), args.control_target)
        after_snapshot = get_scale_snapshot(pixart)

        psnrs, ssims, lpips_list = [], [], []

        pbar = tqdm(loader, desc=f"control_mult={m:g}", dynamic_ncols=True)
        for _rep in range(int(args.val_repeat)):
            for batch in pbar:
                hr11 = batch["hr_img_11"].to(DEVICE).float()  # [1,3,512,512]
                path = batch["path"][0]

                pred01, gt01 = sample_one(
                    hr11, path,
                    pixart, adapter, vae,
                    y_embed, data_info,
                    degrade_mode=args.val_degrade_mode,
                    num_steps=int(args.num_steps),
                    sde_strength=float(args.sde_strength),
                    fixed_noise_seed=int(args.fixed_noise_seed),
                    injection_mode=args.injection_mode,
                )

                p, s, l = compute_metrics(pred01, gt01, metric_y=bool(args.metric_y), shave=int(args.shave), lpips_fn=lpips_fn)
                psnrs.append(p); ssims.append(s); lpips_list.append(l)

                if USE_METRICS and len(psnrs) > 0:
                    pbar.set_postfix({
                        "PSNR": f"{sum(psnrs)/len(psnrs):.2f}",
                        "SSIM": f"{sum(ssims)/len(ssims):.4f}",
                        "LPIPS": f"{(sum(lpips_list)/len(lpips_list)):.4f}" if all(math.isfinite(x) for x in lpips_list) else "NA"
                    })

        restore2()  # 恢复参数

        avg_psnr = sum(psnrs) / len(psnrs) if len(psnrs) > 0 else float("nan")
        avg_ssim = sum(ssims) / len(ssims) if len(ssims) > 0 else float("nan")
        finite_lp = [x for x in lpips_list if math.isfinite(x)]
        avg_lp   = sum(finite_lp) / len(finite_lp) if len(finite_lp) > 0 else float("nan")

        results["curve"].append({
            "control_mult": float(m),
            "n_samples": int(len(psnrs)),
            "psnr": float(avg_psnr),
            "ssim": float(avg_ssim),
            "lpips": float(avg_lp),
            "debug_effective_scales_before": base_snapshot,
            "debug_effective_scales_after": after_snapshot,
        })

        # 尽量释放碎片
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved json to: {out_json}")


if __name__ == "__main__":
    main()
