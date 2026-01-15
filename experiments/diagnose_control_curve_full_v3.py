# experiments/diagnose_control_curve_full_v3.py adapter注入强度对结果的影响 结果：目前情况下刚好为1强度是最好的结果
import os
import sys
import glob
import io
import json
import math
import random
import hashlib
import argparse
from contextlib import contextmanager
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
from PIL import Image
from tqdm import tqdm

from diffusers import AutoencoderKL, DPMSolverMultistepScheduler

# -------------------------
# 0) 环境硬约束：对齐训练脚本
# -------------------------
torch.backends.cudnn.enabled = False

# -------------------------
# 1) Metrics：对齐训练脚本
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
# 2) 路径：对齐训练脚本
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
# 3) 导入你的模型：对齐训练脚本
# -------------------------
from diffusion.model.nets.PixArtMS import PixArtMS_XL_2
from diffusion.model.nets.adapter import MultiLevelAdapter

# -------------------------
# 4) 工具函数：对齐训练脚本（并修掉 numpy 不可写警告）
# -------------------------
def stable_int_hash(s: str, mod: int = 2**32) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h, 16) % mod

def pil_to_tensor_norm01(pil: Image.Image) -> torch.Tensor:
    # 用 copy=True 避免 “NumPy array is not writable” 警告
    arr = np.array(pil, dtype=np.uint8, copy=True)  # [H,W,3]
    x = torch.from_numpy(arr).permute(2,0,1).float() / 255.0
    return x

def norm01_to_norm11(x01: torch.Tensor) -> torch.Tensor:
    return x01 * 2.0 - 1.0

def transforms_to_pil(x01: torch.Tensor) -> Image.Image:
    x = (x01.clamp(0,1) * 255.0).byte().permute(1,2,0).cpu().numpy()
    return Image.fromarray(x)

def _jpeg_compress_tensor(x11: torch.Tensor, quality: int) -> torch.Tensor:
    x = x11.clamp(-1,1)
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
    w = kernel2d.view(1,1,k,k).repeat(3,1,1,1)
    return F.conv2d(x, w, padding=k//2, groups=3)

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

    blur_k = rng.choice([3,5,7])
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
        lr_small = (lr_small + eps * noise_std).clamp(-1,1)

    jpeg_q = rng.randint(30, 95)
    lr_small_cpu = lr_small.squeeze(0).cpu()
    lr_small_cpu = _jpeg_compress_tensor(lr_small_cpu, jpeg_q).unsqueeze(0)

    lr = F.interpolate(lr_small_cpu, size=(512,512), mode="bicubic", align_corners=False)
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

# -------------------------
# 5) Dataset：对齐训练脚本验证集（用 HR 图像做退化和指标）
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
# 6) 加载：严格参照训练脚本
# -------------------------
def build_text_cond():
    y_embed = torch.load(T5_EMBED_PATH, map_location="cpu")["prompt_embeds"].unsqueeze(1)
    y_embed = y_embed.to(DEVICE).to(DTYPE_PIXART)
    data_info = {
        "img_hw": torch.tensor([[512.,512.]], device=DEVICE, dtype=DTYPE_PIXART),
        "aspect_ratio": torch.tensor([1.], device=DEVICE, dtype=DTYPE_PIXART),
    }
    return y_embed, data_info

def load_inject_state_dict(pixart, inject_sd: Dict[str, torch.Tensor]):
    sd = pixart.state_dict()
    for k, v in inject_sd.items():
        if k in sd:
            sd[k] = v.to(sd[k].dtype)
    pixart.load_state_dict(sd, strict=False)

def ensure_fp32_inject_modules(pixart):
    # 对齐训练脚本 set_trainable_A 的 dtype 处理，避免 LN dtype mismatch
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

@contextmanager
def control_scale_guard(pixart, mult: float, target: str = "both"):
    """
    mult 作用在 LN 之后：injection_scales / cross_attn_scale
    target: "both" | "input" | "cross"
    """
    with torch.no_grad():
        saved_inj = None
        saved_cross = None

        if hasattr(pixart, "injection_scales") and target in ("both", "input"):
            saved_inj = [s.data.clone() for s in pixart.injection_scales]
            for s in pixart.injection_scales:
                s.data = s.data * float(mult)

        if hasattr(pixart, "cross_attn_scale") and target in ("both", "cross"):
            saved_cross = pixart.cross_attn_scale.data.clone()
            pixart.cross_attn_scale.data = pixart.cross_attn_scale.data * float(mult)

    try:
        yield
    finally:
        with torch.no_grad():
            if saved_inj is not None:
                for s, old in zip(pixart.injection_scales, saved_inj):
                    s.data = old
            if saved_cross is not None:
                pixart.cross_attn_scale.data = saved_cross

# -------------------------
# 7) 采样：严格复用训练脚本 validate_epoch 的逻辑
# -------------------------
@torch.no_grad()
def sample_one(
    pixart,
    adapter,
    vae,
    y_embed,
    data_info,
    hr_img_11_cpu: torch.Tensor,  # [3,512,512] CPU
    path: str,
    degrade_mode: str,
    num_steps: int,
    sde_strength: float,
    fixed_noise_seed: int,
    injection_mode: str,
    metric_y: bool,
    shave: int,
    lpips_fn=None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[float], Optional[float], Optional[float]]:
    # 每张图固定退化
    seed_i = (stable_int_hash(path) + 12345) & 0xFFFFFFFF
    rng = random.Random(seed_i)
    torch_gen = torch.Generator(device="cpu").manual_seed(seed_i)

    lr_img_11 = degrade_hr_to_lr_tensor(
        hr_img_11_cpu, degrade_mode, rng, torch_gen=torch_gen
    ).unsqueeze(0)  # [1,3,512,512] CPU

    hr_img_11 = hr_img_11_cpu.unsqueeze(0).to(DEVICE).float()
    lr_img_11 = lr_img_11.to(DEVICE).float()

    # VAE encode
    hr_latent = vae.encode(hr_img_11).latent_dist.sample() * vae.config.scaling_factor
    lr_latent = vae.encode(lr_img_11).latent_dist.sample() * vae.config.scaling_factor

    # scheduler
    scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000, solver_order=2)
    scheduler.set_timesteps(num_steps, device=DEVICE)

    start_t_val = int(1000 * sde_strength)
    run_ts = [t for t in scheduler.timesteps if t <= start_t_val]

    # 固定噪声（对齐训练脚本）
    g = torch.Generator(device=DEVICE).manual_seed(int(fixed_noise_seed))
    latents = lr_latent.to(DEVICE).float()
    noise = torch.randn(latents.shape, generator=g, device=DEVICE, dtype=latents.dtype)
    t_start = torch.tensor([start_t_val], device=DEVICE).long()
    latents = scheduler.add_noise(latents, noise, t_start)

    # adapter_cond：fp32，autocast disabled（对齐训练脚本）
    with torch.cuda.amp.autocast(enabled=False):
        cond = adapter(lr_latent.float())

    # denoise
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

    # metrics
    p = s = l = None
    if USE_METRICS:
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

        if lpips_fn is not None:
            pred_norm = pred01 * 2.0 - 1.0
            gt_norm   = gt01 * 2.0 - 1.0
            l = lpips_fn(pred_norm, gt_norm).item()

    return pred01, gt01, p, s, l

# -------------------------
# 8) 主程序：输出 json 曲线
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--val_hr_dir", type=str, required=True)
    parser.add_argument("--val_degrade_mode", type=str, default="realistic", choices=["realistic","bicubic"])
    parser.add_argument("--num_val_images", type=int, default=100, help="-1 means all")
    parser.add_argument("--val_repeat", type=int, default=1)
    parser.add_argument("--control_mults", type=str, required=True, help="e.g. 0,0.5,1,2,3")
    parser.add_argument("--control_target", type=str, default="both", choices=["both","input","cross"])
    parser.add_argument("--injection_mode", type=str, default="hybrid", choices=["input","cross_attn","hybrid"])
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--sde_strength", type=float, default=0.45)
    parser.add_argument("--fixed_noise_seed", type=int, default=42)
    parser.add_argument("--metric_y", action="store_true")
    parser.add_argument("--shave", type=int, default=4)
    parser.add_argument("--out_json", type=str, required=True)
    args = parser.parse_args()

    control_mults = [float(x) for x in args.control_mults.split(",")]

    # dataset
    max_files = None if args.num_val_images < 0 else int(args.num_val_images)
    ds = ValImageDataset(args.val_hr_dir, max_files=max_files)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=(DEVICE=="cuda"))

    # load pixart base
    print("Loading PixArt base...")
    pixart = PixArtMS_XL_2(input_size=64).to(DEVICE).to(DTYPE_PIXART).eval()
    base = torch.load(PIXART_PATH, map_location="cpu")
    if "state_dict" in base:
        base = base["state_dict"]
    if "pos_embed" in base:
        del base["pos_embed"]
    pixart.load_state_dict(base, strict=False)

    # load adapter
    print("Loading Adapter...")
    adapter = MultiLevelAdapter(in_channels=4, hidden_size=1152).to(DEVICE).eval()

    # load ckpt inject + adapter
    print(f"Loading ckpt (adapter + pixart_inject): {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if "adapter" not in ckpt or "pixart_inject" not in ckpt:
        raise KeyError("ckpt must contain keys: adapter, pixart_inject (use last_full_state.pth).")
    adapter.load_state_dict(ckpt["adapter"], strict=True)
    load_inject_state_dict(pixart, ckpt["pixart_inject"])

    # critical: inject modules fp32 (same as training)
    ensure_fp32_inject_modules(pixart)

    # vae local
    print("Loading VAE (local_files_only=True)...")
    vae = AutoencoderKL.from_pretrained(VAE_PATH, local_files_only=True).to(DEVICE).float().eval()
    vae.enable_slicing()

    # text cond
    y_embed, data_info = build_text_cond()

    # lpips
    lpips_fn = None
    if USE_METRICS:
        lpips_fn = lpips.LPIPS(net="vgg").to(DEVICE).eval()
        for p in lpips_fn.parameters():
            p.requires_grad = False

    # run
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)

    meta = {
        "ckpt": os.path.abspath(args.ckpt),
        "val_hr_dir": os.path.abspath(args.val_hr_dir),
        "val_degrade_mode": args.val_degrade_mode,
        "num_val_images": (len(ds) if args.num_val_images < 0 else int(args.num_val_images)),
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

    curve = []
    for cm in control_mults:
        psnrs, ssims, lpipss = [], [], []
        n_samples = 0

        # 记录“实际注入强度参数”，方便你复核是不是被 LN 洗掉
        eff_inj = [float(s.detach().cpu().item()) for s in getattr(pixart, "injection_scales", [])]
        eff_cross = float(getattr(pixart, "cross_attn_scale", torch.tensor(float("nan"))).detach().cpu().item())

        with control_scale_guard(pixart, cm, target=args.control_target):
            eff_inj_after = [float(s.detach().cpu().item()) for s in getattr(pixart, "injection_scales", [])]
            eff_cross_after = float(getattr(pixart, "cross_attn_scale", torch.tensor(float("nan"))).detach().cpu().item())

            pbar = tqdm(dl, desc=f"control_mult={cm:g}", dynamic_ncols=True)
            for batch in pbar:
                hr_img_11 = batch["hr_img_11"][0]  # [3,512,512] CPU
                path = batch["path"][0]

                for _ in range(int(args.val_repeat)):
                    _, _, p, s, l = sample_one(
                        pixart, adapter, vae, y_embed, data_info,
                        hr_img_11_cpu=hr_img_11,
                        path=path,
                        degrade_mode=args.val_degrade_mode,
                        num_steps=args.num_steps,
                        sde_strength=args.sde_strength,
                        fixed_noise_seed=args.fixed_noise_seed,
                        injection_mode=args.injection_mode,
                        metric_y=bool(args.metric_y),
                        shave=int(args.shave),
                        lpips_fn=lpips_fn,
                    )
                    if p is not None:
                        psnrs.append(p)
                        ssims.append(s)
                        if l is not None:
                            lpipss.append(l)
                    n_samples += 1

                    if len(psnrs) > 0:
                        pbar.set_postfix({
                            "PSNR": f"{sum(psnrs)/len(psnrs):.2f}",
                            "SSIM": f"{sum(ssims)/len(ssims):.4f}",
                            "LPIPS": f"{(sum(lpipss)/len(lpipss)):.4f}" if len(lpipss)>0 else "NA"
                        })

        curve.append({
            "control_mult": float(cm),
            "n_samples": int(n_samples),
            "psnr": (sum(psnrs)/len(psnrs) if len(psnrs)>0 else float("nan")),
            "ssim": (sum(ssims)/len(ssims) if len(ssims)>0 else float("nan")),
            "lpips": (sum(lpipss)/len(lpipss) if len(lpipss)>0 else float("nan")),
            "debug_effective_scales_before": {
                "injection_scales": eff_inj,
                "cross_attn_scale": eff_cross,
            },
            "debug_effective_scales_after": {
                "injection_scales": eff_inj_after,
                "cross_attn_scale": eff_cross_after,
            }
        })

    out = {"meta": meta, "curve": curve}
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved json to: {os.path.abspath(args.out_json)}")

if __name__ == "__main__":
    main()
