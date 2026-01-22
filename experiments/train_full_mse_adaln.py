# -*- coding: utf-8 -*-
# experiments/train_full_mse.py
import os
import sys
import glob
import io
import math
import random
import hashlib  # $$ [MOD-SEED-1] 用稳定 hash，替代 Python 内置 hash（跨进程/跨天不稳定）
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from diffusers import AutoencoderKL, DPMSolverMultistepScheduler
from torch.cuda.amp import GradScaler

import numpy as np  # $$ [MOD-IMG-1] 用 np.asarray(pil) 读取图片，避免 TypedStorage 警告且更标准
from torchvision.transforms import functional as TF

# -------------------------
# 0) 环境硬约束（你自己一直在用）
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
# 2) 路径与超参（按 3070 8G 默认）
# -------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

TRAIN_LATENT_DIR = "/home/never/jietian/PixArt-alpha/dataset/DIV2K_train_latents_v2"
VAL_HR_DIR = os.path.join(PROJECT_ROOT, "dataset", "DIV2K_valid_HR")

PIXART_PATH = os.path.join(PROJECT_ROOT, "output", "pretrained_models", "PixArt-XL-2-512x512.pth")
VAE_PATH    = os.path.join(PROJECT_ROOT, "output", "pretrained_models", "sd-vae-ft-ema")
T5_EMBED_PATH = os.path.join(PROJECT_ROOT, "output", "quality_embed.pth")

OUT_DIR = os.path.join(PROJECT_ROOT, "experiments_results", "train_full_mse_adaln")
os.makedirs(OUT_DIR, exist_ok=True)
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
VIS_DIR  = os.path.join(OUT_DIR, "vis")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE_PIXART = torch.float16  # PixArt 主干用 FP16 省显存
DTYPE_LATENT = torch.float16  # 你离线 latent 就是 FP16
USE_AMP = (DEVICE == "cuda")

EPOCHS = 100
BATCH_SIZE = 1
NUM_WORKERS = 0
LR_ADAPTER = 1e-5
LR_SCALES  = 1e-4
GRAD_ACCUM_STEPS = 1

SMOKE = False
SMOKE_TRAIN_SAMPLES = 20
SMOKE_VAL_SAMPLES = 20

NUM_INFER_STEPS = 20
FIXED_NOISE_SEED = 42

VAL_DEGRADE_MODE = "realistic"  # "realistic" / "bicubic"

PSNR_SWITCH = 24.0
KEEP_LAST_EPOCHS = 3
KEEP_TOPK = 1

# $$ [MOD-METRIC-1] SR 论文常见规范：PSNR/SSIM 在 Y(亮度) 通道算；LPIPS 在 RGB 上算
METRIC_Y_CHANNEL = True
# $$ [MOD-METRIC-2] 常见 SR 评估会 shave border（x4 通常=4）。如果你想“完全不裁边”，设为 0。
METRIC_SHAVE_BORDER = 4

# $$ [MOD-RESUME-0] 额外保存一个“last 全状态”用于可靠续跑（不参与 topK 删除）
LAST_CKPT_PATH = os.path.join(CKPT_DIR, "last_full_state.pth")

# -------------------------
# 3) 导入你的模型
# -------------------------
try:
    from diffusion.model.nets.PixArtMS import PixArtMS_XL_2
    from diffusion.model.nets.adapter import build_adapter
    from diffusion import IDDPM
except ImportError as e:
    print(f"❌ Import failed: {e}")
    raise

# -------------------------
# 4) 数据集：训练用离线 latent（hr_latent/lr_latent）
# -------------------------
class TrainLatentDataset(Dataset):
    def __init__(self, root_dir: str, max_files: Optional[int] = None):
        self.root_dir = root_dir
        self.paths = sorted(glob.glob(os.path.join(root_dir, "*.pt")))
        if max_files is not None:
            self.paths = self.paths[:max_files]
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No .pt files found in: {root_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        d = torch.load(self.paths[idx], map_location="cpu")
        hr = d["hr_latent"]  # [4,64,64] fp16
        lr = d["lr_latent"]  # [4,64,64] fp16
        return {"hr_latent": hr, "lr_latent": lr, "path": self.paths[idx]}


class SingleLatentDataset(Dataset):
    def __init__(self, latent_path: str):
        if not os.path.isfile(latent_path):
            raise FileNotFoundError(f"Latent file not found: {latent_path}")
        self.latent_path = latent_path

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        d = torch.load(self.latent_path, map_location="cpu")
        hr = d["hr_latent"]
        lr = d["lr_latent"]
        return {"hr_latent": hr, "lr_latent": lr, "path": self.latent_path}

def scan_latent_schema(root_dir: str, n: int = 50):
    paths = sorted(glob.glob(os.path.join(root_dir, "*.pt")))
    assert len(paths) > 0, "No train latents found."
    n = min(n, len(paths))
    for p in paths[:n]:
        d = torch.load(p, map_location="cpu")
        if "hr_latent" not in d or "lr_latent" not in d:
            raise KeyError(f"Bad schema in {p}, keys={list(d.keys())}")
        hr = d["hr_latent"]; lr = d["lr_latent"]
        if hr.ndim != 3 or lr.ndim != 3:
            raise ValueError(f"Bad shape in {p}: hr={hr.shape}, lr={lr.shape}")
        if hr.shape != (4,64,64) or lr.shape != (4,64,64):
            raise ValueError(f"Unexpected shape in {p}: hr={hr.shape}, lr={lr.shape}")
    print(f"✅ [SCAN] Train latent schema ok: {n}/{n} checked.")

# -------------------------
# 5) 验证：从 HR 图像生成 LR，再 VAE encode 得到 lr_latent/hr_latent
# -------------------------
def pil_to_tensor_norm01(pil: Image.Image) -> torch.Tensor:
    # $$ [MOD-IMG-1] 用 numpy 转换，避免 TypedStorage 警告；结果是 [3,H,W] float32 in [0,1]
    arr = np.asarray(pil, dtype=np.uint8)  # [H,W,3]
    x = torch.from_numpy(arr).permute(2,0,1).float() / 255.0
    return x

def norm01_to_norm11(x01: torch.Tensor) -> torch.Tensor:
    return x01 * 2.0 - 1.0

def _jpeg_compress_tensor(x11: torch.Tensor, quality: int) -> torch.Tensor:
    # 与离线 latent 生成保持一致（TF.to_pil_image + TF.to_tensor）
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
    torch_gen: Optional[torch.Generator] = None,  # $$ [MOD-SEED-2] 用 generator 固定 torch 随机噪声
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

    blur_k = int(rng.choice([3, 5, 7]))
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
# 6) 只训练 A：Adapter + (PixArt 注入相关小模块)
# -------------------------
def set_trainable_A(pixart, adapter):
    for p in pixart.parameters():
        p.requires_grad = False

    # 训练的注入相关模块全部 FP32（否则 GradScaler 会报 “unscale FP16 gradients”）
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

    # [NEW] AdaLN/FiLM injection MLPs must be fp32 as well.
    if hasattr(pixart, "input_adaln"):
        pixart.input_adaln = pixart.input_adaln.to(torch.float32)

    scale_params = []
    for s in pixart.injection_scales:
        s.requires_grad = True
        scale_params.append(s)

    pixart.cross_attn_scale.requires_grad = True
    scale_params.append(pixart.cross_attn_scale)

    proj_params = []
    for p in pixart.adapter_proj.parameters():
        p.requires_grad = True
        proj_params.append(p)
    for p in pixart.adapter_norm.parameters():
        p.requires_grad = True
        proj_params.append(p)
    if hasattr(pixart, "input_adapter_ln"):
        for p in pixart.input_adapter_ln.parameters():
            p.requires_grad = True
            proj_params.append(p)

    # [NEW] Train AdaLN/FiLM MLPs (zero-init so they start as a no-op)
    if hasattr(pixart, "input_adaln"):
        for p in pixart.input_adaln.parameters():
            p.requires_grad = True
            proj_params.append(p)

    adapter_params = list(adapter.parameters())

    for pp in (adapter_params + proj_params + scale_params):
        if pp.dtype != torch.float32:
            raise ValueError(f"Trainable param is not fp32: dtype={pp.dtype}")

    print(f"🔥 Trainable: Adapter({sum(p.numel() for p in adapter_params)}) | "
          f"Proj/LN({len(proj_params)}) | Scales({len(scale_params)})")

    optimizer = torch.optim.AdamW([
        {"params": adapter_params, "lr": LR_ADAPTER},
        {"params": proj_params,    "lr": LR_ADAPTER},
        {"params": scale_params,   "lr": LR_SCALES},
    ])
    return optimizer

def build_text_cond():
    y_embed = torch.load(T5_EMBED_PATH, map_location="cpu")["prompt_embeds"].unsqueeze(1)
    y_embed = y_embed.to(DEVICE).to(DTYPE_PIXART)
    data_info = {
        "img_hw": torch.tensor([[512.,512.]], device=DEVICE, dtype=DTYPE_PIXART),
        "aspect_ratio": torch.tensor([1.], device=DEVICE, dtype=DTYPE_PIXART),
    }
    return y_embed, data_info

# -------------------------
# $$ [MOD-SEED-3] 稳定哈希：跨进程/跨天/跨机器一致（不会被 Python hash 随机化影响）
# -------------------------
def stable_int_hash(s: str, mod: int = 2**32) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h, 16) % mod

# -------------------------
# $$ [MOD-METRIC-3] RGB->Y：按 SR 常用的 YCbCr(BT.601) 亮度定义（浮点 0..1）
#   y = (16 + 65.481 R + 128.553 G + 24.966 B) / 255
# -------------------------
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

# -------------------------
# 7) 验证：方案B（从纯噪声开始）+ DPMSolver
# -------------------------
@torch.no_grad()
def validate_epoch(
    epoch: int,
    pixart,
    adapter,
    vae,
    val_loader,
    y_embed,
    data_info,
    lpips_fn=None,
    max_vis: int = 1,
):
    pixart.eval()
    adapter.eval()

    psnr_list, ssim_list, lpips_list = [], [], []
    vis_done = 0

    pbar = tqdm(val_loader, desc=f"Valid Ep{epoch+1}", dynamic_ncols=True)
    for it, batch in enumerate(pbar):
        hr_img_11 = batch["hr_img_11"].to(DEVICE).float()  # [B,3,512,512]
        B = hr_img_11.shape[0]

        for bi in range(B):
            item_hr = hr_img_11[bi:bi+1]
            path = batch["path"][bi]

            # $$ [MOD-SEED-4] 每张图：stable hash -> 固定退化超参 + 固定退化噪声
            seed_i = (stable_int_hash(path) + 12345) & 0xFFFFFFFF
            rng = random.Random(seed_i)
            torch_gen = torch.Generator(device="cpu").manual_seed(seed_i)

            # 退化得到 LR（在 CPU 做，保证 PIL/JPEG 兼容且稳定）
            lr_img_11 = degrade_hr_to_lr_tensor(
                item_hr.squeeze(0).detach().cpu(),
                VAL_DEGRADE_MODE,
                rng,
                torch_gen=torch_gen,  # $$ [MOD-SEED-2]
            ).unsqueeze(0)

            lr_img_11 = lr_img_11.to(DEVICE).float()

            # VAE encode（不需要梯度）
            hr_latent = vae.encode(item_hr).latent_dist.sample() * vae.config.scaling_factor   # [1,4,64,64] fp32
            lr_latent = vae.encode(lr_img_11).latent_dist.sample() * vae.config.scaling_factor # [1,4,64,64] fp32

            # 采样：严格对齐你单样本脚本的写法
            scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000, solver_order=2)
            scheduler.set_timesteps(NUM_INFER_STEPS, device=DEVICE)

            # 方案B：主干学习 HR latent 的去噪分布，推理从纯噪声开始
            run_ts = list(scheduler.timesteps)

            g = torch.Generator(device=DEVICE).manual_seed(FIXED_NOISE_SEED)
            latents = torch.randn(lr_latent.shape, generator=g, device=DEVICE, dtype=lr_latent.dtype)

            with torch.cuda.amp.autocast(enabled=False):
                cond = adapter(lr_latent.float())  # fp32 list

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

            pred_img = vae.decode(latents / vae.config.scaling_factor).sample
            pred_img_01 = torch.clamp((pred_img + 1.0) / 2.0, 0.0, 1.0)

            gt_img = vae.decode(hr_latent / vae.config.scaling_factor).sample
            gt_img_01 = torch.clamp((gt_img + 1.0) / 2.0, 0.0, 1.0)

            lr_img_01 = torch.clamp((lr_img_11 + 1.0) / 2.0, 0.0, 1.0)

            if USE_METRICS:
                # $$ [MOD-METRIC-4] PSNR/SSIM 在 Y 通道算（并可按 SR 习惯 shave border）
                if METRIC_Y_CHANNEL:
                    pred_y = rgb01_to_y01(pred_img_01)
                    gt_y   = rgb01_to_y01(gt_img_01)
                    pred_y = shave_border(pred_y, METRIC_SHAVE_BORDER)
                    gt_y   = shave_border(gt_y, METRIC_SHAVE_BORDER)
                    p = psnr(pred_y, gt_y, data_range=1.0).item()
                    s = ssim(pred_y, gt_y, data_range=1.0).item()
                else:
                    p = psnr(pred_img_01, gt_img_01, data_range=1.0).item()
                    s = ssim(pred_img_01, gt_img_01, data_range=1.0).item()

                psnr_list.append(p)
                ssim_list.append(s)

                if lpips_fn is not None:
                    pred_norm = pred_img_01 * 2.0 - 1.0
                    gt_norm   = gt_img_01 * 2.0 - 1.0
                    l = lpips_fn(pred_norm, gt_norm).item()
                    lpips_list.append(l)

            if vis_done < max_vis:
                save_path = os.path.join(VIS_DIR, f"epoch{epoch+1:03d}_it{it:04d}.png")
                lr_np = lr_img_01[0].permute(1,2,0).detach().cpu().numpy()
                gt_np = gt_img_01[0].permute(1,2,0).detach().cpu().numpy()
                pr_np = pred_img_01[0].permute(1,2,0).detach().cpu().numpy()

                plt.figure(figsize=(12,4))
                plt.subplot(1,3,1); plt.imshow(lr_np); plt.title("LR"); plt.axis("off")
                plt.subplot(1,3,2); plt.imshow(gt_np); plt.title("GT"); plt.axis("off")
                plt.subplot(1,3,3); plt.imshow(pr_np); plt.title("Pred"); plt.axis("off")
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
    return avg_psnr, avg_ssim, avg_lp


@torch.no_grad()
def validate_overfit_latent(
    epoch: int,
    pixart,
    adapter,
    vae,
    latent_loader,
    y_embed,
    data_info,
    lpips_fn=None,
):
    pixart.eval()
    adapter.eval()

    psnr_list, ssim_list, lpips_list = [], [], []
    pbar = tqdm(latent_loader, desc=f"Valid Ep{epoch+1}", dynamic_ncols=True)
    for batch in pbar:
        hr_latent = batch["hr_latent"].to(DEVICE).to(torch.float32)
        lr_latent = batch["lr_latent"].to(DEVICE).to(torch.float32)

        scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000, solver_order=2)
        scheduler.set_timesteps(NUM_INFER_STEPS, device=DEVICE)
        # 方案B：主干学习 HR latent 的去噪分布，推理从纯噪声开始
        run_ts = list(scheduler.timesteps)

        g = torch.Generator(device=DEVICE).manual_seed(FIXED_NOISE_SEED)
        latents = torch.randn(lr_latent.shape, generator=g, device=DEVICE, dtype=lr_latent.dtype)

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
                    injection_mode="hybrid",
                )
                if out.shape[1] == 8:
                    out, _ = out.chunk(2, dim=1)
            latents = scheduler.step(out.float(), t, latents.float()).prev_sample

        pred_img = vae.decode(latents / vae.config.scaling_factor).sample
        pred_img_01 = torch.clamp((pred_img + 1.0) / 2.0, 0.0, 1.0)

        gt_img = vae.decode(hr_latent / vae.config.scaling_factor).sample
        gt_img_01 = torch.clamp((gt_img + 1.0) / 2.0, 0.0, 1.0)

        if USE_METRICS:
            if METRIC_Y_CHANNEL:
                pred_y = rgb01_to_y01(pred_img_01)
                gt_y   = rgb01_to_y01(gt_img_01)
                pred_y = shave_border(pred_y, METRIC_SHAVE_BORDER)
                gt_y   = shave_border(gt_y, METRIC_SHAVE_BORDER)
                p = psnr(pred_y, gt_y, data_range=1.0).item()
                s = ssim(pred_y, gt_y, data_range=1.0).item()
            else:
                p = psnr(pred_img_01, gt_img_01, data_range=1.0).item()
                s = ssim(pred_img_01, gt_img_01, data_range=1.0).item()

            psnr_list.append(p)
            ssim_list.append(s)

            if lpips_fn is not None:
                pred_norm = pred_img_01 * 2.0 - 1.0
                gt_norm   = gt_img_01 * 2.0 - 1.0
                l = lpips_fn(pred_norm, gt_norm).item()
                lpips_list.append(l)

        if USE_METRICS and len(psnr_list) > 0:
            pbar.set_postfix({
                "PSNR": f"{sum(psnr_list)/len(psnr_list):.2f}",
                "SSIM": f"{sum(ssim_list)/len(ssim_list):.4f}",
                "LPIPS": f"{(sum(lpips_list)/len(lpips_list)):.4f}" if len(lpips_list)>0 else "NA"
            })

    avg_psnr = sum(psnr_list)/len(psnr_list) if len(psnr_list)>0 else float("nan")
    avg_ssim = sum(ssim_list)/len(ssim_list) if len(ssim_list)>0 else float("nan")
    avg_lp   = sum(lpips_list)/len(lpips_list) if len(lpips_list)>0 else float("nan")
    return avg_psnr, avg_ssim, avg_lp

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
        ):
            keep[k] = v.detach().cpu()
    return keep

def load_inject_state_dict(pixart, inject_sd: Dict[str, torch.Tensor]):
    # 部分加载：把 inject_sd 覆盖进 pixart 的 state_dict，再 strict=False load
    sd = pixart.state_dict()
    for k, v in inject_sd.items():
        if k in sd:
            sd[k] = v.to(sd[k].dtype)
    pixart.load_state_dict(sd, strict=False)

def should_keep_ckpt(psnr_v: float, lpips_v: float) -> Tuple[int, float]:
    if not math.isfinite(psnr_v):
        return (999, float("inf"))
    if psnr_v >= PSNR_SWITCH and math.isfinite(lpips_v):
        return (0, lpips_v)
    return (1, -psnr_v)

# $$ [MOD-RESUME-1] 保存“全状态 last checkpoint”，用于无歧义续跑（包含 optimizer/scaler/rng）
def save_last_full_state(
    epoch: int,
    global_step: int,
    pixart,
    adapter,
    optimizer,
    scaler,
    best_records: List[dict],
):
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "adapter": adapter.state_dict(),
        "pixart_inject": extract_inject_state_dict(pixart),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if (scaler is not None) else None,
        "best_records": best_records,
        # RNG states for strict reproducibility
        "py_random_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(payload, LAST_CKPT_PATH)

def save_topk_checkpoints(epoch: int, pixart, adapter, metrics: Tuple[float,float,float], records: List[dict]):
    psnr_v, ssim_v, lp_v = metrics
    pr, score = should_keep_ckpt(psnr_v, lp_v)

    records = [r for r in records if r["epoch"] >= epoch - (KEEP_LAST_EPOCHS - 1)]

    ckpt_name = f"epoch{epoch+1:03d}_psnr{psnr_v:.2f}_lp{lp_v if math.isfinite(lp_v) else 999:.4f}.pth"
    ckpt_path = os.path.join(CKPT_DIR, ckpt_name)

    payload = {
        "epoch": epoch,
        "psnr": psnr_v,
        "ssim": ssim_v,
        "lpips": lp_v,
        "adapter": adapter.state_dict(),
        "pixart_inject": extract_inject_state_dict(pixart),
        # 注意：这个 topK ckpt 仍然只做“展示/对比”用途；真正续跑用 LAST_CKPT_PATH
    }
    torch.save(payload, ckpt_path)

    records.append({
        "epoch": epoch,
        "path": ckpt_path,
        "priority": pr,
        "score": score,
        "psnr": psnr_v,
        "lpips": lp_v,
    })

    records = sorted(records, key=lambda r: (r["priority"], r["score"]))[:KEEP_TOPK]

    keep_paths = set(r["path"] for r in records)
    for f in glob.glob(os.path.join(CKPT_DIR, "epoch*.pth")):
        if f not in keep_paths:
            try:
                os.remove(f)
            except:
                pass

    return records

# $$ [MOD-RESUME-2] resume 加载：优先支持“last_full_state.pth”；也兼容你现有的 epoch003_*.pth（只恢复权重）
def try_resume(
    resume_path: str,
    pixart,
    adapter,
    optimizer,
    scaler,
):
    ckpt = torch.load(resume_path, map_location="cpu")

    if "adapter" in ckpt:
        adapter.load_state_dict(ckpt["adapter"], strict=True)
    else:
        raise KeyError("Resume ckpt missing key: adapter")

    if "pixart_inject" in ckpt:
        load_inject_state_dict(pixart, ckpt["pixart_inject"])
    else:
        print("⚠️ Resume ckpt has no pixart_inject (unexpected). Continue without it.")

    start_epoch = int(ckpt.get("epoch", -1)) + 1
    global_step = int(ckpt.get("global_step", 0))
    best_records = ckpt.get("best_records", [])

    # 旧格式 epoch003_*.pth：没有 optimizer/scaler/rng，必须重置
    if "optimizer" in ckpt and ckpt["optimizer"] is not None:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as e:
            print(f"⚠️ Optimizer state load failed, will reset. err={e}")

    if scaler is not None and "scaler" in ckpt and ckpt["scaler"] is not None:
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception as e:
            print(f"⚠️ Scaler state load failed, will reset. err={e}")

    if "py_random_state" in ckpt:
        try:
            random.setstate(ckpt["py_random_state"])
        except Exception:
            pass
    if "torch_rng_state" in ckpt:
        try:
            torch.set_rng_state(ckpt["torch_rng_state"])
        except Exception:
            pass
    if torch.cuda.is_available() and ("cuda_rng_state_all" in ckpt) and (ckpt["cuda_rng_state_all"] is not None):
        try:
            torch.cuda.set_rng_state_all(ckpt["cuda_rng_state_all"])
        except Exception:
            pass

    print(f"✅ [RESUME] loaded: {resume_path}")
    print(f"    start_epoch={start_epoch} | global_step={global_step} | best_records={len(best_records)}")
    return start_epoch, global_step, best_records

# -------------------------
# 8) 主训练
# -------------------------
def main():
    global SMOKE
    global SMOKE_TRAIN_SAMPLES
    global SMOKE_VAL_SAMPLES
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="run smoke: 20 train + 20 val with full logic")
    parser.add_argument("--smoke_train", type=int, default=None, help="override smoke train samples")
    parser.add_argument("--smoke_val", type=int, default=None, help="override smoke val samples")
    parser.add_argument("--smoke_epochs", type=int, default=None, help="override smoke epochs")
    parser.add_argument("--adapter_type", type=str, default="fpn_se", choices=["fpn", "fpn_se"])
    parser.add_argument("--overfit_latent_path", type=str, default=None, help="use one latent file for strict overfit train+val")
    parser.add_argument("--resume", type=str, default=None, help="path to resume ckpt (prefer last_full_state.pth)")
    args = parser.parse_args()
    SMOKE = bool(args.smoke)
    if args.smoke_train is not None:
        SMOKE_TRAIN_SAMPLES = int(args.smoke_train)
    if args.smoke_val is not None:
        SMOKE_VAL_SAMPLES = int(args.smoke_val)
    smoke_epochs = int(args.smoke_epochs) if args.smoke_epochs is not None else None

    print(f"DEVICE={DEVICE} | AMP={USE_AMP} | cudnn.enabled={torch.backends.cudnn.enabled}")
    scan_latent_schema(TRAIN_LATENT_DIR, n=50)

    train_max = SMOKE_TRAIN_SAMPLES if SMOKE else None
    val_max   = SMOKE_VAL_SAMPLES if SMOKE else None

    if args.overfit_latent_path:
        train_ds = SingleLatentDataset(args.overfit_latent_path)
    else:
        train_ds = TrainLatentDataset(TRAIN_LATENT_DIR, max_files=train_max)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE=="cuda"),
        drop_last=True
    )

    if args.overfit_latent_path:
        val_ds = SingleLatentDataset(args.overfit_latent_path)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    else:
        val_ds = ValImageDataset(VAL_HR_DIR, max_files=val_max)
        val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False,
            num_workers=0, pin_memory=(DEVICE=="cuda")
        )

    print("Loading PixArt...")
    pixart = PixArtMS_XL_2(input_size=64).to(DEVICE).to(DTYPE_PIXART).train()
    ckpt = torch.load(PIXART_PATH, map_location="cpu")
    if "state_dict" in ckpt: ckpt = ckpt["state_dict"]
    if "pos_embed" in ckpt: del ckpt["pos_embed"]
    pixart.load_state_dict(ckpt, strict=False)

    print("Loading Adapter...")
    adapter = build_adapter(args.adapter_type, in_channels=4, hidden_size=1152).to(DEVICE).train()  # FP32

    optimizer = set_trainable_A(pixart, adapter)
    scaler = GradScaler(enabled=USE_AMP)

    diffusion = IDDPM(str(1000))

    vae = AutoencoderKL.from_pretrained(VAE_PATH, local_files_only=True).to(DEVICE).float().eval()
    vae.enable_slicing()

    y_embed, data_info = build_text_cond()

    lpips_fn = None
    if USE_METRICS:
        lpips_fn = lpips.LPIPS(net="vgg").to(DEVICE).eval()
        for p in lpips_fn.parameters():
            p.requires_grad = False

    best_records = []
    global_step = 0
    start_epoch = 0

    # $$ [MOD-RESUME-3] resume 支持：可从 epoch003_*.pth 继续（至少恢复权重）
    if args.resume is not None:
        start_epoch, global_step, best_records = try_resume(
            args.resume, pixart, adapter, optimizer, scaler
        )

    # 训练
    total_epochs = EPOCHS if not SMOKE else (smoke_epochs if smoke_epochs is not None else 1)
    for epoch in range(start_epoch, total_epochs):
        pixart.train()
        adapter.train()
        pbar = tqdm(train_loader, desc=f"Train Ep{epoch+1}", dynamic_ncols=True)

        for batch in pbar:
            hr_latent = batch["hr_latent"].to(DEVICE).to(DTYPE_LATENT)  # [B,4,64,64] fp16
            lr_latent = batch["lr_latent"].to(DEVICE).to(DTYPE_LATENT)  # [B,4,64,64] fp16

            B = hr_latent.shape[0]
            t = torch.randint(0, 1000, (B,), device=DEVICE).long()
            noise = torch.randn_like(hr_latent)
            noisy = diffusion.q_sample(hr_latent, t, noise)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=False):
                cond = adapter(lr_latent.float())

            with torch.cuda.amp.autocast(enabled=USE_AMP, dtype=DTYPE_PIXART):
                out = pixart(
                    noisy.to(DTYPE_PIXART),
                    t,
                    y_embed,
                    data_info=data_info,
                    adapter_cond=cond,
                    injection_mode="hybrid",
                )
                if out.shape[1] == 8:
                    out, _ = out.chunk(2, dim=1)
                loss = F.mse_loss(out.float(), noise.float())

            if USE_AMP:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            global_step += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "seen": global_step})

        # 验证
        if args.overfit_latent_path:
            vpsnr, vssim, vlp = validate_overfit_latent(
                epoch, pixart, adapter, vae, val_loader, y_embed, data_info, lpips_fn
            )
        else:
            vpsnr, vssim, vlp = validate_epoch(
                epoch, pixart, adapter, vae, val_loader, y_embed, data_info, lpips_fn, max_vis=1
            )
        print(f"[VAL] epoch={epoch+1} PSNR={vpsnr:.2f} SSIM={vssim:.4f} LPIPS={vlp if math.isfinite(vlp) else float('nan'):.4f}")

        # $$ [MOD-RESUME-4] 每个 epoch 保存一份 last_full_state：保证你随时能“精确续跑”
        save_last_full_state(epoch, global_step, pixart, adapter, optimizer, scaler, best_records)

        # topK（最近3个epoch内）
        best_records = save_topk_checkpoints(epoch, pixart, adapter, (vpsnr, vssim, vlp), best_records)

    print("✅ Done. Kept checkpoints (topK within last epochs):")
    for r in best_records:
        print(f"  epoch={r['epoch']+1} psnr={r['psnr']:.2f} lpips={r['lpips'] if math.isfinite(r['lpips']) else float('nan'):.4f} file={os.path.basename(r['path'])}")

    print(f"✅ Resume checkpoint (full state) saved at: {LAST_CKPT_PATH}")

if __name__ == "__main__":
    main()
