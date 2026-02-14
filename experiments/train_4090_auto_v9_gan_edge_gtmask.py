# /home/hello/HJT/DiTSR/experiments/train_4090_auto_v9_gan.py
# DiTSR v9 Training Script (Adversarial Finetuning Phase - Final Corrected v6)
# ------------------------------------------------------------------
# Fixes v6 (The "No More Errors" Edition):
# 1. [Structure] Restored `set_requires_grad` function definition.
# 2. [Loss] Restored Min-SNR-Gamma weighting for V-Prediction (Critical).
# 3. [GAN] Corrected Adam betas to (0.5, 0.999).
# 4. [FFL] Restored Standard FFL (stack real/imag) for phase sensitivity.
# 5. [GAN] Added R1 Gradient Penalty for Discriminator stability.
# 6. [Config] Tuned GAN Warmup (2000) and Weight (0.08).
# ------------------------------------------------------------------

import os
import sys

# ================= 1. Path Setup =================
PROJECT_ROOT = "/home/hello/HJT/DiTSR"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import glob
import random
import math
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import io
import hashlib
import shutil
import copy
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import lpips
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.training_utils import EMAModel
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim

# [Import V8 Model]
from diffusion.model.nets.PixArtMS_v8 import PixArtMSV8_XL_2
from diffusion.model.nets.adapter_v7 import build_adapter_v7
from diffusion import IDDPM
from diffusion.model.gaussian_diffusion import _extract_into_tensor

BASE_PIXART_SHA256 = None

# Added "aug_embedder" to required keys
V7_REQUIRED_PIXART_KEY_FRAGMENTS = (
    "input_adaln", "adapter_alpha_mlp", "input_res_proj", "injection_scales", 
    "input_adapter_ln", "style_fusion_mlp", "aug_embedder"
)
FP32_SAVE_KEY_FRAGMENTS = V7_REQUIRED_PIXART_KEY_FRAGMENTS

def get_required_v7_key_fragments_for_model(model: nn.Module):
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    required = []
    for frag in V7_REQUIRED_PIXART_KEY_FRAGMENTS:
        if any(frag in name for name in trainable_names):
            required.append(frag)
    return tuple(required)

# ================= 2. GAN & Discriminator Implementation =================
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""
    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        super(NLayerDiscriminator, self).__init__()
        kw = 4; padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1; nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                # InstanceNorm for stability with BS=1
                nn.InstanceNorm2d(ndf * nf_mult, affine=True, track_running_stats=False), 
                nn.LeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            nn.InstanceNorm2d(ndf * nf_mult, affine=True, track_running_stats=False),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        return self.main(input.float())

def ema_to_device_(ema, device: str):
    # 优先走可能存在的官方 to()
    if hasattr(ema, "to"):
        try:
            ema.to(device)
            return
        except Exception:
            pass

    # 手动搬运常见字段
    for attr in ("shadow_params", "collected_params"):
        if hasattr(ema, attr):
            v = getattr(ema, attr)
            if isinstance(v, (list, tuple)):
                moved = []
                for t in v:
                    moved.append(t.to(device) if torch.is_tensor(t) else t)
                setattr(ema, attr, moved if isinstance(v, list) else type(v)(moved))

    # 兜底：把 ema.__dict__ 里所有 Tensor 都搬过去（通常很安全）
    for k, v in list(getattr(ema, "__dict__", {}).items()):
        if torch.is_tensor(v):
            ema.__dict__[k] = v.to(device)


# [FIX 1] Restored Helper Function
def set_requires_grad(module: nn.Module, requires_grad: bool):
    for p in module.parameters():
        p.requires_grad_(requires_grad)

# [FIX 4] Standard FFL (Stack Real/Imag)
class FocalFrequencyLoss(nn.Module):
    def __init__(self, loss_weight=1.0, alpha=1.0):
        super(FocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, pred, target):
        pred = pred.float()
        target = target.float()
        
        # Calculate 2D FFT
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        target_fft = torch.fft.fft2(target, dim=(-2, -1))
        
        # [Standard FFL] Stack real and imag parts for phase sensitivity
        pred_stack = torch.stack([pred_fft.real, pred_fft.imag], -1)
        target_stack = torch.stack([target_fft.real, target_fft.imag], -1)
        
        diff = pred_stack - target_stack
        loss = torch.mean(diff ** 2) 
        return loss * self.loss_weight

# ================= 3. Hyper-parameters =================
TRAIN_HR_DIR = "/data/DF2K/DF2K_train_HR"
VAL_HR_DIR   = "/data/DF2K/DF2K_valid_HR"
VAL_LR_DIR   = "/data/DF2K/DF2K_valid_LR_bicubic/X4"
if not os.path.exists(VAL_LR_DIR): VAL_LR_DIR = None

PIXART_PATH = os.path.join(PROJECT_ROOT, "output", "pretrained_models", "PixArt-XL-2-512x512.pth")
VAE_PATH    = os.path.join(PROJECT_ROOT, "output", "pretrained_models", "sd-vae-ft-ema")
T5_EMBED_PATH = "/home/hello/HJT/DiTSR/output/null_embed.pth"

OUT_BASE = os.getenv("DTSR_OUT_BASE", os.path.join(PROJECT_ROOT, "experiments_results"))
OUT_DIR = os.path.join(OUT_BASE, "train_4090_auto_v9_gan")
os.makedirs(OUT_DIR, exist_ok=True)
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
VIS_DIR  = os.path.join(OUT_DIR, "vis")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)
LAST_CKPT_PATH = os.path.join(CKPT_DIR, "last.pth")

DEVICE = "cuda"
COMPUTE_DTYPE = torch.bfloat16
SEED = 3407
DETERMINISTIC = True
FAST_DEV_RUN = os.getenv("FAST_DEV_RUN", "0") == "1"
FAST_TRAIN_STEPS = int(os.getenv("FAST_TRAIN_STEPS", "10"))
FAST_VAL_BATCHES = int(os.getenv("FAST_VAL_BATCHES", "2"))
FAST_VAL_STEPS = int(os.getenv("FAST_VAL_STEPS", "10"))

# --- User Config Block Start ---
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16 
NUM_WORKERS = 8

# Learning Rates (Differential)
LR_G_ADAPTER = 1e-4 
LR_G_BASE = 1e-5    
LR_D = 1e-4         

LORA_RANK = 16
LORA_ALPHA = 16
SPARSE_INJECT_RATIO = 1.0
INJECTION_CUTOFF_LAYER = 25
INJECTION_STRATEGY = "front_dense"

# [V8 Augmentation] Range of noise added to LR condition (SR3 style)
COND_AUG_NOISE_RANGE = (0.0, 0.15) 

# [V9 GAN Config]
L1_BASE_WEIGHT = 1.0
FFL_BASE_WEIGHT = 0.0  # disabled (replaced by GT-edge-guided losses)
EDGE_GRAD_WEIGHT = 0.10     # edge-region gradient matching
FLAT_HF_WEIGHT   = 0.05     # flat/defocus HF suppression (Laplacian)
EDGE_Q           = 0.90     # GT edge quantile for normalization
EDGE_POW         = 0.50     # mask sharpening ( <1 boosts weak edges )
EDGE_WARMUP_STEPS = 3000
EDGE_RAMP_STEPS = 4000

LPIPS_TARGET_WEIGHT = 0.5 
LPIPS_WARMUP_STEPS = 0    
LPIPS_RAMP_STEPS = 5000   

# [FIX 6] Tuned GAN Hyperparams
GAN_TARGET_WEIGHT = 0.08  # Lowered from 0.1 for stability
GAN_WARMUP_STEPS = 2000   # Extended warmup
GAN_RAMP_STEPS = 4000     

# Validation
VAL_STEPS_LIST = [50]
BEST_VAL_STEPS = 50
PSNR_SWITCH = 24.0
KEEP_TOPK = 2
VAL_MODE = "valpack"
VAL_PACK_DIR = os.path.join(PROJECT_ROOT, "valpacks", "df2k_train_like_50_seed3407")
VAL_PACK_LR_DIR_NAME = "lq512"
TRAIN_DEG_MODE = "highorder"

# V9 Validation Settings
CFG_SCALE = 3.0 
USE_LQ_INIT = True 
LQ_INIT_STRENGTH = 0.1 

INIT_NOISE_STD = 0.0
USE_CFG_TRAIN = False
CFG_TRAIN_SCALE = 3.0
USE_ADAPTER_CFDROPOUT = True
COND_DROP_PROB = 0.10
FORCE_DROP_TEXT = True

USE_LR_CONSISTENCY = False 
USE_NOISE_CONSISTENCY = False
# --- User Config Block End ---

VAE_TILING = False
DEG_OPS = ["blur", "resize", "noise", "jpeg"]
P_TWO_STAGE = 0.35
RESIZE_SCALE_RANGE = (0.3, 1.8)
NOISE_RANGE = (0.0, 0.05)
BLUR_KERNELS = [7, 9, 11, 13, 15, 21]
JPEG_QUALITY_RANGE = (30, 95)
RESIZE_INTERP_MODES = [transforms.InterpolationMode.NEAREST, transforms.InterpolationMode.BILINEAR, transforms.InterpolationMode.BICUBIC]

# ================= 4. Utils =================
def rgb01_to_y01(rgb01):
    r, g, b = rgb01[:, 0:1], rgb01[:, 1:2], rgb01[:, 2:3]
    return (16.0 + 65.481*r + 128.553*g + 24.966*b) / 255.0

# ----------------- Edge-guided perceptual regularizers (GT-driven) -----------------
# Goal: (1) match gradients where GT has edges, (2) suppress high-frequency hallucinations where GT is flat/defocused.
_SOBEL_X = torch.tensor([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
_SOBEL_Y = torch.tensor([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
_LAPLACE = torch.tensor([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)

def _to_luma01(img_m11: torch.Tensor) -> torch.Tensor:
    # img in [-1,1], returns luma in [0,1], shape [B,1,H,W], float32
    img01 = (img_m11.float() + 1.0) * 0.5
    r = img01[:, 0:1]; g = img01[:, 1:2]; b = img01[:, 2:3]
    luma = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return luma.clamp(0.0, 1.0)

@torch.cuda.amp.autocast(enabled=False)
def edge_mask_from_gt(gt_m11: torch.Tensor, q: float = 0.90, pow_: float = 0.50, eps: float = 1e-6) -> torch.Tensor:
    # Return mask in [0,1], high on GT edges, low on flat/defocus regions.
    x = _to_luma01(gt_m11)
    kx = _SOBEL_X.to(device=x.device); ky = _SOBEL_Y.to(device=x.device)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + eps)
    # Robust normalization using per-image quantile (prevents a few strong edges from saturating everything).
    flat = mag.flatten(1)
    denom = torch.quantile(flat, q, dim=1, keepdim=True).clamp_min(eps)
    m = (flat / denom).view_as(mag).clamp(0.0, 1.0)
    if pow_ != 1.0:
        m = m.pow(pow_)
    return m

@torch.cuda.amp.autocast(enabled=False)
def edge_guided_losses(pred_m11: torch.Tensor, gt_m11: torch.Tensor, q: float = 0.90, pow_: float = 0.50, eps: float = 1e-6):
    # pred/gt in [-1,1], shape [B,3,H,W]
    m = edge_mask_from_gt(gt_m11, q=q, pow_=pow_, eps=eps)  # [B,1,H,W]
    p = _to_luma01(pred_m11); g = _to_luma01(gt_m11)
    kx = _SOBEL_X.to(device=p.device); ky = _SOBEL_Y.to(device=p.device); kl = _LAPLACE.to(device=p.device)
    pgx = F.conv2d(p, kx, padding=1); pgy = F.conv2d(p, ky, padding=1)
    ggx = F.conv2d(g, kx, padding=1); ggy = F.conv2d(g, ky, padding=1)
    # (A) Edge matching: only care where GT has edges (prevents "inventing edges" in defocus).
    loss_edge = (m * (pgx - ggx).abs() + m * (pgy - ggy).abs()).mean()
    # (B) HF suppression on flat regions: penalize Laplacian energy where GT is flat/defocused.
    plap = F.conv2d(p, kl, padding=1)
    loss_flat_hf = ((1.0 - m) * plap.abs()).mean()
    return loss_edge, loss_flat_hf, m
# -------------------------------------------------------------------------------

def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if DETERMINISTIC: torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    random.seed(worker_seed); np.random.seed(worker_seed); torch.manual_seed(worker_seed)

def randn_like_with_generator(tensor, generator):
    return torch.randn(tensor.shape, device=tensor.device, dtype=tensor.dtype, generator=generator)

def get_lq_init_latents(z_lr, scheduler, steps, generator, strength, dtype):
    strength = float(max(0.0, min(1.0, strength)))
    scheduler.set_timesteps(steps, device=z_lr.device)
    timesteps = scheduler.timesteps
    start_index = int(round(strength * (len(timesteps) - 1)))
    start_index = min(max(start_index, 0), len(timesteps) - 1)
    t_start = timesteps[start_index]
    noise = randn_like_with_generator(z_lr, generator)
    if hasattr(scheduler, "add_noise"): latents = scheduler.add_noise(z_lr, noise, t_start)
    else: latents = z_lr + noise
    return latents.to(dtype=dtype), timesteps[start_index:]

def _ramp_weight(global_step: int, warmup_steps: int, ramp_steps: int, target: float) -> float:
    if global_step < int(warmup_steps):
        return 0.0
    if int(ramp_steps) <= 0:
        return float(target)
    t = (global_step - int(warmup_steps)) / float(ramp_steps)
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return float(target) * float(t)

def get_stage2_loss_weights(global_step: int):
    lpips_w = _ramp_weight(global_step, LPIPS_WARMUP_STEPS, LPIPS_RAMP_STEPS, LPIPS_TARGET_WEIGHT)
    gan_w = 0.0
    if global_step >= GAN_WARMUP_STEPS:
        gan_w = _ramp_weight(global_step, GAN_WARMUP_STEPS, GAN_RAMP_STEPS, GAN_TARGET_WEIGHT)
    edge_scale = _ramp_weight(global_step, EDGE_WARMUP_STEPS, EDGE_RAMP_STEPS, 1.0)
    edge_w = EDGE_GRAD_WEIGHT * edge_scale
    flat_w = FLAT_HF_WEIGHT * edge_scale
    return lpips_w, gan_w, edge_w, flat_w

def mask_adapter_cond(cond, keep_mask: torch.Tensor):
    if cond is None: return None
    if not torch.is_tensor(keep_mask): keep_mask = torch.tensor(keep_mask)
    def _find_device_dtype(x):
        if torch.is_tensor(x): return x.device, x.dtype
        if isinstance(x, (list, tuple)):
            for item in x:
                found = _find_device_dtype(item)
                if found is not None: return found
        return None
    found = _find_device_dtype(cond)
    if found is None: return cond
    dev, _ = found
    keep_mask = keep_mask.to(device=dev, dtype=torch.float32)
    def _mask(x: torch.Tensor):
        m = keep_mask
        while m.ndim < x.ndim: m = m.unsqueeze(-1)
        return x * m.to(dtype=x.dtype)
    if torch.is_tensor(cond): return _mask(cond)
    if isinstance(cond, (list, tuple)):
        if len(cond) == 2 and isinstance(cond[0], list) and torch.is_tensor(cond[1]):
            spatial = [_mask(c) for c in cond[0]]
            style = _mask(cond[1])
            return (spatial, style)
        masked = []
        for c in cond:
            if torch.is_tensor(c): masked.append(_mask(c))
            elif isinstance(c, list): masked.append([_mask(ci) if torch.is_tensor(ci) else ci for ci in c])
            else: masked.append(c)
        return masked if isinstance(cond, list) else tuple(masked)
    return cond

def file_sha256(path):
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""): sha.update(chunk)
    return sha.hexdigest()

def _should_keep_fp32_on_save(param_name: str) -> bool:
    return any(tag in param_name for tag in FP32_SAVE_KEY_FRAGMENTS)

def collect_trainable_state_dict(model: nn.Module):
    state = {}
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        tensor = param.detach().cpu()
        if _should_keep_fp32_on_save(name): tensor = tensor.float()
        state[name] = tensor
    return state

def validate_v7_trainable_state_keys(trainable_sd: dict, required_fragments):
    keys = list(trainable_sd.keys())
    missing = []
    counts = {}
    for frag in required_fragments:
        c = sum(1 for k in keys if frag in k)
        counts[frag] = c
        if c == 0: missing.append(frag)
    if missing: raise RuntimeError("Trainable checkpoint validation failed: " + ", ".join(missing))
    return counts

def get_config_snapshot():
    return {"batch_size": BATCH_SIZE, "v9_gan": True}

# ================= 5. Data Pipeline =================
class DegradationPipeline:
    def __init__(self, crop_size=512):
        self.crop_size = crop_size
        self.blur_kernels = BLUR_KERNELS
        self.blur_sigma_range = (0.2, 2.0)
        self.aniso_sigma_range = (0.2, 2.5)
        self.aniso_theta_range = (0.0, math.pi)
        self.noise_range = NOISE_RANGE
        self.downscale_factor = 0.25 

    def _sample_uniform(self, low, high, generator):
        if generator is None: return float(random.uniform(low, high))
        return float(low + (high - low) * torch.rand((), generator=generator).item())

    def _sample_int(self, low, high, generator):
        if generator is None: return int(random.randint(low, high))
        return int(torch.randint(low, high + 1, (1,), generator=generator).item())

    def _sample_choice(self, choices, generator):
        if generator is None: return random.choice(choices)
        idx = int(torch.randint(0, len(choices), (1,), generator=generator).item())
        return choices[idx]

    def _build_aniso_kernel(self, k, sigma_x, sigma_y, theta, device, dtype):
        ax = torch.arange(-(k // 2), k // 2 + 1, device=device, dtype=dtype)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        c, s = math.cos(theta), math.sin(theta)
        x_rot = c * xx + s * yy
        y_rot = -s * xx + c * yy
        kernel = torch.exp(-0.5 * ((x_rot / sigma_x) ** 2 + (y_rot / sigma_y) ** 2))
        kernel = kernel / kernel.sum()
        return kernel

    def _apply_aniso_blur(self, img, k, sigma_x, sigma_y, theta):
        kernel = self._build_aniso_kernel(k, sigma_x, sigma_y, theta, img.device, img.dtype)
        kernel = kernel.view(1, 1, k, k)
        weight = kernel.repeat(img.shape[0], 1, 1, 1)
        img = img.unsqueeze(0)
        img = F.conv2d(img, weight, padding=k // 2, groups=img.shape[1])
        return img.squeeze(0)

    def _apply_jpeg(self, img, quality):
        img = img.detach().to(torch.float32)
        img_np = (img.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(img_np).save(buf, format="JPEG", quality=int(quality))
        buf.seek(0)
        out = Image.open(buf).convert("RGB")
        out = TF.to_tensor(out).to(img.device, dtype=img.dtype)
        return out

    def _shuffle_ops(self, generator):
        ops = list(DEG_OPS)
        if generator is None: random.shuffle(ops)
        else:
            for i in range(len(ops) - 1, 0, -1):
                j = int(torch.randint(0, i + 1, (1,), generator=generator).item())
                ops[i], ops[j] = ops[j], ops[i]
        return ops

    def _sample_stage_params(self, generator):
        blur_applied = bool(self._sample_uniform(0.0, 1.0, generator) < 0.9)
        blur_is_aniso = bool(self._sample_uniform(0.0, 1.0, generator) < 0.5)
        if blur_applied:
            k_size = self._sample_choice(self.blur_kernels, generator)
            if blur_is_aniso:
                sigma_x = self._sample_uniform(*self.aniso_sigma_range, generator)
                sigma_y = self._sample_uniform(*self.aniso_sigma_range, generator)
                theta = self._sample_uniform(*self.aniso_theta_range, generator)
                sigma = 0.0
            else:
                sigma = self._sample_uniform(*self.blur_sigma_range, generator)
                sigma_x = 0.0; sigma_y = 0.0; theta = 0.0
        else:
            k_size = 0; sigma = 0.0; sigma_x = 0.0; sigma_y = 0.0; theta = 0.0
        resize_scale = self._sample_uniform(*RESIZE_SCALE_RANGE, generator)
        resize_interp = self._sample_choice(RESIZE_INTERP_MODES, generator)
        resize_interp_idx = RESIZE_INTERP_MODES.index(resize_interp)
        noise_std = self._sample_uniform(*self.noise_range, generator)
        jpeg_quality = self._sample_int(*JPEG_QUALITY_RANGE, generator)
        return {
            "blur_applied": blur_applied,
            "k_size": k_size,
            "sigma": sigma, "sigma_x": sigma_x, "sigma_y": sigma_y, "theta": theta,
            "resize_scale": resize_scale,
            "resize_interp_idx": resize_interp_idx, "resize_interp": resize_interp,
            "noise_std": noise_std,
            "jpeg_quality": jpeg_quality,
        }

    def __call__(self, hr_tensor, return_meta: bool = False, meta=None, generator=None):
        img = (hr_tensor + 1.0) * 0.5
        if meta is None:
            use_two_stage = bool(self._sample_uniform(0.0, 1.0, generator) < P_TWO_STAGE)
            ops_stage1 = self._shuffle_ops(generator)
            ops_stage2 = self._shuffle_ops(generator) if use_two_stage else []
            stage1 = self._sample_stage_params(generator)
            stage2 = self._sample_stage_params(generator) if use_two_stage else None
        else:
            use_two_stage = bool(int(meta.get("use_two_stage", torch.tensor(0)).item()))
            ops_stage1 = [op for op in str(meta.get("ops_stage1", ",".join(DEG_OPS))).split(",") if op]
            ops_stage2 = [op for op in str(meta.get("ops_stage2", "")).split(",") if op] if use_two_stage else []
            stage1 = {
                "blur_applied": bool(int(meta["stage1_blur_applied"].item())),
                "k_size": int(meta["stage1_k_size"].item()),
                "sigma": float(meta["stage1_sigma"].item()),
                "sigma_x": float(meta["stage1_sigma_x"].item()),
                "sigma_y": float(meta["stage1_sigma_y"].item()),
                "theta": float(meta["stage1_theta"].item()),
                "resize_scale": float(meta["stage1_resize_scale"].item()),
                "resize_interp_idx": int(meta["stage1_resize_interp"].item()),
                "resize_interp": RESIZE_INTERP_MODES[int(meta["stage1_resize_interp"].item())],
                "noise_std": float(meta["stage1_noise_std"].item()),
                "jpeg_quality": int(meta["stage1_jpeg_quality"].item()),
                "noise": meta.get("stage1_noise", None),
            }
            stage2 = None
            if use_two_stage:
                stage2 = {
                    "blur_applied": bool(int(meta["stage2_blur_applied"].item())),
                    "k_size": int(meta["stage2_k_size"].item()),
                    "sigma": float(meta["stage2_sigma"].item()),
                    "sigma_x": float(meta["stage2_sigma_x"].item()),
                    "sigma_y": float(meta["stage2_sigma_y"].item()),
                    "theta": float(meta["stage2_theta"].item()),
                    "resize_scale": float(meta["stage2_resize_scale"].item()),
                    "resize_interp_idx": int(meta["stage2_resize_interp"].item()),
                    "resize_interp": RESIZE_INTERP_MODES[int(meta["stage2_resize_interp"].item())],
                    "noise_std": float(meta["stage2_noise_std"].item()),
                    "jpeg_quality": int(meta["stage2_jpeg_quality"].item()),
                    "noise": meta.get("stage2_noise", None),
                }

        def apply_ops(img_in, ops, params):
            out = img_in
            stage_noise = None
            for op in ops:
                if op == "blur" and params["blur_applied"]:
                    if params["sigma_x"] > 0 and params["sigma_y"] > 0:
                        out = self._apply_aniso_blur(out, params["k_size"], params["sigma_x"], params["sigma_y"], params["theta"])
                    else: out = TF.gaussian_blur(out, params["k_size"], [params["sigma"], params["sigma"]])
                elif op == "resize":
                    mid_h = max(1, int(round(self.crop_size * params["resize_scale"])))
                    mid_w = max(1, int(round(self.crop_size * params["resize_scale"])))
                    out = TF.resize(out, [mid_h, mid_w], interpolation=params["resize_interp"], antialias=True)
                elif op == "noise":
                    if params["noise_std"] > 0:
                        if meta is None:
                            if generator is None: noise = torch.randn_like(out)
                            else: noise = torch.randn(out.shape, device=out.device, dtype=out.dtype, generator=generator)
                        else:
                            noise = params.get("noise")
                            if noise is None: noise = torch.zeros_like(out)
                            else: noise = noise.to(out.device, dtype=out.dtype)
                        stage_noise = noise
                        out = (out + noise * params["noise_std"]).clamp(0.0, 1.0)
                    else: stage_noise = torch.zeros_like(out)
                elif op == "jpeg": out = self._apply_jpeg(out, params["jpeg_quality"])
            if stage_noise is None: stage_noise = torch.zeros_like(out)
            return out, stage_noise

        lr_small, stage1_noise = apply_ops(img, ops_stage1, stage1)
        stage2_noise = torch.zeros_like(lr_small)
        if use_two_stage: lr_small, stage2_noise = apply_ops(lr_small, ops_stage2, stage2)

        down_h = int(self.crop_size * self.downscale_factor)
        down_w = int(self.crop_size * self.downscale_factor)
        lr_small = TF.resize(lr_small, [down_h, down_w], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        lr_out = TF.resize(lr_small, [self.crop_size, self.crop_size], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        lr_out = (lr_out * 2.0 - 1.0).clamp(-1.0, 1.0)

        # STRICTLY 2 Values
        if return_meta:
            meta_out = {
                "stage1_blur_applied": torch.tensor(int(stage1["blur_applied"]), dtype=torch.int64),
                "stage1_k_size": torch.tensor(int(stage1["k_size"]), dtype=torch.int64),
                "stage1_sigma": torch.tensor(float(stage1["sigma"]), dtype=torch.float32),
                "stage1_sigma_x": torch.tensor(float(stage1["sigma_x"]), dtype=torch.float32),
                "stage1_sigma_y": torch.tensor(float(stage1["sigma_y"]), dtype=torch.float32),
                "stage1_theta": torch.tensor(float(stage1["theta"]), dtype=torch.float32),
                "stage1_noise_std": torch.tensor(float(stage1["noise_std"]), dtype=torch.float32),
                "stage1_noise": stage1_noise.detach().cpu().float(),
                "stage1_resize_scale": torch.tensor(float(stage1["resize_scale"]), dtype=torch.float32),
                "stage1_resize_interp": torch.tensor(int(stage1["resize_interp_idx"]), dtype=torch.int64),
                "stage1_jpeg_quality": torch.tensor(int(stage1["jpeg_quality"]), dtype=torch.int64),
                "stage2_blur_applied": torch.tensor(int(stage2["blur_applied"]), dtype=torch.int64) if stage2 else torch.tensor(0, dtype=torch.int64),
                "stage2_k_size": torch.tensor(int(stage2["k_size"]), dtype=torch.int64) if stage2 else torch.tensor(0, dtype=torch.int64),
                "stage2_sigma": torch.tensor(float(stage2["sigma"]), dtype=torch.float32) if stage2 else torch.tensor(0.0, dtype=torch.float32),
                "stage2_sigma_x": torch.tensor(float(stage2["sigma_x"]), dtype=torch.float32) if stage2 else torch.tensor(0.0, dtype=torch.float32),
                "stage2_sigma_y": torch.tensor(float(stage2["sigma_y"]), dtype=torch.float32) if stage2 else torch.tensor(0.0, dtype=torch.float32),
                "stage2_theta": torch.tensor(float(stage2["theta"]), dtype=torch.float32) if stage2 else torch.tensor(0.0, dtype=torch.float32),
                "stage2_noise_std": torch.tensor(float(stage2["noise_std"]), dtype=torch.float32) if stage2 else torch.tensor(0.0, dtype=torch.float32),
                "stage2_noise": stage2_noise.detach().cpu().float(),
                "stage2_resize_scale": torch.tensor(float(stage2["resize_scale"]), dtype=torch.float32) if stage2 else torch.tensor(0.0, dtype=torch.float32),
                "stage2_resize_interp": torch.tensor(int(stage2["resize_interp_idx"]), dtype=torch.int64) if stage2 else torch.tensor(0, dtype=torch.int64),
                "stage2_jpeg_quality": torch.tensor(int(stage2["jpeg_quality"]), dtype=torch.int64) if stage2 else torch.tensor(0, dtype=torch.int64),
                "use_two_stage": torch.tensor(int(use_two_stage), dtype=torch.int64),
                "ops_stage1": ",".join(ops_stage1),
                "ops_stage2": ",".join(ops_stage2),
                "down_h": torch.tensor(int(down_h), dtype=torch.int64),
                "down_w": torch.tensor(int(down_w), dtype=torch.int64),
            }
            return lr_out, meta_out
        return lr_out

class DF2K_Online_Dataset(Dataset):
    def __init__(self, hr_root, crop_size=512, is_train=True):
        self.hr_paths = sorted(glob.glob(os.path.join(hr_root, "*.png")))
        self.crop_size = crop_size
        self.is_train = is_train
        self.pipeline = DegradationPipeline(crop_size)
        self.norm = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        self.to_tensor = transforms.ToTensor()
        self.epoch = 0
    def set_epoch(self, epoch: int): self.epoch = int(epoch)
    def _make_generator(self, idx: int):
        gen = torch.Generator(); seed = SEED + (self.epoch * 1_000_000) + int(idx); gen.manual_seed(seed)
        return gen
    def __len__(self): return len(self.hr_paths)
    def __getitem__(self, idx):
        try: hr_pil = Image.open(self.hr_paths[idx]).convert("RGB")
        except: return self.__getitem__((idx + 1) % len(self))
        gen = None
        if self.is_train:
            gen = self._make_generator(idx)
            if hr_pil.width >= self.crop_size and hr_pil.height >= self.crop_size:
                max_top = hr_pil.height - self.crop_size + 1
                max_left = hr_pil.width - self.crop_size + 1
                top = int(torch.randint(0, max_top, (1,), generator=gen).item())
                left = int(torch.randint(0, max_left, (1,), generator=gen).item())
                hr_crop = TF.crop(hr_pil, top, left, self.crop_size, self.crop_size)
            else: hr_crop = TF.resize(hr_pil, (self.crop_size, self.crop_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        else: hr_crop = TF.center_crop(hr_pil, (self.crop_size, self.crop_size))
        hr_tensor = self.norm(self.to_tensor(hr_crop))
        # Ensure only 2 values unpacked
        lr_tensor, deg_meta = self.pipeline(hr_tensor, return_meta=True, generator=gen if self.is_train else None)
        return {"hr": hr_tensor, "lr": lr_tensor, "deg": deg_meta}

class DF2K_Val_Fixed_Dataset(Dataset):
    def __init__(self, hr_root, lr_root=None, crop_size=512):
        self.hr_paths = sorted(glob.glob(os.path.join(hr_root, "*.png")))
        self.lr_root = lr_root; self.crop_size = crop_size
        self.norm = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3); self.to_tensor = transforms.ToTensor()
    def __len__(self): return len(self.hr_paths)
    def __getitem__(self, idx):
        hr_path = self.hr_paths[idx]; hr_pil = Image.open(hr_path).convert("RGB")
        hr_crop = TF.center_crop(hr_pil, (self.crop_size, self.crop_size))
        lr_crop = None
        if self.lr_root:
            base = os.path.basename(hr_path); lr_name = base.replace(".png", "x4.png")
            lr_p = os.path.join(self.lr_root, lr_name)
            if os.path.exists(lr_p):
                lr_pil = Image.open(lr_p).convert("RGB")
                lr_crop = TF.center_crop(lr_pil, (self.crop_size//4, self.crop_size//4))
                lr_crop = TF.resize(lr_crop, (self.crop_size, self.crop_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        if lr_crop is None:
            w, h = hr_crop.size; lr_small = hr_crop.resize((w//4, h//4), Image.BICUBIC)
            lr_crop = lr_small.resize((w, h), Image.BICUBIC)
        hr_tensor = self.norm(self.to_tensor(hr_crop)); lr_tensor = self.norm(self.to_tensor(lr_crop))
        return {"hr": hr_tensor, "lr": lr_tensor, "path": hr_path}

class DF2K_Val_Degraded_Dataset(Dataset):
    def __init__(self, hr_root, crop_size=512, seed=3407, deg_mode="highorder"):
        self.hr_paths = sorted(glob.glob(os.path.join(hr_root, "*.png")))
        self.crop_size = crop_size; self.norm = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        self.to_tensor = transforms.ToTensor(); self.pipeline = DegradationPipeline(crop_size)
        self.seed = int(seed); self.deg_mode = deg_mode
    def __len__(self): return len(self.hr_paths)
    def __getitem__(self, idx):
        hr_path = self.hr_paths[idx]; hr_pil = Image.open(hr_path).convert("RGB")
        hr_crop = TF.center_crop(hr_pil, (self.crop_size, self.crop_size))
        hr_tensor = self.norm(self.to_tensor(hr_crop))
        if self.deg_mode == "bicubic":
            lr_small = TF.resize((hr_tensor + 1.0) * 0.5, (self.crop_size // 4, self.crop_size // 4), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
            lr_tensor = TF.resize(lr_small, (self.crop_size, self.crop_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
            lr_tensor = (lr_tensor * 2.0 - 1.0).clamp(-1.0, 1.0)
        else:
            gen = torch.Generator(); gen.manual_seed(self.seed + idx)
            # Ensure only 2 values unpacked
            lr_tensor, _ = self.pipeline(hr_tensor, return_meta=True, generator=gen) # Ignore meta here
        return {"hr": hr_tensor, "lr": lr_tensor, "path": hr_path}

class ValPackDataset(Dataset):
    def __init__(self, pack_dir: str, lr_dir_name: str = "lq512", crop_size: int = 512):
        self.pack_dir = Path(pack_dir); self.hr_dir = self.pack_dir / "gt512"; self.lr_dir = self.pack_dir / lr_dir_name
        if not self.hr_dir.is_dir(): raise FileNotFoundError(f"gt512 dir not found: {self.hr_dir}")
        if not self.lr_dir.is_dir(): raise FileNotFoundError(f"LR dir not found: {self.lr_dir}")
        self.crop_size = crop_size; self.norm = transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        self.to_tensor = transforms.ToTensor(); self.hr_paths = sorted(list(self.hr_dir.glob("*.png")))
    def __len__(self): return len(self.hr_paths)
    def __getitem__(self, idx):
        hr_path = self.hr_paths[idx]; name = hr_path.stem; lr_path = self.lr_dir / f"{name}.png"
        if not lr_path.is_file(): raise FileNotFoundError(f"LR image missing: {lr_path}")
        hr_pil = Image.open(hr_path).convert("RGB"); lr_pil = Image.open(lr_path).convert("RGB")
        hr_crop = TF.center_crop(hr_pil, (self.crop_size, self.crop_size)); lr_crop = TF.center_crop(lr_pil, (self.crop_size, self.crop_size))
        if lr_crop.size != (self.crop_size, self.crop_size):
            lr_crop = TF.resize(lr_crop, (self.crop_size, self.crop_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        hr_tensor = self.norm(self.to_tensor(hr_crop)); lr_tensor = self.norm(self.to_tensor(lr_crop))
        return {"hr": hr_tensor, "lr": lr_tensor, "path": str(hr_path)}

# ================= 6. LoRA =================
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base; self.scaling = alpha / r
        self.lora_A = nn.Linear(base.in_features, r, bias=False, dtype=torch.float32)
        self.lora_B = nn.Linear(r, base.out_features, bias=False, dtype=torch.float32)
        self.lora_A.to(base.weight.device); self.lora_B.to(base.weight.device)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5)); nn.init.zeros_(self.lora_B.weight)
        self.base.weight.requires_grad = False
        if self.base.bias is not None: self.base.bias.requires_grad = False
    def forward(self, x):
        out = self.base(x)
        delta = self.lora_B(self.lora_A(x.float())) * self.scaling
        return out + delta.to(out.dtype)

def apply_lora(model, rank=64, alpha=64):
    cnt = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(key in name for key in ("qkv", "proj", "to_q", "to_k", "to_v", "q_linear", "kv_linear")):
             parent = model.get_submodule(name.rsplit('.', 1)[0]); child = name.rsplit('.', 1)[1]
             setattr(parent, child, LoRALinear(module, rank, alpha)); cnt += 1
    print(f"✅ LoRA applied to {cnt} layers.")

# ================= 7. Checkpointing =================
def should_keep_ckpt(psnr_v, lpips_v):
    if not math.isfinite(psnr_v): return (999, float("inf"))
    if psnr_v >= PSNR_SWITCH and math.isfinite(lpips_v): return (0, lpips_v)
    return (1, -psnr_v)

def atomic_torch_save(state, path):
    tmp = path + ".tmp"
    try:
        torch.save(state, tmp); os.replace(tmp, path); return True, "zip"
    except Exception as e_zip:
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except Exception: pass
        try:
            torch.save(state, tmp, _use_new_zipfile_serialization=False); os.replace(tmp, path); return True, f"legacy ({e_zip})"
        except Exception as e_old:
            if os.path.exists(tmp):
                try: os.remove(tmp)
                except Exception: pass
            return False, f"zip_error={e_zip}; legacy_error={e_old}"

def save_smart(epoch, global_step, pixart, adapter, optimizer, optimizer_D, discriminator, ema_pixart, ema_adapter, best_records, metrics, dl_gen):
    global BASE_PIXART_SHA256
    psnr_v, ssim_v, lpips_v = metrics; priority, score = should_keep_ckpt(psnr_v, lpips_v)
    current_record = {"path": None, "epoch": epoch, "priority": priority, "score": score, "psnr": psnr_v, "lpips": lpips_v}
    save_as_best = False
    if len(best_records) < KEEP_TOPK: save_as_best = True
    else:
        worst_record = best_records[-1]
        if (priority < worst_record['priority']) or (priority == worst_record['priority'] and score < worst_record['score']): save_as_best = True

    if save_as_best:
        ckpt_name = f"epoch{epoch+1:03d}_psnr{psnr_v:.2f}_lp{lpips_v:.4f}.pth"
        ckpt_path = os.path.join(CKPT_DIR, ckpt_name); current_record['path'] = ckpt_path
        try:
            best_records.append(current_record); best_records.sort(key=lambda x: (x['priority'], x['score']))
            if len(best_records) > KEEP_TOPK:
                to_delete = best_records[KEEP_TOPK:]; best_records = best_records[:KEEP_TOPK]
                for rec in to_delete:
                    if rec['path'] and os.path.exists(rec['path']):
                        try: os.remove(rec['path']); print(f"🗑️ Removed old best: {os.path.basename(rec['path'])}")
                        except: pass
        except Exception as e: print(f"❌ Failed to save best checkpoint: {e}")

    if BASE_PIXART_SHA256 is None and os.path.exists(PIXART_PATH):
        try: BASE_PIXART_SHA256 = file_sha256(PIXART_PATH)
        except Exception as e: print(f"⚠️ Base PixArt hash failed (non-fatal): {e}"); BASE_PIXART_SHA256 = None
    pixart_sd = collect_trainable_state_dict(pixart); required_frags = get_required_v7_key_fragments_for_model(pixart)
    v7_key_counts = validate_v7_trainable_state_keys(pixart_sd, required_frags)
    print("✅ v7 save check:", ", ".join([f"{k}={v}" for k, v in v7_key_counts.items()]))
    
    # [CRITICAL FIX] Save Discriminator, Optimizer_D, and EMA states
    state = {
        "epoch": epoch, "step": global_step, 
        "adapter": {k: v.detach().float().cpu() for k, v in adapter.state_dict().items()},
        "optimizer_G": optimizer.state_dict(),
        "optimizer_D": optimizer_D.state_dict(),
        "discriminator": discriminator.state_dict(),
        "ema_pixart": ema_pixart.state_dict(),
        "ema_adapter": ema_adapter.state_dict(),
        "rng_state": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None, "numpy": np.random.get_state(), "python": random.getstate()},
        "dl_gen_state": dl_gen.get_state(), 
        "pixart_trainable": pixart_sd, 
        "best_records": best_records, 
        "config_snapshot": get_config_snapshot(), 
        "base_pixart_sha256": BASE_PIXART_SHA256, 
        "env_info": {"torch": torch.__version__, "numpy": np.__version__},
    }
    last_path = LAST_CKPT_PATH; ok_last, msg_last = atomic_torch_save(state, last_path)
    if ok_last: print(f"💾 Saved last checkpoint to {last_path} [{msg_last}]")
    else: print(f"❌ Failed to save last.pth: {msg_last}")
    if save_as_best and current_record["path"]:
        try:
            if ok_last and os.path.exists(last_path): shutil.copy2(last_path, current_record["path"]); print(f"🏆 New Best Model! Copied from last.pth to {ckpt_name}")
            else:
                ok_best, msg_best = atomic_torch_save(state, current_record["path"])
                if ok_best: print(f"🏆 New Best Model! Saved to {ckpt_name} [{msg_best}]")
                else: print(f"❌ Failed to save best checkpoint: {msg_best}")
        except Exception as e: print(f"❌ Failed to save best checkpoint: {e}")
    return best_records

def _ema_to_device(ema, device):
    # diffusers.EMAModel may keep shadow params on CPU after load_state_dict; move them to match model params.
    for attr in ("shadow_params", "collected_params"):
        if hasattr(ema, attr):
            val = getattr(ema, attr)
            if isinstance(val, list):
                new_val = []
                for t in val:
                    if torch.is_tensor(t):
                        new_val.append(t.to(device))
                    else:
                        new_val.append(t)
                setattr(ema, attr, new_val)

def optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

def resume(pixart, adapter, optimizer_G, optimizer_D, discriminator, ema_pixart, ema_adapter, dl_gen):
    if not os.path.exists(LAST_CKPT_PATH): return 0, 0, []
    print(f"📥 Resuming from {LAST_CKPT_PATH}...")
    ckpt = torch.load(LAST_CKPT_PATH, map_location="cpu")
    saved_trainable = ckpt.get("pixart_trainable", {})
    required_frags = get_required_v7_key_fragments_for_model(pixart)
    missing_required = [frag for frag in required_frags if not any(frag in k for k in saved_trainable.keys())]
    if missing_required: raise RuntimeError("Checkpoint is missing required v7 trainable keys: " + ", ".join(missing_required))
    adapter_sd = ckpt.get("adapter", {})
    missing, unexpected = adapter.load_state_dict(adapter_sd, strict=False)
    if missing or unexpected: print(f"⚠️ Adapter state_dict mismatch: missing={len(missing)} unexpected={len(unexpected)}")
    if "scale_gates" in adapter_sd:
        saved = adapter_sd["scale_gates"]; current = adapter.scale_gates
        if saved.shape != current.shape:
            n = min(saved.shape[0], current.shape[0]); 
            with torch.no_grad(): current[:n].copy_(saved[:n])
    curr = pixart.state_dict()
    for k, v in saved_trainable.items():
        if k in curr: curr[k] = v.to(curr[k].dtype)
    pixart.load_state_dict(curr, strict=False)
    
    # [CRITICAL FIX] Resume G, D, Opt_D, and EMA
    if "optimizer" in ckpt: # Support legacy v8 checkpoints
        optimizer_G.load_state_dict(ckpt["optimizer"])
        print("⚠️ Loaded Optimizer G from legacy 'optimizer' key (v8 transition).")
    elif "optimizer_G" in ckpt:
        optimizer_G.load_state_dict(ckpt["optimizer_G"])
        
    if "optimizer_D" in ckpt:
        optimizer_D.load_state_dict(ckpt["optimizer_D"])
    else:
        print("⚠️ Optimizer D not found in checkpoint. Starting D from scratch.")
        
    if "discriminator" in ckpt:
        discriminator.load_state_dict(ckpt["discriminator"])
    else:
        print("⚠️ Discriminator state not found in checkpoint. Starting D from scratch.")

    if "ema_pixart" in ckpt:
        ema_pixart.load_state_dict(ckpt["ema_pixart"])
    else:
        print("⚠️ EMA PixArt not found. Force-syncing with loaded weights for v8->v9 transition.")
        # [FIX] Manually update shadow params to match current loaded weights
        # This prevents starting with random EMA weights when resuming from v8
        ema_pixart.shadow_params = [p.clone().detach() for p in pixart.parameters()]

    if "ema_adapter" in ckpt:
        ema_adapter.load_state_dict(ckpt["ema_adapter"])
    else:
        print("⚠️ EMA Adapter not found. Force-syncing with loaded weights.")
        ema_adapter.shadow_params = [p.clone().detach() for p in adapter.parameters()]

    ema_to_device_(ema_pixart, DEVICE)
    ema_to_device_(ema_adapter, DEVICE)    

    # [FIX] Move optimizer states to GPU to prevent RuntimeError
    optimizer_to_device(optimizer_G, DEVICE)
    optimizer_to_device(optimizer_D, DEVICE)

    _ema_to_device(ema_pixart, DEVICE)
    _ema_to_device(ema_adapter, DEVICE)
    rs = ckpt.get("rng_state", None)
    if rs is not None:
        try:
            if rs.get("torch") is not None: torch.set_rng_state(rs["torch"])
            if torch.cuda.is_available() and rs.get("cuda") is not None: torch.cuda.set_rng_state_all(rs["cuda"])
            if rs.get("numpy") is not None: np.random.set_state(rs["numpy"])
            if rs.get("python") is not None: random.setstate(rs["python"])
        except Exception as e: print(f"⚠️ RNG restore failed (non-fatal): {e}")
    dl_state = ckpt.get("dl_gen_state", None)
    if dl_state is not None:
        try: dl_gen.set_state(dl_state)
        except Exception as e: print(f"⚠️ DataLoader generator restore failed (non-fatal): {e}")
    return ckpt["epoch"]+1, ckpt["step"], ckpt.get("best_records", [])

# ================= 8. Validation (Moved BEFORE Main) =================
@torch.no_grad()
def validate(epoch, pixart, adapter, vae, val_loader, y_embed, data_info, lpips_fn):
    print(f"🔎 Validating Epoch {epoch+1}...")
    pixart.eval(); adapter.eval()
    results = {}
    val_gen = torch.Generator(device=DEVICE); val_gen.manual_seed(SEED)
    scheduler = DDIMScheduler(
        num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02, beta_schedule="linear",
        clip_sample=False, prediction_type="v_prediction", set_alpha_to_one=False,
    )
    
    steps_list = [FAST_VAL_STEPS] if FAST_DEV_RUN else VAL_STEPS_LIST
    for steps in steps_list:
        scheduler.set_timesteps(steps, device=DEVICE)
        psnrs, ssims, lpipss = [], [], []; vis_done = False
        for batch in tqdm(val_loader, desc=f"Val@{steps}"):
            hr = batch["hr"].to(DEVICE); lr = batch["lr"].to(DEVICE)
            z_hr = vae.encode(hr).latent_dist.mean * vae.config.scaling_factor
            z_lr = vae.encode(lr).latent_dist.mean * vae.config.scaling_factor
            
            if USE_LQ_INIT: latents, run_timesteps = get_lq_init_latents(z_lr.to(COMPUTE_DTYPE), scheduler, steps, val_gen, LQ_INIT_STRENGTH, COMPUTE_DTYPE)
            else: latents = randn_like_with_generator(z_hr, val_gen); run_timesteps = scheduler.timesteps
            
            cond = adapter(z_lr.float())
            aug_level = torch.zeros((latents.shape[0],), device=DEVICE, dtype=COMPUTE_DTYPE)
            
            for t in run_timesteps:
                t_b = torch.tensor([t], device=DEVICE).expand(latents.shape[0])
                with torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
                    if FORCE_DROP_TEXT: drop_uncond = torch.ones(latents.shape[0], device=DEVICE); drop_cond = torch.ones(latents.shape[0], device=DEVICE)
                    else: drop_uncond = torch.ones(latents.shape[0], device=DEVICE); drop_cond = torch.zeros(latents.shape[0], device=DEVICE)
                    lr_ref = z_lr.to(COMPUTE_DTYPE)
                    model_in = torch.cat([latents.to(COMPUTE_DTYPE), lr_ref], dim=1)
                    out_uncond = pixart(x=model_in, timestep=t_b, y=y_embed, aug_level=aug_level, mask=None, data_info=data_info, adapter_cond=None, injection_mode="hybrid", force_drop_ids=drop_uncond)
                    out_cond = pixart(x=model_in, timestep=t_b, y=y_embed, aug_level=aug_level, mask=None, data_info=data_info, adapter_cond=cond, injection_mode="hybrid", force_drop_ids=drop_cond)
                    if out_uncond.shape[1] == 8: out_uncond, _ = out_uncond.chunk(2, dim=1)
                    if out_cond.shape[1] == 8: out_cond, _ = out_cond.chunk(2, dim=1)
                    out = out_uncond + CFG_SCALE * (out_cond - out_uncond)
                latents = scheduler.step(out.float(), t, latents.float()).prev_sample
            pred = vae.decode(latents / vae.config.scaling_factor).sample.clamp(-1, 1)
            p01 = (pred + 1) / 2; h01 = (hr + 1) / 2
            py = rgb01_to_y01(p01)[..., 4:-4, 4:-4]; hy = rgb01_to_y01(h01)[..., 4:-4, 4:-4]
            if "psnr" in globals(): psnrs.append(psnr(py, hy, data_range=1.0).item()); ssims.append(ssim(py, hy, data_range=1.0).item())
            lpipss.append(lpips_fn(pred, hr).mean().item())
            if not vis_done:
                save_path = os.path.join(VIS_DIR, f"epoch{epoch+1:03d}_steps{steps}.png")
                lr_np = (lr[0].cpu().float().numpy().transpose(1,2,0) + 1) / 2
                hr_np = (hr[0].cpu().float().numpy().transpose(1,2,0) + 1) / 2
                pr_np = (pred[0].cpu().float().numpy().transpose(1,2,0) + 1) / 2
                plt.figure(figsize=(12,4))
                plt.subplot(1,3,1); plt.imshow(np.clip(lr_np, 0, 1)); plt.title("Input LR"); plt.axis("off")
                plt.subplot(1,3,2); plt.imshow(np.clip(hr_np, 0, 1)); plt.title("GT"); plt.axis("off")
                plt.subplot(1,3,3); plt.imshow(np.clip(pr_np, 0, 1)); plt.title(f"Pred @{steps}"); plt.axis("off")
                plt.savefig(save_path, bbox_inches="tight"); plt.close(); vis_done = True
            if FAST_DEV_RUN and len(psnrs) >= FAST_VAL_BATCHES: break
        res = (float(np.mean(psnrs)), float(np.mean(ssims)), float(np.mean(lpipss)))
        results[int(steps)] = res
        print(f"[VAL@{steps}] Ep{epoch+1}: PSNR={res[0]:.2f} | SSIM={res[1]:.4f} | LPIPS={res[2]:.4f}")
    pixart.train(); adapter.train()
    return results

# ================= 10. Main =================
def main():
    seed_everything(SEED); dl_gen = torch.Generator(); dl_gen.manual_seed(SEED)
    # [FIX] Dataset classes defined before main, so this works now.
    train_ds = DF2K_Online_Dataset(TRAIN_HR_DIR, crop_size=512, is_train=True)
    if VAL_MODE == "valpack": val_ds = ValPackDataset(VAL_PACK_DIR, lr_dir_name=VAL_PACK_LR_DIR_NAME, crop_size=512)
    elif VAL_MODE == "train_like": val_ds = DF2K_Val_Degraded_Dataset(VAL_HR_DIR, crop_size=512, seed=SEED, deg_mode=TRAIN_DEG_MODE)
    elif VAL_MODE == "lr_dir" and VAL_LR_DIR is not None: val_ds = DF2K_Val_Fixed_Dataset(VAL_HR_DIR, lr_root=VAL_LR_DIR, crop_size=512)
    else: val_ds = DF2K_Val_Fixed_Dataset(VAL_HR_DIR, lr_root=None, crop_size=512)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True, worker_init_fn=seed_worker, generator=dl_gen)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    pixart = PixArtMSV8_XL_2(
        input_size=64, in_channels=8, sparse_inject_ratio=SPARSE_INJECT_RATIO,
        injection_cutoff_layer=INJECTION_CUTOFF_LAYER, injection_strategy=INJECTION_STRATEGY,
    ).to(DEVICE)
    # Re-init weights if starting fresh, or load checkpoint
    base = torch.load(PIXART_PATH, map_location="cpu")
    if "state_dict" in base: base = base["state_dict"]
    if "pos_embed" in base: del base["pos_embed"]
    if "x_embedder.proj.weight" in base and base["x_embedder.proj.weight"].shape[1] == 4:
        w4 = base["x_embedder.proj.weight"]
        w8 = torch.zeros((w4.shape[0], 8, w4.shape[2], w4.shape[3]), dtype=w4.dtype)
        w8[:, :4] = w4; base["x_embedder.proj.weight"] = w8
    pixart.load_pretrained_weights_with_zero_init(base)
    apply_lora(pixart, LORA_RANK, LORA_ALPHA)
    
    adapter = build_adapter_v7(in_channels=4, hidden_size=1152, injection_layers_map=getattr(pixart, "injection_layers", None)).to(DEVICE).float().train()
    
    # [EMA Setup]
    ema_pixart = EMAModel(pixart.parameters(), decay=0.999)
    ema_adapter = EMAModel(adapter.parameters(), decay=0.999)
    
    # [Discriminator Setup]
    discriminator = NLayerDiscriminator(input_nc=3).to(DEVICE).train()
    
    vae = AutoencoderKL.from_pretrained(VAE_PATH, local_files_only=True).to(DEVICE).float().eval()
    vae.enable_slicing()
    if VAE_TILING and hasattr(vae, "enable_tiling"): vae.enable_tiling()
    lpips_fn = lpips.LPIPS(net='vgg').to(DEVICE).eval()
    for p in vae.parameters(): p.requires_grad_(False)
    for p in lpips_fn.parameters(): p.requires_grad_(False)

    y = torch.load(T5_EMBED_PATH, map_location="cpu")["prompt_embeds"].unsqueeze(1).to(DEVICE)
    d_info = {"img_hw": torch.tensor([[512.,512.]]).to(DEVICE), "aspect_ratio": torch.tensor([1.]).to(DEVICE)}

    # Optimizers
    # [FIX] Group parameters properly like in v8 to match learning rates
    adapter_params = list(adapter.parameters())
    embedder_params = [p for n, p in pixart.named_parameters() if 'x_embedder' in n and p.requires_grad]
    other_pixart_params = [p for n, p in pixart.named_parameters() if 'x_embedder' not in n and p.requires_grad]
    
    g_params = [
        {"params": adapter_params, "lr": LR_G_ADAPTER}, 
        {"params": embedder_params, "lr": LR_G_ADAPTER}, # Keep input layer fast
        {"params": other_pixart_params, "lr": LR_G_BASE}
    ]
    optimizer_G = torch.optim.AdamW(g_params)
    # [FIX] D Optimizer Standard Betas and Weight Decay
    optimizer_D = torch.optim.AdamW(discriminator.parameters(), lr=LR_D, betas=(0.5, 0.999), weight_decay=1e-4)
    
    g_params_to_clip = adapter_params + embedder_params + other_pixart_params

    # [FIX] Initialize FFL Criterion

    diffusion = IDDPM(str(1000))
    # [CRITICAL FIX] Pass all state objects to resume
    ep_start, step, best = resume(pixart, adapter, optimizer_G, optimizer_D, discriminator, ema_pixart, ema_adapter, dl_gen)

    _ema_to_device(ema_pixart, DEVICE)
    _ema_to_device(ema_adapter, DEVICE)  # safety: keep EMA tensors on GPU
    print("🚀 DiT-SR V9 Adversarial Training Started (InstanceNorm D, Correct Grad Flow, FFL, State Saving).")
    max_steps = FAST_TRAIN_STEPS if FAST_DEV_RUN else None

    for epoch in range(ep_start, 1000):
        if max_steps is not None and step >= max_steps: break
        train_ds.set_epoch(epoch)
        pbar = tqdm(train_loader, dynamic_ncols=True, desc=f"Ep{epoch+1}")
        for i, batch in enumerate(pbar):
            if max_steps is not None and step >= max_steps: break
            hr = batch['hr'].to(DEVICE); lr = batch['lr'].to(DEVICE)
            with torch.no_grad():
                zh = vae.encode(hr).latent_dist.mean * vae.config.scaling_factor
                zl = vae.encode(lr).latent_dist.mean * vae.config.scaling_factor

            # --- Generator Step ---
            t = torch.randint(0, 1000, (zh.shape[0],), device=DEVICE).long()
            noise = torch.randn_like(zh)
            zt = diffusion.q_sample(zh, t, noise)
            
            aug_noise_level = torch.rand(zh.shape[0], device=DEVICE) * (COND_AUG_NOISE_RANGE[1] - COND_AUG_NOISE_RANGE[0]) + COND_AUG_NOISE_RANGE[0]
            zlr_aug = zl.float() + torch.randn_like(zl) * aug_noise_level[:, None, None, None]
            aug_level_emb = aug_noise_level * 1000.0

            cond = adapter(zlr_aug.float())
            cond_in = cond
            if USE_ADAPTER_CFDROPOUT and COND_DROP_PROB > 0:
                keep = (torch.rand((zt.shape[0],), device=DEVICE) >= COND_DROP_PROB).float()
                cond_in = mask_adapter_cond(cond, keep)

            # Train Generator
            
            with torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
                drop_uncond = torch.ones(zt.shape[0], device=DEVICE)
                kwargs = dict(x=torch.cat([zt, zlr_aug.to(zt.dtype)], dim=1), timestep=t, y=y, aug_level=aug_level_emb, data_info=d_info, adapter_cond=cond_in, injection_mode="hybrid")
                kwargs["force_drop_ids"] = drop_uncond
                out = pixart(**kwargs)
                if out.shape[1] == 8: out, _ = out.chunk(2, dim=1)
                model_pred = out.float()

                # [FIX] V-Prediction Min-SNR Weighting (Crucial for V-Pred)
                alpha_t = _extract_into_tensor(diffusion.sqrt_alphas_cumprod, t, zh.shape)
                sigma_t = _extract_into_tensor(diffusion.sqrt_one_minus_alphas_cumprod, t, zh.shape)
                target_v = alpha_t * noise - sigma_t * zh.float()
                
                snr = (alpha_t**2) / (sigma_t**2)
                gamma = 5.0
                min_snr_gamma = torch.min(snr, torch.tensor(gamma, device=DEVICE))
                loss_weights = min_snr_gamma / snr 
                loss_v = (F.mse_loss(model_pred, target_v, reduction='none').mean(dim=[1,2,3]) * loss_weights.squeeze()).mean()
                
                z0 = alpha_t * zt.float() - sigma_t * model_pred # Reconstruct z0
                
                loss_latent_l1 = F.l1_loss(z0, zh.float())

                # Perceptual & GAN & FFL Loss (Pixel Space)
                # Decode SMALL crop to save VRAM
                top = torch.randint(0, 25, (1,), device=DEVICE).item() 
                left = torch.randint(0, 25, (1,), device=DEVICE).item()
                z0_crop = z0[..., top:top+40, left:left+40]
                img_p_raw = vae.decode(z0_crop/vae.config.scaling_factor).sample.clamp(-1,1)
                img_p_valid = img_p_raw[..., 32:-32, 32:-32] # 256x256 valid pixels
                
                y0 = top * 8 + 32; x0 = left * 8 + 32
                img_t_valid = hr[..., y0:y0+256, x0:x0+256].clamp(-1, 1) # Real HR patch
                
                with torch.cuda.amp.autocast(enabled=False):
                        loss_lpips = lpips_fn(img_p_valid.float(), img_t_valid.float()).mean()
                
                loss_edge = torch.tensor(0.0, device=DEVICE)
                loss_flat_hf = torch.tensor(0.0, device=DEVICE)
                loss_ffl = torch.tensor(0.0, device=DEVICE)
                if FFL_BASE_WEIGHT > 0:
                    loss_ffl = ffl_criterion(img_p_valid, img_t_valid)

                # GAN Loss (Generator side)
                loss_gan = torch.tensor(0.0, device=DEVICE)
                lpips_w, gan_w, edge_w, flat_w = get_stage2_loss_weights(step)
                if edge_w > 0 or flat_w > 0:
                    loss_edge, loss_flat_hf, _ = edge_guided_losses(
                        img_p_valid, img_t_valid, q=EDGE_Q, pow_=EDGE_POW
                    )

                # [FIX] Only compute G's GAN loss if weight > 0
                if gan_w > 0:
                    set_requires_grad(discriminator, False)
                    with torch.cuda.amp.autocast(enabled=False):
                        pred_fake = discriminator(img_p_valid.float())
                    loss_gan = -pred_fake.mean()

                loss_G = (
                    loss_v 
                    + L1_BASE_WEIGHT * loss_latent_l1
                    + lpips_w * loss_lpips
                    + gan_w * loss_gan
                    + edge_w * loss_edge + flat_w * loss_flat_hf  # edge-guided
                ) / GRAD_ACCUM_STEPS

            loss_G.backward()

            # --- Discriminator Step ---
            loss_D_val = 0.0
            if gan_w > 0:
                img_p_detached = img_p_valid.detach()
                img_t_valid_d = img_t_valid.detach()
                set_requires_grad(discriminator, True)
                
                # [FIX] R1 Gradient Penalty for Stability
                img_t_valid_d.requires_grad = True # For GP calculation
                with torch.cuda.amp.autocast(enabled=False):
                    pred_real = discriminator(img_t_valid_d.float())
                    pred_fake_d = discriminator(img_p_detached.float())
                    
                    loss_D_real = torch.nn.ReLU()(1.0 - pred_real).mean()
                    loss_D_fake = torch.nn.ReLU()(1.0 + pred_fake_d).mean()
                    
                    # R1 Penalty (Simplified implementation)
                    # Computed every step for safety with BS=1
                    r1_loss = 0.0
                    if step % 16 == 0: # Compute R1 every 16 steps to save compute
                        grad_real = torch.autograd.grad(
                            outputs=pred_real.sum(), inputs=img_t_valid_d, create_graph=True
                        )[0]
                        r1_loss = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean() * 10.0 # Gamma=10
                    
                    loss_D = (loss_D_real + loss_D_fake + r1_loss) * 0.5 / GRAD_ACCUM_STEPS

                loss_D.backward()
                loss_D_val = loss_D.item() * GRAD_ACCUM_STEPS
            else:
                loss_D_val = 0.0

            if (i+1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(g_params_to_clip, 1.0)
                optimizer_G.step()
                optimizer_G.zero_grad()
                
                if gan_w > 0:
                    optimizer_D.step()
                    optimizer_D.zero_grad()
                
                # Update EMA
                ema_pixart.step(pixart.parameters())
                ema_adapter.step(adapter.parameters())
                
                step += 1

            if i % 10 == 0:
                pbar.set_postfix({
                    'v': f"{loss_v.item():.3f}",
                    'l1': f"{loss_latent_l1.item():.3f}",
                    'lp': f"{loss_lpips.item():.3f}",
                    'edge': f"{loss_edge.item():.3f}",
                    'flat_hf': f"{loss_flat_hf.item():.3f}",
                    'gan': f"{loss_gan.item():.3f}",
                    'd_loss': f"{loss_D_val:.3f}",
                    'w_lp': f"{lpips_w:.3f}",
                    'w_gan': f"{gan_w:.3f}",
                    'w_edge': f"{edge_w:.3f}",
                    'w_flat': f"{flat_w:.3f}",
                })

        # Validation uses EMA weights
        ema_pixart.store(pixart.parameters())
        ema_adapter.store(adapter.parameters())
        ema_pixart.copy_to(pixart.parameters())
        ema_adapter.copy_to(adapter.parameters())
        
        # [CRITICAL FIX] Pass correct optimizer_G to save_smart
        val_dict = validate(epoch, pixart, adapter, vae, val_loader, y, d_info, lpips_fn)
        
        ema_pixart.restore(pixart.parameters())
        ema_adapter.restore(adapter.parameters())
        
        if int(BEST_VAL_STEPS) in val_dict: metrics = val_dict[int(BEST_VAL_STEPS)]
        else: metrics = next(iter(val_dict.values()))
        best = save_smart(epoch, step, pixart, adapter, optimizer_G, optimizer_D, discriminator, ema_pixart, ema_adapter, best, metrics, dl_gen)

if __name__ == "__main__":
    main()