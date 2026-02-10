# /home/hello/HJT/DiTSR/experiments/train_4090_auto_v6.py
# DiTSR v6 Training Script
# Goal: strong restoration-first training with fixed valpack validation.
# Strategy: no text conditioning, strong LR-consistency, LQ-init enabled.

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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import lpips
from diffusers import AutoencoderKL, DDIMScheduler
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim

from diffusion.model.nets.PixArtMS_v6 import PixArtMSV6_XL_2
from diffusion.model.nets.adapter_v6 import build_adapter_v6
from diffusion import IDDPM
from diffusion.model.gaussian_diffusion import _extract_into_tensor

BASE_PIXART_SHA256 = None

# v6 critical module key fragments that must be present in saved trainable weights.
V6_REQUIRED_PIXART_KEY_FRAGMENTS = (
    "input_adaln",
    "adapter_alpha_mlp",
    "input_res_proj",
    "injection_scales",
    "input_adapter_ln",
    "style_fusion_mlp",
)
FP32_SAVE_KEY_FRAGMENTS = (
    "input_adaln",
    "adapter_alpha_mlp",
    "input_res_proj",
    "style_fusion_mlp",
)


def get_required_v6_key_fragments_for_model(model: nn.Module):
    """Only require fragments that actually have trainable parameters in current model."""
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    required = []
    for frag in V6_REQUIRED_PIXART_KEY_FRAGMENTS:
        if any(frag in name for name in trainable_names):
            required.append(frag)
    return tuple(required)


# ================= 2. Hyper-parameters =================
# Paths
TRAIN_HR_DIR = "/data/DF2K/DF2K_train_HR"
VAL_HR_DIR   = "/data/DF2K/DF2K_valid_HR"
VAL_LR_DIR   = "/data/DF2K/DF2K_valid_LR_bicubic/X4"
if not os.path.exists(VAL_LR_DIR): VAL_LR_DIR = None

PIXART_PATH = os.path.join(PROJECT_ROOT, "output", "pretrained_models", "PixArt-XL-2-512x512.pth")
VAE_PATH    = os.path.join(PROJECT_ROOT, "output", "pretrained_models", "sd-vae-ft-ema")
T5_EMBED_PATH = os.path.join(PROJECT_ROOT, "output", "quality_embed.pth")

OUT_DIR = os.path.join(PROJECT_ROOT, "experiments_results", "train_4090_auto_v6")
os.makedirs(OUT_DIR, exist_ok=True)
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
VIS_DIR  = os.path.join(OUT_DIR, "vis")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)
LAST_CKPT_PATH = os.path.join(CKPT_DIR, "last.pth")  # unified with save_smart/resume

DEVICE = "cuda"
COMPUTE_DTYPE = torch.bfloat16
SEED = 3407
DETERMINISTIC = True
FAST_DEV_RUN = os.getenv("FAST_DEV_RUN", "0") == "1"
FAST_TRAIN_STEPS = int(os.getenv("FAST_TRAIN_STEPS", "10"))
FAST_VAL_BATCHES = int(os.getenv("FAST_VAL_BATCHES", "2"))
FAST_VAL_STEPS = int(os.getenv("FAST_VAL_STEPS", "10"))

# Optimization Dynamics
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16 # Effective Batch = 16
NUM_WORKERS = 8
LR_BASE = 1e-5
LORA_RANK = 16
LORA_ALPHA = 16
SPARSE_INJECT_RATIO = 0.5

# [Curriculum Logic]
# Step 0-5000: L1=1.0, LPIPS=0.0 (Fix Color/Structure)
# Step 5000-10000: LPIPS ramps up to 0.5
WARMUP_STEPS = 0
RAMP_UP_STEPS = 3500
TARGET_LPIPS_WEIGHT = 0.30  # LPIPS as primary perceptual objective after warmup
LPIPS_BASE_WEIGHT = 0.01
L1_BASE_WEIGHT = 1.0
L1_MIN_WEIGHT = 0.5

# Validation
VAL_STEPS_LIST = [50]   # training-time validation uses ONLY 50-step sampling (main metric / best ckpt)
BEST_VAL_STEPS = 50         # user choice: select best checkpoint by VAL@50
PSNR_SWITCH = 24.0
KEEP_TOPK = 1
VAL_MODE = "valpack"  # fixed valpack validation for alignment with eval pipeline
VAL_PACK_DIR = os.path.join(PROJECT_ROOT, "valpacks", "df2k_train_like_50_seed3407")
VAL_PACK_LR_DIR_NAME = "lq512"  # "lq512" (restoration) or "lq128" (x4 SR)
TRAIN_DEG_MODE = "highorder"  # "bicubic" / "highorder"
CFG_SCALE = 3.0
USE_LQ_INIT = True
INIT_NOISE_STD = 0.05
LQ_INIT_STRENGTH = 0.5  # 0~1, start from this noise level when using LQ-init
USE_CFG_TRAIN = False
CFG_TRAIN_SCALE = 3.0
USE_ADAPTER_CFDROPOUT = True  # adapter-only classifier-free dropout (train-time)
COND_DROP_PROB = 0.10         # probability to drop adapter condition per-sample
FORCE_DROP_TEXT = True        # drop text conditioning in train/val
USE_NOISE_CONSISTENCY = True
NOISE_CONS_WARMUP = 2000
NOISE_CONS_RAMP = 6000
NOISE_CONS_WEIGHT = 0.1
NOISE_CONS_PROB = 0.5
USE_LR_CONSISTENCY = True
LR_CONS_PROB = 1.0  # compute strict LR-consistency loss every step
LR_CONS_WARMUP = 1000
LR_CONS_RAMP = 5000
LR_CONS_WEIGHT = 1.0
# LR-consistency replay mode:
# - "patch": low-memory patch-only replay (default, stable on 24GB)
# - "full": full-image replay (may OOM on 24GB)
# - "mixed": full replay every N steps, patch otherwise
LR_CONS_MODE = "patch"  # "patch" | "full" | "mixed"
LR_CONS_FULL_EVERY = 0  # only used when LR_CONS_MODE="mixed"
VAE_TILING = False      # enable VAE tiling for full replay (reduces peak mem)
DEG_OPS = ["blur", "resize", "noise", "jpeg"]
P_TWO_STAGE = 0.35
RESIZE_SCALE_RANGE = (0.3, 1.8)
NOISE_RANGE = (0.0, 0.05)
BLUR_KERNELS = [7, 9, 11, 13, 15, 21]
JPEG_QUALITY_RANGE = (30, 95)
RESIZE_INTERP_MODES = [
    transforms.InterpolationMode.NEAREST,
    transforms.InterpolationMode.BILINEAR,
    transforms.InterpolationMode.BICUBIC,
]
# ================= 3. Logic Functions =================
def get_loss_weights(global_step):
    # Anchor losses are always active (diffusion eps MSE + image L1).
    # LPIPS is introduced later and weaker to reduce early hallucination.
    # LR-consistency (strict replayed degradation) ramps up BEFORE LPIPS to anchor details.
    weights = {'mse': 1.0, 'l1': L1_BASE_WEIGHT}

    # --- LR consistency schedule (optional) ---
    if USE_LR_CONSISTENCY:
        if global_step < LR_CONS_WARMUP:
            weights['cons'] = 0.0
        elif global_step < (LR_CONS_WARMUP + LR_CONS_RAMP):
            p = (global_step - LR_CONS_WARMUP) / LR_CONS_RAMP
            weights['cons'] = LR_CONS_WEIGHT * p
        else:
            weights['cons'] = LR_CONS_WEIGHT
    else:
        weights['cons'] = 0.0

    # --- LPIPS schedule (base weight + linear warmup) ---
    if global_step < (WARMUP_STEPS + RAMP_UP_STEPS):
        progress = (global_step - WARMUP_STEPS) / RAMP_UP_STEPS if global_step >= WARMUP_STEPS else 0.0
        progress = max(0.0, min(1.0, progress))
        weights['lpips'] = LPIPS_BASE_WEIGHT + (TARGET_LPIPS_WEIGHT - LPIPS_BASE_WEIGHT) * progress
    else:
        weights['lpips'] = TARGET_LPIPS_WEIGHT

    # --- L1 decay after LPIPS warmup begins (keep structure, reduce dominance) ---
    if global_step >= WARMUP_STEPS:
        decay_p = min(1.0, (global_step - WARMUP_STEPS) / max(RAMP_UP_STEPS, 1))
        weights['l1'] = L1_BASE_WEIGHT - decay_p * (L1_BASE_WEIGHT - L1_MIN_WEIGHT)

    # --- Noise-level consistency schedule ---
    if global_step < NOISE_CONS_WARMUP:
        weights['noise_cons'] = 0.0
    elif global_step < (NOISE_CONS_WARMUP + NOISE_CONS_RAMP):
        progress = (global_step - NOISE_CONS_WARMUP) / NOISE_CONS_RAMP
        weights['noise_cons'] = NOISE_CONS_WEIGHT * progress
    else:
        weights['noise_cons'] = NOISE_CONS_WEIGHT
    return weights

def rgb01_to_y01(rgb01):
    r, g, b = rgb01[:, 0:1], rgb01[:, 1:2], rgb01[:, 2:3]
    return (16.0 + 65.481*r + 128.553*g + 24.966*b) / 255.0

def shave_border(x, shave=4):
    return x[..., shave:-shave, shave:-shave]

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if DETERMINISTIC:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def randn_like_with_generator(tensor, generator):
    return torch.randn(
        tensor.shape,
        device=tensor.device,
        dtype=tensor.dtype,
        generator=generator,
    )


def get_lq_init_latents(z_lr, scheduler, steps, generator, strength, dtype):
    """Img2img-style init: add noise at a start timestep and run remaining steps."""
    strength = float(max(0.0, min(1.0, strength)))
    scheduler.set_timesteps(steps, device=z_lr.device)
    timesteps = scheduler.timesteps
    start_index = int(round(strength * (len(timesteps) - 1)))
    start_index = min(max(start_index, 0), len(timesteps) - 1)
    t_start = timesteps[start_index]
    noise = randn_like_with_generator(z_lr, generator)
    if hasattr(scheduler, "add_noise"):
        latents = scheduler.add_noise(z_lr, noise, t_start)
    else:
        latents = z_lr + noise
    return latents.to(dtype=dtype), timesteps[start_index:]


def differentiable_degrade_patch(img, meta, pipeline):
    """Differentiable replay: only blur/resize/noise (skip JPEG)."""
    img = (img + 1.0) * 0.5  # [-1,1] -> [0,1]
    ops_stage1 = [op for op in str(meta.get("ops_stage1", ",".join(DEG_OPS))).split(",") if op]
    ops_stage2 = [op for op in str(meta.get("ops_stage2", "")).split(",") if op]
    use_two_stage = bool(int(meta.get("use_two_stage", torch.tensor(0)).item()))
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
            "noise": meta.get("stage2_noise", None),
        }

    def apply_ops(img_in, ops, params):
        out = img_in
        for op in ops:
            if op == "blur" and params["blur_applied"]:
                if params["sigma_x"] > 0 and params["sigma_y"] > 0:
                    out = pipeline._apply_aniso_blur(out, params["k_size"], params["sigma_x"], params["sigma_y"], params["theta"])
                else:
                    out = TF.gaussian_blur(out, params["k_size"], [params["sigma"], params["sigma"]])
            elif op == "resize":
                mid_h = max(1, int(round(pipeline.crop_size * params["resize_scale"])))
                mid_w = max(1, int(round(pipeline.crop_size * params["resize_scale"])))
                out = TF.resize(out, [mid_h, mid_w], interpolation=params["resize_interp"], antialias=True)
            elif op == "noise":
                if params["noise_std"] > 0:
                    if params.get("noise") is not None:
                        noise = params["noise"].to(out.device, dtype=out.dtype)
                        if noise.shape != out.shape:
                            noise = noise[: out.shape[0], : out.shape[1], : out.shape[2]]
                    else:
                        noise = torch.randn_like(out)
                    out = (out + noise * params["noise_std"]).clamp(0.0, 1.0)
            # skip JPEG to keep differentiable
        return out

    lr_small = apply_ops(img, ops_stage1, stage1)
    if use_two_stage and stage2 is not None:
        lr_small = apply_ops(lr_small, ops_stage2, stage2)
    down_h = int(pipeline.crop_size * pipeline.downscale_factor)
    down_w = int(pipeline.crop_size * pipeline.downscale_factor)
    lr_small = TF.resize(lr_small, [down_h, down_w], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
    lr = TF.resize(lr_small, [pipeline.crop_size, pipeline.crop_size], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
    return (lr * 2.0 - 1.0).clamp(-1.0, 1.0)


def mask_adapter_cond(cond, keep_mask: torch.Tensor):
    """Mask adapter condition per-sample.
    Supports:
      - Tensor
      - List[Tensor]
      - Tuple/List of (List[Tensor], Tensor) for v6 adapter outputs.
    keep_mask: [B] float/bool where 1=keep, 0=drop.
    """
    if cond is None:
        return None
    if not torch.is_tensor(keep_mask):
        keep_mask = torch.tensor(keep_mask)

    def _find_device_dtype(x):
        if torch.is_tensor(x):
            return x.device, x.dtype
        if isinstance(x, (list, tuple)):
            for item in x:
                found = _find_device_dtype(item)
                if found is not None:
                    return found
        return None

    found = _find_device_dtype(cond)
    if found is None:
        return cond
    dev, _ = found
    keep_mask = keep_mask.to(device=dev, dtype=torch.float32)

    def _mask(x: torch.Tensor):
        m = keep_mask
        while m.ndim < x.ndim:
            m = m.unsqueeze(-1)
        return x * m.to(dtype=x.dtype)

    if torch.is_tensor(cond):
        return _mask(cond)

    if isinstance(cond, (list, tuple)):
        # v6 signature: (spatial_list, style_vec)
        if len(cond) == 2 and isinstance(cond[0], list) and torch.is_tensor(cond[1]):
            spatial = [_mask(c) for c in cond[0]]
            style = _mask(cond[1])
            return (spatial, style)

        masked = []
        for c in cond:
            if torch.is_tensor(c):
                masked.append(_mask(c))
            elif isinstance(c, list):
                masked.append([_mask(ci) if torch.is_tensor(ci) else ci for ci in c])
            else:
                masked.append(c)
        return masked if isinstance(cond, list) else tuple(masked)

    # fallback (unexpected type)
    return cond

def file_sha256(path):
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _should_keep_fp32_on_save(param_name: str) -> bool:
    return any(tag in param_name for tag in FP32_SAVE_KEY_FRAGMENTS)


def collect_trainable_state_dict(model: nn.Module):
    """Collect all trainable parameters from model with v6-sensitive fp32 casting on save."""
    state = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        tensor = param.detach().cpu()
        if _should_keep_fp32_on_save(name):
            tensor = tensor.float()
        state[name] = tensor
    return state


def validate_v6_trainable_state_keys(trainable_sd: dict, required_fragments):
    """Defensive validation to ensure v6 non-LoRA trainables are persisted."""
    keys = list(trainable_sd.keys())
    missing = []
    counts = {}
    for frag in required_fragments:
        c = sum(1 for k in keys if frag in k)
        counts[frag] = c
        if c == 0:
            missing.append(frag)
    if missing:
        raise RuntimeError(
            "v6 trainable checkpoint validation failed; missing required key fragments: "
            + ", ".join(missing)
        )
    return counts

def get_config_snapshot():
    return {
        "batch_size": BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "lr_base": LR_BASE,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "sparse_inject_ratio": SPARSE_INJECT_RATIO,
        "warmup_steps": WARMUP_STEPS,
        "ramp_up_steps": RAMP_UP_STEPS,
        "target_lpips_weight": TARGET_LPIPS_WEIGHT,
        "lpips_base_weight": LPIPS_BASE_WEIGHT,
        "l1_base_weight": L1_BASE_WEIGHT,
        "l1_min_weight": L1_MIN_WEIGHT,
        "val_steps_list": VAL_STEPS_LIST,
        "best_val_steps": BEST_VAL_STEPS,
        "psnr_switch": PSNR_SWITCH,
        "keep_topk": KEEP_TOPK,
        "val_mode": VAL_MODE,
        "val_pack_dir": VAL_PACK_DIR if VAL_MODE == "valpack" else None,
        "val_pack_lr_dir_name": VAL_PACK_LR_DIR_NAME if VAL_MODE == "valpack" else None,
        "train_deg_mode": TRAIN_DEG_MODE,
        "cfg_scale": CFG_SCALE,
        "use_lq_init": USE_LQ_INIT,
        "init_noise_std": INIT_NOISE_STD,
        "lq_init_strength": LQ_INIT_STRENGTH,
        "use_cfg_train": USE_CFG_TRAIN,
        "cfg_train_scale": CFG_TRAIN_SCALE,
        "use_adapter_cfdropout": USE_ADAPTER_CFDROPOUT,
        "cond_drop_prob": COND_DROP_PROB,
        "force_drop_text": FORCE_DROP_TEXT,
        "use_noise_consistency": USE_NOISE_CONSISTENCY,
        "noise_cons_warmup": NOISE_CONS_WARMUP,
        "noise_cons_ramp": NOISE_CONS_RAMP,
        "noise_cons_weight": NOISE_CONS_WEIGHT,
        "noise_cons_prob": NOISE_CONS_PROB,
        "use_lr_consistency": USE_LR_CONSISTENCY,
        "lr_cons_prob": LR_CONS_PROB,
        "lr_cons_warmup": LR_CONS_WARMUP,
        "lr_cons_ramp": LR_CONS_RAMP,
        "lr_cons_weight": LR_CONS_WEIGHT,
        "lr_cons_mode": LR_CONS_MODE,
        "lr_cons_full_every": LR_CONS_FULL_EVERY,
        "vae_tiling": VAE_TILING,
        "deg_ops": DEG_OPS,
        "p_two_stage": P_TWO_STAGE,
        "resize_scale_range": RESIZE_SCALE_RANGE,
        "noise_range": NOISE_RANGE,
        "blur_kernels": BLUR_KERNELS,
        "jpeg_quality_range": JPEG_QUALITY_RANGE,
        "resize_interp_modes": [mode.name for mode in RESIZE_INTERP_MODES],
        "seed": SEED,
        "deterministic": DETERMINISTIC,
    }

# ================= 4. Data Pipeline =================
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
        if generator is None:
            return float(random.uniform(low, high))
        return float(low + (high - low) * torch.rand((), generator=generator).item())

    def _sample_int(self, low, high, generator):
        if generator is None:
            return int(random.randint(low, high))
        return int(torch.randint(low, high + 1, (1,), generator=generator).item())

    def _sample_choice(self, choices, generator):
        if generator is None:
            return random.choice(choices)
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
        img = img.detach().to(torch.float32)  # ★关键：numpy 不支持 BF16
        img_np = (img.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(img_np).save(buf, format="JPEG", quality=int(quality))
        buf.seek(0)
        out = Image.open(buf).convert("RGB")
        out = TF.to_tensor(out).to(img.device, dtype=img.dtype)
        return out

    def _shuffle_ops(self, generator):
        ops = list(DEG_OPS)
        if generator is None:
            random.shuffle(ops)
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
                sigma_x = 0.0
                sigma_y = 0.0
                theta = 0.0
        else:
            k_size = 0
            sigma = 0.0
            sigma_x = 0.0
            sigma_y = 0.0
            theta = 0.0

        resize_scale = self._sample_uniform(*RESIZE_SCALE_RANGE, generator)
        resize_interp = self._sample_choice(RESIZE_INTERP_MODES, generator)
        resize_interp_idx = RESIZE_INTERP_MODES.index(resize_interp)
        noise_std = self._sample_uniform(*self.noise_range, generator)
        jpeg_quality = self._sample_int(*JPEG_QUALITY_RANGE, generator)
        return {
            "blur_applied": blur_applied,
            "k_size": k_size,
            "sigma": sigma,
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "theta": theta,
            "resize_scale": resize_scale,
            "resize_interp_idx": resize_interp_idx,
            "resize_interp": resize_interp,
            "noise_std": noise_std,
            "jpeg_quality": jpeg_quality,
        }

    def __call__(self, hr_tensor, return_meta: bool = False, meta=None, generator=None):
        """
        hr_tensor: [C,H,W] in [-1,1].
        If meta is None: sample degradation and optionally return meta for strict replay.
        If meta is provided: deterministically apply the EXACT same degradation (including noise realization).
        """
        img = (hr_tensor + 1.0) * 0.5  # -> [0,1]

        # -------- sample or read params --------
        if meta is None:
            use_two_stage = bool(self._sample_uniform(0.0, 1.0, generator) < P_TWO_STAGE)
            ops_stage1 = self._shuffle_ops(generator)
            ops_stage2 = self._shuffle_ops(generator) if use_two_stage else []
            stage1 = self._sample_stage_params(generator)
            stage2 = self._sample_stage_params(generator) if use_two_stage else None
        else:
            # meta keys are tensors; convert safely
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

            def _match_chw(t, ref):
                """Center crop/pad a [C,H,W] tensor t to match ref shape. Used to avoid rare 1px mismatch in strict replay."""
                if t is None:
                    return torch.zeros_like(ref)
                if t.shape == ref.shape:
                    return t
                # Channel align (rare, but keep safe)
                if t.shape[0] != ref.shape[0]:
                    c = min(int(t.shape[0]), int(ref.shape[0]))
                    t = t[:c]
                    if c < ref.shape[0]:
                        pad_c = int(ref.shape[0] - c)
                        t = torch.cat([t, torch.zeros((pad_c, t.shape[1], t.shape[2]), device=t.device, dtype=t.dtype)], dim=0)
                H, W = int(ref.shape[1]), int(ref.shape[2])
                h, w = int(t.shape[1]), int(t.shape[2])
                # Height
                if h > H:
                    top = (h - H) // 2
                    t = t[:, top:top+H, :]
                elif h < H:
                    pad_top = (H - h) // 2
                    pad_bottom = (H - h) - pad_top
                    t = F.pad(t, (0, 0, pad_top, pad_bottom))
                # Width
                h, w = int(t.shape[1]), int(t.shape[2])
                if w > W:
                    left = (w - W) // 2
                    t = t[:, :, left:left+W]
                elif w < W:
                    pad_left = (W - w) // 2
                    pad_right = (W - w) - pad_left
                    t = F.pad(t, (pad_left, pad_right, 0, 0))
                return t

            stage_noise = None
            for op in ops:
                if op == "blur" and params["blur_applied"]:
                    if params["sigma_x"] > 0 and params["sigma_y"] > 0:
                        out = self._apply_aniso_blur(out, params["k_size"], params["sigma_x"], params["sigma_y"], params["theta"])
                    else:
                        out = TF.gaussian_blur(out, params["k_size"], [params["sigma"], params["sigma"]])
                elif op == "resize":
                    mid_h = max(1, int(round(self.crop_size * params["resize_scale"])))
                    mid_w = max(1, int(round(self.crop_size * params["resize_scale"])))
                    out = TF.resize(out, [mid_h, mid_w], interpolation=params["resize_interp"], antialias=True)
                elif op == "noise":
                    if params["noise_std"] > 0:
                        if meta is None:
                            if generator is None:
                                noise = torch.randn_like(out)
                            else:
                                noise = torch.randn(
                                    out.shape,
                                    device=out.device,
                                    dtype=out.dtype,
                                    generator=generator,
                                )
                        else:
                            noise = params.get("noise")
                            if noise is None:
                                noise = torch.zeros_like(out)
                            else:
                                noise = noise.to(out.device, dtype=out.dtype)
                                noise = _match_chw(noise, out)
                        stage_noise = noise
                        out = (out + noise * params["noise_std"]).clamp(0.0, 1.0)
                    else:
                        stage_noise = torch.zeros_like(out)
                elif op == "jpeg":
                    out = self._apply_jpeg(out, params["jpeg_quality"])
            if stage_noise is None:
                stage_noise = torch.zeros_like(out)
            return out, stage_noise

        lr_small, stage1_noise = apply_ops(img, ops_stage1, stage1)
        stage2_noise = torch.zeros_like(lr_small)
        if use_two_stage:
            lr_small, stage2_noise = apply_ops(lr_small, ops_stage2, stage2)

        # 5. Final downsample to x4 then upsample back to crop size
        down_h = int(self.crop_size * self.downscale_factor)
        down_w = int(self.crop_size * self.downscale_factor)
        lr_small = TF.resize(lr_small, [down_h, down_w], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        lr = TF.resize(lr_small, [self.crop_size, self.crop_size], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        lr_out = (lr * 2.0 - 1.0).clamp(-1.0, 1.0)

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

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _make_generator(self, idx: int):
        gen = torch.Generator()
        seed = SEED + (self.epoch * 1_000_000) + int(idx)
        gen.manual_seed(seed)
        return gen

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        try:
            hr_pil = Image.open(self.hr_paths[idx]).convert("RGB")
        except:
            return self.__getitem__((idx + 1) % len(self))

        gen = None
        if self.is_train:
            gen = self._make_generator(idx)
            if hr_pil.width >= self.crop_size and hr_pil.height >= self.crop_size:
                max_top = hr_pil.height - self.crop_size + 1
                max_left = hr_pil.width - self.crop_size + 1
                top = int(torch.randint(0, max_top, (1,), generator=gen).item())
                left = int(torch.randint(0, max_left, (1,), generator=gen).item())
                hr_crop = TF.crop(hr_pil, top, left, self.crop_size, self.crop_size)
            else:
                hr_crop = TF.resize(hr_pil, (self.crop_size, self.crop_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        else:
            hr_crop = TF.center_crop(hr_pil, (self.crop_size, self.crop_size))

        hr_tensor = self.norm(self.to_tensor(hr_crop))
        if self.is_train and TRAIN_DEG_MODE == "bicubic":
            lr_small = TF.resize((hr_tensor + 1.0) * 0.5, (self.crop_size // 4, self.crop_size // 4),
                                 interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
            lr_tensor = TF.resize(lr_small, (self.crop_size, self.crop_size),
                                  interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
            lr_tensor = (lr_tensor * 2.0 - 1.0).clamp(-1.0, 1.0)
            deg_meta = None
        else:
            lr_tensor, deg_meta = self.pipeline(hr_tensor, return_meta=True, generator=gen if self.is_train else None)
        return {"hr": hr_tensor, "lr": lr_tensor, "deg": deg_meta}

class DF2K_Val_Fixed_Dataset(Dataset):
    def __init__(self, hr_root, lr_root=None, crop_size=512):
        self.hr_paths = sorted(glob.glob(os.path.join(hr_root, "*.png")))
        self.lr_root = lr_root
        self.crop_size = crop_size
        self.norm = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        hr_path = self.hr_paths[idx]
        hr_pil = Image.open(hr_path).convert("RGB")
        hr_crop = TF.center_crop(hr_pil, (self.crop_size, self.crop_size))

        lr_crop = None
        if self.lr_root:
            base = os.path.basename(hr_path)
            lr_name = base.replace(".png", "x4.png")
            lr_p = os.path.join(self.lr_root, lr_name)
            if os.path.exists(lr_p):
                lr_pil = Image.open(lr_p).convert("RGB")
                lr_crop = TF.center_crop(lr_pil, (self.crop_size//4, self.crop_size//4))
                lr_crop = TF.resize(lr_crop, (self.crop_size, self.crop_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)

        if lr_crop is None:
            w, h = hr_crop.size
            lr_small = hr_crop.resize((w//4, h//4), Image.BICUBIC)
            lr_crop = lr_small.resize((w, h), Image.BICUBIC)

        hr_tensor = self.norm(self.to_tensor(hr_crop))
        lr_tensor = self.norm(self.to_tensor(lr_crop))
        return {"hr": hr_tensor, "lr": lr_tensor, "path": hr_path}

class DF2K_Val_Degraded_Dataset(Dataset):
    def __init__(self, hr_root, crop_size=512, seed=3407, deg_mode="highorder"):
        self.hr_paths = sorted(glob.glob(os.path.join(hr_root, "*.png")))
        self.crop_size = crop_size
        self.norm = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        self.to_tensor = transforms.ToTensor()
        self.pipeline = DegradationPipeline(crop_size)
        self.seed = int(seed)
        self.deg_mode = deg_mode

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        hr_path = self.hr_paths[idx]
        hr_pil = Image.open(hr_path).convert("RGB")
        hr_crop = TF.center_crop(hr_pil, (self.crop_size, self.crop_size))
        hr_tensor = self.norm(self.to_tensor(hr_crop))
        if self.deg_mode == "bicubic":
            lr_small = TF.resize((hr_tensor + 1.0) * 0.5, (self.crop_size // 4, self.crop_size // 4),
                                 interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
            lr_tensor = TF.resize(lr_small, (self.crop_size, self.crop_size),
                                  interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
            lr_tensor = (lr_tensor * 2.0 - 1.0).clamp(-1.0, 1.0)
        else:
            gen = torch.Generator()
            gen.manual_seed(self.seed + idx)
            lr_tensor = self.pipeline(hr_tensor, return_meta=False, meta=None, generator=gen)
        return {"hr": hr_tensor, "lr": lr_tensor, "path": hr_path}


class ValPackDataset(Dataset):
    def __init__(self, pack_dir: str, lr_dir_name: str = "lq512", crop_size: int = 512):
        self.pack_dir = Path(pack_dir)
        self.hr_dir = self.pack_dir / "gt512"
        self.lr_dir = self.pack_dir / lr_dir_name
        if not self.hr_dir.is_dir():
            raise FileNotFoundError(f"gt512 dir not found: {self.hr_dir}")
        if not self.lr_dir.is_dir():
            raise FileNotFoundError(f"LR dir not found: {self.lr_dir}")
        self.crop_size = crop_size
        self.norm = transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        self.to_tensor = transforms.ToTensor()
        self.hr_paths = sorted(list(self.hr_dir.glob("*.png")))

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        hr_path = self.hr_paths[idx]
        name = hr_path.stem
        lr_path = self.lr_dir / f"{name}.png"
        if not lr_path.is_file():
            raise FileNotFoundError(f"LR image missing: {lr_path}")

        hr_pil = Image.open(hr_path).convert("RGB")
        lr_pil = Image.open(lr_path).convert("RGB")
        hr_crop = TF.center_crop(hr_pil, (self.crop_size, self.crop_size))
        lr_crop = TF.center_crop(lr_pil, (self.crop_size, self.crop_size))

        if lr_crop.size != (self.crop_size, self.crop_size):
            lr_crop = TF.resize(lr_crop, (self.crop_size, self.crop_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)

        hr_tensor = self.norm(self.to_tensor(hr_crop))
        lr_tensor = self.norm(self.to_tensor(lr_crop))
        return {"hr": hr_tensor, "lr": lr_tensor, "path": str(hr_path)}

# ================= 5. LoRA =================
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base; self.scaling = alpha / r
        self.lora_A = nn.Linear(base.in_features, r, bias=False, dtype=torch.float32)
        self.lora_B = nn.Linear(r, base.out_features, bias=False, dtype=torch.float32)
        self.lora_A.to(base.weight.device); self.lora_B.to(base.weight.device)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5)); nn.init.zeros_(self.lora_B.weight)
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

    def forward(self, x):
        out = self.base(x)
        delta = self.lora_B(self.lora_A(x.float())) * self.scaling
        return out + delta.to(out.dtype)

def apply_lora(model, rank=64, alpha=64):
    cnt = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(key in name for key in ("qkv", "proj", "to_q", "to_k", "to_v", "q_linear", "kv_linear")):
             parent = model.get_submodule(name.rsplit('.', 1)[0])
             child = name.rsplit('.', 1)[1]
             setattr(parent, child, LoRALinear(module, rank, alpha))
             cnt += 1
    print(f"✅ LoRA applied to {cnt} layers.")

# ================= 6. Checkpointing =================
def should_keep_ckpt(psnr_v, lpips_v):
    if not math.isfinite(psnr_v): return (999, float("inf"))
    if psnr_v >= PSNR_SWITCH and math.isfinite(lpips_v): return (0, lpips_v)
    return (1, -psnr_v)


def atomic_torch_save(state, path):
    """Atomic checkpoint save with fallback serialization to avoid zip writer failures on large files."""
    tmp = path + ".tmp"
    # Try modern zip serialization first.
    try:
        torch.save(state, tmp)
        os.replace(tmp, path)
        return True, "zip"
    except Exception as e_zip:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass
        # Fallback: legacy serialization is often more robust for very large checkpoints.
        try:
            torch.save(state, tmp, _use_new_zipfile_serialization=False)
            os.replace(tmp, path)
            return True, f"legacy ({e_zip})"
        except Exception as e_old:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass
            return False, f"zip_error={e_zip}; legacy_error={e_old}"

def save_smart(epoch, global_step, pixart, adapter, optimizer, best_records, metrics, dl_gen):
    global BASE_PIXART_SHA256

    # 1. 判断是否是 Top-K (Best)
    psnr_v, ssim_v, lpips_v = metrics
    priority, score = should_keep_ckpt(psnr_v, lpips_v)

    # 构建当前记录
    current_record = {
        "path": None, # 暂时不填路径，确定要存再填
        "epoch": epoch,
        "priority": priority,
        "score": score,
        "psnr": psnr_v,
        "lpips": lpips_v
    }

    # 逻辑：
    # 如果记录没满 K 个 -> 存
    # 如果记录满了，但当前比最差的那个好 -> 存，并删掉最差的

    save_as_best = False
    if len(best_records) < KEEP_TOPK:
        save_as_best = True
    else:
        # best_records 是排好序的，最后一个是最差的
        worst_record = best_records[-1]
        # 比较：priority 越小越好；同 priority 下 score 越小越好
        if (priority < worst_record['priority']) or \
           (priority == worst_record['priority'] and score < worst_record['score']):
            save_as_best = True

    if save_as_best:
        ckpt_name = f"epoch{epoch+1:03d}_psnr{psnr_v:.2f}_lp{lpips_v:.4f}.pth"
        ckpt_path = os.path.join(CKPT_DIR, ckpt_name)
        current_record['path'] = ckpt_path

        try:
            # 更新记录表
            best_records.append(current_record)
            # 重新排序
            best_records.sort(key=lambda x: (x['priority'], x['score']))

            # 删除多余的
            if len(best_records) > KEEP_TOPK:
                to_delete = best_records[KEEP_TOPK:]
                best_records = best_records[:KEEP_TOPK]
                for rec in to_delete:
                    if rec['path'] and os.path.exists(rec['path']):
                        try:
                            os.remove(rec['path'])
                            print(f"🗑️ Removed old best: {os.path.basename(rec['path'])}")
                        except: pass
        except Exception as e:
            print(f"❌ Failed to save best checkpoint: {e}")

    # 2. 准备 State Dict (约 7.5GB)
    # Strategy-1: save ALL PixArt params that are actually trainable (requires_grad=True), not name-based filtering.
    if BASE_PIXART_SHA256 is None and os.path.exists(PIXART_PATH):
        try:
            BASE_PIXART_SHA256 = file_sha256(PIXART_PATH)
        except Exception as e:
            print(f"⚠️ Base PixArt hash failed (non-fatal): {e}")
            BASE_PIXART_SHA256 = None

    pixart_sd = collect_trainable_state_dict(pixart)
    required_frags = get_required_v6_key_fragments_for_model(pixart)
    v6_key_counts = validate_v6_trainable_state_keys(pixart_sd, required_frags)
    print("✅ v6 save check:", ", ".join([f"{k}={v}" for k, v in v6_key_counts.items()]))
    state = {
        "epoch": epoch,
        "step": global_step,
        "adapter": adapter.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
        "dl_gen_state": dl_gen.get_state(),
        "pixart_trainable": pixart_sd,
        "best_records": best_records,
        "config_snapshot": get_config_snapshot(),
        "base_pixart_sha256": BASE_PIXART_SHA256,
        "env_info": {
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }

    # 3. 始终保存 last.pth (这是为了断点续训，必须有)
    # 使用临时文件名再重命名，防止写入一半被中断导致文件损坏
    last_path = LAST_CKPT_PATH
    ok_last, msg_last = atomic_torch_save(state, last_path)
    if ok_last:
        print(f"💾 Saved last checkpoint to {last_path} [{msg_last}]")
    else:
        print(f"❌ Failed to save last.pth: {msg_last}")

    if save_as_best and current_record["path"]:
        try:
            # Reuse serialized last checkpoint to avoid writing another >2GB stream in the same step.
            if ok_last and os.path.exists(last_path):
                shutil.copy2(last_path, current_record["path"])
                print(f"🏆 New Best Model! Copied from last.pth to {ckpt_name}")
            else:
                ok_best, msg_best = atomic_torch_save(state, current_record["path"])
                if ok_best:
                    print(f"🏆 New Best Model! Saved to {ckpt_name} [{msg_best}]")
                else:
                    print(f"❌ Failed to save best checkpoint: {msg_best}")
        except Exception as e:
            print(f"❌ Failed to save best checkpoint: {e}")

    return best_records

def build_stage_meta_from_deg(deg, index):
    ops_stage1 = deg.get("ops_stage1", ",".join(DEG_OPS))
    if isinstance(ops_stage1, (list, tuple)):
        ops_stage1 = ops_stage1[index]
    ops_stage2 = deg.get("ops_stage2", "")
    if isinstance(ops_stage2, (list, tuple)):
        ops_stage2 = ops_stage2[index]
    return {
        "stage1_blur_applied": deg["stage1_blur_applied"][index].cpu(),
        "stage1_k_size": deg["stage1_k_size"][index].cpu(),
        "stage1_sigma": deg["stage1_sigma"][index].cpu(),
        "stage1_sigma_x": deg["stage1_sigma_x"][index].cpu(),
        "stage1_sigma_y": deg["stage1_sigma_y"][index].cpu(),
        "stage1_theta": deg["stage1_theta"][index].cpu(),
        "stage1_noise_std": deg["stage1_noise_std"][index].cpu(),
        "stage1_noise": deg["stage1_noise"][index].cpu(),
        "stage1_resize_scale": deg["stage1_resize_scale"][index].cpu(),
        "stage1_resize_interp": deg["stage1_resize_interp"][index].cpu(),
        "stage1_jpeg_quality": deg["stage1_jpeg_quality"][index].cpu(),
        "stage2_blur_applied": deg["stage2_blur_applied"][index].cpu(),
        "stage2_k_size": deg["stage2_k_size"][index].cpu(),
        "stage2_sigma": deg["stage2_sigma"][index].cpu(),
        "stage2_sigma_x": deg["stage2_sigma_x"][index].cpu(),
        "stage2_sigma_y": deg["stage2_sigma_y"][index].cpu(),
        "stage2_theta": deg["stage2_theta"][index].cpu(),
        "stage2_noise_std": deg["stage2_noise_std"][index].cpu(),
        "stage2_noise": deg["stage2_noise"][index].cpu(),
        "stage2_resize_scale": deg["stage2_resize_scale"][index].cpu(),
        "stage2_resize_interp": deg["stage2_resize_interp"][index].cpu(),
        "stage2_jpeg_quality": deg["stage2_jpeg_quality"][index].cpu(),
        "use_two_stage": deg["use_two_stage"][index].cpu() if "use_two_stage" in deg else torch.tensor(0),
        "ops_stage1": ops_stage1,
        "ops_stage2": ops_stage2,
    }

def resume(pixart, adapter, optimizer, dl_gen):
    if not os.path.exists(LAST_CKPT_PATH): return 0, 0, []
    print(f"📥 Resuming from {LAST_CKPT_PATH}...")
    ckpt = torch.load(LAST_CKPT_PATH, map_location="cpu")
    saved_trainable = ckpt.get("pixart_trainable", {})
    required_frags = get_required_v6_key_fragments_for_model(pixart)
    missing_required = [frag for frag in required_frags if not any(frag in k for k in saved_trainable.keys())]
    if missing_required:
        raise RuntimeError(
            "Checkpoint is missing required v6 trainable keys: " + ", ".join(missing_required)
        )
    adapter_sd = ckpt.get("adapter", {})
    missing, unexpected = adapter.load_state_dict(adapter_sd, strict=False)
    if missing or unexpected:
        print(f"⚠️ Adapter state_dict mismatch: missing={len(missing)} unexpected={len(unexpected)}")
    # Handle scale_gates size change when expanding adapter outputs.
    if "scale_gates" in adapter_sd:
        saved = adapter_sd["scale_gates"]
        current = adapter.scale_gates
        if saved.shape != current.shape:
            n = min(saved.shape[0], current.shape[0])
            with torch.no_grad():
                current[:n].copy_(saved[:n])
    curr = pixart.state_dict()
    for k, v in saved_trainable.items():
        if k in curr: curr[k] = v.to(curr[k].dtype)
    pixart.load_state_dict(curr, strict=False)
    optimizer.load_state_dict(ckpt["optimizer"])
    # Strict resume: restore RNG states if present
    rs = ckpt.get("rng_state", None)
    if rs is not None:
        try:
            if rs.get("torch") is not None: torch.set_rng_state(rs["torch"])
            if torch.cuda.is_available() and rs.get("cuda") is not None: torch.cuda.set_rng_state_all(rs["cuda"])
            if rs.get("numpy") is not None: np.random.set_state(rs["numpy"])
            if rs.get("python") is not None: random.setstate(rs["python"])
        except Exception as e:
            print(f"⚠️ RNG restore failed (non-fatal): {e}")
    dl_state = ckpt.get("dl_gen_state", None)
    if dl_state is not None:
        try:
            dl_gen.set_state(dl_state)
        except Exception as e:
            print(f"⚠️ DataLoader generator restore failed (non-fatal): {e}")
    return ckpt["epoch"]+1, ckpt["step"], ckpt.get("best_records", [])

# ================= 7. Validation (Deterministic, 50-step only) =================
@torch.no_grad()
def validate(epoch, pixart, adapter, vae, val_loader, y_embed, data_info, lpips_fn):
    """
    Validation uses deterministic init (LQ-init or fixed noise).
    We report VAL@50 only (main tracking & best ckpt).
    """
    print(f"🔎 Validating Epoch {epoch+1}...")
    pixart.eval(); adapter.eval()
    results = {}

    val_gen = torch.Generator(device=DEVICE)
    val_gen.manual_seed(SEED)

    steps_list = [FAST_VAL_STEPS] if FAST_DEV_RUN else VAL_STEPS_LIST
    for steps in steps_list:
        scheduler = DDIMScheduler(
            num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02, beta_schedule="linear",
            clip_sample=False, prediction_type="epsilon", set_alpha_to_one=False,
        )
        scheduler.set_timesteps(steps, device=DEVICE)

        psnrs, ssims, lpipss = [], [], []
        vis_done = False

        for batch in tqdm(val_loader, desc=f"Val@{steps}"):
            hr = batch["hr"].to(DEVICE)
            lr = batch["lr"].to(DEVICE)
            z_hr = vae.encode(hr).latent_dist.mean * vae.config.scaling_factor
            z_lr = vae.encode(lr).latent_dist.mean * vae.config.scaling_factor

            # Deterministic init: img2img-style LQ-init (aligned to scheduler) or fixed noise.
            if USE_LQ_INIT:
                latents, run_timesteps = get_lq_init_latents(
                    z_lr.to(COMPUTE_DTYPE), scheduler, steps, val_gen, LQ_INIT_STRENGTH, COMPUTE_DTYPE
                )
            else:
                latents = randn_like_with_generator(z_hr, val_gen)
                run_timesteps = scheduler.timesteps
            cond = adapter(z_lr.float())

            for t in run_timesteps:
                t_b = torch.tensor([t], device=DEVICE).expand(latents.shape[0])
                with torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
                    if FORCE_DROP_TEXT:
                        drop_uncond = torch.ones(latents.shape[0], device=DEVICE)
                        drop_cond = torch.ones(latents.shape[0], device=DEVICE)
                    else:
                        drop_uncond = torch.ones(latents.shape[0], device=DEVICE)
                        drop_cond = torch.zeros(latents.shape[0], device=DEVICE)
                    out_uncond = pixart(
                        x=latents.to(COMPUTE_DTYPE), timestep=t_b, y=y_embed,
                        mask=None, data_info=data_info, adapter_cond=None,
                        injection_mode="hybrid", force_drop_ids=drop_uncond
                    )
                    out_cond = pixart(
                        x=latents.to(COMPUTE_DTYPE), timestep=t_b, y=y_embed,
                        mask=None, data_info=data_info, adapter_cond=cond,
                        injection_mode="hybrid", force_drop_ids=drop_cond
                    )
                    if out_uncond.shape[1] == 8: out_uncond, _ = out_uncond.chunk(2, dim=1)
                    if out_cond.shape[1] == 8: out_cond, _ = out_cond.chunk(2, dim=1)
                    out = out_uncond + CFG_SCALE * (out_cond - out_uncond)
                latents = scheduler.step(out.float(), t, latents.float()).prev_sample

            pred = vae.decode(latents / vae.config.scaling_factor).sample.clamp(-1, 1)

            # Metrics (Y channel, shave border=4 as before)
            p01 = (pred + 1) / 2; h01 = (hr + 1) / 2
            py = rgb01_to_y01(p01)[..., 4:-4, 4:-4]
            hy = rgb01_to_y01(h01)[..., 4:-4, 4:-4]
            if "psnr" in globals():
                psnrs.append(psnr(py, hy, data_range=1.0).item())
                ssims.append(ssim(py, hy, data_range=1.0).item())
            lpipss.append(lpips_fn(pred, hr).mean().item())

            # Save only one visualization per steps (first batch)
            if not vis_done:
                save_path = os.path.join(VIS_DIR, f"epoch{epoch+1:03d}_steps{steps}.png")
                lr_np = (lr[0].cpu().float().numpy().transpose(1,2,0) + 1) / 2
                hr_np = (hr[0].cpu().float().numpy().transpose(1,2,0) + 1) / 2
                pr_np = (pred[0].cpu().float().numpy().transpose(1,2,0) + 1) / 2
                plt.figure(figsize=(12,4))
                plt.subplot(1,3,1); plt.imshow(np.clip(lr_np, 0, 1)); plt.title("Input LR"); plt.axis("off")
                plt.subplot(1,3,2); plt.imshow(np.clip(hr_np, 0, 1)); plt.title("GT"); plt.axis("off")
                plt.subplot(1,3,3); plt.imshow(np.clip(pr_np, 0, 1)); plt.title(f"Pred @{steps}"); plt.axis("off")
                plt.savefig(save_path, bbox_inches="tight")
                plt.close()
                vis_done = True
            if FAST_DEV_RUN and len(psnrs) >= FAST_VAL_BATCHES:
                break

        res = (float(np.mean(psnrs)), float(np.mean(ssims)), float(np.mean(lpipss)))
        results[int(steps)] = res
        print(f"[VAL@{steps}] Ep{epoch+1}: PSNR={res[0]:.2f} | SSIM={res[1]:.4f} | LPIPS={res[2]:.4f}")

    # For downstream: choose BEST_VAL_STEPS as the main metric tuple
    pixart.train(); adapter.train()
    return results

# ================= 8. Main =================
def main():
    # Setup
    seed_everything(SEED)
    dl_gen = torch.Generator()
    dl_gen.manual_seed(SEED)
    train_ds = DF2K_Online_Dataset(TRAIN_HR_DIR, crop_size=512, is_train=True)
    # Smaller degradation pipeline for patch-level LR consistency (avoids full 512 decode OOM).
    patch_pipeline = DegradationPipeline(crop_size=256)
    if VAL_MODE == "valpack":
        val_ds = ValPackDataset(VAL_PACK_DIR, lr_dir_name=VAL_PACK_LR_DIR_NAME, crop_size=512)
    elif VAL_MODE == "train_like":
        val_ds = DF2K_Val_Degraded_Dataset(VAL_HR_DIR, crop_size=512, seed=SEED, deg_mode=TRAIN_DEG_MODE)
    elif VAL_MODE == "lr_dir" and VAL_LR_DIR is not None:
        val_ds = DF2K_Val_Fixed_Dataset(VAL_HR_DIR, lr_root=VAL_LR_DIR, crop_size=512)
    else:
        val_ds = DF2K_Val_Fixed_Dataset(VAL_HR_DIR, lr_root=None, crop_size=512)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=dl_gen,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # Models
    pixart = PixArtMSV6_XL_2(input_size=64, sparse_inject_ratio=SPARSE_INJECT_RATIO).to(DEVICE)
    base = torch.load(PIXART_PATH, map_location="cpu")
    if "state_dict" in base: base = base["state_dict"]
    if "pos_embed" in base: del base["pos_embed"]
    pixart.load_state_dict(base, strict=False)
    apply_lora(pixart, LORA_RANK, LORA_ALPHA)
    pixart.train()

    adapter = build_adapter_v6(in_channels=4, hidden_size=1152).to(DEVICE).float().train()
    vae = AutoencoderKL.from_pretrained(VAE_PATH, local_files_only=True).to(DEVICE).float().eval()
    vae.enable_slicing()
    if VAE_TILING and hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
    lpips_fn = lpips.LPIPS(net='vgg').to(DEVICE).eval()
    # Freeze VAE/LPIPS weights (keep gradients to inputs if decode is used for losses)
    for p in vae.parameters():
        p.requires_grad_(False)
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    if os.environ.get('DEBUG_SANITY', '0') == '1':
        import inspect
        print('PixArtMS loaded from:', inspect.getfile(pixart.__class__))
        print('injection_layers:', getattr(pixart, 'injection_layers', None))
        if hasattr(pixart, 'cross_gate_ms'):
            cg = pixart.cross_gate_ms.detach().float()
            print('cross_gate_ms min/max:', float(cg.min()), float(cg.max()))
            # If using cross/hybrid injection, a zero gate will kill gradients to the cross-attn conditioning branch.
            if float(cg.abs().max()) == 0.0:
                print('⚠️ WARNING: cross_gate_ms is all zeros; cross-attn conditioning will be disabled.')


    y = torch.load(T5_EMBED_PATH, map_location="cpu")["prompt_embeds"].unsqueeze(1).to(DEVICE)
    d_info = {"img_hw": torch.tensor([[512.,512.]]).to(DEVICE), "aspect_ratio": torch.tensor([1.]).to(DEVICE)}

    # Optimizer
    params = list(adapter.parameters()) + [p for p in pixart.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR_BASE)
    diffusion = IDDPM(str(1000))

    ep_start, step, best = resume(pixart, adapter, optimizer, dl_gen)

    print("🚀 Auto-Curriculum Training Started.")
    max_steps = FAST_TRAIN_STEPS if FAST_DEV_RUN else None
    if FAST_DEV_RUN:
        print(
            "⚡ FAST_DEV_RUN enabled: "
            f"max_steps={FAST_TRAIN_STEPS}, val_batches={FAST_VAL_BATCHES}, val_steps={FAST_VAL_STEPS}"
        )

    for epoch in range(ep_start, 1000):
        if max_steps is not None and step >= max_steps:
            break
        train_ds.set_epoch(epoch)
        pbar = tqdm(train_loader, dynamic_ncols=True, desc=f"Ep{epoch+1}")
        for i, batch in enumerate(pbar):
            if max_steps is not None and step >= max_steps:
                break
            hr = batch['hr'].to(DEVICE); lr = batch['lr'].to(DEVICE)
            with torch.no_grad():
                zh = vae.encode(hr).latent_dist.mean * vae.config.scaling_factor
                zl = vae.encode(lr).latent_dist.mean * vae.config.scaling_factor

            t = torch.randint(0, 1000, (zh.shape[0],), device=DEVICE).long()
            noise = torch.randn_like(zh)
            zt = diffusion.q_sample(zh, t, noise)

            cond = adapter(zl.float())

            # --- Adapter-only classifier-free dropout (per-sample) ---
            # We keep text conditioning fixed, and randomly drop ONLY the adapter condition.
            cond_in = cond
            if USE_ADAPTER_CFDROPOUT and COND_DROP_PROB > 0:
                keep = (torch.rand((zt.shape[0],), device=DEVICE) >= COND_DROP_PROB).float()
                cond_in = mask_adapter_cond(cond, keep)

            with torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
                if USE_CFG_TRAIN:
                    if FORCE_DROP_TEXT:
                        drop_uncond = torch.ones(zt.shape[0], device=DEVICE)
                        drop_cond = torch.ones(zt.shape[0], device=DEVICE)
                    else:
                        drop_uncond = torch.ones(zt.shape[0], device=DEVICE)
                        drop_cond = torch.zeros(zt.shape[0], device=DEVICE)
                    out_uncond = pixart(
                        x=zt, timestep=t, y=y, data_info=d_info, adapter_cond=None,
                        injection_mode="hybrid", force_drop_ids=drop_uncond
                    )
                    out_cond = pixart(
                        x=zt, timestep=t, y=y, data_info=d_info, adapter_cond=cond_in,
                        injection_mode="hybrid", force_drop_ids=drop_cond
                    )
                    if out_uncond.shape[1] == 8: out_uncond, _ = out_uncond.chunk(2, dim=1)
                    if out_cond.shape[1] == 8: out_cond, _ = out_cond.chunk(2, dim=1)
                    eps = out_uncond + CFG_TRAIN_SCALE * (out_cond - out_uncond)
                else:
                    kwargs = dict(x=zt, timestep=t, y=y, data_info=d_info, adapter_cond=cond_in, injection_mode="hybrid")
                    if FORCE_DROP_TEXT:
                        kwargs["force_drop_ids"] = torch.ones((zt.shape[0],), device=DEVICE)
                    out = pixart(**kwargs)
                    if out.shape[1] == 8: out, _ = out.chunk(2, dim=1)
                    eps = out.float()

                # --- Automated Loss Weights ---
                w = get_loss_weights(step)

                loss_mse = F.mse_loss(eps, noise.float())

                # x0 Reconstruction
                with torch.no_grad():
                    c1 = _extract_into_tensor(diffusion.sqrt_recip_alphas_cumprod, t, zt.shape)
                    c2 = _extract_into_tensor(diffusion.sqrt_recipm1_alphas_cumprod, t, zt.shape)
                z0 = c1 * zt.float() - c2 * eps

                # Patch Crop (32x32)
                top = torch.randint(0, 33, (1,), device=DEVICE).item()
                left = torch.randint(0, 33, (1,), device=DEVICE).item()
                z0_c = z0[..., top:top+32, left:left+32]
                zh_c = zh.float()[..., top:top+32, left:left+32]

                img_p = vae.decode(z0_c/vae.config.scaling_factor).sample.clamp(-1,1)
                # IMPORTANT: use real HR pixel patch as target (not VAE-reconstructed HR)
                y0 = int(top) * 8; x0 = int(left) * 8
                img_t = hr[..., y0:y0+256, x0:x0+256].clamp(-1, 1)

                loss_l1 = F.l1_loss(img_p, img_t)

                # --- Noise-level consistency (z0 consistency across t1/t2) ---
                if USE_NOISE_CONSISTENCY and w.get("noise_cons", 0.0) > 0 and torch.rand((), device=DEVICE) < NOISE_CONS_PROB:
                    t2 = torch.randint(0, 1000, (zh.shape[0],), device=DEVICE).long()
                    zt2 = diffusion.q_sample(zh, t2, noise)
                    with torch.no_grad():
                        if USE_CFG_TRAIN:
                            out2_uncond = pixart(
                                x=zt2, timestep=t2, y=y, data_info=d_info, adapter_cond=None,
                                injection_mode="hybrid", force_drop_ids=drop_uncond
                            )
                            out2_cond = pixart(
                                x=zt2, timestep=t2, y=y, data_info=d_info, adapter_cond=cond_in,
                                injection_mode="hybrid", force_drop_ids=drop_cond
                            )
                            if out2_uncond.shape[1] == 8: out2_uncond, _ = out2_uncond.chunk(2, dim=1)
                            if out2_cond.shape[1] == 8: out2_cond, _ = out2_cond.chunk(2, dim=1)
                            eps2 = out2_uncond + CFG_TRAIN_SCALE * (out2_cond - out2_uncond)
                        else:
                            kwargs2 = dict(x=zt2, timestep=t2, y=y, data_info=d_info, adapter_cond=cond_in, injection_mode="hybrid")
                            if FORCE_DROP_TEXT:
                                kwargs2["force_drop_ids"] = torch.ones((zt.shape[0],), device=DEVICE)
                            out2 = pixart(**kwargs2)
                            if out2.shape[1] == 8: out2, _ = out2.chunk(2, dim=1)
                            eps2 = out2.float()
                        c1b = _extract_into_tensor(diffusion.sqrt_recip_alphas_cumprod, t2, zt2.shape)
                        c2b = _extract_into_tensor(diffusion.sqrt_recipm1_alphas_cumprod, t2, zt2.shape)
                        z0_t2 = c1b * zt2.float() - c2b * eps2
                    loss_noise_cons = F.l1_loss(z0, z0_t2)
                else:
                    loss_noise_cons = torch.tensor(0.0, device=DEVICE)

                # --- Strict LR consistency (optional) ---
                # Supports patch-level replay (default) and optional full replay.
                if USE_LR_CONSISTENCY and w.get("cons", 0.0) > 0 and (torch.rand((), device=DEVICE) < LR_CONS_PROB):
                    deg = batch.get("deg", None)
                    if deg is None:
                        # Should not happen in training dataset; keep safe
                        loss_cons = torch.tensor(0.0, device=DEVICE)
                    else:
                        use_full = False
                        if LR_CONS_MODE == "full":
                            use_full = True
                        elif LR_CONS_MODE == "mixed" and LR_CONS_FULL_EVERY and (step % LR_CONS_FULL_EVERY == 0):
                            use_full = True

                        if use_full:
                            # Full-image replay (may be heavy; enable VAE tiling if needed).
                            img_full = vae.decode(z0 / vae.config.scaling_factor).sample.clamp(-1, 1)
                            lr_hat_list = []
                            for b in range(img_full.shape[0]):
                                meta_b = build_stage_meta_from_deg(deg, b)
                                img_cpu = img_full[b].detach().float().cpu()
                                lr_hat_b = train_ds.pipeline(img_cpu, return_meta=False, meta=meta_b)
                                lr_hat_b = lr_hat_b.to(DEVICE)
                                lr_hat_list.append(lr_hat_b)
                            lr_hat = torch.stack(lr_hat_list, dim=0).to(DEVICE)
                            y0 = top * 8; x0 = left * 8
                            lr_hat_p = lr_hat[..., y0:y0+256, x0:x0+256]
                            lr_in_p  = lr[..., y0:y0+256, x0:x0+256]
                            loss_cons = F.l1_loss(lr_hat_p, lr_in_p)
                        else:
                            lr_hat_list = []
                            for b in range(img_p.shape[0]):
                                meta_b = build_stage_meta_from_deg(deg, b)
                                img_patch = img_p[b].float()  # patch-sized replay (256x256), keep grad
                                lr_hat_b = differentiable_degrade_patch(img_patch, meta_b, patch_pipeline)
                                lr_hat_list.append(lr_hat_b)
                            lr_hat = torch.stack(lr_hat_list, dim=0).to(DEVICE)
                            # Compare patch-sized LR
                            y0 = top * 8; x0 = left * 8
                            lr_in_p  = lr[..., y0:y0+256, x0:x0+256]
                            loss_cons = F.l1_loss(lr_hat, lr_in_p)
                else:
                    loss_cons = torch.tensor(0.0, device=DEVICE)

                loss_lpips = lpips_fn(img_p, img_t).mean() if w['lpips'] > 0 else torch.tensor(0.0, device=DEVICE)

                loss = (
                    w['mse']*loss_mse
                    + w['l1']*loss_l1
                    + w['lpips']*loss_lpips
                    + w.get('cons',0.0)*loss_cons
                    + w.get('noise_cons', 0.0)*loss_noise_cons
                ) / GRAD_ACCUM_STEPS

            loss.backward()

            if (i+1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0) # Safety
                optimizer.step()
                optimizer.zero_grad()
                step += 1

            if i % 10 == 0:
                pbar.set_postfix({
                    'mse': f"{loss_mse:.3f}",
                    'l1': f"{loss_l1:.3f}",
                    'lp': f"{loss_lpips:.3f}",
                    'cons': f"{loss_cons:.3f}",
                    'ncons': f"{loss_noise_cons:.3f}",
                })

        val_dict = validate(epoch, pixart, adapter, vae, val_loader, y, d_info, lpips_fn)
        # Choose metrics from BEST_VAL_STEPS for checkpoint selection
        if int(BEST_VAL_STEPS) in val_dict:
            metrics = val_dict[int(BEST_VAL_STEPS)]
        else:
            metrics = next(iter(val_dict.values()))
        best = save_smart(epoch, step, pixart, adapter, optimizer, best, metrics, dl_gen)

if __name__ == "__main__":
    main()
