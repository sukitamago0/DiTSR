# /home/hello/HJT/DiTSR/experiments/train_4090_auto_v9_gan.py
# DiTSR v9 Training Script (Adversarial Finetuning Phase)
# ------------------------------------------------------------------
# Phase: High-Fidelity Generation (Paper Ready)
# Base: V8 Structure (V-Pred, Copy-Init, Augmentation)
# Additions:
# 1. PatchGAN Discriminator: Adds Adversarial Loss to fix "oil painting" artifacts.
# 2. EMA (Exponential Moving Average): Stabilizes training, prevents oscillation.
# 3. Refined Loss Balance: L1 (1.0) + LPIPS (1.0) + GAN (0.02).
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

# [Import V8 Model - Keeping the strong foundation]
from diffusion.model.nets.PixArtMS_v8 import PixArtMSV8_XL_2
from diffusion.model.nets.adapter_v7 import build_adapter_v7
from diffusion import IDDPM
from diffusion.model.gaussian_diffusion import _extract_into_tensor

BASE_PIXART_SHA256 = None
V7_REQUIRED_PIXART_KEY_FRAGMENTS = ("input_adaln", "adapter_alpha_mlp", "input_res_proj", "injection_scales", "input_adapter_ln", "style_fusion_mlp", "aug_embedder")
FP32_SAVE_KEY_FRAGMENTS = V7_REQUIRED_PIXART_KEY_FRAGMENTS

def get_required_v7_key_fragments_for_model(model: nn.Module):
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    required = []
    for frag in V7_REQUIRED_PIXART_KEY_FRAGMENTS:
        if any(frag in name for name in trainable_names): required.append(frag)
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
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        return self.main(input)

# ================= 3. Hyper-parameters =================
TRAIN_HR_DIR = "/data/DF2K/DF2K_train_HR"
VAL_HR_DIR   = "/data/DF2K/DF2K_valid_HR"
VAL_LR_DIR   = "/data/DF2K/DF2K_valid_LR_bicubic/X4"
if not os.path.exists(VAL_LR_DIR): VAL_LR_DIR = None

PIXART_PATH = os.path.join(PROJECT_ROOT, "output", "pretrained_models", "PixArt-XL-2-512x512.pth")
VAE_PATH    = os.path.join(PROJECT_ROOT, "output", "pretrained_models", "sd-vae-ft-ema")
T5_EMBED_PATH = "/home/hello/HJT/DiTSR/output/null_embed.pth"

OUT_DIR = os.path.join(PROJECT_ROOT, "experiments_results", "train_4090_auto_v9_gan")
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

BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16 
NUM_WORKERS = 8

# Learning Rates (Differential)
LR_G_ADAPTER = 5e-5 # Slightly lower for finetuning
LR_G_BASE = 5e-6    # Slightly lower for finetuning
LR_D = 1e-4         # Discriminator LR

LORA_RANK = 16
LORA_ALPHA = 16
SPARSE_INJECT_RATIO = 1.0
INJECTION_CUTOFF_LAYER = 25
INJECTION_STRATEGY = "front_dense"

# Augmentation (Reduced for refinement phase)
COND_AUG_NOISE_RANGE = (0.0, 0.05) 

# [V9 GAN Config]
# GAN Weight: Controls "Realism". Too high = artifacts, Too low = blur.
GAN_WEIGHT = 0.02 
TARGET_LPIPS_WEIGHT = 1.0 # Can be higher now because GAN suppresses artifacts
L1_BASE_WEIGHT = 1.0

# Validation
VAL_STEPS_LIST = [50]
BEST_VAL_STEPS = 50
PSNR_SWITCH = 22.5
KEEP_TOPK = 1
VAL_MODE = "valpack"
VAL_PACK_DIR = os.path.join(PROJECT_ROOT, "valpacks", "df2k_train_like_50_seed3407")
VAL_PACK_LR_DIR_NAME = "lq512"
TRAIN_DEG_MODE = "highorder"
CFG_SCALE = 1.5

USE_LQ_INIT = True 
LQ_INIT_STRENGTH = 0.0 # Strict generation mode

INIT_NOISE_STD = 0.0
USE_CFG_TRAIN = False
CFG_TRAIN_SCALE = 3.0
USE_ADAPTER_CFDROPOUT = True
COND_DROP_PROB = 0.10
FORCE_DROP_TEXT = True

USE_LR_CONSISTENCY = False 
USE_NOISE_CONSISTENCY = False

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

# ================= 6. Datasets =================
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
            lr_tensor = self.pipeline(hr_tensor, return_meta=False, meta=None, generator=gen)
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

# ================= 7. LoRA =================
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

# ================= 8. Checkpointing =================
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

def save_smart(epoch, global_step, pixart, adapter, optimizer, best_records, metrics, dl_gen):
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
    state = {
        "epoch": epoch, "step": global_step, "adapter": {k: v.detach().float().cpu() for k, v in adapter.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "rng_state": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None, "numpy": np.random.get_state(), "python": random.getstate()},
        "dl_gen_state": dl_gen.get_state(), "pixart_trainable": pixart_sd, "best_records": best_records, "config_snapshot": get_config_snapshot(), "base_pixart_sha256": BASE_PIXART_SHA256, "env_info": {"torch": torch.__version__, "numpy": np.__version__},
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

def resume(pixart, adapter, optimizer, dl_gen):
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
    optimizer.load_state_dict(ckpt["optimizer"])
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

# ================= 9. Validation =================
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
    g_params = [{"params": list(adapter.parameters()), "lr": LR_G_ADAPTER}, {"params": [p for p in pixart.parameters() if p.requires_grad], "lr": LR_G_BASE}]
    optimizer_G = torch.optim.AdamW(g_params)
    optimizer_D = torch.optim.AdamW(discriminator.parameters(), lr=LR_D, betas=(0.5, 0.999))
    
    g_params_to_clip = list(adapter.parameters()) + [p for p in pixart.parameters() if p.requires_grad]

    diffusion = IDDPM(str(1000))
    ep_start, step, best = resume(pixart, adapter, optimizer_G, dl_gen)
    # Note: resume function only loads G optim. In a real long run, you'd save D state too. 
    # For now, starting D from scratch at Ep34 is fine as it learns fast.

    print("🚀 DiT-SR V9 Adversarial Training Started.")
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
            # We skip D training every other step to balance speed if needed, but standard is 1:1.
            
            with torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
                drop_uncond = torch.ones(zt.shape[0], device=DEVICE)
                kwargs = dict(x=torch.cat([zt, zlr_aug.to(zt.dtype)], dim=1), timestep=t, y=y, aug_level=aug_level_emb, data_info=d_info, adapter_cond=cond_in, injection_mode="hybrid")
                kwargs["force_drop_ids"] = drop_uncond
                out = pixart(**kwargs)
                if out.shape[1] == 8: out, _ = out.chunk(2, dim=1)
                model_pred = out.float()

                alpha_t = _extract_into_tensor(diffusion.sqrt_alphas_cumprod, t, zh.shape)
                sigma_t = _extract_into_tensor(diffusion.sqrt_one_minus_alphas_cumprod, t, zh.shape)
                target_v = alpha_t * noise - sigma_t * zh.float()
                
                loss_v = F.mse_loss(model_pred, target_v) # Simplified V-loss
                z0 = alpha_t * zt.float() - sigma_t * model_pred # Reconstruct z0
                
                loss_latent_l1 = F.l1_loss(z0, zh.float())

                # Perceptual & GAN Loss (Pixel Space)
                # Decode SMALL crop to save VRAM
                top = torch.randint(0, 25, (1,), device=DEVICE).item() 
                left = torch.randint(0, 25, (1,), device=DEVICE).item()
                z0_crop = z0[..., top:top+40, left:left+40]
                img_p_raw = vae.decode(z0_crop/vae.config.scaling_factor).sample.clamp(-1,1)
                img_p_valid = img_p_raw[..., 32:-32, 32:-32] # 256x256 valid pixels
                
                y0 = top * 8 + 32; x0 = left * 8 + 32
                img_t_valid = hr[..., y0:y0+256, x0:x0+256].clamp(-1, 1) # Real HR patch
                
                loss_lpips = lpips_fn(img_p_valid, img_t_valid).mean()
                
                # GAN Loss (Generator side)
                # We want D to classify fake as real (1)
                pred_fake = discriminator(img_p_valid)
                loss_gan = -torch.mean(pred_fake) # Hinge-like generator loss or WGAN style

                # Total G Loss
                loss_G = (
                    loss_v 
                    + L1_BASE_WEIGHT * loss_latent_l1
                    + TARGET_LPIPS_WEIGHT * loss_lpips
                    + GAN_WEIGHT * loss_gan
                ) / GRAD_ACCUM_STEPS

            loss_G.backward()

            # --- Discriminator Step ---
            # Detach z0 to stop gradients flowing back to G
            img_p_detached = img_p_valid.detach()
            
            with torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
                pred_real = discriminator(img_t_valid)
                pred_fake_d = discriminator(img_p_detached)
                
                # Hinge Loss for D
                loss_D_real = torch.nn.ReLU()(1.0 - pred_real).mean()
                loss_D_fake = torch.nn.ReLU()(1.0 + pred_fake_d).mean()
                loss_D = (loss_D_real + loss_D_fake) * 0.5 / GRAD_ACCUM_STEPS

            loss_D.backward()

            if (i+1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(g_params_to_clip, 1.0)
                optimizer_G.step()
                optimizer_G.zero_grad()
                optimizer_D.step()
                optimizer_D.zero_grad()
                
                # Update EMA
                ema_pixart.step(pixart.parameters())
                ema_adapter.step(adapter.parameters())
                
                step += 1

            if i % 10 == 0:
                pbar.set_postfix({
                    'v': f"{loss_v:.3f}",
                    'lp': f"{loss_lpips:.3f}",
                    'gan': f"{loss_gan:.3f}",
                    'd': f"{loss_D.item()*GRAD_ACCUM_STEPS:.3f}"
                })

        # Validation uses EMA weights
        ema_pixart.store(pixart.parameters())
        ema_adapter.store(adapter.parameters())
        ema_pixart.copy_to(pixart.parameters())
        ema_adapter.copy_to(adapter.parameters())
        
        val_dict = validate(epoch, pixart, adapter, vae, val_loader, y, d_info, lpips_fn)
        
        ema_pixart.restore(pixart.parameters())
        ema_adapter.restore(adapter.parameters())
        
        if int(BEST_VAL_STEPS) in val_dict: metrics = val_dict[int(BEST_VAL_STEPS)]
        else: metrics = next(iter(val_dict.values()))
        best = save_smart(epoch, step, pixart, adapter, optimizer_G, best, metrics, dl_gen)

if __name__ == "__main__":
    main()