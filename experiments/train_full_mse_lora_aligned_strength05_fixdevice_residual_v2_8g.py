# experiments/train_full_mse_lora_aligned_strength05_fixdevice_residual_v2_8g.py
# NOTE: 8G-friendly residual SR variant (optional VAE/LPIPS on CPU, staged losses).

import os
import sys
import glob
import io
import math
import random
import hashlib  # $$ [MOD-SEED-1] stable hash for deterministic per-image seeds
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable, Set

import torch
import torch.nn as nn  # $$ [MOD-LORA-1]
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from diffusers import AutoencoderKL, DDIMScheduler  # $$ [MOD-SCHEDULE-1] use DDIM baseline for aligned validation
from torch.cuda.amp import GradScaler

import numpy as np  # $$ [MOD-IMG-1] np.asarray(pil) avoids TypedStorage warning
from torchvision.transforms import functional as TF

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
# 2) Paths & hparams
# -------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

TRAIN_LATENT_DIR = "/home/never/jietian/PixArt-alpha/dataset/DIV2K_train_latents_v2"
VAL_HR_DIR = os.path.join(PROJECT_ROOT, "dataset", "DIV2K_valid_HR")

PIXART_PATH = os.path.join(PROJECT_ROOT, "output", "pretrained_models", "PixArt-XL-2-512x512.pth")
VAE_PATH    = os.path.join(PROJECT_ROOT, "output", "pretrained_models", "sd-vae-ft-ema")
T5_EMBED_PATH = os.path.join(PROJECT_ROOT, "output", "quality_embed.pth")

OUT_DIR = os.path.join(PROJECT_ROOT, "experiments_results", "train_full_mse_lora_aligned_residual_v2")
os.makedirs(OUT_DIR, exist_ok=True)
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
VIS_DIR  = os.path.join(OUT_DIR, "vis")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Use fp16 only on CUDA; keep CPU runs functional.
DTYPE_PIXART = torch.float16 if DEVICE == "cuda" else torch.float32
DTYPE_LATENT = torch.float16 if DEVICE == "cuda" else torch.float32
USE_AMP = (DEVICE == "cuda")

EPOCHS = 100
BATCH_SIZE = 1
NUM_WORKERS = 0
LR_ADAPTER = 1e-5
LR_SCALES  = 1e-4
GRAD_ACCUM_STEPS = 4

# $$ [MOD-LORA-2] LoRA defaults (your chosen option A)
LORA_ENABLE = True
LORA_RANK = 8
LORA_ALPHA = 8
LORA_DROPOUT = 0.0
LORA_LAST_K_BLOCKS = 4
LR_LORA = 1e-4  # conservative LoRA LR for fast overfit; you can lower to 1e-5 if color drift increases

# keep your previous optional unfreeze pathway (default 0)
LR_UNFREEZE = 5e-6

SMOKE = False
SMOKE_TRAIN_SAMPLES = 20
SMOKE_VAL_SAMPLES = 20

NUM_INFER_STEPS = 40
FIXED_NOISE_SEED = 42

VAL_DEGRADE_MODE = "realistic"  # "realistic" / "bicubic"

# $$ [MOD-SCHEDULE-2] inference strength (0..1), default 0.5 as you requested
INFER_STRENGTH = 0.5
RESIDUAL_MODE = True
RESIDUAL_X0_L1_WEIGHT = 0.1
PIXEL_L1_WEIGHT = 0.1
HF_L1_WEIGHT = 0.05
LPIPS_WEIGHT = 0.1
LPIPS_EVERY = 5
STAGE1_EPOCHS = 5
T_TRAIN_MAX = 300

PSNR_SWITCH = 24.0
KEEP_LAST_EPOCHS = 3
KEEP_TOPK = 1

METRIC_Y_CHANNEL = True
METRIC_SHAVE_BORDER = 4

LAST_CKPT_PATH = os.path.join(CKPT_DIR, "last_full_state.pth")

# -------------------------
# 3) Import your model
# -------------------------
try:
    from diffusion.model.nets.PixArtMS import PixArtMS_XL_2
    from diffusion.model.nets.adapter import build_adapter
    from diffusion import IDDPM
    from diffusion.model.gaussian_diffusion import _extract_into_tensor
except ImportError as e:
    print(f"❌ Import failed: {e}")
    raise

# -------------------------
# 4) Dataset: offline latents
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
# 5) Validation: HR image -> degrade -> VAE encode
# -------------------------
def pil_to_tensor_norm01(pil: Image.Image) -> torch.Tensor:
    arr = np.asarray(pil, dtype=np.uint8)  # [H,W,3]
    x = torch.from_numpy(arr).permute(2,0,1).float() / 255.0
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


def laplacian_highpass(x: torch.Tensor) -> torch.Tensor:
    kernel = torch.tensor(
        [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
        device=x.device,
        dtype=x.dtype,
    )
    weight = kernel.view(1, 1, 3, 3).repeat(x.shape[1], 1, 1, 1)
    return F.conv2d(x, weight, padding=1, groups=x.shape[1])

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
            eps = torch.randn(lr_small.shape, generator=torch_gen, device=lr_small.device, dtype=lr_small.dtype)
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
# 6) Stable hash + metrics helpers
# -------------------------
def stable_int_hash(s: str, mod: int = 2**32) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h, 16) % mod

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
# 7) $$ [MOD-LORA-3] LoRA implementation + patcher (cross-attn only, last K blocks)
# -------------------------
class LoRALinear(nn.Module):
    """
    Wrap a base nn.Linear with a low-rank update: y = Wx + (B(Ax))*scale.
    - base weights stay frozen
    - A/B are fp32 trainable
    """
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got: {type(base)}")
        self.base = base
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = (self.alpha / max(1, self.r))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_f = base.in_features
        out_f = base.out_features

        # LoRA params in fp32
        self.lora_A = nn.Linear(in_f, self.r, bias=False, dtype=torch.float32)
        self.lora_B = nn.Linear(self.r, out_f, bias=False, dtype=torch.float32)
        # [FIX] Ensure LoRA weights are created on the same device as the wrapped Linear.
        # Without this, inputs on CUDA will hit "found at least two devices, cuda and cpu".
        dev = self.base.weight.device
        self.lora_A = self.lora_A.to(device=dev)
        self.lora_B = self.lora_B.to(device=dev)

        # init: B zeros => start as no-op
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # freeze base
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        # compute LoRA delta in fp32 (avoid AMP scaler issues), then cast back
        dx = self.dropout(x).float()
        with torch.cuda.amp.autocast(enabled=False):
            delta = self.lora_B(self.lora_A(dx)) * self.scaling
        return y + delta.to(dtype=y.dtype)

def _maybe_wrap_linear(parent: nn.Module, attr: str, r: int, alpha: float, dropout: float) -> int:
    if not hasattr(parent, attr):
        return 0
    m = getattr(parent, attr)
    if isinstance(m, LoRALinear):
        return 0
    if isinstance(m, nn.Linear):
        setattr(parent, attr, LoRALinear(m, r=r, alpha=alpha, dropout=dropout))
        return 1
    return 0

def _maybe_wrap_linear_in_sequential(parent: nn.Module, attr: str, r: int, alpha: float, dropout: float) -> int:
    if not hasattr(parent, attr):
        return 0
    m = getattr(parent, attr)
    if isinstance(m, nn.Sequential):
        replaced = 0
        for i, sub in enumerate(m):
            if isinstance(sub, LoRALinear):
                continue
            if isinstance(sub, nn.Linear):
                m[i] = LoRALinear(sub, r=r, alpha=alpha, dropout=dropout)
                replaced += 1
        return replaced
    return 0

def apply_lora_to_cross_attn_module(ca: nn.Module, r: int, alpha: float, dropout: float) -> int:
    """
    Patch a single cross-attn module:
      - if has to_q/to_k/to_v: wrap them
      - else if has qkv: wrap qkv (fused QKV)
      - wrap out proj if present (proj/out_proj/to_out/...)
    """
    replaced = 0

    # common naming variants
    replaced += _maybe_wrap_linear(ca, "to_q", r, alpha, dropout)
    replaced += _maybe_wrap_linear(ca, "to_k", r, alpha, dropout)
    replaced += _maybe_wrap_linear(ca, "to_v", r, alpha, dropout)

    replaced += _maybe_wrap_linear(ca, "q_proj", r, alpha, dropout)
    replaced += _maybe_wrap_linear(ca, "k_proj", r, alpha, dropout)
    replaced += _maybe_wrap_linear(ca, "v_proj", r, alpha, dropout)

    # fused QKV (single Linear producing 3*dim)
    if replaced == 0:
        replaced += _maybe_wrap_linear(ca, "qkv", r, alpha, dropout)

    # output proj variants
    replaced += _maybe_wrap_linear(ca, "proj", r, alpha, dropout)
    replaced += _maybe_wrap_linear(ca, "out_proj", r, alpha, dropout)
    replaced += _maybe_wrap_linear(ca, "o_proj", r, alpha, dropout)
    replaced += _maybe_wrap_linear(ca, "proj_out", r, alpha, dropout)
    replaced += _maybe_wrap_linear_in_sequential(ca, "to_out", r, alpha, dropout)

    return replaced

def apply_lora_cross_attn_last_k_blocks(
    pixart: nn.Module,
    last_k: int,
    r: int,
    alpha: float,
    dropout: float,
) -> int:
    if not hasattr(pixart, "blocks"):
        raise AttributeError("PixArt model has no attribute `blocks`; cannot apply LoRA by block index.")
    blocks = list(pixart.blocks)
    if last_k <= 0:
        return 0
    last_k = min(last_k, len(blocks))
    target = blocks[-last_k:]

    total_replaced = 0
    for bi, blk in enumerate(target):
        # try common attribute name first
        candidates: List[nn.Module] = []
        for attr in ["cross_attn", "attn2", "cross_attention", "crossattn"]:
            if hasattr(blk, attr):
                m = getattr(blk, attr)
                if isinstance(m, nn.Module):
                    candidates.append(m)

        # also scan by module name (robust to refactors)
        for name, m in blk.named_modules():
            lname = name.lower()
            if ("cross" in lname) and ("attn" in lname):
                candidates.append(m)

        # unique
        uniq: List[nn.Module] = []
        seen: Set[int] = set()
        for m in candidates:
            mid = id(m)
            if mid not in seen:
                seen.add(mid)
                uniq.append(m)

        for ca in uniq:
            total_replaced += apply_lora_to_cross_attn_module(ca, r=r, alpha=alpha, dropout=dropout)

    return total_replaced

# -------------------------
# 8) Trainable set A: Adapter + PixArt injection + LoRA
# -------------------------
def set_trainable_A(pixart, adapter, unfreeze_blocks: int = 0):
    for p in pixart.parameters():
        p.requires_grad = False

    # injection-related trainables in fp32
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

    # AdaLN/FiLM injection MLPs fp32 (if present)
    if hasattr(pixart, "input_adaln"):
        pixart.input_adaln = pixart.input_adaln.to(torch.float32)

    scale_params = []
    for s in getattr(pixart, "injection_scales", []):
        s.requires_grad = True
        scale_params.append(s)

    if hasattr(pixart, "cross_attn_scale"):
        pixart.cross_attn_scale.requires_grad = True
        scale_params.append(pixart.cross_attn_scale)

    proj_params = []
    if hasattr(pixart, "adapter_proj"):
        for p in pixart.adapter_proj.parameters():
            p.requires_grad = True
            proj_params.append(p)
    if hasattr(pixart, "adapter_norm"):
        for p in pixart.adapter_norm.parameters():
            p.requires_grad = True
            proj_params.append(p)
    if hasattr(pixart, "input_adapter_ln"):
        for p in pixart.input_adapter_ln.parameters():
            p.requires_grad = True
            proj_params.append(p)
    if hasattr(pixart, "input_adaln"):
        for p in pixart.input_adaln.parameters():
            p.requires_grad = True
            proj_params.append(p)

    # $$ [MOD-LORA-4] collect LoRA params (already fp32)
    lora_params = []
    for name, p in pixart.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad = True
            if p.dtype != torch.float32:
                p.data = p.data.float()
            lora_params.append(p)

    adapter_params = list(adapter.parameters())
    extra_params = []
    if unfreeze_blocks > 0 and hasattr(pixart, "blocks"):
        blocks = pixart.blocks
        if unfreeze_blocks > len(blocks):
            raise ValueError(f"unfreeze_blocks={unfreeze_blocks} exceeds depth={len(blocks)}")
        for blk in list(blocks)[-unfreeze_blocks:]:
            for p in blk.parameters():
                p.requires_grad = True
                if p.dtype != torch.float32:
                    p.data = p.data.float()
                extra_params.append(p)

    # strict dtype check (prevents AMP scaler errors)
    for pp in (adapter_params + proj_params + scale_params + lora_params + extra_params):
        if pp.requires_grad and pp.dtype != torch.float32:
            raise ValueError(f"Trainable param is not fp32: dtype={pp.dtype}")

    print(f"🔥 Trainable: Adapter({sum(p.numel() for p in adapter_params)}) | "
          f"Proj/LN({len(proj_params)}) | Scales({len(scale_params)}) | "
          f"LoRA({sum(p.numel() for p in lora_params)}) | "
          f"UnfrozenBlocks({len(extra_params)})")

    optimizer = torch.optim.AdamW([
        {"params": adapter_params, "lr": LR_ADAPTER},
        {"params": proj_params,    "lr": LR_ADAPTER},
        {"params": scale_params,   "lr": LR_SCALES},
        {"params": lora_params,    "lr": LR_LORA},       # $$ [MOD-LORA-5]
        {"params": extra_params,   "lr": LR_UNFREEZE},
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
# 9) $$ [MOD-SCHEDULE-3] Build an aligned validation scheduler from training betas
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
    """
    lr_latent: [B,4,64,64], acts as the coarse init (already aligned shape).
    strength in [0,1]: higher => more noise => more freedom.
    Returns: (latents, timesteps_to_run)
    """
    strength = float(strength)
    strength = max(0.0, min(1.0, strength))

    num_steps = int(scheduler.num_inference_steps)
    init_steps = int(num_steps * strength)
    init_steps = max(1, min(num_steps, init_steps))
    t_start_idx = num_steps - init_steps
    timesteps = scheduler.timesteps[t_start_idx:]  # descending

    g = torch.Generator(device=lr_latent.device).manual_seed(int(noise_seed))
    noise = torch.randn(lr_latent.shape, generator=g, device=lr_latent.device, dtype=lr_latent.dtype)

    t_start = timesteps[0]
    if not torch.is_tensor(t_start):
        t_start = torch.tensor(t_start, device=lr_latent.device, dtype=torch.long)
    t_batch = t_start.expand(lr_latent.shape[0])

    latents = scheduler.add_noise(lr_latent, noise, t_batch)
    return latents, timesteps

# -------------------------
# 10) Checkpoint utilities (keep your "full-state last" + topK display ckpt)
# -------------------------
def extract_inject_state_dict(pixart) -> Dict[str, torch.Tensor]:
    """
    Save:
      - known injection modules
      - any trainable parameters in pixart (this automatically includes LoRA params)
    """
    sd = pixart.state_dict()
    keep = {}

    trainable_names = set()
    for n, p in pixart.named_parameters():
        if p.requires_grad:
            trainable_names.add(n)

    for k, v in sd.items():
        if (
            k.startswith("injection_scales")
            or k.startswith("adapter_proj")
            or k.startswith("adapter_norm")
            or k.startswith("cross_attn_scale")
            or k.startswith("input_adapter_ln")
            or k.startswith("input_adaln")
            or k in trainable_names  # $$ [MOD-RESUME-KEEP-TRAINABLE]
        ):
            keep[k] = v.detach().cpu()
    return keep

def load_inject_state_dict(pixart, inject_sd: Dict[str, torch.Tensor]):
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
        print("⚠️ Resume ckpt has no pixart_inject. Continue without it.")

    start_epoch = int(ckpt.get("epoch", -1)) + 1
    global_step = int(ckpt.get("global_step", 0))
    best_records = ckpt.get("best_records", [])

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
# 11) Validation (aligned DDIM + strength=0.5 init from lr_latent)
# -------------------------
@torch.no_grad()
def validate_epoch(
    epoch: int,
    pixart,
    adapter,
    vae,
    vae_device: str,
    val_loader,
    y_embed,
    data_info,
    diffusion,
    infer_strength: float,
    lpips_fn=None,
    max_vis: int = 1,
):
    pixart.eval()
    adapter.eval()

    scheduler = build_val_scheduler(diffusion, NUM_INFER_STEPS, DEVICE)

    psnr_list, ssim_list, lpips_list = [], [], []
    vis_done = 0

    pbar = tqdm(val_loader, desc=f"Valid Ep{epoch+1}", dynamic_ncols=True)
    for it, batch in enumerate(pbar):
        hr_img_11 = batch["hr_img_11"].to(DEVICE).float()  # [B,3,512,512]
        B = hr_img_11.shape[0]

        for bi in range(B):
            item_hr = hr_img_11[bi:bi+1]
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

            hr_latent = vae.encode(item_hr).latent_dist.sample() * vae.config.scaling_factor   # [1,4,64,64] fp32
            lr_latent = vae.encode(lr_img_11).latent_dist.sample() * vae.config.scaling_factor # [1,4,64,64] fp32

            # adapter cond in fp32
            with torch.cuda.amp.autocast(enabled=False):
                cond = adapter(lr_latent.float())

            # $$ [MOD-SCHEDULE-4] img2img-style init:
            # residual mode -> start from zero delta + noise(strength)
            latents, run_ts = prepare_img2img_latents(
                scheduler,
                torch.zeros_like(lr_latent).to(dtype=torch.float32),
                strength=infer_strength,
                noise_seed=FIXED_NOISE_SEED,
            )

            # sampling loop (epsilon prediction)
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
            pred_latent = pred_latent.to(vae_device)
            pred_img = vae.decode(pred_latent / vae.config.scaling_factor).sample
            pred_img_01 = torch.clamp((pred_img + 1.0) / 2.0, 0.0, 1.0)

            gt_latent = hr_latent.to(vae_device)
            gt_img = vae.decode(gt_latent / vae.config.scaling_factor).sample
            gt_img_01 = torch.clamp((gt_img + 1.0) / 2.0, 0.0, 1.0)

            lr_img_01 = torch.clamp((lr_img_11 + 1.0) / 2.0, 0.0, 1.0)

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
    vae_device: str,
    latent_loader,
    y_embed,
    data_info,
    diffusion,
    infer_strength: float,
    lpips_fn=None,
):
    pixart.eval()
    adapter.eval()

    scheduler = build_val_scheduler(diffusion, NUM_INFER_STEPS, DEVICE)

    psnr_list, ssim_list, lpips_list = [], [], []
    pbar = tqdm(latent_loader, desc=f"Valid Ep{epoch+1}", dynamic_ncols=True)
    for batch in pbar:
        hr_latent = batch["hr_latent"].to(DEVICE).to(torch.float32)
        lr_latent = batch["lr_latent"].to(DEVICE).to(torch.float32)

        with torch.cuda.amp.autocast(enabled=False):
            cond = adapter(lr_latent.float())

        latents, run_ts = prepare_img2img_latents(
            scheduler,
            torch.zeros_like(lr_latent),
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

        pred_latent = (latents + lr_latent).to(vae_device)
        pred_img = vae.decode(pred_latent / vae.config.scaling_factor).sample
        pred_img_01 = torch.clamp((pred_img + 1.0) / 2.0, 0.0, 1.0)

        gt_latent = hr_latent.to(vae_device)
        gt_img = vae.decode(gt_latent / vae.config.scaling_factor).sample
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

# -------------------------
# 12) Main training
# -------------------------
def main():
    global LR_UNFREEZE, SMOKE, SMOKE_TRAIN_SAMPLES, SMOKE_VAL_SAMPLES, LR_LORA, INFER_STRENGTH
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="run smoke: 20 train + 20 val with full logic")
    parser.add_argument("--smoke_train", type=int, default=None, help="override smoke train samples")
    parser.add_argument("--smoke_val", type=int, default=None, help="override smoke val samples")
    parser.add_argument("--smoke_epochs", type=int, default=None, help="override smoke epochs")
    parser.add_argument("--adapter_type", type=str, default="fpn_hf", choices=["fpn", "fpn_se", "fpn_hf"])
    parser.add_argument("--overfit_latent_path", type=str, default=None, help="use one latent file for strict overfit train+val")
    parser.add_argument("--resume", type=str, default=None, help="path to resume ckpt (prefer last_full_state.pth)")
    parser.add_argument("--grad_accum_steps", type=int, default=GRAD_ACCUM_STEPS, help="gradient accumulation steps")
    parser.add_argument("--unfreeze_blocks", type=int, default=0, help="unfreeze last N PixArt blocks (0 keeps backbone frozen)")
    parser.add_argument("--lr_unfreeze", type=float, default=LR_UNFREEZE, help="lr for unfrozen PixArt blocks")
    # $$ [MOD-LORA-6] expose LoRA LR and inference strength (minimal knobs for smoke)
    parser.add_argument("--lr_lora", type=float, default=LR_LORA, help="lr for LoRA params")
    parser.add_argument("--infer_strength", type=float, default=INFER_STRENGTH, help="DDIM img2img strength in validation (0..1)")
    parser.add_argument("--vae_on_cpu", action="store_true", help="run VAE on CPU to save GPU memory")
    parser.add_argument("--lpips_on_cpu", action="store_true", help="run LPIPS on CPU to save GPU memory")
    parser.add_argument("--no_lpips", action="store_true", help="disable LPIPS loss for lower memory use")
    parser.add_argument("--lpips_weight", type=float, default=LPIPS_WEIGHT, help="LPIPS loss weight (stage2)")
    parser.add_argument("--lpips_every", type=int, default=LPIPS_EVERY, help="LPIPS loss frequency (steps)")
    args = parser.parse_args()

    SMOKE = bool(args.smoke)
    if args.smoke_train is not None:
        SMOKE_TRAIN_SAMPLES = int(args.smoke_train)
    if args.smoke_val is not None:
        SMOKE_VAL_SAMPLES = int(args.smoke_val)
    smoke_epochs = int(args.smoke_epochs) if args.smoke_epochs is not None else None
    grad_accum_steps = max(1, int(args.grad_accum_steps))
    LR_UNFREEZE = float(args.lr_unfreeze)
    LR_LORA = float(args.lr_lora)
    INFER_STRENGTH = float(args.infer_strength)
    lpips_weight = float(args.lpips_weight)
    lpips_every = int(args.lpips_every)
    print(f"DEVICE={DEVICE} | AMP={USE_AMP} | cudnn.enabled={torch.backends.cudnn.enabled}")
    print(
        f"[Config] residual_mode={RESIDUAL_MODE} | num_infer_steps={NUM_INFER_STEPS} | "
        f"infer_strength={INFER_STRENGTH} | residual_x0_l1_weight={RESIDUAL_X0_L1_WEIGHT} | "
        f"pixel_l1_weight={PIXEL_L1_WEIGHT} | hf_l1_weight={HF_L1_WEIGHT} | "
        f"lpips_weight={lpips_weight} | stage1_epochs={STAGE1_EPOCHS} | t_train_max={T_TRAIN_MAX} | "
        f"vae_on_cpu={bool(args.vae_on_cpu)} | lpips_on_cpu={bool(args.lpips_on_cpu)} | no_lpips={bool(args.no_lpips)}"
    )
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
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=(DEVICE=="cuda"))

    print("Loading PixArt...")
    pixart = PixArtMS_XL_2(input_size=64).to(DEVICE).to(DTYPE_PIXART).train()
    ckpt = torch.load(PIXART_PATH, map_location="cpu")
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if "pos_embed" in ckpt:
        del ckpt["pos_embed"]
    pixart.load_state_dict(ckpt, strict=False)

    # $$ [MOD-LORA-7] apply LoRA wrappers BEFORE building optimizer / resume
    if LORA_ENABLE:
        replaced = apply_lora_cross_attn_last_k_blocks(
            pixart,
            last_k=LORA_LAST_K_BLOCKS,
            r=LORA_RANK,
            alpha=LORA_ALPHA,
            dropout=LORA_DROPOUT,
        )
        if replaced == 0:
            raise RuntimeError(
                "LoRA patching found 0 Linear layers. "
                "This likely means cross-attn module naming differs. "
                "Search one block's named_modules() to update the patch rules."
            )
        print(f"✅ [LoRA] patched {replaced} Linear layers in cross-attn across last {LORA_LAST_K_BLOCKS} blocks "
              f"(rank={LORA_RANK}, alpha={LORA_ALPHA}).")

    print("Loading Adapter...")
    adapter = build_adapter(args.adapter_type, in_channels=4, hidden_size=1152).to(DEVICE).train()  # FP32

    optimizer = set_trainable_A(pixart, adapter, unfreeze_blocks=int(args.unfreeze_blocks))
    scaler = GradScaler(enabled=USE_AMP)

    diffusion = IDDPM(str(1000))

    vae_device = "cpu" if args.vae_on_cpu else DEVICE
    vae = AutoencoderKL.from_pretrained(VAE_PATH, local_files_only=True).to(vae_device).float().eval()
    vae.enable_slicing()

    y_embed, data_info = build_text_cond()

    lpips_fn = None
    if USE_METRICS and (not args.no_lpips):
        lpips_device = "cpu" if args.lpips_on_cpu else DEVICE
        lpips_fn = lpips.LPIPS(net="vgg").to(lpips_device).eval()
        for p in lpips_fn.parameters():
            p.requires_grad = False

    best_records: List[dict] = []
    global_step = 0
    start_epoch = 0

    if args.resume is not None:
        start_epoch, global_step, best_records = try_resume(args.resume, pixart, adapter, optimizer, scaler)

    total_epochs = EPOCHS if not SMOKE else (smoke_epochs if smoke_epochs is not None else 1)

    for epoch in range(start_epoch, total_epochs):
        pixart.train()
        adapter.train()
        stage = 1 if (epoch < STAGE1_EPOCHS) else 2
        pbar = tqdm(train_loader, desc=f"Train Ep{epoch+1}", dynamic_ncols=True)

        optimizer.zero_grad(set_to_none=True)
        for step_idx, batch in enumerate(pbar):
            hr_latent = batch["hr_latent"].to(DEVICE).to(DTYPE_LATENT)  # [B,4,64,64] fp16
            lr_latent = batch["lr_latent"].to(DEVICE).to(DTYPE_LATENT)

            B = hr_latent.shape[0]
            t = torch.randint(0, T_TRAIN_MAX, (B,), device=DEVICE).long()
            noise = torch.randn_like(hr_latent)
            target_latent = hr_latent - lr_latent
            noisy = diffusion.q_sample(target_latent, t, noise)

            if step_idx % grad_accum_steps == 0:
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
                eps_pred = out.float()
                loss = F.mse_loss(eps_pred, noise.float()) / float(grad_accum_steps)
                with torch.cuda.amp.autocast(enabled=False):
                    coef1 = _extract_into_tensor(diffusion.sqrt_recip_alphas_cumprod, t, noisy.shape)
                    coef2 = _extract_into_tensor(diffusion.sqrt_recipm1_alphas_cumprod, t, noisy.shape)
                    x0_pred = coef1 * noisy.float() - coef2 * eps_pred
                if RESIDUAL_X0_L1_WEIGHT > 0:
                    loss = loss + (
                        RESIDUAL_X0_L1_WEIGHT
                        * F.l1_loss(x0_pred, target_latent.float())
                        / float(grad_accum_steps)
                    )
                if stage >= 2 and PIXEL_L1_WEIGHT > 0:
                    pred_latent = x0_pred + lr_latent.float()
                    loss = loss + (
                        PIXEL_L1_WEIGHT
                        * F.l1_loss(pred_latent, hr_latent.float())
                        / float(grad_accum_steps)
                    )
                if stage >= 2 and HF_L1_WEIGHT > 0:
                    pred_hf = laplacian_highpass(x0_pred)
                    target_hf = laplacian_highpass(target_latent.float())
                    loss = loss + (HF_L1_WEIGHT * F.l1_loss(pred_hf, target_hf) / float(grad_accum_steps))

            if stage >= 2 and lpips_weight > 0 and lpips_fn is not None and (global_step % lpips_every == 0):
                with torch.cuda.amp.autocast(enabled=False):
                    pred_latent = x0_pred + lr_latent.float()
                    pred_img = vae.decode(pred_latent.to(vae_device) / vae.config.scaling_factor).sample
                    gt_img = vae.decode(hr_latent.float().to(vae_device) / vae.config.scaling_factor).sample
                    pred_img_01 = torch.clamp((pred_img + 1.0) / 2.0, 0.0, 1.0)
                    gt_img_01 = torch.clamp((gt_img + 1.0) / 2.0, 0.0, 1.0)
                    pred_norm = pred_img_01 * 2.0 - 1.0
                    gt_norm = gt_img_01 * 2.0 - 1.0
                    lpips_device = next(lpips_fn.parameters()).device
                    loss = loss + (
                        lpips_weight
                        * lpips_fn(pred_norm.to(lpips_device), gt_norm.to(lpips_device)).mean()
                        / float(grad_accum_steps)
                    )

            if USE_AMP:
                scaler.scale(loss).backward()
                if (step_idx + 1) % grad_accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
            else:
                loss.backward()
                if (step_idx + 1) % grad_accum_steps == 0:
                    optimizer.step()

            if (step_idx + 1) % grad_accum_steps == 0:
                global_step += 1
            pbar.set_postfix({"loss": f"{(loss.item() * grad_accum_steps):.4f}", "seen": global_step})

        if (step_idx + 1) % grad_accum_steps != 0:
            if USE_AMP:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

        # validation
        if args.overfit_latent_path:
            vpsnr, vssim, vlp = validate_overfit_latent(
                epoch, pixart, adapter, vae, vae_device, val_loader, y_embed, data_info,
                diffusion=diffusion, infer_strength=INFER_STRENGTH, lpips_fn=lpips_fn
            )
        else:
            vpsnr, vssim, vlp = validate_epoch(
                epoch, pixart, adapter, vae, vae_device, val_loader, y_embed, data_info,
                diffusion=diffusion, infer_strength=INFER_STRENGTH, lpips_fn=lpips_fn, max_vis=1
            )
        print(f"[VAL] epoch={epoch+1} PSNR={vpsnr:.2f} SSIM={vssim:.4f} LPIPS={vlp if math.isfinite(vlp) else float('nan'):.4f}")

        save_last_full_state(epoch, global_step, pixart, adapter, optimizer, scaler, best_records)
        best_records = save_topk_checkpoints(epoch, pixart, adapter, (vpsnr, vssim, vlp), best_records)

    print("✅ Done. Kept checkpoints (topK within last epochs):")
    for r in best_records:
        print(f"  epoch={r['epoch']+1} psnr={r['psnr']:.2f} lpips={r['lpips'] if math.isfinite(r['lpips']) else float('nan'):.4f} file={os.path.basename(r['path'])}")
    print(f"✅ Resume checkpoint (full state) saved at: {LAST_CKPT_PATH}")

if __name__ == "__main__":
    main()
