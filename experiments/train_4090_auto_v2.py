# /home/hello/HJT/DiTSR/experiments/train_4090_auto.py
# Scientific SOTA Training Script for DiT-SR
# Paradigm: Automated Curriculum Learning (L1 -> LPIPS)
# Hardware: RTX 4090 (12G VRAM)
# Status: REVIEWED & FIXED (Signature Mismatch Resolved)

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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import lpips
from diffusers import AutoencoderKL, DDIMScheduler

try:
    from torchmetrics.functional import peak_signal_noise_ratio as psnr
    from torchmetrics.functional import structural_similarity_index_measure as ssim
except ImportError:
    print("⚠️ torchmetrics not found.")

try:
    from diffusion.model.nets.PixArtMS import PixArtMS_XL_2
    from diffusion.model.nets.adapter import build_adapter
    from diffusion import IDDPM
    from diffusion.model.gaussian_diffusion import _extract_into_tensor
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# ================= 2. Hyper-parameters =================
# Paths
TRAIN_HR_DIR = "/data/DF2K/DF2K_train_HR"
VAL_HR_DIR   = "/data/DF2K/DF2K_valid_HR" 
VAL_LR_DIR   = "/data/DF2K/DF2K_valid_LR_bicubic/X4" 
if not os.path.exists(VAL_LR_DIR): VAL_LR_DIR = None

PIXART_PATH = os.path.join(PROJECT_ROOT, "output", "pretrained_models", "PixArt-XL-2-512x512.pth")
VAE_PATH    = os.path.join(PROJECT_ROOT, "output", "pretrained_models", "sd-vae-ft-ema")
T5_EMBED_PATH = os.path.join(PROJECT_ROOT, "output", "quality_embed.pth")

OUT_DIR = os.path.join(PROJECT_ROOT, "experiments_results", "train_4090_auto_v2_steps50_strictcons")
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
LORA_RANK = 64
LORA_ALPHA = 64

# [Curriculum Logic]
# Step 0-5000: L1=1.0, LPIPS=0.0 (Fix Color/Structure)
# Step 5000-10000: LPIPS ramps up to 0.5
WARMUP_STEPS = 8000       
RAMP_UP_STEPS = 8000      
TARGET_LPIPS_WEIGHT = 0.30  # more conservative than 0.5

# Validation
VAL_STEPS_LIST = [50]   # training-time validation uses ONLY 50-step sampling (main metric / best ckpt)
BEST_VAL_STEPS = 50         # user choice: select best checkpoint by VAL@50
PSNR_SWITCH = 24.0
KEEP_TOPK = 1
CFG_SCALE = 3.0
USE_LQ_INIT = False
INIT_NOISE_STD = 0.0
# ================= 3. Logic Functions =================
def get_loss_weights(global_step):
    # Anchor losses are always active (diffusion eps MSE + image L1).
    # LPIPS is introduced later and weaker to reduce early hallucination.
    # LR-consistency (strict replayed degradation) ramps up BEFORE LPIPS to anchor details.
    weights = {'mse': 1.0, 'l1': 1.0}

    # --- LR consistency schedule ---
    # Step 0-1000: cons=0.0  (let structure/color settle)
    # Step 1000-6000: cons ramps to 1.0
    CONS_WARMUP = 1000
    CONS_RAMP   = 5000
    if global_step < CONS_WARMUP:
        weights['cons'] = 0.0
    elif global_step < (CONS_WARMUP + CONS_RAMP):
        p = (global_step - CONS_WARMUP) / CONS_RAMP
        weights['cons'] = 1.0 * p
    else:
        weights['cons'] = 1.0

    # --- LPIPS schedule ---
    if global_step < WARMUP_STEPS:
        weights['lpips'] = 0.0
    elif global_step < (WARMUP_STEPS + RAMP_UP_STEPS):
        progress = (global_step - WARMUP_STEPS) / RAMP_UP_STEPS
        weights['lpips'] = TARGET_LPIPS_WEIGHT * progress
    else:
        weights['lpips'] = TARGET_LPIPS_WEIGHT
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

# ================= 4. Data Pipeline =================
class DegradationPipeline:
    def __init__(self, crop_size=512):
        self.crop_size = crop_size
        self.blur_kernels = [3, 5, 7]
        self.blur_sigma_range = (0.2, 1.2)
        self.noise_range = (0.0, 0.02)
        self.downscale_factor = 0.25 

    def __call__(self, hr_tensor, return_meta: bool = False, meta=None, generator=None):
        """
        hr_tensor: [C,H,W] in [-1,1].
        If meta is None: sample degradation and optionally return meta for strict replay.
        If meta is provided: deterministically apply the EXACT same degradation (including noise realization).
        """
        img = (hr_tensor + 1.0) * 0.5  # -> [0,1]

        # -------- sample or read params --------
        if meta is None:
            if generator is None:
                blur_applied = (random.random() < 0.8)
            else:
                blur_applied = bool(torch.rand((), generator=generator).item() < 0.8)
            if blur_applied:
                if generator is None:
                    k_size = int(random.choice(self.blur_kernels))
                    sigma = float(random.uniform(*self.blur_sigma_range))
                else:
                    kernel_idx = int(torch.randint(0, len(self.blur_kernels), (1,), generator=generator).item())
                    k_size = int(self.blur_kernels[kernel_idx])
                    sigma = float(
                        torch.empty((), generator=generator).uniform_(
                            self.blur_sigma_range[0], self.blur_sigma_range[1]
                        ).item()
                    )
            else:
                k_size = 0
                sigma = 0.0
            if generator is None:
                noise_std = float(random.uniform(*self.noise_range))
            else:
                noise_std = float(
                    torch.empty((), generator=generator).uniform_(self.noise_range[0], self.noise_range[1]).item()
                )
        else:
            # meta keys are tensors; convert safely
            blur_applied = bool(int(meta["blur_applied"].item()))
            k_size = int(meta["k_size"].item())
            sigma = float(meta["sigma"].item())
            noise_std = float(meta["noise_std"].item())

        # 1. Blur
        if blur_applied:
            # torchvision gaussian_blur expects kernel size odd and >=3
            img = TF.gaussian_blur(img, k_size, [sigma, sigma])

        # 2. Downsample
        down_h = int(self.crop_size * self.downscale_factor)
        down_w = int(self.crop_size * self.downscale_factor)
        lr_small = TF.resize(img, [down_h, down_w], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)

        # 3. Noise (strict replay requires the exact noise tensor)
        if noise_std > 0:
            if meta is None:
                if generator is None:
                    noise = torch.randn_like(lr_small)
                else:
                    noise = torch.randn(
                        lr_small.shape,
                        device=lr_small.device,
                        dtype=lr_small.dtype,
                        generator=generator,
                    )
            else:
                noise = meta["noise"].to(lr_small.device, dtype=lr_small.dtype)
            lr_small = lr_small + noise * noise_std
            lr_small = torch.clamp(lr_small, 0.0, 1.0)
        else:
            noise = torch.zeros_like(lr_small)

        # 4. Upsample
        lr = TF.resize(lr_small, [self.crop_size, self.crop_size], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        lr_out = (lr * 2.0 - 1.0).clamp(-1.0, 1.0)

        if return_meta:
            meta_out = {
                "blur_applied": torch.tensor(int(blur_applied), dtype=torch.int64),
                "k_size": torch.tensor(int(k_size), dtype=torch.int64),
                "sigma": torch.tensor(float(sigma), dtype=torch.float32),
                "noise_std": torch.tensor(float(noise_std), dtype=torch.float32),
                "noise": noise.detach().cpu().float(),
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
    
    def forward(self, x):
        out = self.base(x)
        delta = self.lora_B(self.lora_A(x.float())) * self.scaling
        return out + delta.to(out.dtype)

def apply_lora(model, rank=64, alpha=64):
    cnt = 0
    for name, module in model.named_modules():
        if ("attn" in name or "cross" in name) and isinstance(module, nn.Linear):
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

def save_smart(epoch, global_step, pixart, adapter, optimizer, best_records, metrics, dl_gen):
    # 1. 准备 State Dict (约 7.5GB)
    # Strategy-1: save ALL PixArt params that are actually trainable (requires_grad=True), not name-based filtering.
    trainable_names = {n for n, p in pixart.named_parameters() if p.requires_grad}
    full_sd = pixart.state_dict()
    pixart_sd = {k: v.detach().cpu() for k, v in full_sd.items() if k in trainable_names}
    state = {
        "epoch": epoch, "step": global_step,
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
        "best_records": best_records
    }
    
    # 2. 始终保存 last.pth (这是为了断点续训，必须有)
    # 使用临时文件名再重命名，防止写入一半被中断导致文件损坏
    last_path = LAST_CKPT_PATH
    temp_path = LAST_CKPT_PATH + ".tmp"
    try:
        torch.save(state, temp_path)
        if os.path.exists(last_path):
            os.remove(last_path)
        os.rename(temp_path, last_path)
        print(f"💾 Saved last checkpoint to {last_path}")
    except Exception as e:
        print(f"❌ Failed to save last.pth: {e}")
        if os.path.exists(temp_path): os.remove(temp_path)

    # 3. 判断是否是 Top-K (Best)
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
            # 这里我们不需要重新 torch.save，直接复制 last.pth 即可，速度快且不占额外临时空间
            # 但为了代码简单，这里还是 save 一次，因为 state 还在内存里
            torch.save(state, ckpt_path)
            print(f"🏆 New Best Model! Saved to {ckpt_name}")
            
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

    return best_records

def resume(pixart, adapter, optimizer, dl_gen):
    if not os.path.exists(LAST_CKPT_PATH): return 0, 0, []
    print(f"📥 Resuming from {LAST_CKPT_PATH}...")
    ckpt = torch.load(LAST_CKPT_PATH, map_location="cpu")
    adapter.load_state_dict(ckpt["adapter"])
    curr = pixart.state_dict()
    for k, v in ckpt["pixart_trainable"].items():
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

            # Deterministic init: LQ-init (optional) or fixed noise.
            if USE_LQ_INIT:
                latents = z_lr.clone()
                if INIT_NOISE_STD > 0:
                    latents = latents + INIT_NOISE_STD * randn_like_with_generator(latents, val_gen)
            else:
                latents = randn_like_with_generator(z_hr, val_gen)
            cond = adapter(z_lr.float())

            for t in scheduler.timesteps:
                t_b = torch.tensor([t], device=DEVICE).expand(latents.shape[0])
                with torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
                    drop_uncond = torch.ones(latents.shape[0], device=DEVICE)
                    drop_cond = torch.zeros(latents.shape[0], device=DEVICE)
                    out_uncond = pixart(
                        x=latents.to(COMPUTE_DTYPE), timestep=t_b, y=y_embed,
                        mask=None, data_info=data_info, adapter_cond=cond,
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
    val_ds = DF2K_Val_Fixed_Dataset(VAL_HR_DIR, lr_root=VAL_LR_DIR, crop_size=512)
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
    pixart = PixArtMS_XL_2(input_size=64).to(DEVICE)
    base = torch.load(PIXART_PATH, map_location="cpu")
    if "state_dict" in base: base = base["state_dict"]
    if "pos_embed" in base: del base["pos_embed"]
    pixart.load_state_dict(base, strict=False)
    apply_lora(pixart, LORA_RANK, LORA_ALPHA)
    pixart.train()
    
    adapter = build_adapter("fpn_se", 4, 1152).to(DEVICE).float().train()
    vae = AutoencoderKL.from_pretrained(VAE_PATH, local_files_only=True).to(DEVICE).float().eval()
    vae.enable_slicing()
    lpips_fn = lpips.LPIPS(net='vgg').to(DEVICE).eval()
    
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
                zh = vae.encode(hr).latent_dist.sample() * vae.config.scaling_factor
                zl = vae.encode(lr).latent_dist.sample() * vae.config.scaling_factor
            
            t = torch.randint(0, 1000, (zh.shape[0],), device=DEVICE).long()
            noise = torch.randn_like(zh)
            zt = diffusion.q_sample(zh, t, noise)
            
            cond = adapter(zl.float())
            
            with torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
                out = pixart(x=zt, timestep=t, y=y, data_info=d_info, adapter_cond=cond, injection_mode="hybrid")
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
                img_t = vae.decode(zh_c/vae.config.scaling_factor).sample.clamp(-1,1)
                
                loss_l1 = F.l1_loss(img_p, img_t)

                # --- Strict LR consistency (replay EXACT same degradation used to generate LR) ---
                # Decode FULL z0 (64x64 latent -> 512x512 image) to keep degradation context consistent.
                if w.get("cons", 0.0) > 0:
                    img_full = vae.decode(z0 / vae.config.scaling_factor).sample.clamp(-1, 1)
                    # Replay degradation on pred using per-sample meta from dataset
                    deg = batch.get("deg", None)
                    if deg is None:
                        # Should not happen in training dataset; keep safe
                        loss_cons = torch.tensor(0.0, device=DEVICE)
                    else:
                        # Apply degradation per sample (supports batch dimension)
                        lr_hat_list = []
                        for b in range(img_full.shape[0]):
                            meta_b = {
                                "blur_applied": deg["blur_applied"][b].cpu(),
                                "k_size": deg["k_size"][b].cpu(),
                                "sigma": deg["sigma"][b].cpu(),
                                "noise_std": deg["noise_std"][b].cpu(),
                                "noise": deg["noise"][b].cpu(),
                            }
                            lr_hat_b = train_ds.pipeline(img_full[b], return_meta=False, meta=meta_b)
                            lr_hat_list.append(lr_hat_b)
                        lr_hat = torch.stack(lr_hat_list, dim=0).to(DEVICE)

                        # Match the SAME spatial region as the decoded patch (latent crop -> image crop).
                        y0 = top * 8; x0 = left * 8
                        lr_hat_p = lr_hat[..., y0:y0+256, x0:x0+256]
                        lr_in_p  = lr[..., y0:y0+256, x0:x0+256]
                        loss_cons = F.l1_loss(lr_hat_p, lr_in_p)
                else:
                    loss_cons = torch.tensor(0.0, device=DEVICE)

                loss_lpips = lpips_fn(img_p, img_t).mean() if w['lpips'] > 0 else torch.tensor(0.0, device=DEVICE)
                
                loss = (w['mse']*loss_mse + w['l1']*loss_l1 + w['lpips']*loss_lpips + w.get('cons',0.0)*loss_cons) / GRAD_ACCUM_STEPS
            
            loss.backward()
            
            if (i+1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0) # Safety
                optimizer.step()
                optimizer.zero_grad()
                step += 1
            
            if i % 10 == 0:
                pbar.set_postfix({'mse': f"{loss_mse:.3f}", 'l1': f"{loss_l1:.3f}", 'lp': f"{loss_lpips:.3f}", 'cons': f"{loss_cons:.3f}"})
        
        val_dict = validate(epoch, pixart, adapter, vae, val_loader, y, d_info, lpips_fn)
        # Choose metrics from BEST_VAL_STEPS for checkpoint selection
        if int(BEST_VAL_STEPS) in val_dict:
            metrics = val_dict[int(BEST_VAL_STEPS)]
        else:
            metrics = next(iter(val_dict.values()))
        best = save_smart(epoch, step, pixart, adapter, optimizer, best, metrics, dl_gen)

if __name__ == "__main__":
    main()
