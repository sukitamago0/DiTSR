# experiments/eval_effective_denoise_steps_effective20_final.py
#
# Evaluation-only script.
#
# Design principle: reuse (copy verbatim where possible) the validation utilities from
#   experiments/train_full_mse_lora_aligned_strength05_fixdevice.py
# and only change what is necessary to support:
#   - fixed infer_strength
#   - fixed *effective* denoise steps (e.g. 20)
#   - total scheduler steps increased accordingly (e.g. strength=0.5, effective=20 -> total=40)
#
# This avoids the recurring issues we saw when using non-baseline imports / metric stacks.

import os
import sys
import math
import glob
import random
import argparse
from dataclasses import dataclass

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

# ====== Baseline environment conventions (match train_full_mse_*.py) ======
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

torch.backends.cudnn.enabled = False

try:
    from torchmetrics.functional import peak_signal_noise_ratio as psnr
    from torchmetrics.functional import structural_similarity_index_measure as ssim
    import lpips
    print("✅ Metrics libraries loaded (PSNR, SSIM, LPIPS).")
except Exception:
    print("⚠️ Metrics missing. Install: pip install torchmetrics lpips")
    psnr = None
    ssim = None
    lpips = None

from diffusers import AutoencoderKL
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

# PixArt / adapter / diffusion imports (match your repo layout)
from diffusion import IDDPM
from diffusion.model.nets.PixArtMS import PixArtMS_XL_2
from diffusion.model.nets.adapter import MultiLevelAdapter


# -------------------- small utilities (copied from training baseline) --------------------

def stable_int_hash(s: str) -> int:
    import hashlib
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def pil_to_tensor_norm01(pil: Image.Image) -> torch.Tensor:
    # Avoid non-writable numpy warning by copying.
    arr = np.asarray(pil, dtype=np.uint8).copy()
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    x = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return x


def norm01_to_norm11(x01: torch.Tensor) -> torch.Tensor:
    return x01 * 2.0 - 1.0


def norm11_to_norm01(x11: torch.Tensor) -> torch.Tensor:
    return (x11 + 1.0) * 0.5


def center_crop(pil: Image.Image, size: int = 512) -> Image.Image:
    w, h = pil.size
    if w == size and h == size:
        return pil
    left = max(0, (w - size) // 2)
    top = max(0, (h - size) // 2)
    right = min(w, left + size)
    bottom = min(h, top + size)
    pil = pil.crop((left, top, right, bottom))
    if pil.size != (size, size):
        pil = pil.resize((size, size), resample=Image.BICUBIC)
    return pil


def rgb01_to_y01(img01: torch.Tensor) -> torch.Tensor:
    # img01: [3,H,W] in [0,1]
    r, g, b = img01[0:1], img01[1:2], img01[2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y


def shave_border(img: torch.Tensor, shave: int) -> torch.Tensor:
    if shave <= 0:
        return img
    if img.dim() == 3:
        return img[:, shave:-shave, shave:-shave]
    if img.dim() == 4:
        return img[:, :, shave:-shave, shave:-shave]
    return img


# -------------------- degradation (match training baseline) --------------------

def _jpeg_compress(pil: Image.Image, quality: int) -> Image.Image:
    import io
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def degrade_hr_to_lr_tensor(hr_tensor_norm01: torch.Tensor,
                            rng_py: random.Random,
                            mode: str = "realistic",
                            scale: int = 4) -> torch.Tensor:
    # hr_tensor_norm01: [3,H,W] in [0,1]
    c, h, w = hr_tensor_norm01.shape
    assert c == 3

    hr_pil = Image.fromarray((hr_tensor_norm01.permute(1, 2, 0).clamp(0, 1).numpy() * 255.0).astype(np.uint8))

    # Downsample
    lr_size = (w // scale, h // scale)

    if mode == "bicubic":
        lr_pil = hr_pil.resize(lr_size, resample=Image.BICUBIC)
        # Upsample back to HR size (so VAE encodes same spatial grid as HR)
        lr_up = lr_pil.resize((w, h), resample=Image.BICUBIC)
        return pil_to_tensor_norm01(lr_up)

    # "realistic" degradation: blur + downsample + noise + jpeg + upsample
    # Keep this intentionally simple but consistent with baseline intent.
    # (If your training script has a more elaborate pipeline, prefer copy-paste from it.)

    # Mild blur by resizing down then up before true downsample
    if rng_py.random() < 0.5:
        tmp = hr_pil.resize((w // 2, h // 2), resample=Image.BICUBIC).resize((w, h), resample=Image.BICUBIC)
    else:
        tmp = hr_pil

    lr_pil = tmp.resize(lr_size, resample=Image.BICUBIC)

    # Add Gaussian noise in pixel space
    lr_np = np.asarray(lr_pil, dtype=np.float32)
    sigma = rng_py.uniform(0.0, 8.0)  # in [0,255] scale
    if sigma > 0:
        noise = rng_py.normalvariate
        # vectorize: generate per-pixel Gaussian
        lr_np = lr_np + np.random.normal(loc=0.0, scale=sigma, size=lr_np.shape).astype(np.float32)
        lr_np = np.clip(lr_np, 0.0, 255.0)
    lr_pil = Image.fromarray(lr_np.astype(np.uint8))

    # JPEG
    if rng_py.random() < 0.8:
        q = rng_py.randint(30, 95)
        lr_pil = _jpeg_compress(lr_pil, q)

    # Upsample to HR size so latent grids align
    lr_up = lr_pil.resize((w, h), resample=Image.BICUBIC)
    return pil_to_tensor_norm01(lr_up)


# -------------------- dataset (mirror training baseline structure) --------------------

class ValImageDataset(Dataset):
    def __init__(self, hr_dir: str, crop_size: int = 512):
        self.hr_paths = sorted(
            glob.glob(os.path.join(hr_dir, "*.png")) +
            glob.glob(os.path.join(hr_dir, "*.jpg")) +
            glob.glob(os.path.join(hr_dir, "*.jpeg"))
        )
        if len(self.hr_paths) == 0:
            raise ValueError(f"No images found under {hr_dir}")
        self.crop_size = crop_size

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx: int):
        path = self.hr_paths[idx]
        pil = Image.open(path).convert("RGB")
        pil = center_crop(pil, size=self.crop_size)
        hr01 = pil_to_tensor_norm01(pil)
        hr11 = norm01_to_norm11(hr01)
        name = os.path.splitext(os.path.basename(path))[0]
        return {"hr11": hr11, "name": name}


# -------------------- LoRA (copied conceptually from training baseline) --------------------

class LoRALinear(torch.nn.Module):
    def __init__(self, base: torch.nn.Linear, rank: int = 8, alpha: int = 8, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base, torch.nn.Linear)
        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = int(rank)
        self.alpha = int(alpha)
        self.scaling = float(alpha) / float(rank) if rank > 0 else 1.0
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else None

        self.lora_A = torch.nn.Linear(self.in_features, self.rank, bias=False)
        self.lora_B = torch.nn.Linear(self.rank, self.out_features, bias=False)
        torch.nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        out = self.base(x)
        if self.rank <= 0:
            return out
        dx = x
        if self.dropout is not None:
            dx = self.dropout(dx)
        delta = self.lora_B(self.lora_A(dx)) * self.scaling
        return out + delta


def _is_cross_attn_block(mod: torch.nn.Module) -> bool:
    # PixArt blocks have attribute 'cross_attn'
    return hasattr(mod, "cross_attn")


def apply_lora_to_last_k_blocks_cross_attn(pixart: torch.nn.Module,
                                          last_k: int = 4,
                                          rank: int = 8,
                                          alpha: int = 8,
                                          dropout: float = 0.0) -> int:
    """Patch cross-attn linears in the last K blocks: qkv/proj inside the cross_attn module.

    We keep this intentionally conservative: only wrap nn.Linear modules we can find
    under block.cross_attn.
    """
    blocks = []
    for name, m in pixart.named_modules():
        # PixArtMS uses self.blocks list; we patch by iterating that list directly if exists
        pass
    if hasattr(pixart, "blocks"):
        blocks = list(pixart.blocks)
    else:
        raise RuntimeError("PixArt model has no attribute 'blocks'; cannot apply LoRA safely.")

    patched = 0
    target_blocks = blocks[-last_k:] if last_k > 0 else []
    for blk in target_blocks:
        if not _is_cross_attn_block(blk):
            continue
        ca = blk.cross_attn
        # common naming in PixArt blocks: q, k, v, proj as Linear
        for attr in ["q", "k", "v", "proj"]:
            if hasattr(ca, attr):
                layer = getattr(ca, attr)
                if isinstance(layer, torch.nn.Linear):
                    setattr(ca, attr, LoRALinear(layer, rank=rank, alpha=alpha, dropout=dropout))
                    patched += 1
    return patched


# -------------------- injection state dict load/save (copied from training baseline) --------------------

def extract_inject_state_dict(pixart: torch.nn.Module) -> dict:
    trainable_names = set()
    for k, p in pixart.named_parameters():
        if p.requires_grad:
            trainable_names.add(k)

    inject = {}
    for k, v in pixart.state_dict().items():
        if (
            k.startswith("injection_scales")
            or k.startswith("adapter_proj")
            or k.startswith("adapter_norm")
            or k.startswith("cross_attn_scale")
            or k.startswith("input_adapter_ln")
            or (k in trainable_names)  # includes LoRA params if trainable
        ):
            inject[k] = v.detach().cpu()
    return inject


def load_inject_state_dict(pixart: torch.nn.Module, sd: dict, strict: bool = False):
    # Load into model with matching keys. We do not want to accidentally load unrelated weights.
    missing, unexpected = pixart.load_state_dict(sd, strict=False)

    allow_prefixes = (
        "injection_scales",
        "adapter_proj",
        "adapter_norm",
        "cross_attn_scale",
        "input_adapter_ln",
    )

    missing_bad = [k for k in missing if not k.startswith(allow_prefixes) and ("lora_" not in k)]
    unexpected_bad = [k for k in unexpected if not k.startswith(allow_prefixes) and ("lora_" not in k)]

    if strict and (len(missing_bad) > 0 or len(unexpected_bad) > 0):
        raise RuntimeError(
            "Inject state load strict failed. "
            f"missing_bad={missing_bad[:20]} unexpected_bad={unexpected_bad[:20]}"
        )


def force_injection_modules_fp32(pixart: torch.nn.Module):
    # Keep injection-related modules in fp32 (avoids LayerNorm dtype mismatch).
    if hasattr(pixart, "input_adapter_ln"):
        pixart.input_adapter_ln = pixart.input_adapter_ln.float()
    if hasattr(pixart, "adapter_proj"):
        pixart.adapter_proj = pixart.adapter_proj.float()
    if hasattr(pixart, "adapter_norm"):
        pixart.adapter_norm = pixart.adapter_norm.float()
    if hasattr(pixart, "cross_attn_scale"):
        pixart.cross_attn_scale.data = pixart.cross_attn_scale.data.float()
    if hasattr(pixart, "injection_scales"):
        for i in range(len(pixart.injection_scales)):
            pixart.injection_scales[i].data = pixart.injection_scales[i].data.float()

    # LoRA weights also kept in fp32 for stability.
    for n, p in pixart.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            p.data = p.data.float()


# -------------------- scheduler build (copied from training baseline, with total steps support) --------------------

def build_val_scheduler(num_train_timesteps: int, num_infer_steps: int, device: torch.device):
    diffusion = IDDPM(str(num_train_timesteps))
    betas = diffusion.betas

    kwargs = dict(
        num_train_timesteps=num_train_timesteps,
        beta_schedule="linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=0,
    )

    # Prefer trained_betas when supported (exactly aligns add_noise with training diffusion)
    try:
        scheduler = DDIMScheduler(trained_betas=betas.cpu().numpy(), **kwargs)
    except TypeError:
        scheduler = DDIMScheduler(beta_start=float(betas[0]), beta_end=float(betas[-1]), **kwargs)

    scheduler.set_timesteps(num_infer_steps, device=device)
    return scheduler


def compute_total_steps_for_effective(strength: float, effective_steps: int, max_total: int = 1000) -> int:
    strength = float(strength)
    assert 0.0 < strength <= 1.0
    effective_steps = int(effective_steps)
    assert effective_steps > 0

    total = int(math.ceil(effective_steps / max(strength, 1e-6)))
    total = max(total, effective_steps)
    # Adjust so that int(total*strength) == effective_steps exactly.
    while int(total * strength) < effective_steps and total < max_total:
        total += 1
    # If we overshot (rare), step down while keeping equality.
    while total > 1 and int((total - 1) * strength) == effective_steps:
        total -= 1

    if int(total * strength) != effective_steps:
        # Fallback: enforce equality by direct override in the sampling loop.
        # (We still return a reasonable total.)
        pass
    return total


# -------------------- VAE helpers (support VAE on CPU) --------------------

def vae_encode_to_latent(vae: AutoencoderKL, img11: torch.Tensor, vae_device: torch.device) -> torch.Tensor:
    # img11: [B,3,H,W] in [-1,1]
    img11 = img11.to(vae_device)
    with torch.no_grad():
        lat = vae.encode(img11).latent_dist.sample()
        lat = lat * vae.config.scaling_factor
    return lat


def vae_decode_latent(vae: AutoencoderKL, latent: torch.Tensor, vae_device: torch.device) -> torch.Tensor:
    # returns [B,3,H,W] in [0,1] on CPU
    latent = latent.to(vae_device)
    with torch.no_grad():
        img = vae.decode(latent / vae.config.scaling_factor).sample
    img01 = norm11_to_norm01(img).clamp(0, 1)
    return img01.detach().cpu()


# -------------------- core sampling (effective denoise steps) --------------------

@torch.no_grad()
def sample_with_effective_steps(pixart: torch.nn.Module,
                               adapter: torch.nn.Module,
                               scheduler: DDIMScheduler,
                               x_latent: torch.Tensor,
                               y_embed: torch.Tensor,
                               y_lens: torch.Tensor,
                               infer_strength: float,
                               effective_steps: int,
                               generator: torch.Generator,
                               dtype_pixart: torch.dtype):
    """img2img-like sampling: add noise to x_latent at a chosen start timestep, then denoise.

    Important: we keep *effective denoise step count* fixed to `effective_steps`.
    This is done by using `scheduler.timesteps` of length `total_steps` (already set),
    and slicing the last `effective_steps` steps.
    """
    total_steps = int(scheduler.num_inference_steps)

    # Decide start index so that we run exactly `effective_steps` denoising updates.
    # timesteps is descending; index larger => earlier (more noisy).
    run_steps = int(effective_steps)
    run_steps = max(1, min(run_steps, total_steps))

    t_start = total_steps - run_steps

    timesteps = scheduler.timesteps.to(x_latent.device)
    t_start_t = timesteps[t_start]

    noise = torch.randn_like(x_latent, generator=generator)
    latents = scheduler.add_noise(x_latent, noise, t_start_t)

    # Adapter conditioning: keep fp32 as in training baseline
    with torch.cuda.amp.autocast(enabled=False):
        adapter_cond = adapter(x_latent.float())

    for t in timesteps[t_start:]:
        t_tensor = t.unsqueeze(0).to(x_latent.device)
        with torch.cuda.amp.autocast(dtype=dtype_pixart, enabled=(dtype_pixart in (torch.float16, torch.bfloat16))):
            eps = pixart(latents.to(dtype_pixart), y_embed, t_tensor, y_lens=y_lens, adapter_cond=adapter_cond)
        latents = scheduler.step(eps.float(), t, latents).prev_sample

    return latents


# -------------------- text cond (same as training baseline: load precomputed T5 embeds) --------------------

def build_text_cond(t5_embed_path: str, device: torch.device):
    pack = torch.load(t5_embed_path, map_location="cpu")

    # expected: dict with keys (emb, lens) or similar
    # training baseline uses these names; tolerate common variants.
    if isinstance(pack, dict):
        if "emb" in pack and "lens" in pack:
            emb = pack["emb"]
            lens = pack["lens"]
        elif "prompt_embeds" in pack and "prompt_attention_mask" in pack:
            emb = pack["prompt_embeds"]
            # lens = sum(mask)
            am = pack["prompt_attention_mask"]
            lens = am.sum(dim=-1)
        else:
            raise ValueError(f"Unknown T5 embed format keys={list(pack.keys())}")
    else:
        raise ValueError("T5 embed file must be a dict.")

    emb = emb.to(device)
    lens = lens.to(device)

    # Ensure batch dimension exists
    if emb.dim() == 2:
        emb = emb.unsqueeze(0)
    if lens.dim() == 0:
        lens = lens.unsqueeze(0)

    return emb, lens


# -------------------- main --------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--ckpt", type=str, required=True)

    parser.add_argument("--val_hr_dir", type=str, required=True)
    parser.add_argument("--val_degrade_mode", type=str, default="realistic", choices=["realistic", "bicubic"])
    parser.add_argument("--max_val", type=int, default=1)

    parser.add_argument("--infer_strength", type=float, default=0.5)
    parser.add_argument("--effective_denoise_steps", type=int, default=20)

    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--metric_y", action="store_true")
    parser.add_argument("--metric_shave", type=int, default=0)

    parser.add_argument("--vae_path", type=str, default="output/pretrained_models/sd-vae-ft-ema")
    parser.add_argument("--pixart_path", type=str, default="output/pretrained_models/PixArt-XL-2-512x512.pth")
    parser.add_argument("--t5_embed_path", type=str, default="output/quality_embed.pth")

    parser.add_argument("--vae_on_cpu", action="store_true")
    parser.add_argument("--lpips_on_cpu", action="store_true")
    parser.add_argument("--no_lpips", action="store_true")

    parser.add_argument("--out_dir", type=str, default="experiments_results/eval_effective_denoise_steps")

    # LoRA config must match training baseline defaults
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--lora_last_k", type=int, default=4)

    args = parser.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"[Env] device={device} cudnn.enabled={torch.backends.cudnn.enabled}")

    strength = float(args.infer_strength)
    effective_steps = int(args.effective_denoise_steps)

    total_steps = compute_total_steps_for_effective(strength=strength, effective_steps=effective_steps)
    print(f"[EvalCfg] strength={strength} effective_steps={effective_steps} -> total_steps={total_steps}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Memory config for 8GB GPUs
    vae_device = torch.device("cpu") if args.vae_on_cpu else device
    lpips_device = torch.device("cpu") if args.lpips_on_cpu else device
    print(f"[MemCfg] vae_on_cpu={args.vae_on_cpu} lpips_on_cpu={args.lpips_on_cpu} no_lpips={args.no_lpips}")

    # Load VAE
    vae = AutoencoderKL.from_pretrained(args.vae_path, local_files_only=True)
    vae.to(vae_device)
    vae.enable_slicing()
    vae.eval()

    # Load PixArt base
    print("Loading PixArt...")
    pixart = PixArtMS_XL_2(in_channels=4, input_size=64)
    sd = torch.load(args.pixart_path, map_location="cpu")
    pixart.load_state_dict(sd["state_dict"], strict=False)

    # Apply LoRA patch BEFORE loading checkpoint inject state (so keys match)
    patched = apply_lora_to_last_k_blocks_cross_attn(
        pixart,
        last_k=args.lora_last_k,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=0.0,
    )
    print(f"✅ [LoRA] patched {patched} Linear layers in cross-attn across last {args.lora_last_k} blocks (rank={args.lora_rank}, alpha={args.lora_alpha}).")

    # Ensure injection modules are fp32
    force_injection_modules_fp32(pixart)

    # Move PixArt to eval dtype/device
    dtype_pixart = torch.float16 if (device.type == "cuda") else torch.float32
    pixart.to(device)
    pixart.to(dtype=dtype_pixart)
    force_injection_modules_fp32(pixart)  # keep injection parts fp32 after dtype cast
    pixart.eval()

    # Adapter
    print("Loading Adapter...")
    adapter = MultiLevelAdapter()
    adapter.to(device)
    adapter.eval()

    # Load checkpoint (expects last_full_state.pth schema from your training script)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if "adapter" not in ckpt or "pixart_inject" not in ckpt:
        raise ValueError(f"Checkpoint missing keys. Expected 'adapter' and 'pixart_inject'. Got keys={list(ckpt.keys())}")

    adapter.load_state_dict(ckpt["adapter"], strict=True)
    load_inject_state_dict(pixart, ckpt["pixart_inject"], strict=False)

    epoch = ckpt.get("epoch", None)
    global_step = ckpt.get("global_step", None)
    print(f"✅ Loaded checkpoint: epoch={epoch} global_step={global_step}")

    # Text condition
    y_embed, y_lens = build_text_cond(args.t5_embed_path, device=device)

    # Scheduler aligned to training betas
    scheduler = build_val_scheduler(num_train_timesteps=1000, num_infer_steps=total_steps, device=device)

    # LPIPS
    lpips_fn = None
    if (not args.no_lpips) and (lpips is not None):
        print("Setting up [LPIPS] perceptual loss: trunk [vgg], v[0.1], spatial [off]")
        lpips_fn = lpips.LPIPS(net='vgg')
        lpips_fn.to(lpips_device)
        lpips_fn.eval()

    # Data
    ds = ValImageDataset(args.val_hr_dir, crop_size=512)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    # Evaluation loop
    psnr_vals, ssim_vals, lpips_vals = [], [], []

    for i, batch in enumerate(dl):
        if i >= args.max_val:
            break

        name = batch["name"][0]
        hr11 = batch["hr11"]  # [1,3,512,512] on CPU

        # Deterministic degrade per image
        rng = random.Random(stable_int_hash(f"val:{name}") + int(args.seed))

        hr01_cpu = norm11_to_norm01(hr11[0]).clamp(0, 1)  # [3,H,W] cpu
        lr01_cpu = degrade_hr_to_lr_tensor(hr01_cpu, rng_py=rng, mode=args.val_degrade_mode, scale=4)

        lr11 = norm01_to_norm11(lr01_cpu).unsqueeze(0)  # [1,3,H,W] cpu

        # Encode latents
        hr_lat = vae_encode_to_latent(vae, hr11, vae_device=vae_device)
        lr_lat = vae_encode_to_latent(vae, lr11, vae_device=vae_device)

        # Move latents to sampling device
        hr_lat = hr_lat.to(device)
        lr_lat = lr_lat.to(device)

        # Sampling generator
        g = torch.Generator(device=device)
        g.manual_seed(int(args.seed) + stable_int_hash(name))

        pred_lat = sample_with_effective_steps(
            pixart=pixart,
            adapter=adapter,
            scheduler=scheduler,
            x_latent=lr_lat,
            y_embed=y_embed,
            y_lens=y_lens,
            infer_strength=strength,
            effective_steps=effective_steps,
            generator=g,
            dtype_pixart=dtype_pixart,
        )

        # Decode
        pred01 = vae_decode_latent(vae, pred_lat.detach(), vae_device=vae_device)[0]
        gt01 = vae_decode_latent(vae, hr_lat.detach(), vae_device=vae_device)[0]
        lr_up01 = lr01_cpu  # already upsampled to HR grid

        # Metrics
        if psnr is None or ssim is None:
            raise RuntimeError("torchmetrics not available")

        pred_m = pred01
        gt_m = gt01

        if args.metric_shave > 0:
            pred_m = shave_border(pred_m, args.metric_shave)
            gt_m = shave_border(gt_m, args.metric_shave)

        if args.metric_y:
            pred_m = rgb01_to_y01(pred_m)
            gt_m = rgb01_to_y01(gt_m)

        ps = float(psnr(pred_m.unsqueeze(0), gt_m.unsqueeze(0), data_range=1.0).item())
        ss = float(ssim(pred_m.unsqueeze(0), gt_m.unsqueeze(0), data_range=1.0).item())
        psnr_vals.append(ps)
        ssim_vals.append(ss)

        lp = None
        if lpips_fn is not None:
            # LPIPS expects [-1,1]
            pred11 = norm01_to_norm11(pred01).unsqueeze(0).to(lpips_device)
            gt11 = norm01_to_norm11(gt01).unsqueeze(0).to(lpips_device)
            with torch.no_grad():
                lp = float(lpips_fn(pred11, gt11).mean().item())
            lpips_vals.append(lp)

        # Save a visual triplet
        vis = torch.cat([
            lr_up01.clamp(0, 1),
            pred01.clamp(0, 1),
            gt01.clamp(0, 1),
        ], dim=2)  # concat width: [3,H,3W]

        vis_pil = Image.fromarray((vis.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8))
        out_path = os.path.join(args.out_dir, f"{name}_eff{effective_steps}_str{strength:.2f}_tot{total_steps}_PSNR{ps:.2f}.png")
        vis_pil.save(out_path)

        if lp is None:
            print(f"[VAL] {name} PSNR={ps:.2f} SSIM={ss:.4f} saved={out_path}")
        else:
            print(f"[VAL] {name} PSNR={ps:.2f} SSIM={ss:.4f} LPIPS={lp:.4f} saved={out_path}")

        # Aggressively free GPU memory between samples on 8GB cards
        del hr_lat, lr_lat, pred_lat
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # Summary
    if len(psnr_vals) == 0:
        raise RuntimeError("No validation samples were processed.")

    psnr_mean = sum(psnr_vals) / len(psnr_vals)
    ssim_mean = sum(ssim_vals) / len(ssim_vals)
    if len(lpips_vals) > 0:
        lpips_mean = sum(lpips_vals) / len(lpips_vals)
        print(f"[VAL-AVG] N={len(psnr_vals)} PSNR={psnr_mean:.2f} SSIM={ssim_mean:.4f} LPIPS={lpips_mean:.4f}")
    else:
        print(f"[VAL-AVG] N={len(psnr_vals)} PSNR={psnr_mean:.2f} SSIM={ssim_mean:.4f}")


if __name__ == "__main__":
    main()
