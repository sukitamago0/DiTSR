# experiments/diagnose_C_inject_stats_lowvram.py
import os, sys, io, glob, json, random, hashlib, gc
from typing import Optional, Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm

from diffusers import AutoencoderKL, DPMSolverMultistepScheduler

# ---- match your train script convention
torch.backends.cudnn.enabled = False

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

PIXART_PATH = os.path.join(PROJECT_ROOT, "output", "pretrained_models", "PixArt-XL-2-512x512.pth")
VAE_PATH    = os.path.join(PROJECT_ROOT, "output", "pretrained_models", "sd-vae-ft-ema")
T5_EMBED_PATH = os.path.join(PROJECT_ROOT, "output", "quality_embed.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE_PIXART = torch.float16
USE_AMP = (DEVICE == "cuda")

from diffusion.model.nets.PixArtMS import PixArtMS_XL_2
from diffusion.model.nets.adapter import MultiLevelAdapter


def stable_int_hash(s: str, mod: int = 2**32) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h, 16) % mod

def pil_to_tensor_norm01(pil: Image.Image) -> torch.Tensor:
    # copy() avoids "numpy array is not writable" warning
    arr = np.asarray(pil, dtype=np.uint8).copy()
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0

def norm01_to_norm11(x01: torch.Tensor) -> torch.Tensor:
    return x01 * 2.0 - 1.0

def transforms_to_pil(x01: torch.Tensor) -> Image.Image:
    x = (x01.clamp(0, 1) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(x)

def _jpeg_compress_tensor(x11: torch.Tensor, quality: int) -> torch.Tensor:
    x = x11.clamp(-1, 1)
    x01 = (x + 1.0) / 2.0
    pil = transforms_to_pil(x01.cpu())
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    pil2 = Image.open(buf).convert("RGB")
    x01b = pil_to_tensor_norm01(pil2)
    return norm01_to_norm11(x01b)

def center_crop(pil: Image.Image, size: int = 512) -> Image.Image:
    w, h = pil.size
    if w < size or h < size:
        pil = pil.resize((max(size, w), max(size, h)), resample=Image.BICUBIC)
        w, h = pil.size
    left = (w - size) // 2
    top = (h - size) // 2
    return pil.crop((left, top, left + size, top + size))

def gaussian_kernel2d(k: int, sigma: float, device, dtype):
    ax = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    ker = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return ker / ker.sum()

def depthwise_conv2d(x: torch.Tensor, kernel2d: torch.Tensor) -> torch.Tensor:
    k = kernel2d.shape[0]
    w = kernel2d.view(1, 1, k, k).repeat(3, 1, 1, 1)
    return F.conv2d(x, w, padding=k // 2, groups=3)

def degrade_hr_to_lr_tensor(
    hr11: torch.Tensor,
    mode: str,
    rng: random.Random,
    torch_gen: Optional[torch.Generator],
) -> torch.Tensor:
    """
    hr11: [3,512,512] in [-1,1] cpu
    return lr11: [3,512,512] in [-1,1] cpu
    """
    if mode == "bicubic":
        hr = hr11.unsqueeze(0)
        lr_small = F.interpolate(hr, scale_factor=0.25, mode="bicubic", align_corners=False)
        lr = F.interpolate(lr_small, size=(512, 512), mode="bicubic", align_corners=False)
        return lr.squeeze(0)

    blur_k = rng.choice([3, 5, 7])
    blur_sigma = rng.uniform(0.2, 1.2)

    hr = hr11.unsqueeze(0)
    ker = gaussian_kernel2d(blur_k, blur_sigma, device=hr.device, dtype=hr.dtype)
    hr_blur = depthwise_conv2d(hr, ker)

    lr_small = F.interpolate(hr_blur, scale_factor=0.25, mode="bicubic", align_corners=False)

    noise_std = rng.uniform(0.0, 0.02)
    if noise_std > 0:
        eps = torch.randn(lr_small.shape, generator=torch_gen, device=lr_small.device, dtype=lr_small.dtype)
        lr_small = (lr_small + eps * noise_std).clamp(-1, 1)

    jpeg_q = rng.randint(30, 95)
    lr_small_cpu = _jpeg_compress_tensor(lr_small.squeeze(0).cpu(), jpeg_q).unsqueeze(0)

    lr = F.interpolate(lr_small_cpu, size=(512, 512), mode="bicubic", align_corners=False)
    return lr.squeeze(0)

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

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        pil = center_crop(Image.open(p).convert("RGB"), 512)
        hr01 = pil_to_tensor_norm01(pil)
        hr11 = norm01_to_norm11(hr01)
        return {"hr_img_11": hr11, "path": p}

def build_text_cond():
    y_embed = torch.load(T5_EMBED_PATH, map_location="cpu")["prompt_embeds"].unsqueeze(1)
    y_embed = y_embed.to(DEVICE).to(DTYPE_PIXART)
    data_info = {
        "img_hw": torch.tensor([[512., 512.]], device=DEVICE, dtype=DTYPE_PIXART),
        "aspect_ratio": torch.tensor([1.], device=DEVICE, dtype=DTYPE_PIXART),
    }
    return y_embed, data_info

def ensure_inject_modules_fp32(pixart):
    # keep LN / scales / proj in fp32 (same rule you enforce in train script)
    if hasattr(pixart, "injection_scales"):
        for s in pixart.injection_scales:
            s.data = s.data.float()
    if hasattr(pixart, "cross_attn_scale"):
        pixart.cross_attn_scale.data = pixart.cross_attn_scale.data.float()
    if hasattr(pixart, "adapter_proj"):
        pixart.adapter_proj = pixart.adapter_proj.to(torch.float32)
    if hasattr(pixart, "adapter_norm"):
        pixart.adapter_norm = pixart.adapter_norm.to(torch.float32)
    if hasattr(pixart, "input_adapter_ln"):
        pixart.input_adapter_ln = pixart.input_adapter_ln.to(torch.float32)

def load_inject_state_dict(pixart, inject_sd: Dict[str, torch.Tensor]):
    sd = pixart.state_dict()
    for k, v in inject_sd.items():
        if k in sd:
            sd[k] = v.to(sd[k].dtype)
    pixart.load_state_dict(sd, strict=False)

def apply_control_multiplier(pixart, mult: float, target: str):
    """
    Rescale learned scales without changing PixArtMS code.
    """
    assert target in ["both", "input", "cross"]
    orig_inj = [s.data.clone() for s in pixart.injection_scales]
    orig_cross = pixart.cross_attn_scale.data.clone()
    with torch.no_grad():
        if target in ["both", "input"]:
            for i, s in enumerate(pixart.injection_scales):
                s.data = orig_inj[i] * float(mult)
        if target in ["both", "cross"]:
            pixart.cross_attn_scale.data = orig_cross * float(mult)
        if target == "input":
            pixart.cross_attn_scale.data = orig_cross * 0.0
        if target == "cross":
            for i, s in enumerate(pixart.injection_scales):
                s.data = orig_inj[i]
    def restore():
        with torch.no_grad():
            for i, s in enumerate(pixart.injection_scales):
                s.data = orig_inj[i]
            pixart.cross_attn_scale.data = orig_cross
    return restore

def vae_param_dtype(vae: AutoencoderKL) -> torch.dtype:
    return next(vae.parameters()).dtype

@torch.no_grad()
def vae_encode_scaled(vae: AutoencoderKL, x11_float32: torch.Tensor) -> torch.Tensor:
    """
    Align input dtype to VAE param dtype to avoid: Input float vs bias half.
    Return float32 scaled latent.
    """
    dt = vae_param_dtype(vae)
    x = x11_float32.to(dt)
    z = vae.encode(x).latent_dist.sample()
    z = z.to(torch.float32) * float(vae.config.scaling_factor)
    return z

class LayerStats:
    def __init__(self):
        self.sum_norm_x_before = 0.0
        self.sum_norm_inj = 0.0
        self.sum_ratio = 0.0
        self.sum_cos_ln = 0.0
        self.sum_delta_ln = 0.0
        self.count = 0

    def update(self, norm_xb, norm_inj, ratio, cos_ln, delta_ln):
        self.sum_norm_x_before += norm_xb
        self.sum_norm_inj += norm_inj
        self.sum_ratio += ratio
        self.sum_cos_ln += cos_ln
        self.sum_delta_ln += delta_ln
        self.count += 1

    def as_dict(self):
        if self.count == 0:
            return {
                "mean_norm_x_before": float("nan"),
                "mean_norm_inj": float("nan"),
                "mean_inj_over_x_before": float("nan"),
                "mean_cos_ln_before_after": float("nan"),
                "mean_delta_ln_l2": float("nan"),
                "n": 0,
            }
        return {
            "mean_norm_x_before": self.sum_norm_x_before / self.count,
            "mean_norm_inj": self.sum_norm_inj / self.count,
            "mean_inj_over_x_before": self.sum_ratio / self.count,
            "mean_cos_ln_before_after": self.sum_cos_ln / self.count,
            "mean_delta_ln_l2": self.sum_delta_ln / self.count,
            "n": self.count,
        }

def tokenwise_l2_mean(x: torch.Tensor) -> torch.Tensor:
    n = torch.linalg.vector_norm(x, ord=2, dim=-1)  # [B,N]
    return n.mean()

def tokenwise_cos_mean(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dot = (a * b).sum(-1)
    na = torch.linalg.vector_norm(a, dim=-1)
    nb = torch.linalg.vector_norm(b, dim=-1)
    cos = dot / (na * nb + eps)
    return cos.mean()

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--val_hr_dir", type=str, required=True)
    ap.add_argument("--val_degrade_mode", type=str, default="realistic", choices=["realistic", "bicubic"])
    ap.add_argument("--num_val_images", type=int, default=50)

    ap.add_argument("--control_target", type=str, default="both", choices=["both", "input", "cross"])
    ap.add_argument("--control_mult", type=float, default=1.0)

    ap.add_argument("--num_steps", type=int, default=20)
    ap.add_argument("--sde_strength", type=float, default=0.45)
    ap.add_argument("--fixed_noise_seed", type=int, default=42)

    ap.add_argument("--vae_fp16", action="store_true")
    ap.add_argument("--out_json", type=str, required=True)
    args = ap.parse_args()

    ckpt_path = os.path.abspath(os.path.join(PROJECT_ROOT, args.ckpt)) if not os.path.isabs(args.ckpt) else os.path.abspath(args.ckpt)
    val_hr_dir = os.path.abspath(os.path.join(PROJECT_ROOT, args.val_hr_dir)) if not os.path.isabs(args.val_hr_dir) else os.path.abspath(args.val_hr_dir)
    out_json = os.path.abspath(os.path.join(PROJECT_ROOT, args.out_json)) if not os.path.isabs(args.out_json) else os.path.abspath(args.out_json)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)

    print(f"DEVICE={DEVICE} | AMP={USE_AMP} | cudnn.enabled={torch.backends.cudnn.enabled}")

    print("Loading PixArt base weights...")
    pixart = PixArtMS_XL_2(input_size=64).to(DEVICE).to(DTYPE_PIXART).eval()
    base = torch.load(PIXART_PATH, map_location="cpu")
    if "state_dict" in base:
        base = base["state_dict"]
    if "pos_embed" in base:
        del base["pos_embed"]
    pixart.load_state_dict(base, strict=False)

    print("Loading Adapter...")
    adapter = MultiLevelAdapter(in_channels=4, hidden_size=1152).to(DEVICE).eval()

    print(f"Loading finetuned ckpt: {ckpt_path}")
    fin = torch.load(ckpt_path, map_location="cpu")
    adapter.load_state_dict(fin["adapter"], strict=True)
    if "pixart_inject" in fin:
        load_inject_state_dict(pixart, fin["pixart_inject"])

    ensure_inject_modules_fp32(pixart)
    restore = apply_control_multiplier(pixart, float(args.control_mult), args.control_target)

    print("Loading VAE (local_files_only=True)...")
    vae = AutoencoderKL.from_pretrained(VAE_PATH, local_files_only=True).to(DEVICE)
    vae.enable_slicing()
    try:
        vae.enable_tiling()
    except Exception:
        pass
    vae = vae.to(torch.float16 if args.vae_fp16 else torch.float32).eval()

    y_embed, data_info = build_text_cond()

    ds = ValImageDataset(val_hr_dir, max_files=args.num_val_images if args.num_val_images >= 0 else None)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=(DEVICE == "cuda"))

    injection_layers = getattr(pixart, "injection_layers", [0, 7, 14, 21])
    blocks = getattr(pixart, "blocks", None)
    if blocks is None:
        raise AttributeError("pixart has no attribute 'blocks' — cannot register hooks without touching PixArtMS code.")

    stats_by_layer = {int(L): LayerStats() for L in injection_layers}

    CAPTURE = {"on": False}
    INJ_CACHE: Dict[int, torch.Tensor] = {}  # <-- keep the dict forever, only clear()

    def make_pre_hook(layer_idx: int):
        def _hook(module, inputs):
            if not CAPTURE["on"]:
                return
            x_after = inputs[0]
            if not torch.is_tensor(x_after):
                return
            inj = INJ_CACHE.get(int(layer_idx), None)
            if inj is None:
                return

            xa = x_after.float()
            inj32 = inj
            xb = xa - inj32

            norm_xb = tokenwise_l2_mean(xb)
            norm_inj = tokenwise_l2_mean(inj32)
            ratio = norm_inj / (norm_xb + 1e-6)

            xb_ln = F.layer_norm(xb, (xb.shape[-1],))
            xa_ln = F.layer_norm(xa, (xa.shape[-1],))
            cos_ln = tokenwise_cos_mean(xb_ln, xa_ln)
            delta_ln = tokenwise_l2_mean(xa_ln - xb_ln)

            stats_by_layer[int(layer_idx)].update(
                float(norm_xb.item()),
                float(norm_inj.item()),
                float(ratio.item()),
                float(cos_ln.item()),
                float(delta_ln.item()),
            )
        return _hook

    hooks = []
    for L in injection_layers:
        hooks.append(blocks[int(L)].register_forward_pre_hook(make_pre_hook(int(L))))

    scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000, solver_order=2)
    scheduler.set_timesteps(int(args.num_steps), device=DEVICE)

    start_t_val = int(1000 * float(args.sde_strength))
    run_ts = [t for t in scheduler.timesteps if int(t.item()) <= start_t_val]
    if len(run_ts) == 0:
        raise ValueError(f"Empty run_ts. sde_strength={args.sde_strength} produced start_t_val={start_t_val}")
    target_step_idx = len(run_ts) - 1

    # ---- critical: inference_mode = no autograd graph, much lower VRAM
    with torch.inference_mode():
        pbar = tqdm(loader, desc="[C InjectStats LowVRAM]", dynamic_ncols=True)
        for batch in pbar:
            hr11 = batch["hr_img_11"].to(DEVICE).float()
            path = batch["path"][0]

            seed_i = (stable_int_hash(path) + 12345) & 0xFFFFFFFF
            rng = random.Random(seed_i)
            torch_gen = torch.Generator(device="cpu").manual_seed(seed_i)

            lr11 = degrade_hr_to_lr_tensor(
                hr11.squeeze(0).detach().cpu(),
                args.val_degrade_mode,
                rng,
                torch_gen
            ).unsqueeze(0).to(DEVICE).float()

            lr_latent = vae_encode_scaled(vae, lr11)  # float32 scaled

            # adapter_cond + INJ_CACHE in fp32, autocast disabled
            with torch.cuda.amp.autocast(enabled=False):
                cond_list: List[torch.Tensor] = adapter(lr_latent.float())  # fp32 list
                INJ_CACHE.clear()
                for i, L in enumerate(injection_layers):
                    feat = cond_list[i]  # [1,1152,32,32]
                    feat_flat = feat.flatten(2).transpose(1, 2).contiguous()  # [1,N,1152]
                    feat_ln = pixart.input_adapter_ln(feat_flat.float())      # fp32 LN
                    scale = pixart.injection_scales[i].float()               # fp32 scalar
                    INJ_CACHE[int(L)] = feat_ln * scale

            g = torch.Generator(device=DEVICE).manual_seed(int(args.fixed_noise_seed))
            noise = torch.randn(lr_latent.shape, generator=g, device=DEVICE, dtype=lr_latent.dtype)
            t_start = torch.tensor([start_t_val], device=DEVICE).long()
            latents = scheduler.add_noise(lr_latent, noise, t_start)

            for si, t in enumerate(run_ts):
                CAPTURE["on"] = (si == target_step_idx)
                t_tensor = t.unsqueeze(0).to(DEVICE)
                with torch.cuda.amp.autocast(enabled=USE_AMP, dtype=DTYPE_PIXART):
                    out = pixart(
                        latents.to(DTYPE_PIXART),
                        t_tensor,
                        y_embed,
                        data_info=data_info,
                        adapter_cond=cond_list,
                        injection_mode="hybrid",
                    )
                    if out.shape[1] == 8:
                        out, _ = out.chunk(2, dim=1)
                latents = scheduler.step(out.float(), t, latents.float()).prev_sample
                CAPTURE["on"] = False

            # cleanup (IMPORTANT: do NOT del INJ_CACHE)
            del lr11, lr_latent, cond_list, latents, noise, out
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

            msg = {}
            for L in injection_layers:
                d = stats_by_layer[int(L)].as_dict()
                msg[f"L{int(L)}cos"] = f"{d['mean_cos_ln_before_after']:.4f}"
            pbar.set_postfix(msg)

    for h in hooks:
        h.remove()
    restore()

    out = {
        "meta": {
            "ckpt": ckpt_path,
            "val_hr_dir": val_hr_dir,
            "val_degrade_mode": args.val_degrade_mode,
            "num_val_images": int(args.num_val_images),
            "control_target": args.control_target,
            "control_mult": float(args.control_mult),
            "num_steps": int(args.num_steps),
            "sde_strength": float(args.sde_strength),
            "fixed_noise_seed": int(args.fixed_noise_seed),
            "capture": "last_step_of_real_path",
            "injection_layers": [int(x) for x in injection_layers],
            "device": DEVICE,
            "dtype_pixart": str(DTYPE_PIXART),
            "use_amp": bool(USE_AMP),
            "vae_fp16": bool(args.vae_fp16),
            "vae_param_dtype": str(vae_param_dtype(vae)),
        },
        "stats_by_layer": {str(int(L)): stats_by_layer[int(L)].as_dict() for L in injection_layers}
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved json to: {out_json}")

if __name__ == "__main__":
    main()
