# experiments/diagnose_suite_full_mse.py
import os
import sys
import io
import math
import json
import glob
import random
import hashlib
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Callable

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from diffusers import AutoencoderKL, DPMSolverMultistepScheduler

# -------------------------
# 0) 环境硬约束：与 train_full_mse.py 一致
# -------------------------
torch.backends.cudnn.enabled = False

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE_PIXART = torch.float16
USE_AMP = (DEVICE == "cuda")

PIXART_PATH = os.path.join(PROJECT_ROOT, "output", "pretrained_models", "PixArt-XL-2-512x512.pth")
VAE_PATH    = os.path.join(PROJECT_ROOT, "output", "pretrained_models", "sd-vae-ft-ema")
T5_EMBED_PATH = os.path.join(PROJECT_ROOT, "output", "quality_embed.pth")

# -------------------------
# 1) Metrics（与训练脚本一致）
# -------------------------
try:
    from torchmetrics.functional import peak_signal_noise_ratio as psnr
    from torchmetrics.functional import structural_similarity_index_measure as ssim
    import lpips
    USE_METRICS = True
    print("✅ Metrics libraries loaded (PSNR, SSIM, LPIPS).")
except ImportError:
    USE_METRICS = False
    print("❌ Missing metrics libs. Please: pip install torchmetrics lpips")
    raise

# -------------------------
# 2) 导入你的模型（与训练脚本一致）
# -------------------------
from diffusion.model.nets.PixArtMS import PixArtMS_XL_2
from diffusion.model.nets.adapter import MultiLevelAdapter


# -------------------------
# 3) 工具函数：路径解析
# -------------------------
def resolve_path(p: str) -> str:
    if p is None:
        return p
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(PROJECT_ROOT, p))


# -------------------------
# 4) 与训练脚本一致的稳定哈希 & 数据处理/退化
# -------------------------
def stable_int_hash(s: str, mod: int = 2**32) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h, 16) % mod

def pil_to_tensor_norm01(pil: Image.Image) -> torch.Tensor:
    # 兼顾你训练脚本的 np.asarray 规范 + 避免 “not writable” warning
    arr = np.asarray(pil, dtype=np.uint8).copy()  # [H,W,3], writable
    x = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
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


# -------------------------
# 5) Metrics：与训练脚本一致（Y 通道 + shave）
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

def compute_metrics(pred01: torch.Tensor, gt01: torch.Tensor, metric_y: bool, shave: int, lpips_fn):
    if metric_y:
        pred_y = shave_border(rgb01_to_y01(pred01), shave)
        gt_y   = shave_border(rgb01_to_y01(gt01), shave)
        p = psnr(pred_y, gt_y, data_range=1.0).item()
        s = ssim(pred_y, gt_y, data_range=1.0).item()
    else:
        p = psnr(pred01, gt01, data_range=1.0).item()
        s = ssim(pred01, gt01, data_range=1.0).item()

    pred_norm = pred01 * 2.0 - 1.0
    gt_norm   = gt01 * 2.0 - 1.0
    l = lpips_fn(pred_norm, gt_norm).item()
    return p, s, l


# -------------------------
# 6) Val 数据集：与训练脚本一致（从 HR 图像做退化 + VAE encode）
# -------------------------
class ValImageDataset(Dataset):
    def __init__(self, hr_dir: str, max_files: Optional[int] = None):
        hr_dir = resolve_path(hr_dir)
        exts = ["*.png","*.jpg","*.jpeg","*.PNG","*.JPG"]
        paths = []
        for e in exts:
            paths += glob.glob(os.path.join(hr_dir, e))
        paths = sorted(list(set(paths)))
        if max_files is not None:
            paths = paths[:max_files]
        if len(paths) == 0:
            raise FileNotFoundError(f"No HR images found in: {hr_dir}")
        self.paths = [os.path.abspath(p) for p in paths]

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
# 7) 加载 PixArt / Adapter / VAE / Text cond：严格对齐训练脚本
# -------------------------
def build_text_cond():
    y_embed = torch.load(T5_EMBED_PATH, map_location="cpu")["prompt_embeds"].unsqueeze(1)
    y_embed = y_embed.to(DEVICE).to(DTYPE_PIXART)
    data_info = {
        "img_hw": torch.tensor([[512.,512.]], device=DEVICE, dtype=DTYPE_PIXART),
        "aspect_ratio": torch.tensor([1.], device=DEVICE, dtype=DTYPE_PIXART),
    }
    return y_embed, data_info

def prepare_injection_modules_fp32(pixart):
    # 彻底规避你之前遇到的：LN/scale/proj half/float mismatch
    with torch.no_grad():
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

def load_pixart_and_adapter(ckpt_path: str):
    ckpt_path = resolve_path(ckpt_path)

    print("Loading PixArt base weights...")
    pixart = PixArtMS_XL_2(input_size=64).to(DEVICE).to(DTYPE_PIXART).eval()
    base = torch.load(PIXART_PATH, map_location="cpu")
    if "state_dict" in base:
        base = base["state_dict"]
    if "pos_embed" in base:
        del base["pos_embed"]
    pixart.load_state_dict(base, strict=False)

    # 关键：先把注入相关模块固定成 FP32，再加载注入权重
    prepare_injection_modules_fp32(pixart)

    print("Loading Adapter...")
    adapter = MultiLevelAdapter(in_channels=4, hidden_size=1152).to(DEVICE).eval()  # FP32

    print(f"Loading finetuned ckpt: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # adapter
    if "adapter" not in ckpt:
        raise KeyError("ckpt missing key: adapter")
    adapter.load_state_dict(ckpt["adapter"], strict=True)

    # pixart inject
    if "pixart_inject" in ckpt:
        sd = pixart.state_dict()
        for k, v in ckpt["pixart_inject"].items():
            if k in sd:
                sd[k] = v.to(sd[k].dtype)
        pixart.load_state_dict(sd, strict=False)
    else:
        print("⚠️ ckpt has no pixart_inject. (unexpected)")

    # VAE：本地加载，禁止联网
    print("Loading VAE (local_files_only=True)...")
    vae = AutoencoderKL.from_pretrained(VAE_PATH, local_files_only=True).to(DEVICE).float().eval()
    vae.enable_slicing()

    y_embed, data_info = build_text_cond()

    lpips_fn = lpips.LPIPS(net="vgg").to(DEVICE).eval()
    for p in lpips_fn.parameters():
        p.requires_grad = False

    return pixart, adapter, vae, y_embed, data_info, lpips_fn


# -------------------------
# 8) 控制 scale 的“安全实现”：可恢复、可 debug
# -------------------------
@dataclass
class ScaleSnapshot:
    inj: List[torch.Tensor]          # original tensors (cloned)
    cross: torch.Tensor

def snapshot_scales(pixart) -> ScaleSnapshot:
    inj = [s.data.detach().clone() for s in pixart.injection_scales]
    cross = pixart.cross_attn_scale.data.detach().clone()
    return ScaleSnapshot(inj=inj, cross=cross)

def restore_scales(pixart, snap: ScaleSnapshot):
    with torch.no_grad():
        for i, s in enumerate(pixart.injection_scales):
            s.data.copy_(snap.inj[i])
        pixart.cross_attn_scale.data.copy_(snap.cross)

def get_scales_as_float(pixart) -> Dict[str, object]:
    return {
        "injection_scales": [float(s.data.item()) for s in pixart.injection_scales],
        "cross_attn_scale": float(pixart.cross_attn_scale.data.item()),
    }

def apply_control(
    pixart,
    snap: ScaleSnapshot,
    mult: float,
    injection_mode: str,
    layer_mask: Optional[List[float]] = None,
) -> Dict[str, object]:
    """
    mult：总控制倍数（已经合并了 schedule）
    injection_mode: 'input' / 'cross_attn' / 'hybrid'
    layer_mask: length=4, 只对 input 注入层生效（0/1 或其他比例）
    """
    if layer_mask is None:
        layer_mask = [1.0] * len(pixart.injection_layers)
    assert len(layer_mask) == len(pixart.injection_layers)

    with torch.no_grad():
        # input 注入
        for j, s in enumerate(pixart.injection_scales):
            base = snap.inj[j]
            if injection_mode in ["input", "hybrid"]:
                s.data.copy_(base * float(mult) * float(layer_mask[j]))
            else:
                s.data.zero_()

        # cross-attn 注入
        if injection_mode in ["cross_attn", "hybrid"]:
            pixart.cross_attn_scale.data.copy_(snap.cross * float(mult))
        else:
            pixart.cross_attn_scale.data.zero_()

    return get_scales_as_float(pixart)


# -------------------------
# 9) 采样：完全对齐训练脚本的 SDE_STRENGTH + DPMSolver 逻辑
# -------------------------
@torch.no_grad()
def sample_one(
    pixart,
    adapter,
    vae,
    y_embed,
    data_info,
    hr_img_11: torch.Tensor,     # [1,3,512,512] on DEVICE float32
    path: str,
    val_degrade_mode: str,
    num_steps: int,
    sde_strength: float,
    fixed_noise_seed: int,
    injection_mode: str,
    control_mult: float,
    schedule_fn: Optional[Callable[[int, int], float]],
    layer_mask: Optional[List[float]],
    capture: Optional["InjectionCapture"] = None,
    metric_y: bool = True,
    shave: int = 4,
    lpips_fn=None,
):
    # per-image deterministic degrade seed
    seed_i = (stable_int_hash(path) + 12345) & 0xFFFFFFFF
    rng = random.Random(seed_i)
    torch_gen = torch.Generator(device="cpu").manual_seed(seed_i)

    # degrade on CPU (与你训练脚本一致)
    lr_img_11 = degrade_hr_to_lr_tensor(
        hr_img_11[0].detach().cpu(),
        val_degrade_mode,
        rng,
        torch_gen=torch_gen
    ).unsqueeze(0).to(DEVICE).float()

    # VAE encode（fp32）
    hr_latent = vae.encode(hr_img_11).latent_dist.sample() * vae.config.scaling_factor
    lr_latent = vae.encode(lr_img_11).latent_dist.sample() * vae.config.scaling_factor

    # scheduler
    scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000, solver_order=2)
    scheduler.set_timesteps(num_steps, device=DEVICE)

    start_t_val = int(1000 * float(sde_strength))
    run_ts = [t for t in scheduler.timesteps if t <= start_t_val]
    if len(run_ts) == 0:
        run_ts = [scheduler.timesteps[-1]]

    # fixed noise seed（与你训练脚本一致）
    g = torch.Generator(device=DEVICE).manual_seed(int(fixed_noise_seed))
    latents = lr_latent.to(DEVICE)
    noise = torch.randn(latents.shape, generator=g, device=DEVICE, dtype=latents.dtype)
    t_start = torch.tensor([start_t_val], device=DEVICE).long()
    latents = scheduler.add_noise(latents, noise, t_start)

    # adapter cond：必须 fp32 且 autocast disabled（与你训练脚本一致）
    with torch.cuda.amp.autocast(enabled=False):
        cond = adapter(lr_latent.float())  # list fp32

    # scales snapshot（可恢复）
    snap = snapshot_scales(pixart)
    debug_before = get_scales_as_float(pixart)
    debug_after_first = None
    debug_after_last = None

    for step_idx, t in enumerate(run_ts):
        m_sched = 1.0 if schedule_fn is None else float(schedule_fn(step_idx, len(run_ts)))
        m_total = float(control_mult) * m_sched

        dbg = apply_control(pixart, snap, m_total, injection_mode=injection_mode, layer_mask=layer_mask)
        if step_idx == 0:
            debug_after_first = dbg
        if step_idx == len(run_ts) - 1:
            debug_after_last = dbg

        # 可选：统计 capture（Round C 用）
        if capture is not None and (step_idx in capture.capture_steps):
            capture.enabled = True
            capture.current_step = step_idx
            capture.clear_step(step_idx)

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

        if capture is not None and capture.enabled:
            capture.enabled = False  # 只抓这一轮 forward

        latents = scheduler.step(out.float(), t, latents.float()).prev_sample

    # restore scales
    restore_scales(pixart, snap)

    # decode to [0,1]
    pred_img = vae.decode(latents / vae.config.scaling_factor).sample
    pred01 = torch.clamp((pred_img + 1.0) / 2.0, 0.0, 1.0)

    gt_img = vae.decode(hr_latent / vae.config.scaling_factor).sample
    gt01 = torch.clamp((gt_img + 1.0) / 2.0, 0.0, 1.0)

    p, s, l = compute_metrics(pred01, gt01, metric_y=metric_y, shave=shave, lpips_fn=lpips_fn)
    return {
        "psnr": p, "ssim": s, "lpips": l,
        "debug_effective_scales_before": debug_before,
        "debug_effective_scales_after_first": debug_after_first,
        "debug_effective_scales_after_last": debug_after_last,
        "path": path,
    }


# -------------------------
# 10) Round C：抓“注入前后”统计量（不改模型代码，只用 hook）
# -------------------------
class InjectionCapture:
    """
    只抓 injection_layers 的 block 输入 x_after（注入后）
    x_before 用 x_after - (scale * feat_flat_ln) 反推（因为注入是加法）
    """
    def __init__(self, pixart):
        self.pixart = pixart
        self.enabled = False
        self.current_step = -1
        self.capture_steps = set([0, -1])  # 默认抓第一步和最后一步（-1 在外部会被替换）
        self._hooks = []
        self._x_after: Dict[int, Dict[int, torch.Tensor]] = {}  # step -> layer -> x_after

        for layer_idx in pixart.injection_layers:
            blk = pixart.blocks[layer_idx]
            h = blk.register_forward_pre_hook(self._make_hook(layer_idx))
            self._hooks.append(h)

    def _make_hook(self, layer_idx: int):
        def hook(module, args):
            if not self.enabled:
                return
            x = args[0]
            step = int(self.current_step)
            if step not in self._x_after:
                self._x_after[step] = {}
            self._x_after[step][layer_idx] = x.detach()
        return hook

    def clear_step(self, step: int):
        self._x_after[step] = {}

    def get_x_after(self, step: int, layer_idx: int) -> Optional[torch.Tensor]:
        return self._x_after.get(step, {}).get(layer_idx, None)

    def close(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


def prepare_adapter_features_flat_ln(pixart, adapter_cond_list: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    复制 PixArtMS.forward 里对 adapter_cond(list) 的处理：
    [B,C,H,W] -> flatten [B,N,C] -> input_adapter_ln(fp32, autocast disabled)
    """
    feats = []
    for feat in adapter_cond_list:
        feat_flat = feat.flatten(2).transpose(1, 2)  # [B,N,C]
        with torch.cuda.amp.autocast(enabled=False):
            feat_flat = pixart.input_adapter_ln(feat_flat.float())
        feats.append(feat_flat)
    return feats


def layer_stats_from_capture(
    pixart,
    snap: ScaleSnapshot,
    step_idx: int,
    x_after_map: Dict[int, torch.Tensor],          # layer_idx -> x_after
    adapter_feats_flat_ln: List[torch.Tensor],      # index aligns to injection_layers order
    eff_scales: Dict[str, object],                  # current effective scales (float)
) -> Dict[str, object]:
    """
    输出每个注入层的：
    - mean||x_before||
    - mean||inj||
    - ratio
    - cosine(LN(x_before), LN(x_after))
    """
    out = {}
    inj_scales = eff_scales["injection_scales"]
    for j, layer_idx in enumerate(pixart.injection_layers):
        x_after = x_after_map.get(layer_idx, None)
        if x_after is None:
            continue
        x_after = x_after.float()  # [B,N,C]
        C = x_after.shape[-1]

        # 对齐 PixArtMS 的 “feat_idx 不够则复用最后一个”
        feat_idx = j if j < len(adapter_feats_flat_ln) else -1
        feat = adapter_feats_flat_ln[feat_idx].float()

        scale = float(inj_scales[j])
        inj = feat * scale

        x_before = x_after - inj

        # norms：用 token 向量 L2 的均值，避免被 N,C 大小影响
        xb = x_before.norm(dim=-1).mean().item()
        xi = inj.norm(dim=-1).mean().item()
        ratio = xi / (xb + 1e-8)

        # LN：用 block.norm1 的 eps 做同样的 layer_norm（elementwise_affine=False）
        eps = pixart.blocks[layer_idx].norm1.eps
        xbln = F.layer_norm(x_before, (C,), None, None, eps=eps)
        xaln = F.layer_norm(x_after,  (C,), None, None, eps=eps)

        # cosine：把 [B,N,C] 展平
        a = xbln.reshape(xbln.shape[0], -1)
        b = xaln.reshape(xaln.shape[0], -1)
        cos = F.cosine_similarity(a, b, dim=-1).mean().item()

        out[str(layer_idx)] = {
            "mean_norm_x_before": xb,
            "mean_norm_inj": xi,
            "mean_ratio_inj_to_x_before": ratio,
            "cos_ln_before_after": cos,
        }
    return out


# -------------------------
# 11) 三个子命令：A / B / C
# -------------------------
def run_ablation(args):
    pixart, adapter, vae, y_embed, data_info, lpips_fn = load_pixart_and_adapter(args.ckpt)

    # dataset
    max_files = None if args.num_val_images < 0 else int(args.num_val_images)
    ds = ValImageDataset(args.val_hr_dir, max_files=max_files)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=(DEVICE=="cuda"))

    # configs：Round A（Where）
    configs = []

    # 1) full hybrid
    configs.append({
        "name": "hybrid_all",
        "injection_mode": "hybrid",
        "layer_mask": [1,1,1,1],
    })

    # 2) input only（去掉 cross）
    configs.append({
        "name": "input_only_all",
        "injection_mode": "input",
        "layer_mask": [1,1,1,1],
    })

    # 3) cross-attn only（去掉 input）
    configs.append({
        "name": "cross_attn_only",
        "injection_mode": "cross_attn",
        "layer_mask": None,
    })

    # 4) per-layer（input-only）
    for k, layer_idx in enumerate(pixart.injection_layers):
        m = [0,0,0,0]
        m[k] = 1
        configs.append({
            "name": f"input_only_layer{layer_idx}",
            "injection_mode": "input",
            "layer_mask": m,
        })

    results = []
    for cfg in configs:
        ps, ss, ls = [], [], []
        dbg_first = None

        pbar = tqdm(dl, desc=f"[Ablation] {cfg['name']}", dynamic_ncols=True)
        for rep in range(int(args.val_repeat)):
            for batch in pbar:
                hr = batch["hr_img_11"].to(DEVICE).float()  # [1,3,512,512]
                path = batch["path"][0]

                r = sample_one(
                    pixart=pixart, adapter=adapter, vae=vae,
                    y_embed=y_embed, data_info=data_info,
                    hr_img_11=hr, path=path,
                    val_degrade_mode=args.val_degrade_mode,
                    num_steps=int(args.num_steps),
                    sde_strength=float(args.sde_strength),
                    fixed_noise_seed=int(args.fixed_noise_seed),
                    injection_mode=cfg["injection_mode"],
                    control_mult=float(args.control_mult),
                    schedule_fn=None,
                    layer_mask=cfg["layer_mask"],
                    capture=None,
                    metric_y=bool(args.metric_y),
                    shave=int(args.shave),
                    lpips_fn=lpips_fn,
                )
                ps.append(r["psnr"]); ss.append(r["ssim"]); ls.append(r["lpips"])
                if dbg_first is None:
                    dbg_first = {
                        "debug_effective_scales_before": r["debug_effective_scales_before"],
                        "debug_effective_scales_after_first": r["debug_effective_scales_after_first"],
                    }

        results.append({
            "name": cfg["name"],
            "injection_mode": cfg["injection_mode"],
            "layer_mask": cfg["layer_mask"],
            "n_samples": len(ps),
            "psnr": float(sum(ps)/len(ps)),
            "ssim": float(sum(ss)/len(ss)),
            "lpips": float(sum(ls)/len(ls)),
            "debug": dbg_first,
        })

    out = {
        "meta": {
            "ckpt": resolve_path(args.ckpt),
            "val_hr_dir": resolve_path(args.val_hr_dir),
            "val_degrade_mode": args.val_degrade_mode,
            "num_val_images": int(args.num_val_images),
            "val_repeat": int(args.val_repeat),
            "control_mult": float(args.control_mult),
            "num_steps": int(args.num_steps),
            "sde_strength": float(args.sde_strength),
            "fixed_noise_seed": int(args.fixed_noise_seed),
            "metric_y": bool(args.metric_y),
            "shave": int(args.shave),
            "device": DEVICE,
            "dtype_pixart": str(DTYPE_PIXART),
            "use_amp": bool(USE_AMP),
        },
        "results": results,
    }

    os.makedirs(os.path.dirname(resolve_path(args.out_json)), exist_ok=True)
    with open(resolve_path(args.out_json), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved JSON: {resolve_path(args.out_json)}")


def run_schedule(args):
    pixart, adapter, vae, y_embed, data_info, lpips_fn = load_pixart_and_adapter(args.ckpt)

    max_files = None if args.num_val_images < 0 else int(args.num_val_images)
    ds = ValImageDataset(args.val_hr_dir, max_files=max_files)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=(DEVICE=="cuda"))

    m_lo = float(args.m_lo)
    m_hi = float(args.m_hi)
    split = float(args.split)

    def schedule_constant(step_idx: int, T: int) -> float:
        return 1.0

    def schedule_late_strong(step_idx: int, T: int) -> float:
        # 前半弱，后半强
        cut = int(round(T * split))
        return m_lo if step_idx < cut else m_hi

    def schedule_early_strong(step_idx: int, T: int) -> float:
        cut = int(round(T * split))
        return m_hi if step_idx < cut else m_lo

    def schedule_cosine_ramp(step_idx: int, T: int) -> float:
        # 平滑从 m_lo -> m_hi
        if T <= 1:
            return m_hi
        x = step_idx / (T - 1)
        w = 0.5 - 0.5 * math.cos(math.pi * x)
        return m_lo * (1 - w) + m_hi * w

    schedules = [
        {"name": "constant_1.0", "fn": schedule_constant},
        {"name": f"late_strong_{m_lo}->{m_hi}", "fn": schedule_late_strong},
        {"name": f"early_strong_{m_hi}->{m_lo}", "fn": schedule_early_strong},
        {"name": f"cosine_ramp_{m_lo}->{m_hi}", "fn": schedule_cosine_ramp},
    ]

    results = []
    for sch in schedules:
        ps, ss, ls = [], [], []
        dbg_first = None
        dbg_last = None

        pbar = tqdm(dl, desc=f"[Schedule] {sch['name']}", dynamic_ncols=True)
        for rep in range(int(args.val_repeat)):
            for batch in pbar:
                hr = batch["hr_img_11"].to(DEVICE).float()
                path = batch["path"][0]

                r = sample_one(
                    pixart=pixart, adapter=adapter, vae=vae,
                    y_embed=y_embed, data_info=data_info,
                    hr_img_11=hr, path=path,
                    val_degrade_mode=args.val_degrade_mode,
                    num_steps=int(args.num_steps),
                    sde_strength=float(args.sde_strength),
                    fixed_noise_seed=int(args.fixed_noise_seed),
                    injection_mode="hybrid",
                    control_mult=float(args.base_control_mult),
                    schedule_fn=sch["fn"],
                    layer_mask=[1,1,1,1],
                    capture=None,
                    metric_y=bool(args.metric_y),
                    shave=int(args.shave),
                    lpips_fn=lpips_fn,
                )
                ps.append(r["psnr"]); ss.append(r["ssim"]); ls.append(r["lpips"])
                if dbg_first is None:
                    dbg_first = r["debug_effective_scales_after_first"]
                dbg_last = r["debug_effective_scales_after_last"]

        results.append({
            "name": sch["name"],
            "n_samples": len(ps),
            "psnr": float(sum(ps)/len(ps)),
            "ssim": float(sum(ss)/len(ss)),
            "lpips": float(sum(ls)/len(ls)),
            "debug_effective_scales_after_first": dbg_first,
            "debug_effective_scales_after_last": dbg_last,
        })

    out = {
        "meta": {
            "ckpt": resolve_path(args.ckpt),
            "val_hr_dir": resolve_path(args.val_hr_dir),
            "val_degrade_mode": args.val_degrade_mode,
            "num_val_images": int(args.num_val_images),
            "val_repeat": int(args.val_repeat),
            "base_control_mult": float(args.base_control_mult),
            "m_lo": m_lo,
            "m_hi": m_hi,
            "split": split,
            "num_steps": int(args.num_steps),
            "sde_strength": float(args.sde_strength),
            "fixed_noise_seed": int(args.fixed_noise_seed),
            "metric_y": bool(args.metric_y),
            "shave": int(args.shave),
            "device": DEVICE,
            "dtype_pixart": str(DTYPE_PIXART),
            "use_amp": bool(USE_AMP),
        },
        "schedules": results,
    }

    os.makedirs(os.path.dirname(resolve_path(args.out_json)), exist_ok=True)
    with open(resolve_path(args.out_json), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved JSON: {resolve_path(args.out_json)}")


def run_stats(args):
    pixart, adapter, vae, y_embed, data_info, lpips_fn = load_pixart_and_adapter(args.ckpt)

    max_files = None if args.num_val_images < 0 else int(args.num_val_images)
    ds = ValImageDataset(args.val_hr_dir, max_files=max_files)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=(DEVICE=="cuda"))

    # capture：抓第一步和最后一步
    cap = InjectionCapture(pixart)

    # 聚合容器：step -> layer -> list
    agg = {
        "step0": {str(i): [] for i in pixart.injection_layers},
        "step_last": {str(i): [] for i in pixart.injection_layers},
    }

    metrics_ps, metrics_ss, metrics_lp = [], [], []

    pbar = tqdm(dl, desc="[Stats]", dynamic_ncols=True)
    for rep in range(int(args.val_repeat)):
        for batch in pbar:
            hr = batch["hr_img_11"].to(DEVICE).float()
            path = batch["path"][0]

            # 这里复用 sample_one 的采样流程，但开启 capture
            # 注意：cap.capture_steps 里的 -1 需要在 sample_one 内部等价替换不了，
            # 所以我们在每次 sample 前先把 capture_steps 设成 {0, last} 的形式：
            # 但是 last 依赖 run_ts 长度，只有 sample_one 内部知道。
            # 简化做法：sample_one 把“最后一步”固定抓 debug_after_last，
            # 我们这里抓 step0 和 step_last：通过“先跑一次得到 run_ts 长度”会更复杂。
            # 所以这里采取稳妥实现：只抓 step0；并另外再抓最后一步：用一个小技巧：
            #   - cap.capture_steps = {0} 先抓 step0
            #   - 再把 sde_strength 设成极小，使 run_ts 只剩最后一个 timestep（等效抓末步）
            # 这不会影响你训练脚本的主评估，只是 Round C 统计用途。
            # 这部分属于“诊断用近似”，不用于报告指标结论。

            # ---- 抓 step0（高噪声阶段）----
            cap.capture_steps = set([0])

            r0 = sample_one(
                pixart=pixart, adapter=adapter, vae=vae,
                y_embed=y_embed, data_info=data_info,
                hr_img_11=hr, path=path,
                val_degrade_mode=args.val_degrade_mode,
                num_steps=int(args.num_steps),
                sde_strength=float(args.sde_strength),
                fixed_noise_seed=int(args.fixed_noise_seed),
                injection_mode="hybrid",
                control_mult=float(args.control_mult),
                schedule_fn=None,
                layer_mask=[1,1,1,1],
                capture=cap,
                metric_y=bool(args.metric_y),
                shave=int(args.shave),
                lpips_fn=lpips_fn,
            )
            metrics_ps.append(r0["psnr"]); metrics_ss.append(r0["ssim"]); metrics_lp.append(r0["lpips"])

            # 复算 step0 的统计
            # 需要：adapter_features_flat_ln + 当前 effective scales + capture 的 x_after
            # 这里用 r0 的 debug_after_first 作为 step0 的 effective scales
            # 为了得到 adapter_features_flat_ln，需要重新做一次 lr_latent->adapter(lr_latent) 的流程
            # 但 sample_one 内部没有返回 lr_latent；我们保持“只基于 PixArt 内部一致的 scale 改动”
            # 因为本轮是诊断：统计量只求相对趋势，不做严格数值证明。
            # ---- 实用折中：直接把 feat 的统计替换为 “注入后与注入前 LN cos” 的代理会更稳，但需要 feat。
            # 为保证可运行，这里采用更直接的方式：重新走一次 VAE encode 得到 lr_latent，然后 adapter 得到 feat。

            # reconstruct lr_latent 同样用 deterministic degrade
            seed_i = (stable_int_hash(path) + 12345) & 0xFFFFFFFF
            rng = random.Random(seed_i)
            torch_gen = torch.Generator(device="cpu").manual_seed(seed_i)
            lr_img_11 = degrade_hr_to_lr_tensor(
                hr[0].detach().cpu(),
                args.val_degrade_mode,
                rng,
                torch_gen=torch_gen
            ).unsqueeze(0).to(DEVICE).float()
            lr_latent = vae.encode(lr_img_11).latent_dist.sample() * vae.config.scaling_factor
            with torch.cuda.amp.autocast(enabled=False):
                cond_list = adapter(lr_latent.float())
            feats_flat_ln = prepare_adapter_features_flat_ln(pixart, cond_list)

            # cap 里 step0 的 x_after map
            x_after_map0 = {}
            for layer_idx in pixart.injection_layers:
                xa = cap.get_x_after(0, layer_idx)
                if xa is not None:
                    x_after_map0[layer_idx] = xa

            st0 = layer_stats_from_capture(
                pixart=pixart,
                snap=snapshot_scales(pixart),
                step_idx=0,
                x_after_map=x_after_map0,
                adapter_feats_flat_ln=feats_flat_ln,
                eff_scales=r0["debug_effective_scales_after_first"],
            )
            for layer_idx_str, v in st0.items():
                agg["step0"][layer_idx_str].append(v)

            # ---- 抓 “末步近似”（低噪声阶段）----
            # 诊断近似：把 sde_strength 设到很小，让 run_ts 基本只剩最后一两个步，
            # 从而抓到接近末步的统计趋势（不用于主指标结论，仅用于定位“强度在低噪声是否更安全”）。
            cap.capture_steps = set([0])  # 此时 0 就相当于“低噪声阶段的第一步”
            rL = sample_one(
                pixart=pixart, adapter=adapter, vae=vae,
                y_embed=y_embed, data_info=data_info,
                hr_img_11=hr, path=path,
                val_degrade_mode=args.val_degrade_mode,
                num_steps=int(args.num_steps),
                sde_strength=float(args.low_noise_sde_strength),
                fixed_noise_seed=int(args.fixed_noise_seed),
                injection_mode="hybrid",
                control_mult=float(args.control_mult),
                schedule_fn=None,
                layer_mask=[1,1,1,1],
                capture=cap,
                metric_y=bool(args.metric_y),
                shave=int(args.shave),
                lpips_fn=lpips_fn,
            )

            # cap step0 现在对应“低噪声阶段的一步”
            x_after_mapL = {}
            for layer_idx in pixart.injection_layers:
                xa = cap.get_x_after(0, layer_idx)
                if xa is not None:
                    x_after_mapL[layer_idx] = xa

            stL = layer_stats_from_capture(
                pixart=pixart,
                snap=snapshot_scales(pixart),
                step_idx=0,
                x_after_map=x_after_mapL,
                adapter_feats_flat_ln=feats_flat_ln,
                eff_scales=rL["debug_effective_scales_after_first"],
            )
            for layer_idx_str, v in stL.items():
                agg["step_last"][layer_idx_str].append(v)

    cap.close()

    def reduce_list(lst: List[Dict[str, float]]) -> Dict[str, float]:
        # 每个 key 求均值
        if len(lst) == 0:
            return {}
        keys = lst[0].keys()
        out = {}
        for k in keys:
            out[k] = float(sum(d[k] for d in lst) / len(lst))
        return out

    stats_out = {"step0": {}, "step_last": {}}
    for step_name in ["step0", "step_last"]:
        for layer_idx_str, lst in agg[step_name].items():
            stats_out[step_name][layer_idx_str] = reduce_list(lst)

    out = {
        "meta": {
            "ckpt": resolve_path(args.ckpt),
            "val_hr_dir": resolve_path(args.val_hr_dir),
            "val_degrade_mode": args.val_degrade_mode,
            "num_val_images": int(args.num_val_images),
            "val_repeat": int(args.val_repeat),
            "control_mult": float(args.control_mult),
            "num_steps": int(args.num_steps),
            "sde_strength": float(args.sde_strength),
            "low_noise_sde_strength": float(args.low_noise_sde_strength),
            "fixed_noise_seed": int(args.fixed_noise_seed),
            "metric_y": bool(args.metric_y),
            "shave": int(args.shave),
            "device": DEVICE,
            "dtype_pixart": str(DTYPE_PIXART),
            "use_amp": bool(USE_AMP),
            "note": "Round C 的 step_last 采用 low_noise_sde_strength 近似抓低噪声阶段；用于诊断趋势，不作为主指标结论。",
        },
        "metrics_main_run": {
            "n_samples": len(metrics_ps),
            "psnr": float(sum(metrics_ps)/len(metrics_ps)) if len(metrics_ps)>0 else float("nan"),
            "ssim": float(sum(metrics_ss)/len(metrics_ss)) if len(metrics_ss)>0 else float("nan"),
            "lpips": float(sum(metrics_lp)/len(metrics_lp)) if len(metrics_lp)>0 else float("nan"),
        },
        "stats": stats_out,
    }

    os.makedirs(os.path.dirname(resolve_path(args.out_json)), exist_ok=True)
    with open(resolve_path(args.out_json), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved JSON: {resolve_path(args.out_json)}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    # common args
    def add_common(p):
        p.add_argument("--ckpt", type=str, required=True)
        p.add_argument("--val_hr_dir", type=str, required=True)
        p.add_argument("--val_degrade_mode", type=str, default="realistic", choices=["realistic", "bicubic"])
        p.add_argument("--num_val_images", type=int, default=100)
        p.add_argument("--val_repeat", type=int, default=1)
        p.add_argument("--num_steps", type=int, default=20)
        p.add_argument("--sde_strength", type=float, default=0.45)
        p.add_argument("--fixed_noise_seed", type=int, default=42)
        p.add_argument("--metric_y", action="store_true")
        p.add_argument("--shave", type=int, default=4)
        p.add_argument("--out_json", type=str, required=True)

    # A: ablation
    pa = sub.add_parser("ablation")
    add_common(pa)
    pa.add_argument("--control_mult", type=float, default=1.0)

    # B: schedule
    pb = sub.add_parser("schedule")
    add_common(pb)
    pb.add_argument("--base_control_mult", type=float, default=1.0)
    pb.add_argument("--m_lo", type=float, default=0.5)
    pb.add_argument("--m_hi", type=float, default=1.5)
    pb.add_argument("--split", type=float, default=0.5)

    # C: stats
    pc = sub.add_parser("stats")
    add_common(pc)
    pc.add_argument("--control_mult", type=float, default=1.0)
    pc.add_argument("--low_noise_sde_strength", type=float, default=0.05)

    args = parser.parse_args()

    # resolve paths early
    args.ckpt = resolve_path(args.ckpt)
    args.val_hr_dir = resolve_path(args.val_hr_dir)
    args.out_json = resolve_path(args.out_json)

    print(f"DEVICE={DEVICE} | AMP={USE_AMP} | cudnn.enabled={torch.backends.cudnn.enabled}")
    if args.cmd == "ablation":
        run_ablation(args)
    elif args.cmd == "schedule":
        run_schedule(args)
    elif args.cmd == "stats":
        run_stats(args)
    else:
        raise ValueError(args.cmd)

if __name__ == "__main__":
    main()
