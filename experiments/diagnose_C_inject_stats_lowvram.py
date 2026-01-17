# experiments/diagnose_C_inject_stats_lowvram.py
import os
import sys
import glob
import io
import json
import math
import random
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import numpy as np
from tqdm import tqdm

from diffusers import AutoencoderKL, DPMSolverMultistepScheduler

# -------------------------
# 0) 环境硬约束（与训练脚本一致）
# -------------------------
torch.backends.cudnn.enabled = False

# -------------------------
# 1) 路径（与训练脚本一致）
# -------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

PIXART_PATH = os.path.join(PROJECT_ROOT, "output", "pretrained_models", "PixArt-XL-2-512x512.pth")
VAE_PATH    = os.path.join(PROJECT_ROOT, "output", "pretrained_models", "sd-vae-ft-ema")
T5_EMBED_PATH = os.path.join(PROJECT_ROOT, "output", "quality_embed.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE_PIXART = torch.float16
USE_AMP = (DEVICE == "cuda")

# 你目前的 baseline 注入层（按你“永远记住”的设定）
INJECTION_LAYERS = [0, 7, 14, 21]  # block indices

# -------------------------
# 2) 导入你的模型（与训练脚本一致）
# -------------------------
from diffusion.model.nets.PixArtMS import PixArtMS_XL_2
from diffusion.model.nets.adapter import MultiLevelAdapter

# -------------------------
# 3) 与训练脚本一致的图像/退化工具
# -------------------------
def pil_to_tensor_norm01(pil: Image.Image) -> torch.Tensor:
    # 用 copy() 避免 non-writable numpy warning
    arr = np.asarray(pil, dtype=np.uint8).copy()  # [H,W,3]
    x = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return x

def norm01_to_norm11(x01: torch.Tensor) -> torch.Tensor:
    return x01 * 2.0 - 1.0

def transforms_to_pil(x01: torch.Tensor) -> Image.Image:
    x = (x01.clamp(0, 1) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(x)

def _jpeg_compress_tensor(x11: torch.Tensor, quality: int) -> torch.Tensor:
    x = x11.clamp(-1, 1)
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
    w = kernel2d.view(1, 1, k, k).repeat(3, 1, 1, 1)
    return F.conv2d(x, w, padding=k // 2, groups=3)

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
        lr = F.interpolate(lr_small, size=(512, 512), mode="bicubic", align_corners=False)
        return lr.squeeze(0)

    blur_k = rng.choice([3, 5, 7])
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
        lr_small = (lr_small + eps * noise_std).clamp(-1, 1)

    jpeg_q = rng.randint(30, 95)
    lr_small_cpu = lr_small.squeeze(0).cpu()
    lr_small_cpu = _jpeg_compress_tensor(lr_small_cpu, jpeg_q).unsqueeze(0)

    lr = F.interpolate(lr_small_cpu, size=(512, 512), mode="bicubic", align_corners=False)
    return lr.squeeze(0)

# -------------------------
# 4) 稳定哈希（与训练脚本一致）
# -------------------------
def stable_int_hash(s: str, mod: int = 2**32) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h, 16) % mod

# -------------------------
# 5) 验证集读取（与训练脚本一致）
# -------------------------
class ValImageDataset(Dataset):
    def __init__(self, hr_dir: str, max_files: Optional[int] = None):
        exts = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"]
        paths = []
        for e in exts:
            paths += glob.glob(os.path.join(hr_dir, e))
        self.paths = sorted(list(set(paths)))
        if max_files is not None and max_files > 0:
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
# 6) 文本条件（与训练脚本一致）
# -------------------------
def build_text_cond():
    y_embed = torch.load(T5_EMBED_PATH, map_location="cpu")["prompt_embeds"].unsqueeze(1)
    y_embed = y_embed.to(DEVICE).to(DTYPE_PIXART)
    data_info = {
        "img_hw": torch.tensor([[512., 512.]], device=DEVICE, dtype=DTYPE_PIXART),
        "aspect_ratio": torch.tensor([1.], device=DEVICE, dtype=DTYPE_PIXART),
    }
    return y_embed, data_info

# -------------------------
# 7) 关键：把“注入相关模块”强制 fp32（训练脚本里已经证明必须这么做）
# -------------------------
def ensure_inject_modules_fp32(pixart):
    # injection_scales: ParameterList / list of Parameters
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

    # sanity check（只检查这些模块，避免误伤主干 fp16）
    if hasattr(pixart, "input_adapter_ln"):
        for p in pixart.input_adapter_ln.parameters():
            if p.dtype != torch.float32:
                raise RuntimeError(f"input_adapter_ln not fp32: {p.dtype}")

# -------------------------
# 8) 注入权重加载（与训练脚本一致）
# -------------------------
def extract_inject_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    keep = {}
    for k, v in sd.items():
        if (
            k.startswith("injection_scales")
            or k.startswith("adapter_proj")
            or k.startswith("adapter_norm")
            or k.startswith("cross_attn_scale")
            or k.startswith("input_adapter_ln")
        ):
            keep[k] = v
    return keep

def load_inject_state_dict(pixart, inject_sd: Dict[str, torch.Tensor]):
    sd = pixart.state_dict()
    for k, v in inject_sd.items():
        if k in sd:
            sd[k] = v.to(sd[k].device).to(sd[k].dtype)
    pixart.load_state_dict(sd, strict=False)

# -------------------------
# 9) 取 blocks（兼容不同命名）
# -------------------------
def get_blocks_modulelist(pixart):
    for name in ["blocks", "transformer_blocks", "layers"]:
        if hasattr(pixart, name):
            b = getattr(pixart, name)
            if isinstance(b, (torch.nn.ModuleList, list, tuple)):
                return b, name
    raise AttributeError("Cannot find blocks ModuleList in PixArtMS (expected one of: blocks/transformer_blocks/layers)")

# -------------------------
# 10) LN（无 affine）用于 cos 诊断：cos(LN(x_before), LN(x_after))
# -------------------------
def layer_norm_noaffine(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # x: [..., C] float32
    mu = x.mean(dim=-1, keepdim=True)
    var = (x - mu).pow(2).mean(dim=-1, keepdim=True)
    return (x - mu) / torch.sqrt(var + eps)

# -------------------------
# 11) 控制强度：临时缩放，结束后恢复
# -------------------------
@dataclass
class ControlBackup:
    inj_scales: List[float]
    cross_scale: Optional[float]

def apply_control_mult(pixart, control_mult: float, control_target: str) -> ControlBackup:
    inj0 = []
    if hasattr(pixart, "injection_scales"):
        inj0 = [float(s.detach().item()) for s in pixart.injection_scales]
    cross0 = float(pixart.cross_attn_scale.detach().item()) if hasattr(pixart, "cross_attn_scale") else None

    if control_target in ["both", "input"]:
        for s, base in zip(pixart.injection_scales, inj0):
            s.data = torch.tensor(base * control_mult, device=s.device, dtype=s.dtype)

    if control_target in ["both", "cross"] and hasattr(pixart, "cross_attn_scale") and cross0 is not None:
        pixart.cross_attn_scale.data = torch.tensor(cross0 * control_mult, device=pixart.cross_attn_scale.device, dtype=pixart.cross_attn_scale.dtype)

    return ControlBackup(inj_scales=inj0, cross_scale=cross0)

def restore_control(pixart, backup: ControlBackup):
    if hasattr(pixart, "injection_scales") and len(backup.inj_scales) > 0:
        for s, base in zip(pixart.injection_scales, backup.inj_scales):
            s.data = torch.tensor(base, device=s.device, dtype=s.dtype)
    if hasattr(pixart, "cross_attn_scale") and backup.cross_scale is not None:
        pixart.cross_attn_scale.data = torch.tensor(backup.cross_scale, device=pixart.cross_attn_scale.device, dtype=pixart.cross_attn_scale.dtype)

# -------------------------
# 12) Hook 采集器：在注入层 block 的 forward_pre_hook 里抓 x_after
# -------------------------
class InjectStatsCollector:
    def __init__(self, inj_layers: List[int], blocks):
        self.inj_layers = inj_layers
        self.blocks = blocks
        self.handles = []
        self.cur_inj: Dict[int, torch.Tensor] = {}     # layer -> inj tensor [B,N,C] fp32
        self.cur_stats: Dict[int, Dict[str, float]] = {}  # layer -> stats scalars

    def clear_current(self):
        self.cur_inj = {}
        self.cur_stats = {}

    def set_current_inj(self, inj_map: Dict[int, torch.Tensor]):
        self.cur_inj = inj_map

    def _make_hook(self, layer_idx: int):
        def pre_hook(module, inputs):
            # inputs[0] is x (token) for this block
            if len(inputs) == 0:
                return
            x_after = inputs[0]  # [B,N,C], likely fp16
            if layer_idx not in self.cur_inj:
                return

            inj = self.cur_inj[layer_idx]  # fp32
            xa = x_after.float()
            # 反推：x_before = x_after - inj
            xb = xa - inj

            # norm & ratio
            xb_norm = xb.norm(dim=-1).mean()
            inj_norm = inj.norm(dim=-1).mean()
            ratio = (inj.norm(dim=-1) / (xb.norm(dim=-1) + 1e-8)).mean()

            # cos(LN(x_before), LN(x_after))
            xb_ln = layer_norm_noaffine(xb)
            xa_ln = layer_norm_noaffine(xa)
            cos = F.cosine_similarity(xb_ln, xa_ln, dim=-1).mean()

            self.cur_stats[layer_idx] = {
                "x_before_norm": float(xb_norm.item()),
                "inj_norm": float(inj_norm.item()),
                "inj_over_x": float(ratio.item()),
                "cos_ln_before_after": float(cos.item()),
            }
        return pre_hook

    def register(self):
        for li in self.inj_layers:
            if li < 0 or li >= len(self.blocks):
                raise IndexError(f"Injection layer idx out of range: {li} (blocks={len(self.blocks)})")
            h = self.blocks[li].register_forward_pre_hook(self._make_hook(li))
            self.handles.append(h)

    def remove(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []

# -------------------------
# 13) 主流程：只做“单次 forward 触发注入 + hook 统计”
#     注意：这是最初定义的 C，不做低噪声/双采样/指标，避免跑偏
# -------------------------
@torch.no_grad()
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--val_hr_dir", type=str, required=True)
    parser.add_argument("--val_degrade_mode", type=str, default="realistic", choices=["realistic", "bicubic"])
    parser.add_argument("--num_val_images", type=int, default=50)
    parser.add_argument("--control_target", type=str, default="both", choices=["both", "input", "cross"])
    parser.add_argument("--control_mult", type=float, default=1.0)
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--sde_strength", type=float, default=0.45)
    parser.add_argument("--fixed_noise_seed", type=int, default=42)
    parser.add_argument("--vae_fp16", action="store_true")
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument("--max_split_size_mb", type=int, default=None, help="set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:... (best-effort)")
    args = parser.parse_args()

    # best-effort: 只能在 torch allocator 已经初始化后起一点作用（不保证）
    if args.max_split_size_mb is not None and DEVICE == "cuda":
        try:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:{int(args.max_split_size_mb)}"
        except Exception:
            pass

    print(f"DEVICE={DEVICE} | AMP={USE_AMP} | cudnn.enabled={torch.backends.cudnn.enabled}")

    # loader
    ds = ValImageDataset(args.val_hr_dir, max_files=(args.num_val_images if args.num_val_images > 0 else None))
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=(DEVICE == "cuda"))

    # load pixart
    print("Loading PixArt base weights...")
    pixart = PixArtMS_XL_2(input_size=64).to(DEVICE).to(DTYPE_PIXART).eval()

    ckpt_base = torch.load(PIXART_PATH, map_location="cpu")
    if "state_dict" in ckpt_base:
        ckpt_base = ckpt_base["state_dict"]
    if "pos_embed" in ckpt_base:
        del ckpt_base["pos_embed"]
    pixart.load_state_dict(ckpt_base, strict=False)

    # 关键：先把注入模块强制 fp32（对齐训练脚本）
    ensure_inject_modules_fp32(pixart)

    # load adapter
    print("Loading Adapter...")
    adapter = MultiLevelAdapter(in_channels=4, hidden_size=1152).to(DEVICE).eval()  # fp32

    # load finetuned inject + adapter weights
    print(f"Loading finetuned ckpt: {args.ckpt}")
    ft = torch.load(args.ckpt, map_location="cpu")
    if "adapter" in ft:
        adapter.load_state_dict(ft["adapter"], strict=True)
    else:
        raise KeyError("ckpt missing key: adapter")
    if "pixart_inject" in ft:
        load_inject_state_dict(pixart, ft["pixart_inject"])
    else:
        # 兼容：如果直接给 topK ckpt
        inject_sd = extract_inject_keys(ft) if isinstance(ft, dict) else {}
        if len(inject_sd) > 0:
            load_inject_state_dict(pixart, inject_sd)
        else:
            print("⚠️ No pixart_inject found; continue (stats will reflect base inject params).")

    # 再次确保注入模块 fp32（因为 load_state_dict 可能把 dtype 弄乱）
    ensure_inject_modules_fp32(pixart)

    # load vae
    print("Loading VAE (local_files_only=True)...")
    vae = AutoencoderKL.from_pretrained(VAE_PATH, local_files_only=True).to(DEVICE).eval()
    vae.enable_slicing()
    if args.vae_fp16:
        vae = vae.to(torch.float16)
        vae_dtype = torch.float16
    else:
        vae = vae.to(torch.float32)
        vae_dtype = torch.float32

    # text cond
    y_embed, data_info = build_text_cond()

    # scheduler only for add_noise (不 step，避免你之前那个 IndexError 陷阱)
    scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000, solver_order=2)
    scheduler.set_timesteps(args.num_steps, device=DEVICE)

    start_t_val = int(1000 * args.sde_strength)
    t_start = torch.tensor([start_t_val], device=DEVICE).long()

    blocks, blocks_name = get_blocks_modulelist(pixart)
    print(f"Found blocks: {blocks_name} (len={len(blocks)})")
    collector = InjectStatsCollector(INJECTION_LAYERS, blocks)
    collector.register()

    # running sums for mean/std
    stats_keys = ["x_before_norm", "inj_norm", "inj_over_x", "cos_ln_before_after"]
    running = {li: {k: {"sum": 0.0, "sum2": 0.0, "n": 0} for k in stats_keys} for li in INJECTION_LAYERS}

    # debug scales
    base_inj_scales = [float(s.detach().item()) for s in pixart.injection_scales] if hasattr(pixart, "injection_scales") else []
    base_cross = float(pixart.cross_attn_scale.detach().item()) if hasattr(pixart, "cross_attn_scale") else None

    pbar = tqdm(dl, desc="[C InjectStats LowVRAM]", dynamic_ncols=True)
    for batch in pbar:
        hr11 = batch["hr_img_11"][0].cpu()  # [3,512,512] cpu
        path = batch["path"][0]

        # per-image deterministic degrade seed（对齐训练脚本）
        seed_i = (stable_int_hash(path) + 12345) & 0xFFFFFFFF
        rng = random.Random(seed_i)
        torch_gen = torch.Generator(device="cpu").manual_seed(seed_i)

        lr11 = degrade_hr_to_lr_tensor(hr11, args.val_degrade_mode, rng, torch_gen=torch_gen)  # cpu [3,512,512]
        lr11 = lr11.unsqueeze(0).to(DEVICE)  # [1,3,512,512]

        # VAE encode：如果 vae_fp16，则输入也必须 half（否则你之前就遇到 bias half / input float）
        with torch.cuda.amp.autocast(enabled=False):
            lr_in = lr11.to(dtype=vae_dtype)
            lr_latent = vae.encode(lr_in).latent_dist.sample() * vae.config.scaling_factor  # [1,4,64,64] (vae_dtype)

        # adapter cond (fp32, autocast off)
        with torch.cuda.amp.autocast(enabled=False):
            cond = adapter(lr_latent.float())  # list of [1,1152,32,32] fp32

        # build noisy latent exactly like validate_epoch (but只需要 add_noise + 1次 forward)
        g = torch.Generator(device=DEVICE).manual_seed(int(args.fixed_noise_seed))
        lat0 = lr_latent.float()
        noise = torch.randn(lat0.shape, generator=g, device=DEVICE, dtype=lat0.dtype)
        lat_noisy = scheduler.add_noise(lat0, noise, t_start)

        # 准备 inj_map（复刻 PixArtMS 的 input 注入：flatten -> input_adapter_ln(fp32) -> scale * feat）
        collector.clear_current()
        inj_map: Dict[int, torch.Tensor] = {}

        # 只统计 input 注入层，所以这里用 INJECTION_LAYERS 的顺序对应 cond[0..3]
        for idx, li in enumerate(INJECTION_LAYERS):
            if idx >= len(cond):
                continue
            feat = cond[idx]  # [1,1152,32,32] fp32
            feat_flat = feat.flatten(2).transpose(1, 2).contiguous()  # [1,1024,1152] fp32
            # 关键：pixart.input_adapter_ln 必须是 fp32（我们已 ensure）
            with torch.cuda.amp.autocast(enabled=False):
                feat_flat = pixart.input_adapter_ln(feat_flat.float())  # fp32

            # 取对应 injection_scale（注意：这个 scale 已经可能被 control_mult 临时缩放）
            s = float(pixart.injection_scales[idx].detach().item())
            inj = feat_flat * s  # [1,1024,1152] fp32
            inj_map[li] = inj

        collector.set_current_inj(inj_map)

        # 临时控制强度（对齐你前面曲线脚本的逻辑）
        backup = apply_control_mult(pixart, args.control_mult, args.control_target)

        # 触发 forward（只需要一次，hook 会抓到 x_after）
        with torch.cuda.amp.autocast(enabled=USE_AMP, dtype=DTYPE_PIXART):
            _ = pixart(
                lat_noisy.to(DTYPE_PIXART),
                t_start,
                y_embed,
                data_info=data_info,
                adapter_cond=cond,
                injection_mode="hybrid",
            )

        # 恢复参数
        restore_control(pixart, backup)

        # 统计入 running
        for li in INJECTION_LAYERS:
            if li not in collector.cur_stats:
                continue
            st = collector.cur_stats[li]
            for k in stats_keys:
                v = float(st[k])
                running[li][k]["sum"] += v
                running[li][k]["sum2"] += v * v
                running[li][k]["n"] += 1

        # pbar 显示 cos
        show = {}
        for li in INJECTION_LAYERS:
            if li in collector.cur_stats:
                show[f"L{li}cos"] = f"{collector.cur_stats[li]['cos_ln_before_after']:.4f}"
        pbar.set_postfix(show)

        # cleanup（强制释放一些中间量，减轻 8G 压力）
        del lr11, lr_latent, cond, lat0, noise, lat_noisy
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    collector.remove()

    # 汇总 mean/std
    out_layers = {}
    for li in INJECTION_LAYERS:
        out_layers[str(li)] = {}
        for k in stats_keys:
            n = running[li][k]["n"]
            if n == 0:
                out_layers[str(li)][k] = {"mean": float("nan"), "std": float("nan"), "n": 0}
                continue
            s1 = running[li][k]["sum"]
            s2 = running[li][k]["sum2"]
            mean = s1 / n
            var = max(0.0, s2 / n - mean * mean)
            std = math.sqrt(var)
            out_layers[str(li)][k] = {"mean": mean, "std": std, "n": n}

    # effective scales after control (for logging)
    eff_inj = []
    if hasattr(pixart, "injection_scales") and len(base_inj_scales) > 0:
        for base in base_inj_scales[:4]:
            if args.control_target in ["both", "input"]:
                eff_inj.append(base * args.control_mult)
            else:
                eff_inj.append(base)
    eff_cross = base_cross
    if base_cross is not None and args.control_target in ["both", "cross"]:
        eff_cross = base_cross * args.control_mult

    payload = {
        "meta": {
            "ckpt": os.path.abspath(args.ckpt),
            "val_hr_dir": os.path.abspath(args.val_hr_dir),
            "val_degrade_mode": args.val_degrade_mode,
            "num_val_images": len(ds),
            "control_target": args.control_target,
            "control_mult": float(args.control_mult),
            "num_steps": int(args.num_steps),
            "sde_strength": float(args.sde_strength),
            "fixed_noise_seed": int(args.fixed_noise_seed),
            "injection_layers": INJECTION_LAYERS,
            "device": DEVICE,
            "dtype_pixart": str(DTYPE_PIXART),
            "use_amp": bool(USE_AMP),
            "vae_fp16": bool(args.vae_fp16),
            "vae_param_dtype": str(vae_dtype),
            "debug_scales_base": {
                "injection_scales": base_inj_scales[:4],
                "cross_attn_scale": base_cross,
            },
            "debug_scales_effective": {
                "injection_scales": eff_inj,
                "cross_attn_scale": eff_cross,
            },
        },
        "stats_by_layer": out_layers,
        "note": (
            "C InjectStats: hook抓block输入x_after(注入后)，用inj反推x_before，计算 mean||x_before||, mean||inj||, "
            "mean(inj/||x_before||), cos(LN_noaffine(x_before), LN_noaffine(x_after)). "
            "只做1次forward触发注入，不进行多步采样，不引入low-noise近似，避免跑偏。"
        ),
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved JSON to: {args.out_json}")
    print("Done.")

if __name__ == "__main__":
    main()
