# experiments/diag_eval.py
# 目的：把“验证性实验”从 train_full_mse.py 分离出来，避免训练脚本越来越复杂
# 功能：
#   1) 同一 ckpt 重复验证 val_repeat 次（判断验证抖动）
#   2) adapter 控制力扫 mult=0,1,2（判断 adapter 是否到顶）
# 输出：JSON 文件 + 可选可视化图

import os
import sys
import glob
import io
import math
import json
import random
import hashlib
import argparse
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from diffusers import AutoencoderKL, DPMSolverMultistepScheduler

# 你一直在用的硬约束
torch.backends.cudnn.enabled = False

# -------------------------
# Metrics（可选）
# -------------------------
try:
    from torchmetrics.functional import peak_signal_noise_ratio as psnr
    from torchmetrics.functional import structural_similarity_index_measure as ssim
    import lpips
    USE_METRICS = True
except ImportError:
    USE_METRICS = False

# -------------------------
# 项目根目录
# -------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

# -------------------------
# 导入你的模型/噪声
# -------------------------
from diffusion.model.nets.PixArtMS import PixArtMS_XL_2
from diffusion.model.nets.adapter import MultiLevelAdapter
from diffusion import IDDPM


# -------------------------
# $$ [UTIL-SEED] 稳定哈希：跨进程/跨机器一致
# -------------------------
def stable_int_hash(s: str, mod: int = 2**32) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h, 16) % mod


# -------------------------
# $$ [UTIL-IMG] PIL->Tensor，避免 TypedStorage 警告
# -------------------------
def pil_to_tensor_norm01(pil: Image.Image) -> torch.Tensor:
    arr = np.asarray(pil, dtype=np.uint8)  # [H,W,3]
    x = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return x


def norm01_to_norm11(x01: torch.Tensor) -> torch.Tensor:
    return x01 * 2.0 - 1.0


def transforms_to_pil(x01: torch.Tensor) -> Image.Image:
    x = (x01.clamp(0, 1) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(x)


def center_crop(pil: Image.Image, size: int = 512) -> Image.Image:
    w, h = pil.size
    if w < size or h < size:
        pil = pil.resize((max(size, w), max(size, h)), resample=Image.BICUBIC)
        w, h = pil.size
    left = (w - size) // 2
    top = (h - size) // 2
    return pil.crop((left, top, left + size, top + size))


# -------------------------
# $$ [UTIL-DEGRADE] realistic 退化（对齐你训练脚本逻辑：blur + down + noise + jpeg + up）
# -------------------------
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


def _jpeg_compress_tensor(x11: torch.Tensor, quality: int) -> torch.Tensor:
    x = x11.clamp(-1, 1)
    x01 = (x + 1.0) / 2.0
    pil = transforms_to_pil(x01.cpu())
    buffer = io.BytesIO()
    pil.save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    pil2 = Image.open(buffer).convert("RGB")
    x01b = pil_to_tensor_norm01(pil2)
    return norm01_to_norm11(x01b)


def degrade_hr_to_lr_tensor(
    hr11_cpu: torch.Tensor,
    mode: str,
    rng: random.Random,
    torch_gen: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    hr11_cpu: [3,512,512] in [-1,1] on CPU
    return  : [3,512,512] in [-1,1] on CPU
    """
    if mode == "bicubic":
        hr = hr11_cpu.unsqueeze(0)
        lr_small = F.interpolate(hr, scale_factor=0.25, mode="bicubic", align_corners=False)
        lr = F.interpolate(lr_small, size=(512, 512), mode="bicubic", align_corners=False)
        return lr.squeeze(0)

    blur_k = rng.choice([3, 5, 7])
    blur_sigma = rng.uniform(0.2, 1.2)

    hr = hr11_cpu.unsqueeze(0)
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
    lr_small_cpu = _jpeg_compress_tensor(lr_small.squeeze(0).cpu(), jpeg_q).unsqueeze(0)

    lr = F.interpolate(lr_small_cpu, size=(512, 512), mode="bicubic", align_corners=False)
    return lr.squeeze(0)


# -------------------------
# $$ [UTIL-METRIC] RGB->Y（SR 常用亮度）
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


# -------------------------
# $$ [CKPT] 兼容两种 ckpt：
#   - last_full_state.pth（含 adapter + pixart_inject）
#   - epochXXX_*.pth（含 adapter + pixart_inject）
# -------------------------
def extract_inject_keys(pixart_sd: Dict[str, torch.Tensor]) -> List[str]:
    keys = []
    for k in pixart_sd.keys():
        if (
            k.startswith("injection_scales")
            or k.startswith("adapter_proj")
            or k.startswith("adapter_norm")
            or k.startswith("cross_attn_scale")
            or k.startswith("input_adapter_ln")
        ):
            keys.append(k)
    return keys


def load_inject_state_dict(pixart, inject_sd: Dict[str, torch.Tensor]):
    sd = pixart.state_dict()
    for k, v in inject_sd.items():
        if k in sd:
            sd[k] = v.to(sd[k].dtype)
    pixart.load_state_dict(sd, strict=False)


def cast_inject_modules_fp32(pixart):
    # 只把“注入相关的小模块”转到 FP32（不把整个 PixArt 转 FP32，省显存）
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


def build_text_cond(t5_path: str, device: str, dtype_pixart: torch.dtype):
    y_embed = torch.load(t5_path, map_location="cpu")["prompt_embeds"].unsqueeze(1)
    y_embed = y_embed.to(device).to(dtype_pixart)
    data_info = {
        "img_hw": torch.tensor([[512.0, 512.0]], device=device, dtype=dtype_pixart),
        "aspect_ratio": torch.tensor([1.0], device=device, dtype=dtype_pixart),
    }
    return y_embed, data_info


@torch.no_grad()
def sample_with_control(
    pixart,
    adapter,
    vae,
    lr_img_11: torch.Tensor,   # [1,3,512,512] float32
    y_embed,
    data_info,
    num_steps: int,
    sde_strength: float,
    fixed_noise_seed: int,
    mult: float,
    injection_mode: str,
    use_amp: bool,
    dtype_pixart: torch.dtype,
    device: str,
):
    # encode to latent
    lr_latent = vae.encode(lr_img_11).latent_dist.sample() * vae.config.scaling_factor  # fp32

    # scheduler
    scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000, solver_order=2)
    scheduler.set_timesteps(num_steps, device=device)

    start_t_val = int(1000 * sde_strength)

    # 用 slice 找起点（等价于 “过滤 <= start_t”，但更不容易踩 scheduler 边界坑）
    timesteps = scheduler.timesteps
    idxs = (timesteps <= start_t_val).nonzero(as_tuple=True)[0]
    start_idx = int(idxs[0].item()) if len(idxs) > 0 else 0
    run_ts = timesteps[start_idx:]

    # add noise at exact timestep = start_t_val（与你单样本/训练脚本一致）
    g = torch.Generator(device=device).manual_seed(int(fixed_noise_seed))
    latents = lr_latent.to(device)
    noise = torch.randn(latents.shape, generator=g, device=device, dtype=latents.dtype)
    t_start = torch.tensor([start_t_val], device=device).long()
    latents = scheduler.add_noise(latents, noise, t_start)

    # mult=0：真正“拔掉 adapter”，不传 adapter_cond
    adapter_cond = None
    if mult != 0.0:
        with torch.cuda.amp.autocast(enabled=False):
            cond = adapter(lr_latent.float().to(device))
            # 统一缩放 cond（同时影响 input injection + cross-attn injection）
            if isinstance(cond, list):
                cond = [c * mult for c in cond]
            else:
                cond = cond * mult
            adapter_cond = cond

    for t in run_ts:
        t_tensor = t.unsqueeze(0).to(device)
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=dtype_pixart):
            out = pixart(
                latents.to(dtype_pixart),
                t_tensor,
                y_embed,
                data_info=data_info,
                adapter_cond=adapter_cond,
                injection_mode=injection_mode,
            )
            if out.shape[1] == 8:
                out, _ = out.chunk(2, dim=1)

        latents = scheduler.step(out.float(), t, latents.float()).prev_sample

    pred_img = vae.decode(latents / vae.config.scaling_factor).sample
    pred_img_01 = torch.clamp((pred_img + 1.0) / 2.0, 0.0, 1.0)
    return pred_img_01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="path to last_full_state.pth or epochXXX_*.pth")
    ap.add_argument("--val_hr_dir", type=str, default=os.path.join(PROJECT_ROOT, "dataset", "DIV2K_valid_HR"))
    ap.add_argument("--pixart_path", type=str, default=os.path.join(PROJECT_ROOT, "output", "pretrained_models", "PixArt-XL-2-512x512.pth"))
    ap.add_argument("--vae_path", type=str, default=os.path.join(PROJECT_ROOT, "output", "pretrained_models", "sd-vae-ft-ema"))
    ap.add_argument("--t5_path", type=str, default=os.path.join(PROJECT_ROOT, "output", "quality_embed.pth"))

    ap.add_argument("--val_degrade_mode", type=str, default="realistic", choices=["realistic", "bicubic"])
    ap.add_argument("--num_val_images", type=int, default=20)
    ap.add_argument("--val_repeat", type=int, default=1)

    ap.add_argument("--control_mults", type=str, default="0,1,2", help="e.g. 0,1,2 or 0,0.5,1,2")
    ap.add_argument("--injection_mode", type=str, default="hybrid", choices=["hybrid", "input", "cross_attn"])

    ap.add_argument("--num_steps", type=int, default=20)
    ap.add_argument("--sde_strength", type=float, default=0.45)
    ap.add_argument("--fixed_noise_seed", type=int, default=42)

    ap.add_argument("--metric_y", action="store_true", help="PSNR/SSIM on Y channel")
    ap.add_argument("--shave", type=int, default=4)
    ap.add_argument("--out_dir", type=str, default=os.path.join(PROJECT_ROOT, "experiments_results", "train_full_mse", "diag"))
    ap.add_argument("--max_vis", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_pixart = torch.float16
    use_amp = (device == "cuda")

    # 1) load pixart base
    pixart = PixArtMS_XL_2(input_size=64).to(device).to(dtype_pixart).eval()
    base = torch.load(args.pixart_path, map_location="cpu")
    if "state_dict" in base:
        base = base["state_dict"]
    if "pos_embed" in base:
        del base["pos_embed"]
    pixart.load_state_dict(base, strict=False)

    # 2) load adapter
    adapter = MultiLevelAdapter(in_channels=4, hidden_size=1152).to(device).eval()

    # 3) load ckpt (adapter + pixart_inject)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if "adapter" not in ckpt or "pixart_inject" not in ckpt:
        raise KeyError("ckpt must contain keys: adapter, pixart_inject")
    adapter.load_state_dict(ckpt["adapter"], strict=True)
    load_inject_state_dict(pixart, ckpt["pixart_inject"])

    # 让注入相关模块保持 fp32（对齐你训练时的做法，不需要改训练脚本）
    cast_inject_modules_fp32(pixart)

    # 4) vae
    vae = AutoencoderKL.from_pretrained(args.vae_path, local_files_only=True).to(device).float().eval()
    vae.enable_slicing()

    # 5) text cond
    y_embed, data_info = build_text_cond(args.t5_path, device, dtype_pixart)

    # 6) lpips
    lpips_fn = None
    if USE_METRICS:
        lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
        for p in lpips_fn.parameters():
            p.requires_grad = False

    # 7) collect val images
    exts = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"]
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(args.val_hr_dir, e))
    paths = sorted(list(set(paths)))
    if len(paths) == 0:
        raise FileNotFoundError(f"No images in {args.val_hr_dir}")
    paths = paths[: min(len(paths), args.num_val_images)]

    mults = [float(x) for x in args.control_mults.split(",")]

    # repeat results container
    all_repeat_summary = []

    for rep in range(args.val_repeat):
        rep_records = []

        pbar = tqdm(paths, desc=f"[diag] repeat {rep+1}/{args.val_repeat}", dynamic_ncols=True)
        for p in pbar:
            pil = Image.open(p).convert("RGB")
            pil = center_crop(pil, 512)
            hr01 = pil_to_tensor_norm01(pil)              # [3,512,512] 0..1
            hr11 = norm01_to_norm11(hr01).unsqueeze(0)    # [1,3,512,512] -1..1

            # 固定每张图退化分布
            seed_i = (stable_int_hash(p) + 12345) & 0xFFFFFFFF
            rng = random.Random(seed_i)
            torch_gen = torch.Generator(device="cpu").manual_seed(int(seed_i))

            lr11 = degrade_hr_to_lr_tensor(
                hr11.squeeze(0).cpu(),
                args.val_degrade_mode,
                rng,
                torch_gen=torch_gen,
            ).unsqueeze(0).to(device).float()

            # GT 用 VAE(hr) 得到（与你训练脚本一致：在 latent 空间对齐）
            with torch.no_grad():
                hr_latent = vae.encode(hr11.to(device).float()).latent_dist.sample() * vae.config.scaling_factor
                gt_img = vae.decode(hr_latent / vae.config.scaling_factor).sample
                gt01 = torch.clamp((gt_img + 1.0) / 2.0, 0.0, 1.0)

            # 每个 mult 跑一次
            mult_outs = {}
            for mult in mults:
                pred01 = sample_with_control(
                    pixart=pixart,
                    adapter=adapter,
                    vae=vae,
                    lr_img_11=lr11,
                    y_embed=y_embed,
                    data_info=data_info,
                    num_steps=args.num_steps,
                    sde_strength=args.sde_strength,
                    fixed_noise_seed=args.fixed_noise_seed,
                    mult=mult,
                    injection_mode=args.injection_mode,
                    use_amp=use_amp,
                    dtype_pixart=dtype_pixart,
                    device=device,
                )
                mult_outs[mult] = pred01

            # metrics + control distance
            base0 = mult_outs.get(0.0, None)
            row = {"path": p, "seed": int(seed_i), "mode": args.val_degrade_mode}

            for mult, pred01 in mult_outs.items():
                m = {"mult": float(mult)}
                if USE_METRICS:
                    if args.metric_y:
                        py = shave_border(rgb01_to_y01(pred01), args.shave)
                        gy = shave_border(rgb01_to_y01(gt01), args.shave)
                        m["psnr"] = float(psnr(py, gy, data_range=1.0).item())
                        m["ssim"] = float(ssim(py, gy, data_range=1.0).item())
                    else:
                        m["psnr"] = float(psnr(pred01, gt01, data_range=1.0).item())
                        m["ssim"] = float(ssim(pred01, gt01, data_range=1.0).item())

                    if lpips_fn is not None:
                        pred_norm = pred01 * 2.0 - 1.0
                        gt_norm = gt01 * 2.0 - 1.0
                        m["lpips"] = float(lpips_fn(pred_norm, gt_norm).item())

                if base0 is not None and mult != 0.0:
                    m["mad_vs_0"] = float((pred01 - base0).abs().mean().item())

                row[str(mult)] = m

            rep_records.append(row)

            # 可视化（只存少量）
            if args.max_vis > 0:
                # 存 mult=1 的图（如果没有 1 就存第一个非0）
                vis_mult = 1.0 if 1.0 in mult_outs else (mults[1] if len(mults) > 1 else mults[0])
                pred_vis = mult_outs[vis_mult][0].permute(1, 2, 0).detach().cpu().numpy()
                lr01 = torch.clamp((lr11 + 1.0) / 2.0, 0.0, 1.0)[0].permute(1, 2, 0).detach().cpu().numpy()
                gt_vis = gt01[0].permute(1, 2, 0).detach().cpu().numpy()

                if rep == 0 and len(rep_records) <= args.max_vis:
                    import matplotlib.pyplot as plt
                    save_path = os.path.join(args.out_dir, f"vis_rep{rep+1}_idx{len(rep_records):03d}_mult{vis_mult}.png")
                    plt.figure(figsize=(12, 4))
                    plt.subplot(1, 3, 1); plt.imshow(lr01); plt.title("LR"); plt.axis("off")
                    plt.subplot(1, 3, 2); plt.imshow(gt_vis); plt.title("GT"); plt.axis("off")
                    plt.subplot(1, 3, 3); plt.imshow(pred_vis); plt.title(f"Pred mult={vis_mult}"); plt.axis("off")
                    plt.tight_layout()
                    plt.savefig(save_path, bbox_inches="tight")
                    plt.close()

        # 汇总 repeat
        def avg_of(key: str, mult: float):
            vals = []
            for r in rep_records:
                if str(mult) in r and key in r[str(mult)]:
                    vals.append(r[str(mult)][key])
            if len(vals) == 0:
                return float("nan")
            return float(sum(vals) / len(vals))

        rep_sum = {"repeat": rep + 1, "num": len(rep_records), "mults": mults}
        for mult in mults:
            rep_sum[f"mult={mult}_psnr"] = avg_of("psnr", mult)
            rep_sum[f"mult={mult}_ssim"] = avg_of("ssim", mult)
            rep_sum[f"mult={mult}_lpips"] = avg_of("lpips", mult)
            rep_sum[f"mult={mult}_mad_vs_0"] = avg_of("mad_vs_0", mult)

        all_repeat_summary.append(rep_sum)

        # 写出本次 repeat 的详细 JSONL
        detail_path = os.path.join(args.out_dir, f"detail_repeat{rep+1}.jsonl")
        with open(detail_path, "w", encoding="utf-8") as f:
            for r in rep_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 写出 summary
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "ckpt": args.ckpt,
                "val_hr_dir": args.val_hr_dir,
                "val_degrade_mode": args.val_degrade_mode,
                "num_val_images": args.num_val_images,
                "val_repeat": args.val_repeat,
                "control_mults": mults,
                "injection_mode": args.injection_mode,
                "num_steps": args.num_steps,
                "sde_strength": args.sde_strength,
                "fixed_noise_seed": args.fixed_noise_seed,
                "metric_y": bool(args.metric_y),
                "shave": int(args.shave),
                "repeat_summary": all_repeat_summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"✅ Wrote: {summary_path}")
    print(f"✅ Details: {os.path.join(args.out_dir, 'detail_repeat*.jsonl')}")


if __name__ == "__main__":
    main()
