#!/usr/bin/env python3
"""Evaluate DiTSR on full DRealSR/RealSR datasets.

Fixes vs earlier draft:
1) No global forced resize of HR/LR to same size.
   Use center-crop protocol aligned with training-style evaluation:
   - HR center-crop to `crop_size` (default 512)
   - LR center-crop to `crop_size/4` when LR is true x4-resolution input
   - then bicubic-upsample LR crop to HR crop size
2) Report PSNR on Y channel (and also RGB for reference).
3) Optionally apply EMA weights saved in v9 checkpoints (`--use-ema`).
4) Use one persistent torch.Generator across samples (no per-image seed reset).

Outputs:
- JSON summary (averaged metrics only).
- Up to N 4-panel previews per dataset: [LR-upsampled | GT(HR) | Pred | Bicubic].
"""

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

import lpips
from diffusers import AutoencoderKL, DDIMScheduler
from torchmetrics.functional import structural_similarity_index_measure as tm_ssim


def load_train_module(script_path: str):
    spec = importlib.util.spec_from_file_location("ditsr_train_mod", str(Path(script_path).resolve()))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def load_png_chw01(path: str) -> torch.Tensor:
    return TF.to_tensor(Image.open(path).convert("RGB"))


def resize_bchw(x: torch.Tensor, size_hw):
    return F.interpolate(x, size=size_hw, mode="bicubic", align_corners=False)


def center_crop_bchw(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    _, _, H, W = x.shape
    h = min(h, H)
    w = min(w, W)
    top = max(0, (H - h) // 2)
    left = max(0, (W - w) // 2)
    return x[..., top:top + h, left:left + w]


def psnr01(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a - b) ** 2).clamp_min(1e-12)
    return float((10.0 * torch.log10(1.0 / mse)).item())


def rgb01_to_y01(rgb01: torch.Tensor) -> torch.Tensor:
    r, g, b = rgb01[:, 0:1], rgb01[:, 1:2], rgb01[:, 2:3]
    return (16.0 + 65.481 * r + 128.553 * g + 24.966 * b) / 255.0


def to_uint8_hwc(x01_bchw: torch.Tensor):
    return (x01_bchw[0].clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).permute(1, 2, 0).cpu().numpy()


def make_4panel(lr01_up, hr01, pred01, bic01, out_path: Path):
    a, b, c, d = [to_uint8_hwc(t) for t in [lr01_up, hr01, pred01, bic01]]
    h = max(a.shape[0], b.shape[0], c.shape[0], d.shape[0])
    w = max(a.shape[1], b.shape[1], c.shape[1], d.shape[1])

    def _pad(img):
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[: img.shape[0], : img.shape[1]] = img
        return canvas

    panel = np.concatenate([_pad(a), _pad(b), _pad(c), _pad(d)], axis=1)
    Image.fromarray(panel).save(out_path)


def apply_ema_shadow_(model: torch.nn.Module, ema_state: Dict):
    shadow = ema_state.get("shadow_params", None)
    if shadow is None:
        return False
    with torch.no_grad():
        for p, s in zip(model.parameters(), shadow):
            if torch.is_tensor(s):
                p.copy_(s.to(device=p.device, dtype=p.dtype))
    return True


def load_ckpt_into_models(ckpt_path: str, pixart, adapter, use_ema: bool = True):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    adapter.load_state_dict(ckpt["adapter"], strict=False)
    curr = pixart.state_dict()
    for k, v in ckpt["pixart_trainable"].items():
        if k in curr:
            curr[k] = v.to(curr[k].dtype)
    pixart.load_state_dict(curr, strict=False)

    ema_applied = {"pixart": False, "adapter": False}
    if use_ema:
        if "ema_pixart" in ckpt:
            ema_applied["pixart"] = apply_ema_shadow_(pixart, ckpt["ema_pixart"])
        if "ema_adapter" in ckpt:
            ema_applied["adapter"] = apply_ema_shadow_(adapter, ckpt["ema_adapter"])
    return ckpt, ema_applied


@torch.no_grad()
def infer_one(mod, pixart, adapter, vae, y_embed, lr_m11, steps: int, cfg_scale: float, gen: torch.Generator):
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        clip_sample=False,
        prediction_type="v_prediction",
        set_alpha_to_one=False,
    )
    scheduler.set_timesteps(steps, device=lr_m11.device)

    z_lr = vae.encode(lr_m11).latent_dist.mean * vae.config.scaling_factor

    if getattr(mod, "USE_LQ_INIT", True):
        latents, run_timesteps = mod.get_lq_init_latents(
            z_lr.to(mod.COMPUTE_DTYPE), scheduler, steps,
            gen,
            float(getattr(mod, "LQ_INIT_STRENGTH", 0.1)), mod.COMPUTE_DTYPE,
        )
    else:
        latents = torch.randn_like(z_lr)
        run_timesteps = scheduler.timesteps

    cond = adapter(z_lr.float())
    aug_level = torch.zeros((latents.shape[0],), device=lr_m11.device, dtype=mod.COMPUTE_DTYPE)
    d_info = {
        "img_hw": torch.tensor([[float(lr_m11.shape[-2]), float(lr_m11.shape[-1])]], device=lr_m11.device),
        "aspect_ratio": torch.tensor([float(lr_m11.shape[-1]) / float(lr_m11.shape[-2])], device=lr_m11.device),
    }

    for t in run_timesteps:
        t_b = torch.tensor([t], device=lr_m11.device).expand(latents.shape[0])
        with torch.autocast(device_type="cuda", dtype=mod.COMPUTE_DTYPE):
            drop_uncond = torch.ones(latents.shape[0], device=lr_m11.device)
            drop_cond = torch.ones(latents.shape[0], device=lr_m11.device)
            model_in = torch.cat([latents.to(mod.COMPUTE_DTYPE), z_lr.to(mod.COMPUTE_DTYPE)], dim=1)
            out_uncond = pixart(x=model_in, timestep=t_b, y=y_embed, aug_level=aug_level,
                                mask=None, data_info=d_info, adapter_cond=None,
                                injection_mode="hybrid", force_drop_ids=drop_uncond)
            out_cond = pixart(x=model_in, timestep=t_b, y=y_embed, aug_level=aug_level,
                              mask=None, data_info=d_info, adapter_cond=cond,
                              injection_mode="hybrid", force_drop_ids=drop_cond)
            if out_uncond.shape[1] == 8:
                out_uncond, _ = out_uncond.chunk(2, dim=1)
            if out_cond.shape[1] == 8:
                out_cond, _ = out_cond.chunk(2, dim=1)
            out = out_uncond + cfg_scale * (out_cond - out_uncond)
        latents = scheduler.step(out.float(), t, latents.float()).prev_sample

    return vae.decode(latents / vae.config.scaling_factor).sample.clamp(-1, 1)


def collect_pairs_drealsr(hr_dir: Path, lr_dir: Path):
    pairs = []
    for hr in sorted(hr_dir.glob("*_x4.png")):
        stem = hr.stem[:-3] if hr.stem.endswith("_x4") else hr.stem
        lr = lr_dir / f"{stem}_x1.png"
        if lr.exists():
            pairs.append({"name": stem, "hr": str(hr), "lr": str(lr)})
    return pairs


def collect_pairs_realsr(dir_path: Path, prefix: str):
    pairs = []
    for hr in sorted(dir_path.glob(f"{prefix}*_HR.png")):
        stem = hr.stem.replace("_HR", "")
        lr = dir_path / f"{stem}_LR4.png"
        if lr.exists():
            pairs.append({"name": stem, "hr": str(hr), "lr": str(lr)})
    return pairs


def build_default_datasets():
    return [
        {"dataset": "DRealSR_Canon_x4", "pairs": collect_pairs_drealsr(Path("/data/DRealSR/HR"), Path("/data/DRealSR/LR"))},
        {"dataset": "RealSR_Canon_x4", "pairs": collect_pairs_realsr(Path("/data/RealSR/Canon/Test/4"), "Canon_")},
        {"dataset": "RealSR_Nikon_x4", "pairs": collect_pairs_realsr(Path("/data/RealSR/Nikon/Test/4"), "Nikon_")},
    ]


def align_pair_train_style(hr01: torch.Tensor, lr01: torch.Tensor, crop_size: int):
    """Mimic train/val style:
    - HR center-crop to crop_size (or min size).
    - LR center-crop to crop_size/4 when LR is true low-res by x4.
    - Upsample LR crop to HR crop size.
    """
    b, c, h_hr, w_hr = hr01.shape
    _, _, h_lr, w_lr = lr01.shape

    tgt = min(crop_size, h_hr, w_hr) if crop_size > 0 else min(h_hr, w_hr)
    tgt = int(max(32, tgt - (tgt % 8)))
    hr_c = center_crop_bchw(hr01, tgt, tgt)

    scale_h = h_hr / max(1, h_lr)
    scale_w = w_hr / max(1, w_lr)
    is_x4 = abs(scale_h - 4.0) < 0.25 and abs(scale_w - 4.0) < 0.25

    if is_x4:
        lr_t = max(8, tgt // 4)
        lr_c = center_crop_bchw(lr01, lr_t, lr_t)
    else:
        lr_c = center_crop_bchw(lr01, tgt, tgt)

    lr_up = resize_bchw(lr_c, (tgt, tgt))
    return hr_c, lr_up


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-script", default="experiments/train_4090_auto_v9_gan_edge_gtmask.py")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--crop-size", type=int, default=512, help="Center crop size in HR space; 0 uses full shortest side")
    ap.add_argument("--crop-border", type=int, default=4)
    ap.add_argument("--datasets-json", default="", help="JSON list: [{dataset, pairs:[{name,hr,lr}]}]")
    ap.add_argument("--save-max-per-dataset", type=int, default=5)
    ap.add_argument("--use-ema", action="store_true", help="Apply ema_pixart/ema_adapter from checkpoint when present")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--out-json", default="experiments_results/realsr_eval/summary.json")
    ap.add_argument("--save-panels-dir", default="experiments_results/realsr_eval/panels")
    args = ap.parse_args()

    mod = load_train_module(args.train_script)
    device = torch.device(getattr(mod, "DEVICE", "cuda"))

    pixart = mod.PixArtMSV8_XL_2(input_size=64, in_channels=8,
                                 sparse_inject_ratio=mod.SPARSE_INJECT_RATIO,
                                 injection_cutoff_layer=mod.INJECTION_CUTOFF_LAYER,
                                 injection_strategy=mod.INJECTION_STRATEGY).to(device)

    base = torch.load(mod.PIXART_PATH, map_location="cpu")
    if "state_dict" in base:
        base = base["state_dict"]
    if "pos_embed" in base:
        del base["pos_embed"]
    if "x_embedder.proj.weight" in base and base["x_embedder.proj.weight"].shape[1] == 4:
        w4 = base["x_embedder.proj.weight"]
        w8 = torch.zeros((w4.shape[0], 8, w4.shape[2], w4.shape[3]), dtype=w4.dtype)
        w8[:, :4] = w4
        base["x_embedder.proj.weight"] = w8
    pixart.load_pretrained_weights_with_zero_init(base)
    mod.apply_lora(pixart, mod.LORA_RANK, mod.LORA_ALPHA)

    adapter = mod.build_adapter_v7(in_channels=4, hidden_size=1152,
                                   injection_layers_map=getattr(pixart, "injection_layers", None)).to(device).float()
    vae = AutoencoderKL.from_pretrained(mod.VAE_PATH, local_files_only=True).to(device).float().eval()
    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    _, ema_applied = load_ckpt_into_models(args.ckpt, pixart, adapter, use_ema=args.use_ema)
    pixart.eval()
    adapter.eval()

    y_embed = torch.load(mod.T5_EMBED_PATH, map_location="cpu")["prompt_embeds"].unsqueeze(1).to(device)
    datasets = json.loads(args.datasets_json) if args.datasets_json else build_default_datasets()
    cb = int(args.crop_border)
    out_panel_root = Path(args.save_panels_dir)
    out_panel_root.mkdir(parents=True, exist_ok=True)

    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed))

    per_dataset = {}
    total_n = 0

    for d in datasets:
        dname = d["dataset"]
        pairs = d.get("pairs", [])
        ddir = out_panel_root / dname
        ddir.mkdir(parents=True, exist_ok=True)

        vals = {k: [] for k in [
            "psnr_y_pred", "psnr_rgb_pred", "ssim_y_pred", "lpips_pred",
            "psnr_y_bic", "psnr_rgb_bic", "ssim_y_bic", "lpips_bic",
        ]}
        saved_count = 0

        for item in pairs:
            hr_path, lr_path = Path(item["hr"]), Path(item["lr"])
            if (not hr_path.exists()) or (not lr_path.exists()):
                continue

            name = item["name"]
            hr = load_png_chw01(str(hr_path)).unsqueeze(0).to(device)
            lr = load_png_chw01(str(lr_path)).unsqueeze(0).to(device)
            hr, lr_up = align_pair_train_style(hr, lr, int(args.crop_size))

            pred_m11 = infer_one(mod, pixart, adapter, vae, y_embed, (lr_up * 2 - 1).clamp(-1, 1), args.steps, args.cfg, gen)
            pred01 = (pred_m11 + 1) * 0.5
            bic01 = lr_up.clone()

            if cb > 0 and hr.shape[-1] > 2 * cb and hr.shape[-2] > 2 * cb:
                pr_c, bi_c, hr_c = pred01[..., cb:-cb, cb:-cb], bic01[..., cb:-cb, cb:-cb], hr[..., cb:-cb, cb:-cb]
            else:
                pr_c, bi_c, hr_c = pred01, bic01, hr

            py, hy, by = rgb01_to_y01(pr_c), rgb01_to_y01(hr_c), rgb01_to_y01(bi_c)

            vals["psnr_y_pred"].append(psnr01(py, hy))
            vals["psnr_rgb_pred"].append(psnr01(pr_c, hr_c))
            vals["ssim_y_pred"].append(float(tm_ssim(py, hy, data_range=1.0).item()))
            vals["lpips_pred"].append(float(lpips_fn((pred01 * 2 - 1).clamp(-1, 1), (hr * 2 - 1).clamp(-1, 1)).mean().item()))

            vals["psnr_y_bic"].append(psnr01(by, hy))
            vals["psnr_rgb_bic"].append(psnr01(bi_c, hr_c))
            vals["ssim_y_bic"].append(float(tm_ssim(by, hy, data_range=1.0).item()))
            vals["lpips_bic"].append(float(lpips_fn((bic01 * 2 - 1).clamp(-1, 1), (hr * 2 - 1).clamp(-1, 1)).mean().item()))

            if saved_count < int(args.save_max_per_dataset):
                make_4panel(lr_up, hr, pred01, bic01, ddir / f"{name}_4panel.png")
                saved_count += 1

        n = len(vals["psnr_y_pred"])
        total_n += n
        if n > 0:
            per_dataset[dname] = {"n": n, **{k: float(np.mean(v)) for k, v in vals.items()}}
        else:
            per_dataset[dname] = {"n": 0}

    valid = [m for m in per_dataset.values() if m.get("n", 0) > 0]
    overall = {
        "n": total_n,
        "crop_size": int(args.crop_size),
        "crop_border": cb,
        "steps": args.steps,
        "cfg": args.cfg,
        "use_ema": bool(args.use_ema),
        "ema_applied": ema_applied,
        "per_dataset": per_dataset,
    }
    if valid:
        denom = sum(m["n"] for m in valid)
        keys = [
            "psnr_y_pred", "psnr_rgb_pred", "ssim_y_pred", "lpips_pred",
            "psnr_y_bic", "psnr_rgb_bic", "ssim_y_bic", "lpips_bic",
        ]
        overall["overall_weighted"] = {k: float(sum(m[k] * m["n"] for m in valid) / denom) for k in keys}

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(overall, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(overall, indent=2, ensure_ascii=False))
    print(f"✅ Saved JSON: {out_json}")
    print(f"✅ Saved panel previews: {out_panel_root} (max {args.save_max_per_dataset}/dataset)")


if __name__ == "__main__":
    main()
