# experiments/eval_valpack_ditsr_img2img_gt512.py
# Img2img evaluation on a "valpack" directory.
# Goal: same DDIM validation loop as experiments/train_4090_auto_v4.py, but initialize latents from LQ image (img2img)
# and compare against GT512 metrics using the existing experiments/eval_valpack_metrics_gt512.py.

import os
import sys
import math
import json
import argparse
import importlib
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

from diffusers import AutoencoderKL, DDIMScheduler

# --- Project root import (match your training scripts style) ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from diffusion.model.nets.PixArtMS import PixArtMS_XL_2
from diffusion.model.nets.adapter import build_adapter


def _parse_dtype(s: str) -> Optional[torch.dtype]:
    s = (s or "off").lower()
    if s in ("off", "none", "no"):
        return None
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"Unknown autocast_dtype: {s} (use off/fp16/bf16)")


def _load_png_as_chw01(path: str) -> torch.Tensor:
    # NOTE: np.asarray avoids TypedStorage warnings and is faster than torchvis transforms
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img)  # HWC uint8
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0  # CHW, [0,1]
    return t


def _save_chw01_as_png(x: torch.Tensor, path: str):
    x = x.detach().clamp(0, 1).cpu()
    arr = (x.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _pick_pack_dirs(pack_dir: str) -> Tuple[str, str, str]:
    """
    We follow the "valpack" convention you used:
      - gt512/ : ground-truth 512x512
      - lq512/ : LR upscaled to 512x512 (model input for your SR setting)
      - lq128/ : LR 128x128 (sometimes used in other baselines)
    """
    pack = Path(pack_dir)
    if not pack.exists():
        raise FileNotFoundError(f"pack_dir not found: {pack_dir}")

    # Preferred explicit names
    gt = pack / "gt512"
    lq512 = pack / "lq512"
    lq128 = pack / "lq128"
    if gt.is_dir() and lq512.is_dir() and lq128.is_dir():
        return str(gt), str(lq512), str(lq128)

    # Fallback: heuristic by size + name hints
    subdirs = [d for d in pack.iterdir() if d.is_dir()]
    cand = []
    for d in subdirs:
        pngs = sorted(d.glob("*.png"))
        if not pngs:
            continue
        p0 = str(pngs[0])
        im = Image.open(p0)
        w, h = im.size
        name = d.name.lower()
        hr_hint = 0
        lr_hint = 0
        if "gt" in name or "hr" in name:
            hr_hint += 50
        if "lq" in name or "lr" in name:
            lr_hint += 50
        cand.append((d, len(pngs), w, h, w * h, hr_hint, lr_hint))

    if not cand:
        raise FileNotFoundError(f"No image subdirs found under {pack_dir}")

    # print candidates
    print("[PACK] Candidate image dirs (dir, n, HxW):")
    for d, n, w, h, area, hr_hint, lr_hint in sorted(cand, key=lambda x: -x[4]):
        print(f"  - {d} | n={n} | {w}x{h} | area={area}")

    # HR: prefer 512x512 with hr_hint, tie-break by name
    hr = sorted(
        [c for c in cand if c[2] == 512 and c[3] == 512],
        key=lambda x: (-(x[5]), x[0].name),
    )
    if not hr:
        hr = sorted(cand, key=lambda x: (-(x[4]), -(x[5])))
    hr_dir = str(hr[0][0])

    # LR model input: prefer 512x512 with lr_hint and NOT the same as hr
    lr = sorted(
        [c for c in cand if c[2] == 512 and c[3] == 512 and str(c[0]) != hr_dir],
        key=lambda x: (-(x[6]), x[0].name),
    )
    if not lr:
        # fallback: largest non-hr
        lr = sorted([c for c in cand if str(c[0]) != hr_dir], key=lambda x: -x[4])
    lr_dir = str(lr[0][0])

    # lq128: prefer 128x128
    lq128_c = sorted([c for c in cand if c[2] == 128 and c[3] == 128], key=lambda x: -x[6])
    lq128_dir = str(lq128_c[0][0]) if lq128_c else lr_dir

    print(f"[PACK] Selected HR dir: {hr_dir}")
    print(f"[PACK] Selected LR dir: {lr_dir}")
    return hr_dir, lr_dir, lq128_dir


def _randn_like_with_generator(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    # Your environment's torch may not support generator= in randn_like, so do it explicitly.
    return torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=gen)


def _load_ckpt(path: str) -> Dict:
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise TypeError(f"Checkpoint must be a dict, got {type(ckpt)}")
    if "pixart_trainable" not in ckpt or "adapter" not in ckpt:
        raise KeyError(f"Checkpoint missing required keys. keys={list(ckpt.keys())[:20]}")
    return ckpt


def build_models_from_train_module(train_module, ckpt: Dict, device: str, vae_device: str, autocast_dtype: Optional[torch.dtype]):
    """
    Mirror experiments/train_4090_auto_v4.py:
      - PixArtMS_XL_2(input_size=64)
      - load base weights from train_module.PIXART_PATH (strict=False, del pos_embed if present)
      - apply_lora() from train_module
      - load pixart_trainable from checkpoint
      - build_adapter("fpn_se", in_ch=4, out_ch=1152) (same as train_4090_auto_v4.py)
      - load adapter weights from checkpoint
      - load fixed text embedding from train_module.T5_EMBED_PATH and unsqueeze(1)
    """
    # ---- PixArt base ----
    pixart = PixArtMS_XL_2(input_size=64).to(device)
    base_sd = torch.load(train_module.PIXART_PATH, map_location="cpu")
    if isinstance(base_sd, dict) and "state_dict" in base_sd:
        base_sd = base_sd["state_dict"]
    if "pos_embed" in base_sd:
        del base_sd["pos_embed"]
    missing, unexpected = pixart.load_state_dict(base_sd, strict=False)
    pred_sigma = getattr(pixart, "pred_sigma", None)
    print(f"[LOAD] base PixArt: missing={len(missing)}, unexpected={len(unexpected)} pred_sigma={pred_sigma}")

    # ---- LoRA injection (must happen BEFORE loading pixart_trainable) ----
    if not hasattr(train_module, "apply_lora"):
        raise AttributeError("train_module has no apply_lora(model, rank, alpha). This script is intended for train_4090_auto_v4.py style.")
    train_module.apply_lora(pixart, rank=16, alpha=16.0)
    print("[LORA] applied via train_module.apply_lora(rank=16, alpha=16.0)")

    # ---- Load trained params ----
    miss2, unexp2 = pixart.load_state_dict(ckpt["pixart_trainable"], strict=False)
    # Strict=False is fine, but missing should be ~0 if arch+LoRA match.
    print(f"[LOAD] pixart_trainable: missing={len(miss2)}, unexpected={len(unexp2)}")

    # ---- Adapter ----
    adapter = build_adapter("fpn_se", 4, 1152).to(device)
    miss_a, unexp_a = adapter.load_state_dict(ckpt["adapter"], strict=False)
    print(f"[LOAD] adapter: missing={len(miss_a)}, unexpected={len(unexp_a)}")

    # ---- VAE ----
    vae = AutoencoderKL.from_pretrained(train_module.VAE_PATH, local_files_only=True).to(vae_device)
    vae.enable_slicing()

    # ---- Text embedding ----
    y = torch.load(train_module.T5_EMBED_PATH, map_location="cpu")  # shape [1, 4096] in your setup
    if y.ndim == 2:
        y = y.unsqueeze(1)  # -> [1, 1, 4096] (matches your train script)
    y = y.to(device)

    pixart.eval()
    adapter.eval()
    vae.eval()

    # Keep adapter+injection LN parts numerically safe: your PixArtMS internally casts to float() before LN.
    return pixart, adapter, vae, y


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_module", type=str, required=True, help="e.g. train_4090_auto_v4 (under experiments/)")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--pack_dir", type=str, required=True)
    ap.add_argument("--mode", type=str, default="text+adapter", choices=["text+adapter", "adapter_only"])
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--strength", type=float, default=0.6)
    ap.add_argument("--steps_mode", type=str, default="standard", choices=["standard", "full"],
                    help="standard: diffusers img2img (fewer denoise steps when strength<1). full: always run --steps denoise steps.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_n", type=int, default=50)
    ap.add_argument("--autocast_dtype", type=str, default="bf16", help="off/fp16/bf16")
    ap.add_argument("--vae_device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--fail_fast", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae_device = args.vae_device
    ac_dtype = _parse_dtype(args.autocast_dtype)

    print(f"[INFO] Using device={device}, vae_device={vae_device}")
    # Import train module exactly like your scripts (experiments.*)
    tm = importlib.import_module(f"experiments.{args.train_module}")
    print(f"[IMPORT] train_module={args.train_module} -> {tm.__file__}")

    ckpt = _load_ckpt(args.ckpt)

    hr_dir, lr_dir, _lq128 = _pick_pack_dirs(args.pack_dir)
    hr_paths = sorted(Path(hr_dir).glob("*.png"))[: args.max_n]
    lr_paths = sorted(Path(lr_dir).glob("*.png"))[: args.max_n]
    if len(hr_paths) != len(lr_paths):
        raise RuntimeError(f"HR/LR counts differ: {len(hr_paths)} vs {len(lr_paths)}")

    out_root = Path(tm.EXP_DIR if hasattr(tm, "EXP_DIR") else "experiments_results") / args.train_module
    pack_name = Path(args.pack_dir).name
    tag = f"img2img_{Path(args.ckpt).stem}_{args.mode}_cfg{args.cfg}_steps{args.steps}_str{args.strength}_{args.steps_mode}_seed{args.seed}"
    pred_dir = out_root / f"eval_valpack_img2img_{pack_name}" / tag / "pred"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Build models
    pixart, adapter, vae, y = build_models_from_train_module(tm, ckpt, device=device, vae_device=vae_device, autocast_dtype=ac_dtype)

    # DDIM scheduler: match train_4090_auto_v4.py validate config
    # (train script uses DDIMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
    #  steps_offset=1, clip_sample=False, set_alpha_to_one=False))
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        steps_offset=1,
        clip_sample=False,
        set_alpha_to_one=False,
    )
    scheduler.set_timesteps(args.steps, device=device)
    timesteps_all = scheduler.timesteps  # descending

    # RNG
    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed))

    failures = []
    saved = 0

    pbar = tqdm(list(zip(hr_paths, lr_paths)), desc="DiTSR img2img valpack inference")
    for hr_p, lr_p in pbar:
        stem = hr_p.stem
        try:
            # Load LR image (512x512) -> latent z_lr
            lr = _load_png_as_chw01(str(lr_p)).unsqueeze(0).to(device)  # [1,3,512,512]
            with torch.autocast(device_type="cuda", dtype=ac_dtype, enabled=(ac_dtype is not None and device == "cuda")):
                z_lr = vae.encode(lr.to(vae_device)).latent_dist.sample().to(device)
                # Scale factor is fixed for SD VAE; follow train script constant
                z_lr = z_lr * 0.18215

            # Adapter cond from LR latent (match train validate: adapter(z_lr.float()))
            with torch.autocast(device_type="cuda", dtype=ac_dtype, enabled=(ac_dtype is not None and device == "cuda")):
                adapter_cond = adapter(z_lr.float())  # list of 4 features

            # --- Build uncond/cond text (match train validate CFG) ---
            y_cond = y
            y_uncond = torch.zeros_like(y_cond)
            y_lens = None
            mask = None
            data_info = None

            # --- Init img2img latent ---
            noise = _randn_like_with_generator(z_lr, gen)
            if args.steps_mode == "standard":
                # Diffusers img2img convention: run only part of timesteps depending on strength.
                init_timestep = min(int(args.steps * args.strength), args.steps)
                t_start = max(args.steps - init_timestep, 0)
                timesteps = timesteps_all[t_start:]
                t_init = timesteps[0]
                # add noise corresponding to t_init
                zt = scheduler.add_noise(z_lr, noise, t_init)
            else:
                # "full" mode: always run args.steps denoise steps, but set a noise level based on strength.
                # We map strength in [0,1] -> a training-time timestep in [0,999].
                t_init_val = int(round(float(args.strength) * 999.0))
                t_init = torch.tensor([t_init_val], device=device, dtype=timesteps_all.dtype)
                zt = scheduler.add_noise(z_lr, noise, t_init)
                timesteps = timesteps_all

            # --- Denoising loop: replicate train validate loop, but starting from zt ---
            for t in timesteps:
                # IMPORTANT: PixArt timestep embed expects shape [B], not scalar
                t_b = torch.full((zt.shape[0],), int(t.item()), device=device, dtype=torch.long)

                with torch.autocast(device_type="cuda", dtype=ac_dtype, enabled=(ac_dtype is not None and device == "cuda")):
                    if args.mode == "adapter_only":
                        out_u = pixart(x=zt, timestep=t_b, y=y_uncond, y_lens=y_lens, data_info=data_info, adapter_cond=adapter_cond)
                        out_c = pixart(x=zt, timestep=t_b, y=y_uncond, y_lens=y_lens, data_info=data_info, adapter_cond=adapter_cond)
                    else:
                        out_u = pixart(x=zt, timestep=t_b, y=y_uncond, y_lens=y_lens, data_info=data_info, adapter_cond=None)
                        out_c = pixart(x=zt, timestep=t_b, y=y_cond, y_lens=y_lens, data_info=data_info, adapter_cond=adapter_cond)

                    eps_u = out_u[:, :4]
                    eps_c = out_c[:, :4]
                    eps = eps_u + float(args.cfg) * (eps_c - eps_u)
                    zt = scheduler.step(eps, t, zt).prev_sample

            # Decode
            with torch.autocast(device_type="cuda", dtype=ac_dtype, enabled=(ac_dtype is not None and device == "cuda")):
                x0 = zt / 0.18215
                x = vae.decode(x0.to(vae_device)).sample.to(device)  # [-1,1]
                x = (x + 1) / 2

            _save_chw01_as_png(x[0], str(pred_dir / f"{stem}.png"))
            saved += 1
            pbar.set_postfix(saved=saved)
        except Exception as e:
            failures.append({"stem": stem, "error": repr(e)})
            if args.fail_fast:
                raise
            continue

    summary = {
        "train_module": args.train_module,
        "ckpt": args.ckpt,
        "pack_dir": args.pack_dir,
        "gt_dir": hr_dir,
        "lr_dir_model_input": lr_dir,
        "mode": args.mode,
        "cfg": args.cfg,
        "steps": args.steps,
        "strength": args.strength,
        "steps_mode": args.steps_mode,
        "seed": args.seed,
        "autocast_dtype": args.autocast_dtype,
        "device": device,
        "vae_device": vae_device,
        "n_saved": saved,
        "failures": failures[:50],
        "pred_dir": str(pred_dir),
        "note": "DDIM loop matches train_4090_auto_v4.py validate(); img2img init uses scheduler.add_noise(z_lr, noise, t_init).",
    }
    out_json = pred_dir.parent / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"✅ Saved: {out_json}")


if __name__ == "__main__":
    main()
