#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Eval DiTSR on valpack (gt512/lq512/lq128).

Key alignment with experiments/train_4090_auto_v4.py and experiments/eval_cfg_sweep.py:
- pixart weights stay in fp32 (DO NOT cast to bf16), because PixArtMS contains
  autocast(enabled=False) + .float() regions for LayerNorm/Linear.
- LoRA must be applied BEFORE loading ckpt["pixart_trainable"].
- y_embed must be prompt_embeds.unsqueeze(1) (same as eval_cfg_sweep.py).
- Inference uses 2-pass CFG (uncond/cond) exactly like your validate/sweep.

Memory strategy:
- allow autocast dtype (bf16/fp16) for compute
- VAE on CPU supported
- LPIPS can be fully disabled; recommend save_pred_only=1 then compute metrics via eval_valpack_metrics.py
"""

import os
import sys
import json
import argparse
import importlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from diffusers import AutoencoderKL, DDIMScheduler


IMG_EXT = ".png"


def _proj_root() -> str:
    return str(Path(__file__).resolve().parents[1])


def _dtype_from_str(s: str):
    s = s.lower()
    if s == "bf16":
        return torch.bfloat16
    if s == "fp16":
        return torch.float16
    if s == "fp32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {s}")


def _load_rgb01(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.array(img, copy=True)
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0


def _to_m11(x01: torch.Tensor) -> torch.Tensor:
    return x01 * 2.0 - 1.0


def _resize_chw(x: torch.Tensor, size_hw=(512, 512)) -> torch.Tensor:
    return F.interpolate(x.unsqueeze(0), size=size_hw, mode="bicubic", align_corners=False).squeeze(0)


def _inspect_pack(pack_dir: str):
    pack = Path(pack_dir)
    if not pack.is_dir():
        raise FileNotFoundError(pack_dir)
    cand = []
    for d in sorted([x for x in pack.iterdir() if x.is_dir()]):
        pngs = sorted([p for p in d.iterdir() if p.suffix.lower() == IMG_EXT])
        if not pngs:
            continue
        w, h = Image.open(pngs[0]).size
        name = d.name.lower()
        hr_hint = 30 if any(k in name for k in ["gt", "hr"]) else -10
        lr_hint = 30 if any(k in name for k in ["lq", "lr"]) else -10
        cand.append({"dir": str(d), "name": d.name, "n": len(pngs), "h": h, "w": w, "area": h * w, "hr_hint": hr_hint, "lr_hint": lr_hint})
    return cand


def _pick_hr_lr(pack_dir: str):
    cand = _inspect_pack(pack_dir)
    if not cand:
        raise FileNotFoundError(f"No image subdirs found under {pack_dir}")

    print("[PACK] Candidate image dirs (showing dir, n, HxW):")
    for c in cand:
        print(f"  - {c['dir']} | n={c['n']} | {c['h']}x{c['w']} | area={c['area']} | hr_hint={c['hr_hint']} lr_hint={c['lr_hint']}")

    # HR: prefer gt/hr by name, then largest area
    hr = sorted(cand, key=lambda x: (x["hr_hint"], x["area"], x["n"]), reverse=True)[0]
    hr_dir = hr["dir"]
    hr_area = hr["area"]

    # LR: for v4, prefer same-res degraded input if exists (lq512)
    lr_cand = [c for c in cand if c["dir"] != hr_dir]
    lr = sorted(lr_cand, key=lambda x: (x["lr_hint"], -abs(x["area"] - hr_area), x["n"]), reverse=True)[0]
    lr_dir = lr["dir"]

    print(f"[PACK] Selected HR dir: {hr_dir}")
    print(f"[PACK] Selected LR dir: {lr_dir}")
    return hr_dir, lr_dir


def _list_stems(dir_path: str):
    d = Path(dir_path)
    files = sorted([p for p in d.iterdir() if p.suffix.lower() == IMG_EXT])
    return {p.stem: str(p) for p in files}


@torch.no_grad()
def _load_ckpt(ckpt_path: str, pixart, adapter):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "adapter" not in ckpt or "pixart_trainable" not in ckpt:
        raise KeyError(f"ckpt must contain keys 'adapter' and 'pixart_trainable'. got={list(ckpt.keys())}")
    adapter.load_state_dict(ckpt["adapter"], strict=True)

    curr = pixart.state_dict()
    for k, v in ckpt["pixart_trainable"].items():
        if k in curr:
            curr[k] = v.to(curr[k].dtype)
    pixart.load_state_dict(curr, strict=False)

    epoch = int(ckpt.get("epoch", -1)) + 1
    step = int(ckpt.get("step", -1))
    return epoch, step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_module", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--pack_dir", type=str, required=True)

    ap.add_argument("--mode", type=str, default="adapter_only", choices=["text+adapter", "adapter_only"])
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--max_n", type=int, default=50)

    # IMPORTANT: this is COMPUTE dtype (autocast), NOT weight dtype
    ap.add_argument("--autocast_dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])

    ap.add_argument("--vae_device", type=str, default="cpu", choices=["cuda", "cpu"])
    ap.add_argument("--save_pred_only", type=int, default=1, choices=[0, 1])
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()

    # path setup to import train scripts in experiments/
    root = _proj_root()
    sys.path.insert(0, root)
    sys.path.insert(0, str(Path(root) / "experiments"))

    M = importlib.import_module(args.train_module)

    device = getattr(M, "DEVICE", "cuda")
    if device != "cuda":
        raise RuntimeError(f"Expected M.DEVICE='cuda', got {device}")

    compute_dtype = _dtype_from_str(args.autocast_dtype)
    use_autocast = (compute_dtype != torch.float32)

    hr_dir, lr_dir = _pick_hr_lr(args.pack_dir)
    hr_map = _list_stems(hr_dir)
    lr_map = _list_stems(lr_dir)
    names = sorted(list(set(hr_map.keys()).intersection(lr_map.keys())))[: args.max_n]
    if not names:
        raise RuntimeError("No matched stems between HR and LR dirs.")

    # Output dirs
    if args.out_dir is None:
        out_dir = Path(getattr(M, "OUT_DIR", "experiments_results")) / f"eval_valpack_{Path(args.pack_dir).name}" / f"{Path(args.ckpt).stem}_{args.mode}_cfg{args.cfg:g}_steps{args.steps}"
    else:
        out_dir = Path(args.out_dir)
    pred_dir = out_dir / "pred"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Build PixArt + LoRA + Adapter exactly like eval_cfg_sweep.py (weights stay fp32)
    pixart = M.PixArtMS_XL_2(input_size=64).to("cuda")
    base = torch.load(M.PIXART_PATH, map_location="cpu")
    if isinstance(base, dict) and "state_dict" in base:
        base = base["state_dict"]
    if isinstance(base, dict) and "pos_embed" in base:
        del base["pos_embed"]
    pixart.load_state_dict(base, strict=False)

    M.apply_lora(pixart, getattr(M, "LORA_RANK", 16), getattr(M, "LORA_ALPHA", 16))

    adapter = M.build_adapter(getattr(M, "ADAPTER_TYPE", "fpn_se"), 4, 1152).to("cuda").float().eval()

    epoch, step = _load_ckpt(args.ckpt, pixart, adapter)
    pixart.eval()
    for p in pixart.parameters():
        p.requires_grad_(False)

    # Text embed (same as eval_cfg_sweep.py)
    y_embed = torch.load(M.T5_EMBED_PATH, map_location="cpu")["prompt_embeds"].unsqueeze(1).to("cuda")

    d_info = {
        "img_hw": torch.tensor([[512.0, 512.0]], device="cuda"),
        "aspect_ratio": torch.tensor([1.0], device="cuda"),
    }

    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        clip_sample=False,
        prediction_type="epsilon",
        set_alpha_to_one=False,
    )
    scheduler.set_timesteps(args.steps, device="cuda")

    vae = AutoencoderKL.from_pretrained(M.VAE_PATH, local_files_only=True)
    if args.vae_device == "cuda":
        vae = vae.to("cuda").float().eval()
        vae.enable_slicing()
    else:
        vae = vae.to("cpu").float().eval()
        vae.enable_slicing()

    val_gen = torch.Generator(device="cuda")
    val_gen.manual_seed(args.seed)

    # follow your train module init style
    use_lq_init = bool(getattr(M, "USE_LQ_INIT", False))
    init_noise_std = float(getattr(M, "INIT_NOISE_STD", 0.0))

    with torch.inference_mode():
        for i, name in enumerate(tqdm(names, desc="DiTSR valpack inference")):
            hr01 = _load_rgb01(hr_map[name])
            lr01 = _load_rgb01(lr_map[name])

            if hr01.shape[-2:] != (512, 512):
                hr01 = _resize_chw(hr01, (512, 512))
            if lr01.shape[-2:] != (512, 512):
                lr01 = _resize_chw(lr01, (512, 512))

            hr = _to_m11(hr01).unsqueeze(0)
            lr = _to_m11(lr01).unsqueeze(0)

            # VAE encode on chosen device (mean, same as eval_cfg_sweep.py)
            if args.vae_device == "cuda":
                z_hr = vae.encode(hr.to("cuda")).latent_dist.mean * vae.config.scaling_factor
                z_lr = vae.encode(lr.to("cuda")).latent_dist.mean * vae.config.scaling_factor
            else:
                z_hr = vae.encode(hr.to("cpu")).latent_dist.mean * vae.config.scaling_factor
                z_lr = vae.encode(lr.to("cpu")).latent_dist.mean * vae.config.scaling_factor
                z_hr = z_hr.to("cuda")
                z_lr = z_lr.to("cuda")

            if use_lq_init:
                latents = z_lr.clone()
                if init_noise_std > 0:
                    latents = latents + init_noise_std * M.randn_like_with_generator(latents, val_gen)
            else:
                latents = M.randn_like_with_generator(z_hr, val_gen)

            cond = adapter(z_lr.float())

            for t in scheduler.timesteps:
                t_b = torch.tensor([t], device="cuda").expand(latents.shape[0])

                with torch.autocast(device_type="cuda", dtype=compute_dtype, enabled=use_autocast):
                    if args.mode == "text+adapter":
                        drop_uncond = torch.ones(latents.shape[0], device="cuda")
                        drop_cond = torch.zeros(latents.shape[0], device="cuda")
                    else:  # adapter_only
                        drop_uncond = torch.zeros(latents.shape[0], device="cuda")
                        drop_cond = torch.zeros(latents.shape[0], device="cuda")

                    out_uncond = pixart(
                        x=latents.to(compute_dtype) if use_autocast else latents,
                        timestep=t_b,
                        y=y_embed,
                        mask=None,
                        data_info=d_info,
                        adapter_cond=None,
                        injection_mode="hybrid",
                        force_drop_ids=drop_uncond,
                    )
                    out_cond = pixart(
                        x=latents.to(compute_dtype) if use_autocast else latents,
                        timestep=t_b,
                        y=y_embed,
                        mask=None,
                        data_info=d_info,
                        adapter_cond=cond,
                        injection_mode="hybrid",
                        force_drop_ids=drop_cond,
                    )

                    if out_uncond.shape[1] == 8:
                        out_uncond, _ = out_uncond.chunk(2, dim=1)
                    if out_cond.shape[1] == 8:
                        out_cond, _ = out_cond.chunk(2, dim=1)

                    out = out_uncond + float(args.cfg) * (out_cond - out_uncond)

                latents = scheduler.step(out.float(), t, latents.float()).prev_sample

            # VAE decode
            if args.vae_device == "cuda":
                pred = vae.decode(latents / vae.config.scaling_factor).sample.clamp(-1, 1)
                pred01 = ((pred[0].detach().cpu() + 1) / 2).clamp(0, 1).numpy().transpose(1, 2, 0)
            else:
                pred = vae.decode((latents / vae.config.scaling_factor).to("cpu")).sample.clamp(-1, 1)
                pred01 = ((pred[0].detach() + 1) / 2).clamp(0, 1).numpy().transpose(1, 2, 0)

            Image.fromarray((pred01 * 255.0 + 0.5).astype(np.uint8)).save(str(pred_dir / f"{name}.png"))

            # free GPU cache per image
            del hr, lr, z_hr, z_lr, latents, cond, out_uncond, out_cond, out, pred
            torch.cuda.empty_cache()

    summary = {
        "train_module": args.train_module,
        "ckpt": args.ckpt,
        "epoch": epoch,
        "step": step,
        "pack_dir": args.pack_dir,
        "hr_dir": hr_dir,
        "lr_dir": lr_dir,
        "mode": args.mode,
        "cfg": args.cfg,
        "steps": args.steps,
        "seed": args.seed,
        "autocast_dtype": args.autocast_dtype,
        "vae_device": args.vae_device,
        "n": len(names),
        "pred_dir": str(pred_dir),
        "save_pred_only": args.save_pred_only,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"✅ Saved: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
