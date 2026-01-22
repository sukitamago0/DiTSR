# -*- coding: utf-8 -*-
"""
Sweep inference modes and noise strengths for SR evaluation.

This script evaluates different inference start modes using a fixed checkpoint:
  - noise: start from pure noise (current scheme B default)
  - lr: start from LR latent with a noise strength (img2img-style)

It reuses utilities from experiments/train_full_mse_adaln.py to stay consistent
with the validation degradation and metrics.
"""

import argparse
import math
import os

import numpy as np
import torch
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler

from experiments import train_full_mse_adaln as tfa


def parse_strengths(text: str):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def load_models(ckpt_path: str, adapter_type: str):
    pixart = tfa.PixArtMS_XL_2(input_size=64).to(tfa.DEVICE).to(tfa.DTYPE_PIXART).eval()
    ckpt = torch.load(tfa.PIXART_PATH, map_location="cpu")
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if "pos_embed" in ckpt:
        del ckpt["pos_embed"]
    pixart.load_state_dict(ckpt, strict=False)

    adapter = tfa.build_adapter(adapter_type, in_channels=4, hidden_size=1152).to(tfa.DEVICE).eval()

    state = torch.load(ckpt_path, map_location="cpu")
    if "adapter" not in state:
        raise KeyError("Checkpoint missing key: adapter")
    adapter.load_state_dict(state["adapter"], strict=True)

    if "pixart_inject" in state:
        tfa.load_inject_state_dict(pixart, state["pixart_inject"])
    else:
        print("⚠️ checkpoint has no pixart_inject; continuing without it.")

    return pixart, adapter


@torch.no_grad()
def eval_mode(
    pixart,
    adapter,
    vae,
    y_embed,
    data_info,
    mode: str,
    strength: float,
    steps: int,
    max_val: int,
    seed: int,
):
    val_ds = tfa.ValImageDataset(tfa.VAL_HR_DIR, max_files=max_val)

    psnr_list, ssim_list, lpips_list = [], [], []
    lpips_fn = None
    if tfa.USE_METRICS:
        lpips_fn = tfa.lpips.LPIPS(net="vgg").to(tfa.DEVICE).eval()
        for p in lpips_fn.parameters():
            p.requires_grad = False

    for item in val_ds:
        hr11 = item["hr_img_11"]
        name = os.path.basename(item["path"])

        # deterministic per-image degradation (aligned with validation)
        rng = np.random.RandomState(tfa.stable_int_hash(name, mod=2**32))
        torch_gen = torch.Generator(device=tfa.DEVICE).manual_seed(
            tfa.stable_int_hash(name, mod=2**31)
        )

        lr_img_11 = tfa.degrade_hr_to_lr_tensor(
            hr11.detach().cpu(),
            tfa.VAL_DEGRADE_MODE,
            rng,
            torch_gen=torch_gen,
        ).unsqueeze(0)
        lr_img_11 = lr_img_11.to(tfa.DEVICE).float()

        hr_img = hr11.unsqueeze(0).to(tfa.DEVICE).float()

        hr_latent = vae.encode(hr_img).latent_dist.sample() * vae.config.scaling_factor
        lr_latent = vae.encode(lr_img_11).latent_dist.sample() * vae.config.scaling_factor

        scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000, solver_order=2)
        scheduler.set_timesteps(steps, device=tfa.DEVICE)
        run_ts = list(scheduler.timesteps)

        g = torch.Generator(device=tfa.DEVICE).manual_seed(seed)
        noise = torch.randn(lr_latent.shape, generator=g, device=tfa.DEVICE, dtype=lr_latent.dtype)

        if mode == "noise":
            latents = noise
        elif mode == "lr":
            if not (0.0 < strength <= 1.0):
                raise ValueError("strength must be in (0, 1]")
            start_idx = int(len(run_ts) * strength)
            start_idx = min(max(start_idx, 0), len(run_ts) - 1)
            t_start = run_ts[start_idx]
            latents = scheduler.add_noise(lr_latent, noise, t_start)
            run_ts = run_ts[start_idx:]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        with torch.cuda.amp.autocast(enabled=False):
            cond = adapter(lr_latent.float())

        for t in run_ts:
            t_tensor = t.unsqueeze(0).to(tfa.DEVICE)
            with torch.cuda.amp.autocast(enabled=tfa.USE_AMP, dtype=tfa.DTYPE_PIXART):
                out = pixart(
                    latents.to(tfa.DTYPE_PIXART),
                    t_tensor,
                    y_embed,
                    data_info=data_info,
                    adapter_cond=cond,
                    injection_mode="hybrid",
                )
                if out.shape[1] == 8:
                    out, _ = out.chunk(2, dim=1)
            latents = scheduler.step(out.float(), t, latents.float()).prev_sample

        pred_img = vae.decode(latents / vae.config.scaling_factor).sample
        pred_img_01 = torch.clamp((pred_img + 1.0) / 2.0, 0.0, 1.0)

        gt_img = vae.decode(hr_latent / vae.config.scaling_factor).sample
        gt_img_01 = torch.clamp((gt_img + 1.0) / 2.0, 0.0, 1.0)

        if tfa.USE_METRICS:
            if tfa.METRIC_Y_CHANNEL:
                pred_y = tfa.rgb01_to_y01(pred_img_01)
                gt_y = tfa.rgb01_to_y01(gt_img_01)
                pred_y = tfa.shave_border(pred_y, tfa.METRIC_SHAVE_BORDER)
                gt_y = tfa.shave_border(gt_y, tfa.METRIC_SHAVE_BORDER)
                p = tfa.psnr(pred_y, gt_y, data_range=1.0).item()
                s = tfa.ssim(pred_y, gt_y, data_range=1.0).item()
            else:
                p = tfa.psnr(pred_img_01, gt_img_01, data_range=1.0).item()
                s = tfa.ssim(pred_img_01, gt_img_01, data_range=1.0).item()

            psnr_list.append(p)
            ssim_list.append(s)

            if lpips_fn is not None:
                pred_norm = pred_img_01 * 2.0 - 1.0
                gt_norm = gt_img_01 * 2.0 - 1.0
                l = lpips_fn(pred_norm, gt_norm).item()
                lpips_list.append(l)

    avg_psnr = sum(psnr_list) / len(psnr_list) if psnr_list else float("nan")
    avg_ssim = sum(ssim_list) / len(ssim_list) if ssim_list else float("nan")
    avg_lp = sum(lpips_list) / len(lpips_list) if lpips_list else float("nan")
    return avg_psnr, avg_ssim, avg_lp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="path to adapter/pixart_inject checkpoint")
    parser.add_argument("--adapter_type", default="fpn_se", choices=["fpn", "fpn_se"])
    parser.add_argument("--mode", choices=["noise", "lr"], default="noise")
    parser.add_argument("--strengths", default="0.3,0.5,0.7,0.9", help="comma list for lr mode")
    parser.add_argument("--steps", type=int, default=tfa.NUM_INFER_STEPS)
    parser.add_argument("--max_val", type=int, default=None)
    parser.add_argument("--seed", type=int, default=tfa.FIXED_NOISE_SEED)
    args = parser.parse_args()

    pixart, adapter = load_models(args.ckpt, args.adapter_type)
    vae = AutoencoderKL.from_pretrained(tfa.VAE_PATH, local_files_only=True).to(tfa.DEVICE).float().eval()
    vae.enable_slicing()

    y_embed, data_info = tfa.build_text_cond()

    if args.mode == "noise":
        psnr_v, ssim_v, lp_v = eval_mode(
            pixart, adapter, vae, y_embed, data_info,
            mode="noise",
            strength=1.0,
            steps=args.steps,
            max_val=args.max_val,
            seed=args.seed,
        )
        print(f"[MODE noise] PSNR={psnr_v:.2f} SSIM={ssim_v:.4f} LPIPS={lp_v:.4f}")
        return

    strengths = parse_strengths(args.strengths)
    for strength in strengths:
        psnr_v, ssim_v, lp_v = eval_mode(
            pixart, adapter, vae, y_embed, data_info,
            mode="lr",
            strength=strength,
            steps=args.steps,
            max_val=args.max_val,
            seed=args.seed,
        )
        print(f"[MODE lr strength={strength:.2f}] PSNR={psnr_v:.2f} SSIM={ssim_v:.4f} LPIPS={lp_v:.4f}")


if __name__ == "__main__":
    main()
