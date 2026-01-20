#!/usr/bin/env python3
import argparse
import math
import os
import random
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusers import AutoencoderKL, DPMSolverMultistepScheduler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from experiments.train_full_mse_adaln import (
    DEVICE,
    DTYPE_PIXART,
    USE_AMP,
    VAL_DEGRADE_MODE,
    NUM_INFER_STEPS,
    SDE_STRENGTH,
    FIXED_NOISE_SEED,
    METRIC_SHAVE_BORDER,
    METRIC_Y_CHANNEL,
    VAE_PATH,
    PIXART_PATH,
    T5_EMBED_PATH,
    ValImageDataset,
    degrade_hr_to_lr_tensor,
    rgb01_to_y01,
    shave_border,
    stable_int_hash,
)

from diffusion.model.nets.PixArtMS import PixArtMS_XL_2
from diffusion.model.nets.adapter import build_adapter

try:
    from torchmetrics.functional import peak_signal_noise_ratio as psnr
    from torchmetrics.functional import structural_similarity_index_measure as ssim
    import lpips
    USE_METRICS = True
except ImportError:
    USE_METRICS = False


def ensure_fp32_inject_modules(pixart):
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
    if hasattr(pixart, "input_adaln"):
        pixart.input_adaln = pixart.input_adaln.to(torch.float32)


def load_ckpt(pixart, adapter, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "adapter" in ckpt:
        adapter.load_state_dict(ckpt["adapter"], strict=True)
    if "pixart_inject" in ckpt:
        from experiments.train_full_mse_adaln import load_inject_state_dict
        load_inject_state_dict(pixart, ckpt["pixart_inject"])


def build_text_cond():
    y_embed = torch.load(T5_EMBED_PATH, map_location="cpu")["prompt_embeds"].unsqueeze(1)
    y_embed = y_embed.to(DEVICE).to(DTYPE_PIXART)
    data_info = {
        "img_hw": torch.tensor([[512., 512.]], device=DEVICE, dtype=DTYPE_PIXART),
        "aspect_ratio": torch.tensor([1.], device=DEVICE, dtype=DTYPE_PIXART),
    }
    return y_embed, data_info


@torch.no_grad()
def run_epoch(mode: str, pixart, adapter, vae, dl, y_embed, data_info, lpips_fn=None):
    psnr_list, ssim_list, lpips_list = [], [], []
    pbar = tqdm(dl, desc=f"[{mode}] Val", dynamic_ncols=True)

    for batch in pbar:
        hr_img_11 = batch["hr_img_11"].to(DEVICE).float()
        B = hr_img_11.shape[0]

        for bi in range(B):
            item_hr = hr_img_11[bi:bi+1]
            path = batch["path"][bi]

            seed_i = (stable_int_hash(path) + 12345) & 0xFFFFFFFF
            rng = random.Random(seed_i)
            torch_gen = torch.Generator(device="cpu").manual_seed(seed_i)

            lr_img_11 = degrade_hr_to_lr_tensor(
                item_hr.squeeze(0).detach().cpu(),
                VAL_DEGRADE_MODE,
                rng,
                torch_gen=torch_gen,
            ).unsqueeze(0)
            lr_img_11 = lr_img_11.to(DEVICE).float()

            lr_latent = vae.encode(lr_img_11).latent_dist.mode() * vae.config.scaling_factor

            if mode == "adapter_only":
                _, recon_base = adapter.forward_with_recon(lr_latent.float())
                pred_img = vae.decode(recon_base / vae.config.scaling_factor).sample
                pred_img_01 = torch.clamp((pred_img + 1.0) / 2.0, 0.0, 1.0)
            else:
                scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000, solver_order=2)
                scheduler.set_timesteps(NUM_INFER_STEPS, device=DEVICE)
                start_t_val = int(1000 * SDE_STRENGTH)
                run_ts = [t for t in scheduler.timesteps if t <= start_t_val]

                g = torch.Generator(device=DEVICE).manual_seed(FIXED_NOISE_SEED)
                latents = lr_latent.to(DEVICE)
                noise = torch.randn(latents.shape, generator=g, device=DEVICE, dtype=latents.dtype)
                t_start = torch.tensor([start_t_val], device=DEVICE).long()
                latents = scheduler.add_noise(latents, noise, t_start)

                with torch.cuda.amp.autocast(enabled=False):
                    cond = None
                    if mode == "joint":
                        cond, _ = adapter.forward_with_recon(lr_latent.float())
                        cond = [feat.float() for feat in cond]

                for t in run_ts:
                    t_tensor = t.unsqueeze(0).to(DEVICE)
                    with torch.cuda.amp.autocast(enabled=USE_AMP, dtype=DTYPE_PIXART):
                        out = pixart(
                            latents.to(DTYPE_PIXART),
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

            gt_img_01 = torch.clamp((item_hr + 1.0) / 2.0, 0.0, 1.0)
            if USE_METRICS:
                if METRIC_Y_CHANNEL:
                    pred_y = rgb01_to_y01(pred_img_01)
                    gt_y = rgb01_to_y01(gt_img_01)
                    pred_y = shave_border(pred_y, METRIC_SHAVE_BORDER)
                    gt_y = shave_border(gt_y, METRIC_SHAVE_BORDER)
                    p = psnr(pred_y, gt_y, data_range=1.0).item()
                    s = ssim(pred_y, gt_y, data_range=1.0).item()
                else:
                    p = psnr(pred_img_01, gt_img_01, data_range=1.0).item()
                    s = ssim(pred_img_01, gt_img_01, data_range=1.0).item()
                psnr_list.append(p)
                ssim_list.append(s)
                if lpips_fn is not None:
                    pred_norm = pred_img_01 * 2.0 - 1.0
                    gt_norm = gt_img_01 * 2.0 - 1.0
                    l = lpips_fn(pred_norm, gt_norm).item()
                    lpips_list.append(l)

        if USE_METRICS and len(psnr_list) > 0:
            pbar.set_postfix({
                "PSNR": f"{sum(psnr_list)/len(psnr_list):.2f}",
                "SSIM": f"{sum(ssim_list)/len(ssim_list):.4f}",
                "LPIPS": f"{(sum(lpips_list)/len(lpips_list)):.4f}" if len(lpips_list)>0 else "NA",
            })

    avg_psnr = sum(psnr_list)/len(psnr_list) if len(psnr_list) > 0 else float("nan")
    avg_ssim = sum(ssim_list)/len(ssim_list) if len(ssim_list) > 0 else float("nan")
    avg_lp = sum(lpips_list)/len(lpips_list) if len(lpips_list) > 0 else float("nan")
    return avg_psnr, avg_ssim, avg_lp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="adapter+pixart_inject checkpoint")
    parser.add_argument("--adapter_type", type=str, default="fpn_se", choices=["fpn", "fpn_se"])
    parser.add_argument("--mode", type=str, default="all", choices=["all", "joint", "pixart_only", "adapter_only"])
    parser.add_argument("--max_samples", type=int, default=20)
    args = parser.parse_args()

    pixart = PixArtMS_XL_2(input_size=64).to(DEVICE).to(DTYPE_PIXART).eval()
    ckpt = torch.load(PIXART_PATH, map_location="cpu")
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if "pos_embed" in ckpt:
        del ckpt["pos_embed"]
    pixart.load_state_dict(ckpt, strict=False)
    ensure_fp32_inject_modules(pixart)

    adapter = build_adapter(args.adapter_type, in_channels=4, hidden_size=1152).to(DEVICE).eval()
    load_ckpt(pixart, adapter, args.ckpt)

    vae = AutoencoderKL.from_pretrained(VAE_PATH, local_files_only=True).to(DEVICE).float().eval()
    vae.enable_slicing()

    y_embed, data_info = build_text_cond()

    lpips_fn = None
    if USE_METRICS:
        lpips_fn = lpips.LPIPS(net="vgg").to(DEVICE).eval()
        for p in lpips_fn.parameters():
            p.requires_grad = False

    val_ds = ValImageDataset(os.path.join(os.path.dirname(__file__), "..", "dataset", "DIV2K_valid_HR"), max_files=args.max_samples)
    dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=(DEVICE=="cuda"))

    modes = ["joint", "pixart_only", "adapter_only"] if args.mode == "all" else [args.mode]
    for mode in modes:
        p, s, l = run_epoch(mode, pixart, adapter, vae, dl, y_embed, data_info, lpips_fn)
        print(f"[{mode}] PSNR={p:.2f} SSIM={s:.4f} LPIPS={l if math.isfinite(l) else float('nan'):.4f}")


if __name__ == "__main__":
    main()
