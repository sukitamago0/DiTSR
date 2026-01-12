import argparse
import glob
import os
import sys

import torch
from diffusers import AutoencoderKL
from tqdm import tqdm

# ================= 🛑 1. 硬件硬约束 (针对 3070) =================
# 修复 RuntimeError: GET was unable to find an engine to execute this computation
torch.backends.cudnn.enabled = False
# ==============================================================


def setup_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_vae(path, device):
    print(f"🔄 [Init] Loading VAE from {path}...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ VAE model not found at {path}")

    try:
        vae = AutoencoderKL.from_pretrained(path, local_files_only=True).to(device).float().eval()
    except Exception as e:
        print(f"⚠️ Load failed: {e}")
        sys.exit(1)

    vae.enable_slicing()
    print("✅ VAE Loaded (FP32 mode) & Slicing Enabled")
    return vae


def encode_lr_latent(lr_img, vae, device):
    lr_img = lr_img.unsqueeze(0).to(device).float()
    with torch.no_grad():
        dist = vae.encode(lr_img).latent_dist
        latents = dist.sample()
        latents = latents * vae.config.scaling_factor
    return latents.squeeze(0).cpu().half()


def process_file(path, vae, device, overwrite=False, output_dir=None):
    data = torch.load(path, map_location="cpu")
    if "lr_img" not in data:
        print(f"⚠️ Skip (missing lr_img): {path}")
        return False
    if "lr_latent" in data and not overwrite:
        return False

    lr_img = data["lr_img"]
    lr_latent = encode_lr_latent(lr_img, vae, device)
    data["lr_latent"] = lr_latent

    save_path = path if output_dir is None else os.path.join(output_dir, os.path.basename(path))
    torch.save(data, save_path)
    return True


def main():
    parser = argparse.ArgumentParser(description="Offline LR latent extraction (aligned with existing HR latents).")
    parser.add_argument("--input_dir", type=str, default="../dataset/DIV2K_train_latents",
                        help="Directory with existing .pt samples (lr_img + hr_latent).")
    parser.add_argument("--vae_path", type=str, default="../output/pretrained_models/sd-vae-ft-ema",
                        help="Path to VAE model.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Optional output directory (default: overwrite in place).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing lr_latent if present.")
    args = parser.parse_args()

    device = setup_device()
    print(f"🚀 Starting LR Latent Extraction on {device}")
    print(f"   Hard Constraint: torch.backends.cudnn.enabled = {torch.backends.cudnn.enabled}")

    if not os.path.exists(args.input_dir):
        print(f"❌ Input Directory not found: {args.input_dir}")
        return

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    vae = load_vae(args.vae_path, device)

    files = sorted(glob.glob(os.path.join(args.input_dir, "*.pt")))
    if not files:
        print(f"❌ No .pt files found in {args.input_dir}")
        return

    updated = 0
    pbar = tqdm(files, desc="Processing", unit="file")
    for path in pbar:
        if process_file(path, vae, device, overwrite=args.overwrite, output_dir=args.output_dir):
            updated += 1
        pbar.set_postfix({"Updated": updated})

    print(f"\n✅ Done! Updated {updated}/{len(files)} files.")


if __name__ == "__main__":
    main()
