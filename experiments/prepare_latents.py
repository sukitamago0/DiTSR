import argparse
import glob
import io
import os
import random
import sys

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms import functional as TF

# ================= 🛑 1. 硬件硬约束 (针对 3070) =================
# 修复 RuntimeError: GET was unable to find an engine to execute this computation
torch.backends.cudnn.enabled = False
# ==============================================================

# ================= 🔧 默认配置 =================
HR_DIR = "../dataset/DIV2K_train_HR"
OUTPUT_DIR = "../dataset/DIV2K_train_latents"
VAE_PATH = "../output/pretrained_models/sd-vae-ft-ema"

CROP_SIZE = 512
STRIDE = 256  # 50% 重叠

DOWNSCALE_FACTOR = 0.25
BLUR_KERNEL_SIZES = [3, 5, 7]
BLUR_SIGMA_RANGE = (0.2, 1.2)
NOISE_STD_RANGE = (0.0, 0.02)
JPEG_QUALITY_RANGE = (30, 95)

SAVE_LR_IMG = False
SAVE_PREVIEW_PATH = None
# ===============================================


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


def _jpeg_compress(tensor, quality):
    img = tensor.clamp(-1.0, 1.0)
    img = (img + 1.0) / 2.0
    img = TF.to_pil_image(img)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    img = Image.open(buffer).convert("RGB")
    img = TF.to_tensor(img)
    img = (img * 2.0) - 1.0
    return img


def degrade_hr_to_lr(hr_crop, rng, crop_size):
    blur_k = rng.choice(BLUR_KERNEL_SIZES)
    blur_sigma = rng.uniform(*BLUR_SIGMA_RANGE)
    hr_blur = TF.gaussian_blur(hr_crop.squeeze(0), blur_k, [blur_sigma, blur_sigma]).unsqueeze(0)

    lr_small = F.interpolate(hr_blur, scale_factor=DOWNSCALE_FACTOR, mode="bicubic", align_corners=False)

    noise_std = rng.uniform(*NOISE_STD_RANGE)
    if noise_std > 0:
        lr_small = lr_small + torch.randn_like(lr_small) * noise_std
        lr_small = lr_small.clamp(-1.0, 1.0)

    jpeg_quality = rng.randint(JPEG_QUALITY_RANGE[0], JPEG_QUALITY_RANGE[1])
    lr_small = _jpeg_compress(lr_small.squeeze(0), jpeg_quality).unsqueeze(0)

    lr_crop = F.interpolate(lr_small, size=(crop_size, crop_size), mode="bicubic", align_corners=False)
    return lr_crop


def save_preview(hr_crop, lr_crop, path):
    hr_img = (hr_crop.squeeze(0).clamp(-1.0, 1.0) + 1.0) / 2.0
    lr_img = (lr_crop.squeeze(0).clamp(-1.0, 1.0) + 1.0) / 2.0
    hr_img = TF.to_pil_image(hr_img)
    lr_img = TF.to_pil_image(lr_img)

    preview = Image.new("RGB", (hr_img.width * 2, hr_img.height))
    preview.paste(hr_img, (0, 0))
    preview.paste(lr_img, (hr_img.width, 0))
    preview.save(path)


def process_single_image(
    img_path,
    vae,
    transform,
    device,
    save_root,
    rng,
    save_lr_img,
    preview_path,
    preview_written,
    crop_size,
    stride,
):
    filename = os.path.basename(img_path).split('.')[0]

    try:
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)  # [1, 3, H, W]
    except Exception as e:
        print(f"❌ Error reading {img_path}: {e}")
        return 0, preview_written

    _, _, h, w = img_tensor.shape
    count = 0

    y_points = list(range(0, h - crop_size + 1, stride))
    if (h - crop_size) % stride != 0:
        y_points.append(h - crop_size)

    x_points = list(range(0, w - crop_size + 1, stride))
    if (w - crop_size) % stride != 0:
        x_points.append(w - crop_size)

    if h < crop_size or w < crop_size:
        return 0, preview_written

    for y in y_points:
        for x in x_points:
            hr_crop = img_tensor[:, :, y:y + crop_size, x:x + crop_size]
            lr_crop = degrade_hr_to_lr(hr_crop, rng, crop_size)

            hr_crop_gpu = hr_crop.to(device).float()
            lr_crop_gpu = lr_crop.to(device).float()

            with torch.no_grad():
                hr_dist = vae.encode(hr_crop_gpu).latent_dist
                hr_latent = hr_dist.sample() * vae.config.scaling_factor

                lr_dist = vae.encode(lr_crop_gpu).latent_dist
                lr_latent = lr_dist.sample() * vae.config.scaling_factor

            save_dict = {
                "hr_latent": hr_latent.squeeze(0).cpu().half(),
                "lr_latent": lr_latent.squeeze(0).cpu().half(),
            }
            if save_lr_img:
                save_dict["lr_img"] = lr_crop.squeeze(0).half()

            save_name = f"{filename}_y{y}_x{x}.pt"
            torch.save(save_dict, os.path.join(save_root, save_name))
            count += 1

            if preview_path and not preview_written:
                save_preview(hr_crop, lr_crop, preview_path)
                preview_written = True

    return count, preview_written


def main():
    parser = argparse.ArgumentParser(description="Prepare HR/LR latents with realistic SR degradation.")
    parser.add_argument("--hr_dir", type=str, default=HR_DIR)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--vae_path", type=str, default=VAE_PATH)
    parser.add_argument("--crop_size", type=int, default=CROP_SIZE)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--save_lr_img", action="store_true", default=SAVE_LR_IMG)
    parser.add_argument(
        "--preview_path",
        type=str,
        default=SAVE_PREVIEW_PATH,
        help="Save a side-by-side HR/LR preview image (first crop only).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = setup_device()
    print(f"🚀 Starting Offline Latent Extraction on {device}")
    print(f"   Hard Constraint: torch.backends.cudnn.enabled = {torch.backends.cudnn.enabled}")

    if not os.path.exists(args.hr_dir):
        print(f"❌ HR Directory not found: {args.hr_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    vae = load_vae(args.vae_path, device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    exts = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG']
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(args.hr_dir, ext)))
    image_paths = sorted(list(set(image_paths)))

    if len(image_paths) == 0:
        print(f"❌ No images found in {args.hr_dir}")
        return

    print(f"📊 Found {len(image_paths)} images. Stride={args.stride}, Crop={args.crop_size}")
    print(f"📂 Output Dir: {args.output_dir}")

    rng = random.Random(args.seed)

    total_generated = 0
    preview_written = False
    pbar = tqdm(image_paths, desc="Processing", unit="img")

    for img_path in pbar:
        num, preview_written = process_single_image(
            img_path,
            vae,
            transform,
            device,
            args.output_dir,
            rng,
            args.save_lr_img,
            args.preview_path,
            preview_written,
            args.crop_size,
            args.stride,
        )
        total_generated += num
        pbar.set_postfix({"Patches": total_generated})

    print(f"\n✅ All Done! Generated {total_generated} latents.")
    if args.preview_path:
        print(f"🖼️ Preview saved to: {args.preview_path}")


if __name__ == "__main__":
    main()
