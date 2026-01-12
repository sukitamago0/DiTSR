import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import sys
import glob
import argparse
from tqdm import tqdm 
from diffusers import AutoencoderKL

# ================= 🛑 1. 硬件硬约束 (针对 3070) =================
# 修复 RuntimeError: GET was unable to find an engine to execute this computation
torch.backends.cudnn.enabled = False
# ==============================================================

# ================= 🔧 配置区域 =================
# 路径配置
HR_DIR = "../dataset/DIV2K_train_HR"      
OUTPUT_DIR = "../dataset/DIV2K_train_latents" 
VAE_PATH = "../output/pretrained_models/sd-vae-ft-ema" 

# 切片策略
CROP_SIZE = 512
STRIDE = 256  # 50% 重叠
# ===============================================

def setup_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def load_vae(path, device):
    print(f"🔄 [Init] Loading VAE from {path}...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ VAE model not found at {path}")
    
    # [Fix] 使用 float32 加载 VAE
    # 3070 8G 显存足够跑 FP32 的 VAE (仅占用约 1.5GB)
    # 这能避免 FP16 下的 cuDNN 卷积算法查找错误
    try:
        vae = AutoencoderKL.from_pretrained(path, local_files_only=True).to(device).float().eval()
    except Exception as e:
        print(f"⚠️ Load failed: {e}")
        sys.exit(1)
    
    # 开启切片推理，进一步节省显存
    vae.enable_slicing()
    print("✅ VAE Loaded (FP32 mode) & Slicing Enabled")
    return vae

def process_single_image(img_path, vae, transform, device, save_root):
    filename = os.path.basename(img_path).split('.')[0]
    
    try:
        # 1. 读取图片
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0) # [1, 3, H, W]
    except Exception as e:
        print(f"❌ Error reading {img_path}: {e}")
        return 0

    _, _, h, w = img_tensor.shape
    count = 0
    
    # 2. 滑动窗口切片 (Sliding Window)
    y_points = list(range(0, h - CROP_SIZE + 1, STRIDE))
    if (h - CROP_SIZE) % STRIDE != 0: 
        y_points.append(h - CROP_SIZE)
        
    x_points = list(range(0, w - CROP_SIZE + 1, STRIDE))
    if (w - CROP_SIZE) % STRIDE != 0: 
        x_points.append(w - CROP_SIZE)

    if h < CROP_SIZE or w < CROP_SIZE:
        return 0

    for y in y_points:
        for x in x_points:
            # 3. 裁剪 (CPU)
            hr_crop = img_tensor[:, :, y:y+CROP_SIZE, x:x+CROP_SIZE]
            
            # 4. 生成 LR (CPU)
            lr_crop = F.interpolate(hr_crop, scale_factor=0.25, mode='bicubic', align_corners=False)
            lr_crop = F.interpolate(lr_crop, size=(CROP_SIZE, CROP_SIZE), mode='bicubic', align_corners=False)
            
            # 5. VAE 编码 (GPU FP32)
            # [Fix] 转为 float() 而不是 half()，避免报错
            hr_crop_gpu = hr_crop.to(device).float() 
            
            with torch.no_grad():
                dist = vae.encode(hr_crop_gpu).latent_dist
                latents = dist.sample()
                # Scaling Factor
                latents = latents * vae.config.scaling_factor
            
            # 6. 保存 (转回 CPU FP16 保存以节省空间)
            # 虽然计算用 FP32，但存储用 FP16 是安全的
            save_dict = {
                "lr_img": lr_crop.squeeze(0).half(),      # [3, 512, 512] FP16
                "hr_latent": latents.squeeze(0).cpu().half() # [4, 64, 64] FP16
            }
            
            save_name = f"{filename}_y{y}_x{x}.pt"
            torch.save(save_dict, os.path.join(save_root, save_name))
            count += 1
            
    return count

def main():
    device = setup_device()
    print(f"🚀 Starting Offline Latent Extraction on {device}")
    print(f"   Hard Constraint: torch.backends.cudnn.enabled = {torch.backends.cudnn.enabled}")
    
    if not os.path.exists(HR_DIR):
        print(f"❌ HR Directory not found: {HR_DIR}")
        return
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    vae = load_vae(VAE_PATH, device)
    
    # 预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    exts = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG']
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(HR_DIR, ext)))
    image_paths = sorted(list(set(image_paths)))
    
    if len(image_paths) == 0:
        print(f"❌ No images found in {HR_DIR}")
        return

    print(f"📊 Found {len(image_paths)} images. Stride={STRIDE}, Crop={CROP_SIZE}")
    print(f"📂 Output Dir: {OUTPUT_DIR}")
    
    total_generated = 0
    pbar = tqdm(image_paths, desc="Processing", unit="img")
    
    for img_path in pbar:
        num = process_single_image(img_path, vae, transform, device, OUTPUT_DIR)
        total_generated += num
        pbar.set_postfix({"Patches": total_generated})
        
    print(f"\n✅ All Done! Generated {total_generated} latents.")

if __name__ == "__main__":
    main()