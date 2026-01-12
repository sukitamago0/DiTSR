import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import sys
import glob
import logging
from datetime import datetime
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# ================= 🛑 1. 硬件硬约束 (严格对齐 A版) =================
torch.backends.cudnn.enabled = False
# =================================================================

import lpips
from torchmetrics.functional import peak_signal_noise_ratio as calc_psnr
from torchmetrics.functional import structural_similarity_index_measure as calc_ssim
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler
from diffusers.optimization import get_scheduler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from diffusion.model.nets.PixArtMS import PixArtMS_XL_2
from diffusion import IDDPM

# ================= ⚙️ 2. 全局配置 =================
class Config:
    # [开关] 开启后只跑单张图，快速验证流程
    DEBUG = True          
    
    # 路径配置 (与训练一致)
    VAL_ROOT = "dataset/DIV2K_valid_HR"
    # 如果 DEBUG=True, 我们通常想看训练集那张图在 Baseline 下的表现
    TRAIN_ROOT = "dataset/DIV2K_train_HR" 
    
    NULL_EMBED_PATH = "output/null_embed.pth"
    PRETRAINED_PIXART = "output/pretrained_models/PixArt-XL-2-512x512.pth"
    VAE_PATH = "output/pretrained_models/sd-vae-ft-ema"
    
    EXP_NAME = f"Baseline_NoAdapter_{datetime.now().strftime('%m%d_%H%M')}"
    if DEBUG: EXP_NAME += "_DEBUG"
    OUTPUT_DIR = f"experiments_results/{EXP_NAME}"
    
    # 显存安全配置
    BATCH_SIZE = 1           
    NUM_WORKERS = 0 # 避免多进程开销
    
    # --- SDEdit 物理设置 ---
    VAL_STEPS = 20           
    # 此处1是全噪声 而0.001为低噪声
    SDE_STRENGTH = 0      
    
    DEVICE = "cuda"
    DTYPE_BACKBONE = torch.float16 
# =============================================================

# --- 3. 辅助函数 (复用 train_final.py) ---
def setup_logger(cfg):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(cfg.OUTPUT_DIR, "vis"), exist_ok=True)
    
    logger = logging.getLogger("BaselineLog")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(f"{cfg.OUTPUT_DIR}/log.txt")
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter); ch.setFormatter(formatter)
        logger.addHandler(fh); logger.addHandler(ch)
    return logger

def encode_latents(vae, img_tensor, device, dtype):
    with torch.no_grad():
        latents = vae.encode(img_tensor.to("cpu").float()).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        return latents.to(device).to(dtype)

def decode_latents(vae, latents, device):
    latents_cpu = latents.to("cpu").float()
    latents_cpu = latents_cpu / vae.config.scaling_factor
    with torch.no_grad():
        image = vae.decode(latents_cpu).sample
    return image.to(device)

# --- 4. 数据集 (复用 train_final.py) ---
class SRDataset(Dataset):
    def __init__(self, root_dir, crop_size=512, debug=False):
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, "*.png")))
        if len(self.image_paths) == 0: self.image_paths = sorted(glob.glob(os.path.join(root_dir, "*.jpg")))
        
        if len(self.image_paths) == 0 and os.path.exists("test.jpg"):
            self.image_paths = ["test.jpg"]
            
        if debug:
            # 锁定单样本 (与训练过拟合时的样本保持一致)
            first_img = self.image_paths[0] if len(self.image_paths) > 0 else "test.jpg"
            print(f"⚠️ [DEBUG] 锁定单样本: {first_img}")
            self.image_paths = [first_img]

        self.norm_mean = [0.5, 0.5, 0.5]
        self.norm_std = [0.5, 0.5, 0.5]

        # 验证模式只做 CenterCrop
        self.transform = transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(self.norm_mean, self.norm_std)
        ])

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            hr_img = Image.open(self.image_paths[idx]).convert("RGB")
            hr_tensor = self.transform(hr_img)
            
            # 制造 LR
            lr_tensor = F.interpolate(hr_tensor.unsqueeze(0), scale_factor=0.25, mode='bicubic', align_corners=False)
            lr_tensor = F.interpolate(lr_tensor, size=(hr_tensor.shape[1], hr_tensor.shape[2]), mode='bicubic', align_corners=False).squeeze(0)
            
            return hr_tensor, lr_tensor
        except Exception as e:
            print(f"Error: {e}")
            return torch.zeros(3, 512, 512), torch.zeros(3, 512, 512)

# --- 5. 指标计算 ---
class MetricsCalculator:
    def __init__(self, device):
        self.device = device
        print("   [Init] Loading LPIPS...")
        self.loss_fn_lpips = lpips.LPIPS(net='vgg').to(device).eval()
        for p in self.loss_fn_lpips.parameters(): p.requires_grad = False

    def calculate(self, img_pred, img_gt):
        p = img_pred.float()
        g = img_gt.float()
        
        val_lpips = self.loss_fn_lpips(p, g).item()
        
        p_01 = ((p + 1) / 2).clamp(0, 1)
        g_01 = ((g + 1) / 2).clamp(0, 1)
        
        val_psnr = calc_psnr(p_01, g_01, data_range=1.0).item()
        val_ssim = calc_ssim(p_01, g_01, data_range=1.0).item()
        return {"psnr": val_psnr, "ssim": val_ssim, "lpips": val_lpips}

# --- 6. 主程序 ---
def main():
    cfg = Config()
    logger = setup_logger(cfg)
    logger.info(f"🚀 启动 Baseline (No Adapter) | Strength: {cfg.SDE_STRENGTH} | Mode: {'🛑 DEBUG' if cfg.DEBUG else '✅ FULL'}")
    
    # 1. 准备数据
    # 如果是 DEBUG，我们通常想对比训练集那张图在没有 Adapter 时的表现
    target_root = cfg.TRAIN_ROOT if cfg.DEBUG else cfg.VAL_ROOT
    dataset = SRDataset(target_root, debug=cfg.DEBUG)
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    logger.info(f"   Data: {len(dataset)} images")

    # 2. 加载模型 (不加载 Adapter)
    metrics_calc = MetricsCalculator(cfg.DEVICE)
    
    vae = AutoencoderKL.from_pretrained(cfg.VAE_PATH, local_files_only=True).to("cpu").to(torch.float32).eval()
    logger.info("   ✅ VAE Loaded")
    
    model = PixArtMS_XL_2(input_size=64).to(cfg.DEVICE).to(cfg.DTYPE_BACKBONE).eval()
    ckpt = torch.load(cfg.PRETRAINED_PIXART, map_location="cpu")
    if "state_dict" in ckpt: ckpt = ckpt["state_dict"]
    if "pos_embed" in ckpt: del ckpt["pos_embed"]
    model.load_state_dict(ckpt, strict=False)
    logger.info(f"   ✅ PixArt Loaded (FP16)")
    
    # Null Embed
    y_embed = torch.load(cfg.NULL_EMBED_PATH, map_location="cpu")["prompt_embeds"].unsqueeze(1).to(cfg.DEVICE).to(cfg.DTYPE_BACKBONE)
    
    data_info = {'img_hw': torch.tensor([[512., 512.]]).to(cfg.DEVICE).to(cfg.DTYPE_BACKBONE), 'aspect_ratio': torch.tensor([1.]).to(cfg.DEVICE).to(cfg.DTYPE_BACKBONE)}

    # 3. Scheduler & Diffusion
    diffusion = IDDPM(str(1000))
    val_scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        algorithm_type="dpmsolver++",
        solver_order=2,
    )
    
    # ================= [Debug] 打印时间步详情 =================
    val_scheduler.set_timesteps(cfg.VAL_STEPS)
    timesteps = val_scheduler.timesteps
    
    # 强制转成 float 列表以便观察，防止 tensor 格式看不清
    ts_list = [float(t) for t in timesteps]
    logger.info(f"🔎 [Debug] All Timesteps: {ts_list}")
    
    target_t_physical = int(1000 * cfg.SDE_STRENGTH)
    logger.info(f"🎯 [Debug] Target Physical T: {target_t_physical}")
    # ========================================================

    is_descending = timesteps[0] > timesteps[-1]
    
    # 🚨 [关键修复] 默认值改为 None，用于检测是否命中
    start_idx = None 
    
    if is_descending: # Case: [999, ... 0]
        for i, t in enumerate(timesteps):
            if t <= target_t_physical:
                start_idx = i
                break
    else: # Case: [0, ... 999]
        # 注意：升序时我们要找第一个 >= target 的位置（或者反过来截取）
        # 这里建议统一逻辑：翻转后按降序处理，最稳妥
        timesteps_rev = torch.flip(timesteps, [0])
        for i, t in enumerate(timesteps_rev):
            if t <= target_t_physical:
                start_idx = i
                break
        # 注意：如果翻转了，切片逻辑也要变。简单起见，我们只改 timesteps 变量本身
        if start_idx is not None:
            timesteps = timesteps_rev

    # ================= [逻辑兜底] =================
    if start_idx is None:
        # 情况A：没找到（通常是因为 strength 设得太小，比最小 timestep 还小）
        # 行为：不进行任何去噪推理（直接输出原图）
        logger.warning("⚠️ Strength too low! Skipping diffusion steps.")
        inference_timesteps = []
        actual_start_t = torch.tensor(0).to(cfg.DEVICE)
    else:
        # 情况B：找到了
        inference_timesteps = timesteps[start_idx:]
        actual_start_t = inference_timesteps[0]
    # =============================================

    logger.info(f"✅ SDE Strength: {cfg.SDE_STRENGTH} | Physical T: {actual_start_t.item()} | Steps: {len(inference_timesteps)}")
    # 4. 推理循环
    metrics_sum = {"psnr": 0, "ssim": 0, "lpips": 0}
    
    for i, (val_hr, val_lr) in enumerate(loader):
        val_hr = val_hr.to(cfg.DEVICE)
        val_lr = val_lr.to(cfg.DEVICE).to(cfg.DTYPE_BACKBONE) # 注意：这里 LR 不需要转 adapter dtype，因为没 adapter
        
        # 1. 准备底图
        lr_latents = encode_latents(vae, val_lr, cfg.DEVICE, cfg.DTYPE_BACKBONE)
        
        # 2. 加噪
        noise = torch.randn_like(lr_latents)
        if len(inference_timesteps) == 0:
            latents = lr_latents
        else:
            t_batch = torch.full((lr_latents.shape[0],), actual_start_t.item(), device=cfg.DEVICE, dtype=torch.long)
            latents = diffusion.q_sample(lr_latents, t_batch, noise)
            
        # 3. 去噪 (无 Adapter)
        with torch.no_grad():
            for t in inference_timesteps:
                t_tensor = t.unsqueeze(0).to(cfg.DEVICE)
                
                # [关键] 传入 adapter_features=None
                out = model(latents, t_tensor, y_embed, data_info=data_info, adapter_features=None)
                
                # PixArt 输出可能有 8 通道，取前 4
                if out.shape[1] == 8: 
                    noise_pred, _ = out.chunk(2, dim=1)
                else: 
                    noise_pred = out
                
                latents = val_scheduler.step(noise_pred, t, latents).prev_sample
        
        # 4. 解码与评估
        pred_img = decode_latents(vae, latents, cfg.DEVICE)
        res = metrics_calc.calculate(pred_img, val_hr)
        
        for k in res: metrics_sum[k] += res[k]
        
        logger.info(f"   Img {i}: PSNR={res['psnr']:.2f} | LPIPS={res['lpips']:.4f}")
        
        # 保存图片
        save_visual_sample(val_lr[0], val_hr[0], pred_img[0], i, cfg.OUTPUT_DIR)

    count = len(dataset)
    avg = {k: v / count for k, v in metrics_sum.items()}
    logger.info(f"📊 [Baseline Result] Average PSNR: {avg['psnr']:.2f} | Average LPIPS: {avg['lpips']:.4f}")


def save_visual_sample(lr, hr, sr, idx, save_dir):
    def prep(x): return ((x.detach().cpu().float() + 1) / 2).clamp(0, 1)
    
    img_lr = prep(lr).permute(1, 2, 0).numpy()
    img_hr = prep(hr).permute(1, 2, 0).numpy()
    img_sr = prep(sr).permute(1, 2, 0).numpy()
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img_lr); axs[0].set_title("Input LR")
    axs[1].imshow(img_hr); axs[1].set_title("Target HR")
    axs[2].imshow(img_sr); axs[2].set_title(f"Baseline SR (No Adapter)")
    
    for ax in axs: ax.axis('off')
    plt.savefig(f"{save_dir}/vis/baseline_{idx}.png", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()