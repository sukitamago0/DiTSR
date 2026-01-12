import torch
import torch.nn as nn
import torch.nn.functional as F
# [1] 导入所需组件
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler
from torch.cuda.amp import GradScaler
from torch.utils.data import Dataset, DataLoader
import os
import sys
import glob
# [2] 强制非交互后端，防止服务器报错
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import gc
from PIL import Image
from torchvision import transforms

# -----------------------------------------------------------------------------
# 1. 环境与配置
# -----------------------------------------------------------------------------
# 显存碎片优化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
torch.backends.cudnn.enabled = False

# 路径配置
TRAIN_DATASET_DIR = "../dataset/DIV2K_train_latents/" 
# 指向包含 HR 原图的验证集文件夹 (png/jpg)
VALID_DATASET_DIR = "../dataset/DIV2K_valid_HR/"

CHECKPOINT_DIR = "../output/checkpoints/hybrid_production_v1/"
VIS_DIR = os.path.join(CHECKPOINT_DIR, "vis")

PIXART_PATH = "../output/pretrained_models/PixArt-XL-2-512x512.pth"
VAE_PATH = "../output/pretrained_models/sd-vae-ft-ema"
T5_EMBED_PATH = "../output/quality_embed.pth" 

DEVICE = "cuda"
DTYPE = torch.float16

# [训练超参]
NUM_EPOCHS = 100        
BATCH_SIZE = 1          # 3070 8G 物理限制
GRAD_ACCUM_STEPS = 4    # 等效 Batch Size = 4
NUM_WORKERS = 2         
LR_ADAPTER = 1e-5       
LR_SCALES = 1e-4         
SAVE_INTERVAL_EPOCH = 1 # 每轮保存并验证
SDE_STRENGTH = 0.6      
TEXTURE_LOSS_WEIGHT = 0.1 # 训练时使用的 Latent 纹理损失权重

# -----------------------------------------------------------------------------
# 2. 导入项目模块
# -----------------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from diffusion.model.nets.PixArtMS import PixArtMS_XL_2
    from diffusion.model.nets.adapter import MultiLevelAdapter
    from diffusion import IDDPM
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# -----------------------------------------------------------------------------
# 3. 评估指标 (仅用于验证阶段，不占用训练显存)
# -----------------------------------------------------------------------------
try:
    from torchmetrics.functional import peak_signal_noise_ratio as psnr
    from torchmetrics.functional import structural_similarity_index_measure as ssim
    import lpips
    USE_METRICS = True
    # 初始化 LPIPS 用于验证打分
    val_lpips_fn = lpips.LPIPS(net='vgg').to(DEVICE).to(DTYPE).eval()
    for p in val_lpips_fn.parameters(): p.requires_grad = False
    print("✅ Metrics libraries loaded.")
except ImportError:
    USE_METRICS = False
    print("⚠️ Metrics libraries missing. Validation will only generate images.")

# ==============================================================================
# 数据集定义
# ==============================================================================
class TrainLatentDataset(Dataset):
    def __init__(self, root_dir):
        self.files = sorted(glob.glob(os.path.join(root_dir, "*.pt")))
        if len(self.files) == 0: raise ValueError(f"❌ No .pt files in {root_dir}")
        print(f"📂 Training Set: Found {len(self.files)} samples.")

    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        # 包含 {'hr_latent': ..., 'lr_img': ...}
        data = torch.load(self.files[idx], map_location="cpu")
        return data

class ValidImageDataset(Dataset):
    def __init__(self, root_dir):
        self.files = sorted(glob.glob(os.path.join(root_dir, "*.png")) + 
                            glob.glob(os.path.join(root_dir, "*.jpg")))
        if len(self.files) == 0: print(f"⚠️ Warning: No images found in {root_dir}")
        else: print(f"📂 Validation Set: Found {len(self.files)} samples.")
        
        self.transform = transforms.Compose([
            transforms.CenterCrop(512), # 验证时中心裁剪，保证尺寸统一
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        img_path = self.files[idx]
        hr_img = Image.open(img_path).convert("RGB")
        hr_tensor = self.transform(hr_img)
        
        # 模拟退化：Downsample -> Upsample
        lr_small = F.interpolate(hr_tensor.unsqueeze(0), scale_factor=0.25, mode='bicubic', align_corners=False)
        lr_tensor = F.interpolate(lr_small, size=(512, 512), mode='bicubic', align_corners=False).squeeze(0)
        
        return {
            'hr_img': hr_tensor, # [-1, 1]
            'lr_img': lr_tensor, # [-1, 1]
            'name': os.path.basename(img_path)
        }

# ==============================================================================
# 主训练流程
# ==============================================================================
def train_full_production():
    print(f"\n🚀 Start Full Training Production Run")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)
    
    # --- 1. Init Models ---
    print("   Loading PixArt...")
    pixart = PixArtMS_XL_2(input_size=64).to(DEVICE).to(DTYPE).train()
    if os.path.exists(PIXART_PATH):
        ckpt = torch.load(PIXART_PATH, map_location="cpu")
        if "state_dict" in ckpt: ckpt = ckpt["state_dict"]
        if "pos_embed" in ckpt: del ckpt["pos_embed"]
        pixart.load_state_dict(ckpt, strict=False)
    
    # 开启梯度检查点省显存
    if hasattr(pixart, 'enable_gradient_checkpointing'): pixart.enable_gradient_checkpointing()
    else: pixart.gradient_checkpointing = True 
    
    print("   Loading Adapter...")
    adapter = MultiLevelAdapter(in_channels=4, hidden_size=1152).to(DEVICE).train()
    
    # --- 2. Freeze & FP32 ---
    for p in pixart.parameters(): p.requires_grad = False
    
    # 强制可训练参数为 FP32 (AMP 要求)
    pixart.injection_scales = pixart.injection_scales.to(torch.float32)
    pixart.adapter_proj = pixart.adapter_proj.to(torch.float32)
    pixart.adapter_norm = pixart.adapter_norm.to(torch.float32)
    pixart.cross_attn_scale.data = pixart.cross_attn_scale.data.float()
    if hasattr(pixart, 'input_adapter_ln'):
        pixart.input_adapter_ln = pixart.input_adapter_ln.to(torch.float32)

    # --- 3. Optimizer ---
    adapter_params = list(adapter.parameters())
    scale_params = []
    for scale in pixart.injection_scales: scale.requires_grad = True; scale_params.append(scale)
    pixart.cross_attn_scale.requires_grad = True; scale_params.append(pixart.cross_attn_scale)
    
    proj_params = []
    for p in pixart.adapter_proj.parameters(): p.requires_grad = True; proj_params.append(p)
    for p in pixart.adapter_norm.parameters(): p.requires_grad = True; proj_params.append(p)
    if hasattr(pixart, 'input_adapter_ln'):
        for p in pixart.input_adapter_ln.parameters(): p.requires_grad = True; proj_params.append(p)

    optimizer = torch.optim.AdamW([
        {'params': adapter_params, 'lr': LR_ADAPTER},
        {'params': proj_params, 'lr': LR_ADAPTER},
        {'params': scale_params, 'lr': LR_SCALES}
    ])
    
    scaler = GradScaler()
    diffusion = IDDPM(str(1000))

    # --- 4. Data Loaders ---
    train_loader = DataLoader(
        TrainLatentDataset(TRAIN_DATASET_DIR), 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    # 验证集 BS=1
    valid_loader = DataLoader(
        ValidImageDataset(VALID_DATASET_DIR), 
        batch_size=1, 
        shuffle=False, 
        num_workers=1
    )

    print("   Loading VAE (for encoding)...")
    vae = AutoencoderKL.from_pretrained(VAE_PATH, local_files_only=True).to(DEVICE).to(DTYPE)
    vae.eval()
    for p in vae.parameters(): p.requires_grad = False

    y_embed = torch.load(T5_EMBED_PATH, map_location="cpu")["prompt_embeds"].unsqueeze(1).to(DEVICE).to(DTYPE)
    data_info = {'img_hw': torch.tensor([[512., 512.]]).to(DEVICE).to(DTYPE), 'aspect_ratio': torch.tensor([1.]).to(DEVICE).to(DTYPE)}

    # --- Helpers ---
    def predict_x0(x_t, eps, t):
        acp = torch.from_numpy(diffusion.alphas_cumprod).to(x_t.device).to(x_t.dtype)
        at = acp[t]
        return (x_t - (1 - at).sqrt().view(-1, 1, 1, 1) * eps) / at.sqrt().view(-1, 1, 1, 1)

    def latent_texture_loss(pred, target):
        p_dx = pred[..., 1:] - pred[..., :-1]; p_dy = pred[..., 1:, :] - pred[..., :-1, :]
        t_dx = target[..., 1:] - target[..., :-1]; t_dy = target[..., 1:, :] - target[..., :-1, :]
        return F.l1_loss(p_dx, t_dx) + F.l1_loss(p_dy, t_dy)

    # -------------------------------------------------------------------------
    # 5. Training Loop
    # -------------------------------------------------------------------------
    for epoch in range(NUM_EPOCHS):
        print(f"\n🌟 Epoch {epoch+1}/{NUM_EPOCHS}")
        
        pixart.train()
        adapter.train()
        
        pbar = tqdm(train_loader, desc="Training")
        optimizer.zero_grad()
        
        for i, batch in enumerate(pbar):
            # Data
            hr_latent = batch["hr_latent"].to(DEVICE).to(DTYPE)
            lr_img = batch["lr_img"].to(DEVICE).to(DTYPE)
            
            # Encode LR (On-the-fly)
            with torch.no_grad():
                dist = vae.encode(lr_img).latent_dist
                lr_latent = dist.sample() * vae.config.scaling_factor
            
            # Noise
            current_bs = hr_latent.shape[0]
            t = torch.randint(0, 1000, (current_bs,), device=DEVICE).long()
            noise = torch.randn_like(hr_latent)
            noisy_input = diffusion.q_sample(hr_latent, t, noise)
            
            batch_y_embed = y_embed.repeat(current_bs, 1, 1, 1)
            batch_data_info = {k: v.repeat(current_bs, 1) for k, v in data_info.items()}
            
            # Forward
            adapter_cond = adapter(lr_latent.float())
            
            with torch.cuda.amp.autocast():
                model_out = pixart(
                    noisy_input, t, batch_y_embed, 
                    data_info=batch_data_info, 
                    adapter_cond=adapter_cond, 
                    injection_mode='hybrid'
                )
                if model_out.shape[1] == 8: model_out, _ = model_out.chunk(2, dim=1)
                
                # Loss Calculation
                loss_mse = F.mse_loss(model_out, noise)
                
                # Texture Loss (Low VRAM alternative to LPIPS)
                pred_x0 = predict_x0(noisy_input.float(), model_out.float(), t)
                loss_tex = latent_texture_loss(pred_x0, hr_latent.float())
                
                loss = loss_mse + TEXTURE_LOSS_WEIGHT * loss_tex
                
                # Gradient Accumulation
                loss = loss / GRAD_ACCUM_STEPS
            
            scaler.scale(loss).backward()
            
            if (i + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # Logging
            if isinstance(adapter_cond, list):
                adapter_abs_mean = np.mean([f.abs().mean().item() for f in adapter_cond])
            else:
                adapter_abs_mean = adapter_cond.abs().mean().item()
            pbar.set_postfix({"Loss": f"{loss.item() * GRAD_ACCUM_STEPS:.5f}", "Adp": f"{adapter_abs_mean:.5f}"})

        # --- Validation & Save ---
        if (epoch + 1) % SAVE_INTERVAL_EPOCH == 0:
            # 1. Save
            save_checkpoint(epoch, adapter, pixart, optimizer, scaler)
            
            # 2. Validate
            print(f"   🔍 Running Full Validation...")
            gc.collect()
            torch.cuda.empty_cache()
            
            run_production_validation(epoch, pixart, adapter, vae, valid_loader, y_embed, data_info)

def run_production_validation(epoch, model, adapter, vae, val_loader, y_embed, data_info):
    model.eval()
    adapter.eval()
    
    # [核心修正] 1. 手动实例化 Scheduler，避免从文件加载配置出错
    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
        solver_order=2
    )
    
    start_t = int(1000 * SDE_STRENGTH)
    
    total_psnr, total_ssim, total_lpips = 0.0, 0.0, 0.0
    count = 0
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_loader, desc="Validating")):
            # [核心修正] 2. 每次处理图片前重置 Timesteps，清除 DPM 历史状态
            scheduler.set_timesteps(20)
            timesteps = [t for t in scheduler.timesteps if t <= start_t]
            t_start_tensor = torch.tensor([start_t], device=DEVICE).long()

            hr_img = batch['hr_img'].to(DEVICE).to(DTYPE)
            lr_img = batch['lr_img'].to(DEVICE).to(DTYPE)
            
            # Encode
            dist = vae.encode(lr_img).latent_dist
            lr_latent = dist.sample() * vae.config.scaling_factor
            
            # Noise
            g = torch.Generator(DEVICE).manual_seed(42 + idx)
            latents = lr_latent.clone()
            noise = torch.randn(latents.shape, generator=g, device=DEVICE, dtype=latents.dtype)
            latents = scheduler.add_noise(latents, noise, t_start_tensor)
            
            # Inference
            cond = adapter(lr_latent.float())
            for t in timesteps:
                t_tensor = t.unsqueeze(0).to(DEVICE)
                with torch.cuda.amp.autocast():
                    out = model(latents, t_tensor, y_embed, data_info=data_info, adapter_cond=cond, injection_mode='hybrid')
                if out.shape[1] == 8: out, _ = out.chunk(2, dim=1)
                latents = scheduler.step(out, t, latents).prev_sample
            
            # Decode & Metrics
            if USE_METRICS or idx == 0:
                pred_img = vae.decode(latents / vae.config.scaling_factor).sample
                pred_img = torch.clamp(pred_img, -1.0, 1.0)
                
                if USE_METRICS:
                    p_01 = (pred_img.float() / 2 + 0.5).clamp(0, 1)
                    g_01 = (hr_img.float() / 2 + 0.5).clamp(0, 1)
                    total_psnr += psnr(p_01, g_01, data_range=1.0).item()
                    total_ssim += ssim(p_01, g_01, data_range=1.0).item()
                    total_lpips += val_lpips_fn(pred_img, hr_img).item()
                
                # Save First Image
                if idx == 0:
                    pred_np = (pred_img[0].permute(1, 2, 0).detach().cpu().float().numpy() / 2 + 0.5).clip(0, 1)
                    lr_np = ((lr_img[0].cpu().permute(1, 2, 0).float().numpy() + 1) / 2).clip(0, 1)
                    plt.figure(figsize=(10, 5))
                    plt.subplot(1, 2, 1); plt.imshow(lr_np); plt.title("Input LR")
                    plt.subplot(1, 2, 2); plt.imshow(pred_np); plt.title(f"Ep {epoch+1}")
                    plt.axis('off')
                    plt.savefig(f"{VIS_DIR}/ep{epoch+1:04d}_sample.png", bbox_inches='tight')
                    plt.close()
            
            count += 1

    # Log Average
    if count > 0:
        print(f"   📊 Val Results (Ep {epoch+1}): PSNR={total_psnr/count:.2f}, SSIM={total_ssim/count:.4f}, LPIPS={total_lpips/count:.4f}")
        with open(os.path.join(CHECKPOINT_DIR, "val_metrics.txt"), "a") as f:
            f.write(f"Epoch {epoch+1}: PSNR={total_psnr/count:.4f}, SSIM={total_ssim/count:.4f}, LPIPS={total_lpips/count:.4f}\n")

def save_checkpoint(epoch, adapter, pixart, optimizer, scaler):
    save_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1:04d}.pth")
    pixart_trainable = {
        'injection_scales': pixart.injection_scales.state_dict(),
        'adapter_proj': pixart.adapter_proj.state_dict(),
        'adapter_norm': pixart.adapter_norm.state_dict(),
        'cross_attn_scale': pixart.cross_attn_scale,
    }
    if hasattr(pixart, 'input_adapter_ln'):
        pixart_trainable['input_adapter_ln'] = pixart.input_adapter_ln.state_dict()

    torch.save({
        'epoch': epoch,
        'adapter_state_dict': adapter.state_dict(),
        'pixart_trainable': pixart_trainable,
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
    }, save_path)

if __name__ == "__main__":
    train_full_production()