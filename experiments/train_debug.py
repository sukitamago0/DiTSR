import torch
import torch.nn as nn
import torch.nn.functional as F
# [1] 恢复使用 DPMSolverMultistepScheduler
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler
from torch.cuda.amp import GradScaler
from torch.utils.data import Dataset, DataLoader
import os
import sys
import glob
import matplotlib
matplotlib.use('Agg') # 强制后台绘图
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import gc
from PIL import Image
from torchvision import transforms

# -----------------------------------------------------------------------------
# 1. 环境与配置
# -----------------------------------------------------------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
torch.backends.cudnn.enabled = False

# 路径配置
TRAIN_DATASET_DIR = "../dataset/DIV2K_train_latents/" 
VALID_DATASET_DIR = "../dataset/DIV2K_valid_HR/"
CHECKPOINT_DIR = "../output/checkpoints/sanity_check/" 
VIS_DIR = os.path.join(CHECKPOINT_DIR, "vis")

PIXART_PATH = "../output/pretrained_models/PixArt-XL-2-512x512.pth"
VAE_PATH = "../output/pretrained_models/sd-vae-ft-ema"
T5_EMBED_PATH = "../output/quality_embed.pth" 

DEVICE = "cuda"
DTYPE = torch.float16

# [极速验证配置]
NUM_EPOCHS = 2          # 跑2轮确保稳定性
BATCH_SIZE = 1          
GRAD_ACCUM_STEPS = 1    
NUM_WORKERS = 0         
LR_ADAPTER = 1e-5       
LR_SCALES = 1e-4         
SDE_STRENGTH = 0.6      
TEXTURE_LOSS_WEIGHT = 0.1

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from diffusion.model.nets.PixArtMS import PixArtMS_XL_2
    from diffusion.model.nets.adapter import MultiLevelAdapter
    # [2] 补全 IDDPM 导入
    from diffusion import IDDPM
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# ==============================================================================
# 数据集 (限制样本数)
# ==============================================================================
class MiniTrainDataset(Dataset):
    def __init__(self, root_dir):
        self.files = sorted(glob.glob(os.path.join(root_dir, "*.pt")))
        # [强制限制] 只取前 4 个样本
        if len(self.files) > 0:
            self.files = self.files[:4]
            print(f"🐛 [Sanity Check] Loaded {len(self.files)} training samples.")
        else:
            raise ValueError(f"❌ No .pt files in {root_dir}")

    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location="cpu")
        return {'hr_latent': data['hr_latent'], 'lr_img': data['lr_img']}

class MiniValidDataset(Dataset):
    def __init__(self, root_dir):
        self.files = sorted(glob.glob(os.path.join(root_dir, "*.png")) + glob.glob(os.path.join(root_dir, "*.jpg")))
        # [强制限制] 只取前 2 个样本 (测试循环是否会崩)
        if len(self.files) > 0:
            self.files = self.files[:2]
            print(f"🐛 [Sanity Check] Loaded {len(self.files)} validation samples.")
        else:
            print(f"⚠️ Warning: No validation images found in {root_dir}")
        
        self.transform = transforms.Compose([
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        img_path = self.files[idx]
        hr_img = Image.open(img_path).convert("RGB")
        hr_tensor = self.transform(hr_img)
        
        lr_small = F.interpolate(hr_tensor.unsqueeze(0), scale_factor=0.25, mode='bicubic', align_corners=False)
        lr_tensor = F.interpolate(lr_small, size=(512, 512), mode='bicubic', align_corners=False).squeeze(0)
        
        return {'hr_img': hr_tensor, 'lr_img': lr_tensor, 'name': os.path.basename(img_path)}

# ==============================================================================
# 核心流程
# ==============================================================================
def run_sanity_check():
    print(f"\n🚀 Start Sanity Check (DPM-Solver Robustness Test)")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)
    
    # --- 1. Init ---
    print("   Loading PixArt...")
    pixart = PixArtMS_XL_2(input_size=64).to(DEVICE).to(DTYPE).train()
    
    if os.path.exists(PIXART_PATH):
        ckpt = torch.load(PIXART_PATH, map_location="cpu")
        if "state_dict" in ckpt: ckpt = ckpt["state_dict"]
        if "pos_embed" in ckpt: del ckpt["pos_embed"]
        pixart.load_state_dict(ckpt, strict=False)
    
    if hasattr(pixart, 'enable_gradient_checkpointing'): pixart.enable_gradient_checkpointing()
    else: pixart.gradient_checkpointing = True
    
    print("   Loading Adapter...")
    adapter = MultiLevelAdapter(in_channels=4, hidden_size=1152).to(DEVICE).train()
    
    # FP32 & Freeze
    for p in pixart.parameters(): p.requires_grad = False
    
    pixart.injection_scales = pixart.injection_scales.to(torch.float32)
    pixart.adapter_proj = pixart.adapter_proj.to(torch.float32)
    pixart.adapter_norm = pixart.adapter_norm.to(torch.float32)
    pixart.cross_attn_scale.data = pixart.cross_attn_scale.data.float()
    if hasattr(pixart, 'input_adapter_ln'):
        pixart.input_adapter_ln = pixart.input_adapter_ln.to(torch.float32)

    # Optimizer (简化)
    params = list(adapter.parameters()) + \
             list(pixart.injection_scales.parameters()) + \
             list(pixart.adapter_proj.parameters()) + \
             list(pixart.adapter_norm.parameters()) + \
             [pixart.cross_attn_scale]
    if hasattr(pixart, 'input_adapter_ln'):
        params += list(pixart.input_adapter_ln.parameters())
        
    optimizer = torch.optim.AdamW(params, lr=LR_ADAPTER)
    scaler = GradScaler()
    diffusion = IDDPM(str(1000))

    # --- 2. Data ---
    train_loader = DataLoader(MiniTrainDataset(TRAIN_DATASET_DIR), batch_size=BATCH_SIZE)
    valid_loader = DataLoader(MiniValidDataset(VALID_DATASET_DIR), batch_size=1)
    
    print("   Loading VAE...")
    vae = AutoencoderKL.from_pretrained(VAE_PATH, local_files_only=True).to(DEVICE).to(DTYPE)
    vae.eval()
    
    y_embed = torch.load(T5_EMBED_PATH, map_location="cpu")["prompt_embeds"].unsqueeze(1).to(DEVICE).to(DTYPE)
    data_info = {'img_hw': torch.tensor([[512., 512.]]).to(DEVICE).to(DTYPE), 'aspect_ratio': torch.tensor([1.]).to(DEVICE).to(DTYPE)}

    # 辅助函数
    def predict_x0(x_t, eps, t):
        acp = torch.from_numpy(diffusion.alphas_cumprod).to(x_t.device).to(x_t.dtype)
        at = acp[t]
        return (x_t - (1 - at).sqrt().view(-1, 1, 1, 1) * eps) / at.sqrt().view(-1, 1, 1, 1)

    def latent_texture_loss(pred, target):
        p_dx = pred[..., 1:] - pred[..., :-1]; p_dy = pred[..., 1:, :] - pred[..., :-1, :]
        t_dx = target[..., 1:] - target[..., :-1]; t_dy = target[..., 1:, :] - target[..., :-1, :]
        return F.l1_loss(p_dx, t_dx) + F.l1_loss(p_dy, t_dy)

    # --- 3. Loop ---
    print("\n🏁 Start Dummy Training Loop...")
    for epoch in range(NUM_EPOCHS):
        # 模拟训练
        pixart.train()
        adapter.train()
        for batch in train_loader:
            optimizer.zero_grad()
            
            hr_latent = batch["hr_latent"].to(DEVICE).to(DTYPE)
            lr_img = batch["lr_img"].to(DEVICE).to(DTYPE)
            
            with torch.no_grad():
                lr_latent = vae.encode(lr_img).latent_dist.sample() * vae.config.scaling_factor
            
            t = torch.randint(0, 1000, (hr_latent.shape[0],), device=DEVICE).long()
            noise = torch.randn_like(hr_latent)
            noisy_input = diffusion.q_sample(hr_latent, t, noise)
            
            adapter_cond = adapter(lr_latent.float())
            
            with torch.cuda.amp.autocast():
                model_out = pixart(noisy_input, t, y_embed, data_info=data_info, adapter_cond=adapter_cond, injection_mode='hybrid')
                if model_out.shape[1] == 8: model_out, _ = model_out.chunk(2, dim=1)
                
                loss_mse = F.mse_loss(model_out, noise)
                pred_x0 = predict_x0(noisy_input.float(), model_out.float(), t)
                loss_tex = latent_texture_loss(pred_x0, hr_latent.float())
                loss = loss_mse + TEXTURE_LOSS_WEIGHT * loss_tex
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            print(f"   Step Done. Loss: {loss.item():.4f}")

        # --- 4. Validation (CRITICAL PART) ---
        print(f"   🔍 Running Validation (DPM-Solver)...")
        run_robust_validation(epoch, pixart, adapter, vae, valid_loader, y_embed, data_info)
        
    print("\n✅ Sanity Check Passed! Flow is safe.")

def run_robust_validation(epoch, model, adapter, vae, val_loader, y_embed, data_info):
    model.eval()
    adapter.eval()
    
    # [恢复] 使用 DPMSolver
    # 只要不使用 from_pretrained，就不会报错
    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085, 
        beta_end=0.012, 
        beta_schedule="scaled_linear",
        solver_order=2 # DPM 是二阶的
    )
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            # [核心修复] 在每次循环开始时重置 timesteps
            # 这会清空 scheduler 内部的 step_index 和 history，防止越界
            scheduler.set_timesteps(20)
            
            start_t = int(1000 * SDE_STRENGTH)
            timesteps = [t for t in scheduler.timesteps if t <= start_t]
            t_start_tensor = torch.tensor([start_t], device=DEVICE).long()
            
            lr_img = batch['lr_img'].to(DEVICE).to(DTYPE)
            dist = vae.encode(lr_img).latent_dist
            lr_latent = dist.sample() * vae.config.scaling_factor
            
            g = torch.Generator(DEVICE).manual_seed(42 + i)
            latents = lr_latent.clone()
            noise = torch.randn(latents.shape, generator=g, device=DEVICE, dtype=latents.dtype)
            
            # 使用 scheduler.add_noise
            latents = scheduler.add_noise(latents, noise, t_start_tensor)
            
            cond = adapter(lr_latent.float())
            
            print(f"      Image {i}: Denoising {len(timesteps)} steps...")
            
            for t in timesteps:
                t_tensor = torch.tensor([t], device=DEVICE)
                
                with torch.cuda.amp.autocast():
                    out = model(latents, t_tensor, y_embed, data_info=data_info, adapter_cond=cond, injection_mode='hybrid')
                if out.shape[1] == 8: out, _ = out.chunk(2, dim=1)
                
                latents = scheduler.step(out, t, latents).prev_sample
            
            print("      Decoding...")
            img = vae.decode(latents / vae.config.scaling_factor).sample
            img = (img / 2 + 0.5).clamp(0, 1)
            
            save_path = f"{VIS_DIR}/debug_ep{epoch}_img{i}.png"
            plt.imsave(save_path, img[0].permute(1, 2, 0).detach().cpu().float().numpy())
            print(f"      ✅ Saved: {save_path}")

if __name__ == "__main__":
    run_sanity_check()