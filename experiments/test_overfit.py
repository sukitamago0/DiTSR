import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler
from torch.cuda.amp import GradScaler
import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# ================= 🛑 硬件约束 =================
torch.backends.cudnn.enabled = False
# ===============================================

# ================= 🔧 SDE SR 核心配置 =================
LATENT_FILE = "../dataset/DIV2K_train_latents/0001_y0_x0.pt" 
PIXART_PATH = "../output/pretrained_models/PixArt-XL-2-512x512.pth"
VAE_PATH = "../output/pretrained_models/sd-vae-ft-ema"
T5_EMBED_PATH = "../output/quality_embed.pth" 

DEVICE = "cuda"
DTYPE = torch.float16
STEPS = 500             # 稍微增加步数确保收敛
LR = 5e-5               # 学习率
SAVE_INTERVAL = 50 

# [核心修正] SDE 强度
# 0.0 = 原图, 1.0 = 纯噪声
# SR 任务通常在 0.4 - 0.7 之间
SDE_STRENGTH = 0.5 
# ====================================================

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from diffusion.model.nets.PixArtMS import PixArtMS_XL_2
    from diffusion.model.nets.adapter import ProgressiveFrequencyAdapter
    from diffusion import IDDPM
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

def run_experiment(injection_mode, experiment_name):
    print(f"\n🚀 开始 SDE SR 过拟合实验: {experiment_name} (Mode: {injection_mode})")
    
    # 1. 初始化模型
    pixart = PixArtMS_XL_2(input_size=64).to(DEVICE).to(DTYPE).train()
    ckpt = torch.load(PIXART_PATH, map_location="cpu")
    if "state_dict" in ckpt: ckpt = ckpt["state_dict"]
    if "pos_embed" in ckpt: del ckpt["pos_embed"]
    pixart.load_state_dict(ckpt, strict=False)
    for p in pixart.parameters(): p.requires_grad = False # 冻结主干
    
    adapter = ProgressiveFrequencyAdapter(in_channels=3, hidden_size=1152).to(DEVICE).train()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=LR)
    scaler = GradScaler()
    diffusion = IDDPM(str(1000))

    # 2. 数据加载
    if not os.path.exists(LATENT_FILE): return []
    data = torch.load(LATENT_FILE)
    hr_latent = data["hr_latent"].unsqueeze(0).to(DEVICE).to(DTYPE) 
    lr_img = data["lr_img"].unsqueeze(0).to(DEVICE).float() # Adapter输入必须是fp32
    
    # 我们也需要 LR 的 Latent 作为 SDE 的起点
    # 这里我们直接从 LR 图片 encode 得到 (模拟真实推理流程)
    vae = AutoencoderKL.from_pretrained(VAE_PATH, local_files_only=True).to("cpu").float()
    with torch.no_grad():
        # 简单的编码 LR
        lr_latent_base = vae.encode(lr_img.cpu()).latent_dist.sample() * vae.config.scaling_factor
        lr_latent_base = lr_latent_base.to(DEVICE).to(DTYPE)

    y_embed = torch.load(T5_EMBED_PATH, map_location="cpu")["prompt_embeds"].unsqueeze(1).to(DEVICE).to(DTYPE)
    data_info = {'img_hw': torch.tensor([[512., 512.]]).to(DEVICE).to(DTYPE), 'aspect_ratio': torch.tensor([1.]).to(DEVICE).to(DTYPE)}

    losses = []
    
    # 3. 训练循环
    pbar = tqdm(range(STEPS))
    for step in pbar:
        optimizer.zero_grad()
        
        # [SDE 修正] 训练时，我们依然覆盖全时间步 [0, 1000]，这能让 Adapter 更鲁棒
        # 但在 SR 任务中，我们也可以偏向于训练 [0, 800] 区间
        t = torch.randint(0, 1000, (1,), device=DEVICE).long()
        
        noise = torch.randn_like(hr_latent)
        noisy_input = diffusion.q_sample(hr_latent, t, noise)
        
        # Adapter Forward
        adapter_cond = adapter(lr_img).to(DTYPE)
        
        with torch.cuda.amp.autocast():
            model_out = pixart(
                noisy_input, t, y_embed, 
                data_info=data_info, 
                adapter_cond=adapter_cond, 
                injection_mode=injection_mode
            )
            if model_out.shape[1] == 8: model_out, _ = model_out.chunk(2, dim=1)
            loss = F.mse_loss(model_out, noise)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())
        pbar.set_postfix({"Loss": f"{loss.item():.6f}"})
        
        # 4. 可视化 (这是修正的重点)
        if (step + 1) % SAVE_INTERVAL == 0:
            # 传入 lr_latent_base，以此为起点加噪
            save_sde_progress(pixart, adapter, vae, lr_latent_base, lr_img, y_embed, data_info, step, experiment_name, injection_mode)

    return losses

def save_sde_progress(model, adapter, vae, lr_latent_base, lr_img, y_embed, data_info, step, exp_name, mode):
    model.eval(); adapter.eval()
    save_dir = f"../experiments_results/overfit_sde/{exp_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000, solver_order=2)
    scheduler.set_timesteps(20)
    
    # [核心修正] SDE 生成逻辑
    # 1. 确定起始时间步
    start_timestep = int(1000 * SDE_STRENGTH)
    # 找到 Scheduler 中最接近的时间点
    timesteps = scheduler.timesteps
    start_idx = 0
    for i, t in enumerate(timesteps):
        if t <= start_timestep:
            start_idx = i
            break
    
    target_timesteps = timesteps[start_idx:]
    actual_start_t = target_timesteps[0]
    
    # 2. 构造起点: LR Latent + Noise(t)
    g = torch.Generator(DEVICE).manual_seed(42)
    noise = torch.randn_like(lr_latent_base)
    
    # 手动加噪公式: x_t = sqrt(alpha_cumprod)*x_0 + sqrt(1-alpha_cumprod)*noise
    # 简单起见，我们使用 scheduler 的 add_noise (如果支持) 或 diffusers 的标准 q_sample
    # 这里为了通用性，我们模拟加噪:
    # 注意: DPM Solver 对 alpha 的定义可能不同，这里我们使用简化的加噪逻辑用于可视化验证
    # 更严谨的做法是实例化一个 IDDPM 来做 q_sample，这里直接用线性插值模拟 SDE 强度
    
    # 简单 SDE 模拟: latents = (1-strength)*LR + strength*Noise
    # 这不是严格的 Diffusion 公式，但在可视化过拟合效果时足够验证 Adapter 是否起作用
    # 严格做法是:
    latents = scheduler.add_noise(lr_latent_base, noise, torch.tensor([actual_start_t]))

    with torch.no_grad():
        cond = adapter(lr_img).to(DTYPE)
        
        # 从 start_t 开始去噪
        for t in target_timesteps:
            t_tensor = t.unsqueeze(0).to(DEVICE)
            out = model(latents, t_tensor, y_embed, data_info=data_info, adapter_cond=cond, injection_mode=mode)
            if out.shape[1] == 8: out, _ = out.chunk(2, dim=1)
            latents = scheduler.step(out, t, latents).prev_sample
            
    # Decode
    img = vae.decode(latents.cpu().float() / vae.config.scaling_factor).sample
    
    # Plot
    img_np = ((img[0].permute(1, 2, 0).detach().cpu().numpy() + 1) / 2).clip(0, 1)
    lr_np = ((lr_img[0].cpu().permute(1, 2, 0).detach().numpy() + 1) / 2).clip(0, 1)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1); plt.imshow(lr_np); plt.title("Input LR")
    plt.subplot(1, 2, 2); plt.imshow(img_np); plt.title(f"SDE {SDE_STRENGTH} (Step {step+1})")
    plt.savefig(f"{save_dir}/step_{step+1:04d}.png")
    plt.close()
    
    model.train(); adapter.train()

def main():
    # 实验 A: Input Injection
    loss_a = run_experiment(injection_mode="input", experiment_name="sde_input_gate")
    
    # 实验 B: Cross-Attn Injection
    loss_b = run_experiment(injection_mode="cross_attn", experiment_name="sde_cross_attn")
    
    # 绘制 Loss
    if loss_a and loss_b:
        plt.figure(figsize=(10, 6))
        plt.plot(loss_a, label="Input Injection")
        plt.plot(loss_b, label="Cross-Attn Injection")
        plt.title(f"SDE Overfitting (Strength={SDE_STRENGTH})")
        plt.legend(); plt.grid(True)
        plt.savefig("../experiments_results/overfit_sde/loss_comparison.png")

if __name__ == "__main__":
    main()