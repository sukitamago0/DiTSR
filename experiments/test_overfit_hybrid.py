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

# -----------------------------------------------------------------------------
# 1. 环境与指标库设置
# -----------------------------------------------------------------------------
try:
    from torchmetrics.functional import peak_signal_noise_ratio as psnr
    from torchmetrics.functional import structural_similarity_index_measure as ssim
    import lpips
    USE_METRICS = True
    print("✅ Metrics libraries loaded (PSNR, SSIM, LPIPS).")
except ImportError:
    USE_METRICS = False
    print("⚠️ Metrics libraries missing. Install via `pip install torchmetrics lpips`. Only Loss will be logged.")

torch.backends.cudnn.enabled = False

# ================= 🔧 实验配置 =================
LATENT_FILE = "../dataset/DIV2K_train_latents/0002_y256_x256.pt" 
PIXART_PATH = "../output/pretrained_models/PixArt-XL-2-512x512.pth"
VAE_PATH = "../output/pretrained_models/sd-vae-ft-ema"
T5_EMBED_PATH = "../output/quality_embed.pth" 

DEVICE = "cuda"
DTYPE = torch.float16
STEPS = 2000            
LR_ADAPTER = 1e-5        # [调整] Adapter 本体给大一点的学习率
LR_SCALES = 1e-4         # [调整] Scales 给大一点，让它敢于从 1.0 变动
SAVE_INTERVAL = 50      
SDE_STRENGTH = 0.5      
# ===========================================

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from diffusion.model.nets.PixArtMS import PixArtMS_XL_2
    from diffusion.model.nets.adapter import ProgressiveFrequencyAdapter
    from diffusion import IDDPM
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

if USE_METRICS:
    lpips_fn = lpips.LPIPS(net='vgg').to(DEVICE).eval()
    for p in lpips_fn.parameters(): p.requires_grad = False

def run_hybrid_experiment():
    print(f"\n🚀 开始 Hybrid SDE 过拟合测试 (v3.1: Seed Fixed & Activation Check)")
    
    # -------------------------------------------------------------------------
    # 2. 模型初始化
    # -------------------------------------------------------------------------
    print("   Loading PixArt...")
    pixart = PixArtMS_XL_2(input_size=64).to(DEVICE).to(DTYPE).train()
    ckpt = torch.load(PIXART_PATH, map_location="cpu")
    if "state_dict" in ckpt: ckpt = ckpt["state_dict"]
    if "pos_embed" in ckpt: del ckpt["pos_embed"]
    pixart.load_state_dict(ckpt, strict=False)
    
    print("   Loading Adapter...")
    # 确保 adapter.py 中的 in_channels=4 且 fusion 有 stride=2
    adapter = ProgressiveFrequencyAdapter(in_channels=4, hidden_size=1152).to(DEVICE).train()
    
    # -------------------------------------------------------------------------
    # 3. 参数冻结与分组
    # -------------------------------------------------------------------------
    for p in pixart.parameters(): p.requires_grad = False
    
    # 显式转 FP32
    pixart.injection_scales = pixart.injection_scales.to(torch.float32)
    pixart.adapter_proj = pixart.adapter_proj.to(torch.float32)
    pixart.adapter_norm = pixart.adapter_norm.to(torch.float32)
    pixart.cross_attn_scale.data = pixart.cross_attn_scale.data.float()

    # 分组优化参数 (不同的 LR)
    adapter_params = list(adapter.parameters())
    
    scale_params = []
    for scale in pixart.injection_scales:
        scale.requires_grad = True
        scale_params.append(scale)
    pixart.cross_attn_scale.requires_grad = True
    scale_params.append(pixart.cross_attn_scale)
    
    proj_params = []
    for p in pixart.adapter_proj.parameters(): p.requires_grad = True; proj_params.append(p)
    for p in pixart.adapter_norm.parameters(): p.requires_grad = True; proj_params.append(p)

    print(f"   🔥 可训练: Adapter({len(adapter_params)}) | Scales({len(scale_params)}) | Proj({len(proj_params)})")
    
    # 使用 Parameter Groups
    optimizer = torch.optim.AdamW([
        {'params': adapter_params, 'lr': LR_ADAPTER},
        {'params': proj_params, 'lr': LR_ADAPTER},
        {'params': scale_params, 'lr': LR_SCALES} # Scale 跑快点
    ])
    
    scaler = GradScaler()
    diffusion = IDDPM(str(1000))

    # -------------------------------------------------------------------------
    # 4. 数据预处理
    # -------------------------------------------------------------------------
    if not os.path.exists(LATENT_FILE): print("❌ 数据缺失"); return
    data = torch.load(LATENT_FILE)
    hr_latent = data["hr_latent"].unsqueeze(0).to(DEVICE).to(DTYPE)
    lr_latent_input = None
    lr_img = None

    if "lr_latent" in data:
        lr_latent_input = data["lr_latent"].unsqueeze(0).to(DEVICE).float()
    elif "lr_img" in data:
        lr_img = data["lr_img"].unsqueeze(0).to(DEVICE).float()
    else:
        print("❌ 缺少 lr_latent 或 lr_img，无法继续")
        return

    vae = AutoencoderKL.from_pretrained(VAE_PATH, local_files_only=True).to("cpu").float()

    with torch.no_grad():
        if lr_latent_input is None:
            print("   Encoding LR image to Latent...")
            dist = vae.encode(lr_img.cpu()).latent_dist
            lr_latent_input = dist.sample() * vae.config.scaling_factor
            lr_latent_input = lr_latent_input.to(DEVICE)
        elif lr_img is None:
            lr_img = vae.decode(lr_latent_input.cpu().float() / vae.config.scaling_factor).sample.to(DEVICE)
            lr_img = torch.clamp((lr_img + 1.0) / 2.0, 0.0, 1.0)

        hr_img_gt = vae.decode(hr_latent.cpu().float() / vae.config.scaling_factor).sample.to(DEVICE)
        hr_img_gt = torch.clamp((hr_img_gt + 1.0) / 2.0, 0.0, 1.0)

    y_embed = torch.load(T5_EMBED_PATH, map_location="cpu")["prompt_embeds"].unsqueeze(1).to(DEVICE).to(DTYPE)
    data_info = {'img_hw': torch.tensor([[512., 512.]]).to(DEVICE).to(DTYPE), 'aspect_ratio': torch.tensor([1.]).to(DEVICE).to(DTYPE)}

    losses = []
    
    # -------------------------------------------------------------------------
    # 5. 训练循环
    # -------------------------------------------------------------------------
    pbar = tqdm(range(STEPS))
    for step in pbar:
        optimizer.zero_grad()
        
        t = torch.randint(0, 1000, (1,), device=DEVICE).long()
        noise = torch.randn_like(hr_latent)
        noisy_input = diffusion.q_sample(hr_latent, t, noise)
        
        # 1. Adapter Forward
        adapter_cond = adapter(lr_latent_input.float()) 
        
        # 2. PixArt Forward
        with torch.cuda.amp.autocast():
            model_out = pixart(
                noisy_input, t, y_embed, 
                data_info=data_info, 
                adapter_cond=adapter_cond, 
                injection_mode='hybrid'
            )
            if model_out.shape[1] == 8: model_out, _ = model_out.chunk(2, dim=1)
            loss = F.mse_loss(model_out, noise)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        losses.append(loss.item())
        
        # [监控] 打印 Adapter 输出强度，确保它不是 0
        adapter_abs_mean = adapter_cond.abs().mean().item()
        pbar.set_postfix({"Loss": f"{loss.item():.5f}", "Adp": f"{adapter_abs_mean:.5f}"})
        
        if (step + 1) % SAVE_INTERVAL == 0:
            input_scales = [f"{s.item():.4f}" for s in pixart.injection_scales]
            attn_scale = pixart.cross_attn_scale.item()
            
            # [监控] 确保 Adapter 输出在变大
            print(f"\n🔍 [Debug] Adapter Activation Mean: {adapter_abs_mean:.6f}")
            if adapter_abs_mean < 1e-6 and step > 100:
                print("⚠️ 警告: Adapter 输出依然接近 0，梯度可能未传导成功！")

            metrics = evaluate_and_save(
                pixart, adapter, vae, 
                lr_latent_input, lr_img, y_embed, data_info, 
                step, hr_img_gt
            )
            
            print(f"📊 Step {step+1} Report:")
            print(f"   📉 Loss: {loss.item():.6f}")
            print(f"   🎛️ Input Scales: {input_scales}")
            print(f"   🎛️ Cross-Attn Scale: {attn_scale:.4f}")
            if metrics:
                print(f"   🖼️ PSNR: {metrics['psnr']:.2f} | SSIM: {metrics['ssim']:.4f} | LPIPS: {metrics['lpips']:.4f}")
            print("-" * 50)

    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.savefig("../experiments_results/overfit_hybrid_v3/loss.png")

def evaluate_and_save(model, adapter, vae, lr_latent_input, lr_img, y_embed, data_info, step, hr_img_gt):
    model.eval(); adapter.eval()
    save_dir = "../experiments_results/overfit_hybrid_v3/vis"
    os.makedirs(save_dir, exist_ok=True)
    
    scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000, solver_order=2)
    scheduler.set_timesteps(20)
    
    start_t_val = int(1000 * SDE_STRENGTH)
    # 获取需要的时间步
    timesteps = [t for t in scheduler.timesteps if t <= start_t_val]
    
    # 构造 Latent
    # [关键修复] 必须使用 Generator 锁死噪声，否则每次都是不同的随机噪声，看不出变化
    g = torch.Generator(DEVICE).manual_seed(42)
    
    latents = lr_latent_input.clone().to(DTYPE)
    # 使用 generator=g 生成固定的噪声
    noise = torch.randn(latents.shape, generator=g, device=DEVICE, dtype=latents.dtype)
    
    t_start_tensor = torch.tensor([start_t_val], device=DEVICE).long()
    latents = scheduler.add_noise(latents, noise, t_start_tensor)
    
    with torch.no_grad():
        cond = adapter(lr_latent_input.float())
        
        for t in timesteps:
            t_tensor = t.unsqueeze(0).to(DEVICE)
            with torch.cuda.amp.autocast():
                out = model(latents, t_tensor, y_embed, data_info=data_info, adapter_cond=cond, injection_mode='hybrid')
            
            if out.shape[1] == 8: out, _ = out.chunk(2, dim=1)
            latents = scheduler.step(out, t, latents).prev_sample
            
        pred_img = vae.decode(latents.cpu().float() / vae.config.scaling_factor).sample.to(DEVICE)
        pred_img = torch.clamp((pred_img + 1.0) / 2.0, 0.0, 1.0)

    metrics = {}
    if USE_METRICS:
        metrics['psnr'] = psnr(pred_img, hr_img_gt, data_range=1.0).item()
        metrics['ssim'] = ssim(pred_img, hr_img_gt, data_range=1.0).item()
        pred_norm = pred_img * 2.0 - 1.0
        gt_norm = hr_img_gt * 2.0 - 1.0
        metrics['lpips'] = lpips_fn(pred_norm, gt_norm).item()

    pred_np = pred_img[0].permute(1, 2, 0).detach().cpu().numpy()
    lr_np = ((lr_img[0].cpu().permute(1, 2, 0) + 1) / 2).clamp(0, 1).numpy()
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1); plt.imshow(lr_np); plt.title("Input LR")
    plt.subplot(1, 2, 2); plt.imshow(pred_np); plt.title(f"Step {step+1}")
    plt.axis('off')
    plt.savefig(f"{save_dir}/step_{step+1:04d}.png", bbox_inches='tight')
    plt.close()
    
    model.train(); adapter.train()
    return metrics

if __name__ == "__main__":
    run_hybrid_experiment()
