# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from diffusers import AutoencoderKL, DPMSolverMultistepScheduler
# from torch.cuda.amp import GradScaler
# import os
# import sys
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import numpy as np

# # -----------------------------------------------------------------------------
# # 1. 环境与指标库设置
# # -----------------------------------------------------------------------------
# try:
#     from torchmetrics.functional import peak_signal_noise_ratio as psnr
#     from torchmetrics.functional import structural_similarity_index_measure as ssim
#     import lpips
#     USE_METRICS = True
#     print("✅ Metrics libraries loaded (PSNR, SSIM, LPIPS).")
# except ImportError:
#     USE_METRICS = False
#     print("⚠️ Metrics libraries missing. Install via `pip install torchmetrics lpips`. Only Loss will be logged.")

# torch.backends.cudnn.enabled = False

# # ================= 🔧 实验配置 =================
# LATENT_FILE = "../dataset/DIV2K_train_latents/0002_y256_x256.pt" 
# PIXART_PATH = "../output/pretrained_models/PixArt-XL-2-512x512.pth"
# VAE_PATH = "../output/pretrained_models/sd-vae-ft-ema"
# T5_EMBED_PATH = "../output/quality_embed.pth" 

# DEVICE = "cuda"
# DTYPE = torch.float16
# STEPS = 2000            
# LR_ADAPTER = 1e-5        # Adapter 本体学习率
# LR_SCALES = 1e-4         # Scale 学习率
# SAVE_INTERVAL = 50      
# SDE_STRENGTH = 0.6      
# # ===========================================

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# try:
#     from diffusion.model.nets.PixArtMS import PixArtMS_XL_2
#     from diffusion.model.nets.adapter import MultiLevelAdapter
#     from diffusion import IDDPM
# except ImportError as e:
#     print(f"❌ 导入失败: {e}")
#     sys.exit(1)

# if USE_METRICS:
#     lpips_fn = lpips.LPIPS(net='vgg').to(DEVICE).eval()
#     for p in lpips_fn.parameters(): p.requires_grad = False

# def run_hybrid_experiment():
#     print(f"\n🚀 开始 Hybrid SDE 过拟合测试 (v3.3: Added LayerNorm Support)")
    
#     # -------------------------------------------------------------------------
#     # 2. 模型初始化
#     # -------------------------------------------------------------------------
#     print("   Loading PixArt...")
#     pixart = PixArtMS_XL_2(input_size=64).to(DEVICE).to(DTYPE).train()
#     ckpt = torch.load(PIXART_PATH, map_location="cpu")
#     if "state_dict" in ckpt: ckpt = ckpt["state_dict"]
#     if "pos_embed" in ckpt: del ckpt["pos_embed"]
#     pixart.load_state_dict(ckpt, strict=False)
    
#     print("   Loading Adapter...")
#     adapter = MultiLevelAdapter(in_channels=4, hidden_size=1152).to(DEVICE).train()
    
#     # -------------------------------------------------------------------------
#     # 3. 参数冻结与分组
#     # -------------------------------------------------------------------------
#     # 3.1 冻结主干
#     for p in pixart.parameters(): p.requires_grad = False
    
#     # 3.2 显式转 FP32 (包括新加的 LN)
#     pixart.injection_scales = pixart.injection_scales.to(torch.float32)
#     pixart.adapter_proj = pixart.adapter_proj.to(torch.float32)
#     pixart.adapter_norm = pixart.adapter_norm.to(torch.float32)
#     pixart.cross_attn_scale.data = pixart.cross_attn_scale.data.float()
    
#     # [关键修改] 确保新加的 Input Injection LayerNorm 也是 FP32
#     if hasattr(pixart, 'input_adapter_ln'):
#         pixart.input_adapter_ln = pixart.input_adapter_ln.to(torch.float32)
#         print("   ✅ 检测到 input_adapter_ln，已转为 FP32")
#     else:
#         print("   ⚠️ 警告: 未检测到 input_adapter_ln，请检查 PixArtMS.py 是否已修改！")

#     # 3.3 构建优化器参数
#     adapter_params = list(adapter.parameters())
    
#     scale_params = []
#     for scale in pixart.injection_scales:
#         scale.requires_grad = True
#         scale_params.append(scale)
#     pixart.cross_attn_scale.requires_grad = True
#     scale_params.append(pixart.cross_attn_scale)
    
#     proj_params = []
#     # 原有的 Cross-Attn Proj/Norm
#     for p in pixart.adapter_proj.parameters(): p.requires_grad = True; proj_params.append(p)
#     for p in pixart.adapter_norm.parameters(): p.requires_grad = True; proj_params.append(p)
    
#     # [关键修改] 将新加的 LN 参数加入优化器
#     if hasattr(pixart, 'input_adapter_ln'):
#         for p in pixart.input_adapter_ln.parameters():
#             p.requires_grad = True
#             proj_params.append(p) # 复用 proj 的学习率配置

#     print(f"   🔥 可训练: Adapter({len(adapter_params)}) | Scales({len(scale_params)}) | Proj/LN({len(proj_params)})")
    
#     optimizer = torch.optim.AdamW([
#         {'params': adapter_params, 'lr': LR_ADAPTER},
#         {'params': proj_params, 'lr': LR_ADAPTER},
#         {'params': scale_params, 'lr': LR_SCALES}
#     ])
    
#     scaler = GradScaler()
#     diffusion = IDDPM(str(1000))

#     # -------------------------------------------------------------------------
#     # 4. 数据预处理
#     # -------------------------------------------------------------------------
#     if not os.path.exists(LATENT_FILE): print("❌ 数据缺失"); return
#     data = torch.load(LATENT_FILE)
#     hr_latent = data["hr_latent"].unsqueeze(0).to(DEVICE).to(DTYPE) 
#     lr_img = data["lr_img"].unsqueeze(0).to(DEVICE).float()
    
#     vae = AutoencoderKL.from_pretrained(VAE_PATH, local_files_only=True).to("cpu").float()
    
#     with torch.no_grad():
#         print("   Encoding LR image to Latent...")
#         dist = vae.encode(lr_img.cpu()).latent_dist
#         lr_latent_input = dist.sample() * vae.config.scaling_factor
#         lr_latent_input = lr_latent_input.to(DEVICE)
        
#         # 简单的诊断打印
#         print(f"   👉 Latent Input Std: {lr_latent_input.std().item():.4f}")
        
#         hr_img_gt = vae.decode(hr_latent.cpu().float() / vae.config.scaling_factor).sample.to(DEVICE)
#         hr_img_gt = torch.clamp((hr_img_gt + 1.0) / 2.0, 0.0, 1.0) 

#     y_embed = torch.load(T5_EMBED_PATH, map_location="cpu")["prompt_embeds"].unsqueeze(1).to(DEVICE).to(DTYPE)
#     data_info = {'img_hw': torch.tensor([[512., 512.]]).to(DEVICE).to(DTYPE), 'aspect_ratio': torch.tensor([1.]).to(DEVICE).to(DTYPE)}

#     losses = []

#     # -------------------------------------------------------------------------
#     # 5. 训练循环
#     # -------------------------------------------------------------------------
#     pbar = tqdm(range(STEPS))
#     for step in pbar:
#         optimizer.zero_grad()
        
#         t = torch.randint(0, 1000, (1,), device=DEVICE).long()
#         noise = torch.randn_like(hr_latent)
#         noisy_input = diffusion.q_sample(hr_latent, t, noise)
        
#         # 1. Adapter Forward
#         adapter_cond = adapter(lr_latent_input.float()) 
        
#         # 2. PixArt Forward
#         with torch.cuda.amp.autocast():
#             model_out = pixart(
#                 noisy_input, t, y_embed, 
#                 data_info=data_info, 
#                 adapter_cond=adapter_cond, 
#                 injection_mode='hybrid'
#             )
#             if model_out.shape[1] == 8: model_out, _ = model_out.chunk(2, dim=1)
#             loss = F.mse_loss(model_out, noise)
        
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
        
#         losses.append(loss.item())
        
                
#         # 👇👇👇 [修改开始] 兼容 List 和 Tensor 的监控逻辑 👇👇👇
#         if isinstance(adapter_cond, list):
#             # 如果是多层级，计算所有层级激活值的平均值
#             adapter_abs_mean = np.mean([f.abs().mean().item() for f in adapter_cond])
#         else:
#             # 如果是单层级 Tensor
#             adapter_abs_mean = adapter_cond.abs().mean().item()
#         # 👆👆👆 [修改结束] 👆👆👆
#         pbar.set_postfix({"Loss": f"{loss.item():.5f}", "Adp": f"{adapter_abs_mean:.5f}"})
        
#         if (step + 1) % SAVE_INTERVAL == 0:
#             input_scales = [f"{s.item():.4f}" for s in pixart.injection_scales]
            
#             print(f"\n🔍 [Monitor] Step {step+1} Adapter Mean: {adapter_abs_mean:.6f}")

#             metrics = evaluate_and_save(
#                 pixart, adapter, vae, 
#                 lr_latent_input, lr_img, y_embed, data_info, 
#                 step, hr_img_gt, 
#                 hr_latent=hr_latent # 传入 GT Latent 进行对比
#             )
            
#             print(f"📊 Report:")
#             print(f"   📉 Loss: {loss.item():.6f}")
#             print(f"   🎛️ Input Scales: {input_scales}")
#             if metrics:
#                 print(f"   🖼️ PSNR: {metrics['psnr']:.2f} | SSIM: {metrics['ssim']:.4f} | LPIPS: {metrics['lpips']:.4f}")
#             print("-" * 50)

#     plt.figure(figsize=(10, 5))
#     plt.plot(losses)
#     plt.savefig("../experiments_results/overfit_hybrid_v4_kick01/loss.png")

# def evaluate_and_save(model, adapter, vae, lr_latent_input, lr_img, y_embed, data_info, step, hr_img_gt, hr_latent):
#     model.eval(); adapter.eval()
#     save_dir = "../experiments_results/overfit_hybrid_v4_kick01/vis"
#     os.makedirs(save_dir, exist_ok=True)
    
#     scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000, solver_order=2)
#     scheduler.set_timesteps(20)
    
#     start_t_val = int(1000 * SDE_STRENGTH)
#     timesteps = [t for t in scheduler.timesteps if t <= start_t_val]
    
#     # 锁死种子
#     g = torch.Generator(DEVICE).manual_seed(42)
#     latents = lr_latent_input.clone().to(DTYPE)
#     noise = torch.randn(latents.shape, generator=g, device=DEVICE, dtype=latents.dtype)
#     t_start_tensor = torch.tensor([start_t_val], device=DEVICE).long()
#     latents = scheduler.add_noise(latents, noise, t_start_tensor)
    
#     with torch.no_grad():
#         cond = adapter(lr_latent_input.float())
#         for t in timesteps:
#             t_tensor = t.unsqueeze(0).to(DEVICE)
#             with torch.cuda.amp.autocast():
#                 out = model(latents, t_tensor, y_embed, data_info=data_info, adapter_cond=cond, injection_mode='hybrid')
            
#             if out.shape[1] == 8: out, _ = out.chunk(2, dim=1)
#             latents = scheduler.step(out, t, latents).prev_sample

#         # 诊断输出
#         gen_mean = latents.float().mean().item()
#         gen_std = latents.float().std().item()
#         gt_mean = hr_latent.float().mean().item()
#         gt_std = hr_latent.float().std().item()
#         print(f"\n🧐 [DIAGNOSTIC] Final Latent Stats | Gen: u={gen_mean:.3f},s={gen_std:.3f} | GT: u={gt_mean:.3f},s={gt_std:.3f}")
            
#         pred_img = vae.decode(latents.cpu().float() / vae.config.scaling_factor).sample.to(DEVICE)
#         pred_img = torch.clamp((pred_img + 1.0) / 2.0, 0.0, 1.0)

#     metrics = {}
#     if USE_METRICS:
#         metrics['psnr'] = psnr(pred_img, hr_img_gt, data_range=1.0).item()
#         metrics['ssim'] = ssim(pred_img, hr_img_gt, data_range=1.0).item()
#         pred_norm = pred_img * 2.0 - 1.0
#         gt_norm = hr_img_gt * 2.0 - 1.0
#         metrics['lpips'] = lpips_fn(pred_norm, gt_norm).item()

#     pred_np = pred_img[0].permute(1, 2, 0).detach().cpu().numpy()
#     lr_np = ((lr_img[0].cpu().permute(1, 2, 0) + 1) / 2).clamp(0, 1).numpy()
    
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1); plt.imshow(lr_np); plt.title("Input LR")
#     plt.subplot(1, 2, 2); plt.imshow(pred_np); plt.title(f"Step {step+1}")
#     plt.axis('off')
#     plt.savefig(f"{save_dir}/step_{step+1:04d}.png", bbox_inches='tight')
#     plt.close()
    
#     model.train(); adapter.train()
#     return metrics

# if __name__ == "__main__":
#     run_hybrid_experiment()

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
LR_ADAPTER = 1e-5        
LR_SCALES = 1e-4         
SAVE_INTERVAL = 50      
SDE_STRENGTH = 0.5
TEXTURE_LOSS_WEIGHT = 0.1 # [新增] 纹理损失权重，推荐 0.1
# ===========================================

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from diffusion.model.nets.PixArtMS import PixArtMS_XL_2
    from diffusion.model.nets.adapter import MultiLevelAdapter
    from diffusion import IDDPM
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# LPIPS 仅用于评估监控 (Eval)，不参与训练梯度回传，不会爆显存
if USE_METRICS:
    lpips_fn = lpips.LPIPS(net='vgg').to(DEVICE).eval()
    for p in lpips_fn.parameters(): p.requires_grad = False

def run_hybrid_experiment():
    print(f"\n🚀 开始 Hybrid SDE 过拟合测试 (v4.5: Aligned Latent Texture Loss)")
    
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
    # [保持一致] 默认初始化 (FP32)
    adapter = MultiLevelAdapter(in_channels=4, hidden_size=1152).to(DEVICE).train()
    
    # -------------------------------------------------------------------------
    # 3. 参数冻结与分组 (严格保持你的参考代码逻辑)
    # -------------------------------------------------------------------------
    # 3.1 冻结主干
    for p in pixart.parameters(): p.requires_grad = False
    
    # 3.2 显式转 FP32 (包括新加的 LN)
    pixart.injection_scales = pixart.injection_scales.to(torch.float32)
    pixart.adapter_proj = pixart.adapter_proj.to(torch.float32)
    pixart.adapter_norm = pixart.adapter_norm.to(torch.float32)
    pixart.cross_attn_scale.data = pixart.cross_attn_scale.data.float()
    
    # 检测 LayerNorm 并转换
    if hasattr(pixart, 'input_adapter_ln'):
        pixart.input_adapter_ln = pixart.input_adapter_ln.to(torch.float32)
        print("   ✅ 检测到 input_adapter_ln，已转为 FP32")

    # 3.3 构建优化器参数
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
    
    if hasattr(pixart, 'input_adapter_ln'):
        for p in pixart.input_adapter_ln.parameters():
            p.requires_grad = True
            proj_params.append(p)

    print(f"   🔥 可训练: Adapter({len(adapter_params)}) | Scales({len(scale_params)}) | Proj/LN({len(proj_params)})")
    
    optimizer = torch.optim.AdamW([
        {'params': adapter_params, 'lr': LR_ADAPTER},
        {'params': proj_params, 'lr': LR_ADAPTER},
        {'params': scale_params, 'lr': LR_SCALES}
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

    # 仅用于评估解码，不用梯度
    vae = AutoencoderKL.from_pretrained(VAE_PATH, local_files_only=True).to("cpu").float()

    with torch.no_grad():
        if lr_latent_input is None:
            print("   Encoding LR image to Latent...")
            # 保持你的参考代码逻辑：在 CPU 做 Encode (或者 GPU)，这里用 CPU 省显存
            dist = vae.encode(lr_img.cpu()).latent_dist
            lr_latent_input = dist.sample() * vae.config.scaling_factor
            lr_latent_input = lr_latent_input.to(DEVICE)  # 最终放到 GPU (FP32 by default from cpu float)
        elif lr_img is None:
            lr_img = vae.decode(lr_latent_input.cpu().float() / vae.config.scaling_factor).sample.to(DEVICE)
            lr_img = torch.clamp((lr_img + 1.0) / 2.0, 0.0, 1.0)

        # 简单的诊断打印
        print(f"   👉 Latent Input Std: {lr_latent_input.std().item():.4f}")

        hr_img_gt = vae.decode(hr_latent.cpu().float() / vae.config.scaling_factor).sample.to(DEVICE)
        hr_img_gt = torch.clamp((hr_img_gt + 1.0) / 2.0, 0.0, 1.0)

    y_embed = torch.load(T5_EMBED_PATH, map_location="cpu")["prompt_embeds"].unsqueeze(1).to(DEVICE).to(DTYPE)
    data_info = {'img_hw': torch.tensor([[512., 512.]]).to(DEVICE).to(DTYPE), 'aspect_ratio': torch.tensor([1.]).to(DEVICE).to(DTYPE)}

    # -------------------------------------------------------------------------
    # 辅助函数: Latent Texture Loss (替代 LPIPS)
    # -------------------------------------------------------------------------
    def predict_x0_from_eps(x_t, eps, t):
        acp = torch.from_numpy(diffusion.alphas_cumprod).to(x_t.device).to(x_t.dtype)
        at = acp[t]
        return (x_t - (1 - at).sqrt().view(-1, 1, 1, 1) * eps) / at.sqrt().view(-1, 1, 1, 1)

    def latent_texture_loss(pred, target):
        # 计算 Latent 空间的一阶差分 (纹理/边缘)
        # 鼓励模型生成的 Latent 具有和 GT 相同的局部变化率
        p_dx = pred[..., 1:] - pred[..., :-1]
        p_dy = pred[..., 1:, :] - pred[..., :-1, :]
        t_dx = target[..., 1:] - target[..., :-1]
        t_dy = target[..., 1:, :] - target[..., :-1, :]
        return F.l1_loss(p_dx, t_dx) + F.l1_loss(p_dy, t_dy)

    # -------------------------------------------------------------------------
    # 5. 训练循环
    # -------------------------------------------------------------------------
    losses = []
    pbar = tqdm(range(STEPS))
    for step in pbar:
        optimizer.zero_grad()
        
        t = torch.randint(0, 1000, (1,), device=DEVICE).long()
        noise = torch.randn_like(hr_latent)
        noisy_input = diffusion.q_sample(hr_latent, t, noise)
        
        # [保持一致] 你的参考代码中 explicitly cast to .float()
        # 这确保了 Adapter 以 FP32 运行，避免了 FP16 溢出或类型不匹配
        adapter_cond = adapter(lr_latent_input.float()) 
        
        with torch.cuda.amp.autocast():
            model_out = pixart(
                noisy_input, t, y_embed, 
                data_info=data_info, 
                adapter_cond=adapter_cond, 
                injection_mode='hybrid'
            )
            if model_out.shape[1] == 8: model_out, _ = model_out.chunk(2, dim=1)
            
            # --- Loss Calculation ---
            # 1. MSE Loss (基础)
            loss_mse = F.mse_loss(model_out, noise)
            
            # 2. Texture Loss (新增，解决油画感)
            # 反推 x0 (Latent Space)
            pred_latents = predict_x0_from_eps(noisy_input.float(), model_out.float(), t)
            
            # 计算纹理损失 (Input: Float32)
            # hr_latent 是 FP16, 需要转 float 对齐
            loss_tex = latent_texture_loss(pred_latents, hr_latent.float())
            
            loss = loss_mse + TEXTURE_LOSS_WEIGHT * loss_tex
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        losses.append(loss.item())
        
        if isinstance(adapter_cond, list):
            adapter_abs_mean = np.mean([f.abs().mean().item() for f in adapter_cond])
        else:
            adapter_abs_mean = adapter_cond.abs().mean().item()
            
        pbar.set_postfix({
            "Loss": f"{loss.item():.4f}", 
            "MSE": f"{loss_mse.item():.4f}",
            "Tex": f"{loss_tex.item():.4f}",
            "Adp": f"{adapter_abs_mean:.4f}"
        })
        
        if (step + 1) % SAVE_INTERVAL == 0:
            input_scales = [f"{s.item():.4f}" for s in pixart.injection_scales]
            print(f"\n🔍 [Monitor] Step {step+1} Adapter Mean: {adapter_abs_mean:.6f}")

            # 评估时使用 LPIPS 监控质量
            metrics = evaluate_and_save(
                pixart, adapter, vae, 
                lr_latent_input, lr_img, y_embed, data_info, 
                step, hr_img_gt, hr_latent
            )
            
            print(f"📊 Report:")
            print(f"   📉 Loss: {loss.item():.6f}")
            print(f"   🎛️ Input Scales: {input_scales}")
            if metrics:
                print(f"   🖼️ PSNR: {metrics['psnr']:.2f} | LPIPS: {metrics['lpips']:.4f}")
            print("-" * 50)

    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.savefig("../experiments_results/overfit_hybrid_v4_texture/loss.png")

def evaluate_and_save(model, adapter, vae, lr_latent_input, lr_img, y_embed, data_info, step, hr_img_gt, hr_latent):
    model.eval(); adapter.eval()
    save_dir = "../experiments_results/overfit_hybrid_v4_texture/vis"
    os.makedirs(save_dir, exist_ok=True)
    
    scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000, solver_order=2)
    scheduler.set_timesteps(20)
    
    start_t_val = int(1000 * SDE_STRENGTH)
    timesteps = [t for t in scheduler.timesteps if t <= start_t_val]
    
    g = torch.Generator(DEVICE).manual_seed(42)
    latents = lr_latent_input.clone().to(DTYPE)
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

        # Diagnostic
        gen_mean = latents.float().mean().item()
        gen_std = latents.float().std().item()
        gt_mean = hr_latent.float().mean().item()
        gt_std = hr_latent.float().std().item()
        print(f"\n🧐 [DIAGNOSTIC] Final Latent Stats | Gen: u={gen_mean:.3f},s={gen_std:.3f} | GT: u={gt_mean:.3f},s={gt_std:.3f}")
            
        pred_img = vae.decode(latents.cpu().float() / vae.config.scaling_factor).sample.to(DEVICE)
        pred_img_clamp = torch.clamp(pred_img, -1.0, 1.0)

    metrics = {}
    if USE_METRICS:
        pred_norm_01 = (pred_img_clamp / 2 + 0.5).clamp(0, 1)
        gt_norm_01 = (hr_img_gt / 2 + 0.5).clamp(0, 1)
        
        metrics['psnr'] = psnr(pred_norm_01, gt_norm_01, data_range=1.0).item()
        metrics['ssim'] = ssim(pred_norm_01, gt_norm_01, data_range=1.0).item()
        metrics['lpips'] = lpips_fn(pred_img_clamp, hr_img_gt).item() # Eval LPIPS uses FP32 images here

    pred_np = pred_img_clamp[0].permute(1, 2, 0).detach().cpu().numpy()
    pred_np = (pred_np / 2 + 0.5).clip(0, 1)
    
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
