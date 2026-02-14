# DiTSR v8/v9 损失函数对比分析与重设计建议

## 1) 先回答核心问题：新 v8/v9 是否“只是引入了 FFL”
不是。脚本层面至少有以下变化：
- `v8_old`：`v-pred + Min-SNR + latent L1 + LPIPS`，无 GAN/无 FFL。
- `v8`：在 `v8_old` 基础上加入 `FFL`（且默认 `FFL_BASE_WEIGHT=0.1`），其余主干逻辑基本一致。
- `v9_gan`：引入 PatchGAN、GAN warmup/ramp、R1 GP、EMA；形成 `recon + perceptual + adversarial` 三元结构。
- `edge_gtmask` 版本：意图用 GT 引导 edge/flat 区域损失替代 FFL（`FFL_BASE_WEIGHT=0`）。

## 2) 你观测到的现象与可能机制

### 2.1 旧 v8 早期 LPIPS 好，后期纹理化/过锐化
这在“LPIPS + GAN（或高频导向损失）逐步上升”类训练中很典型：
- LPIPS/GAN 会奖励感知纹理与局部对比，
- 当缺乏“区域约束”（例如 defocus 区域应保持低频）时，模型会把低频区域“纹理化”来换取感知分数。

### 2.2 新 v8/v9 长期发糊
最常见触发器不是“FFL 一定错”，而是：
- 目标之间的梯度方向冲突（v-loss/latent L1 更偏保真低频，LPIPS/GAN/FFL偏感知高频）；
- 权重调度不合理（感知项生效太晚或过早、GAN过强或过弱）；
- 像素损失采样 patch 太随机且太小，难以稳定驱动结构恢复。

## 3) 对 edge_gtmask 脚本的漏洞审视（重点）

### 3.1 关键逻辑漏洞：edge loss 定义了但未真正参与训练（已修）
在原脚本中：
- `loss_edge/loss_flat_hf` 初始化为 0；
- 但训练环节没有调用 `edge_guided_losses(...)` 给它们赋值；
- 最终总损失里虽然加了 `EDGE_GRAD_WEIGHT * loss_edge + FLAT_HF_WEIGHT * loss_flat_hf`，实际一直等于 0。

这会导致你以为“edge_gtmask 生效了”，其实没有。

### 3.2 日志变量漏洞（已修）
- `loss_ffl` 仅在条件分支内赋值；
- 但 `set_postfix` 无条件引用 `loss_ffl.item()`；
- 当 `FFL_BASE_WEIGHT=0` 时可能触发未定义引用。

### 3.3 调度漏洞：edge 约束缺少 warmup/ramp（已修）
- 原脚本 edge 权重是常量，容易在早期干扰主重建目标；
- 已补充 `EDGE_WARMUP_STEPS/EDGE_RAMP_STEPS`，与 LPIPS/GAN 一样使用渐进注入。

## 4) 已实施修复

### 4.1 `train_4090_auto_v8_edge_gtmask.py`
- 新增 edge 权重调度（warmup+ramp）；
- 在 pixel-space 分支显式调用 `edge_guided_losses`；
- 确保 `loss_ffl` 总是先初始化；
- 日志新增 `edge/flat_hf/w_edge/w_flat`，便于诊断。

### 4.2 `train_4090_auto_v9_gan_edge_gtmask.py`
- `get_stage2_loss_weights` 扩展返回 `edge_w/flat_w`；
- 训练时真正计算 `edge_guided_losses`；
- `loss_ffl` 安全初始化；
- 日志补充 `w_edge/w_flat`。

## 5) 新的损失函数构成建议（可直接落地）

推荐“分阶段 + 区域约束 + 对抗后置”的组合：

### Stage A（0 ~ 3k step）：重建定盘
- `L_v`: 1.0（Min-SNR 保留）
- `L_latent_l1`: 1.0
- `L_lpips`: 0
- `L_edge`: 0
- `L_flat`: 0
- `L_gan`: 0

目标：先学可逆结构与内容对齐，避免早期感知项导致模式漂移。

### Stage B（3k ~ 9k）：感知渐进 + 区域约束
- `L_lpips`: 0 -> 0.35 线性升
- `L_edge`: 0 -> 0.06
- `L_flat`: 0 -> 0.08（建议略高于 edge，优先抑制 defocus 乱纹理）
- `L_gan`: 0

目标：提升纹理但把“允许锐化的位置”限制在 GT edge 区域。

### Stage C（>= 9k）：低权重 GAN 微调
- `L_gan`: 0 -> 0.03（慢 ramp，至少 4k steps）
- `L_lpips`: 0.35~0.45
- `L_edge`: 0.06
- `L_flat`: 0.08~0.10

目标：把 GAN 作为“最后修饰器”，不是主导项。

> 经验上，你现在 `GAN_TARGET_WEIGHT=0.08` 偏大，容易驱动“纹理幻觉”或不稳定。

## 6) FFL 是否不合适：结论
- 不是“绝对不合适”，而是**不建议作为主高频驱动项**。
- 在你这个任务（存在 defocus 区域应保持平滑）里，FFL 很容易把频域误差均匀扩散成“到处补纹理”。
- 若要保留，建议：
  - 仅在 Stage C 以很小权重（如 0.003~0.01）启用；
  - 或只对 edge-mask 区域启用（而非全图）。

## 7) 建议的实验矩阵（最小可行）
1. `No-FFL + EdgeMask + NoGAN`（先验证仅区域损失是否能压糊+抑制幻觉）。
2. 在 1 的基础上加 `Low-GAN(0.03)`。
3. 在 2 的基础上加 `Tiny-FFL(<=0.01, late start)`。

每组至少看：
- LPIPS/PSNR/SSIM；
- 以及你最关心的 defocus 区域可视化（是否被过锐化）。

## 8) 训练监控建议（避免“看起来在学，实际没生效”）
- 每 N step 记录各项损失**未加权值**与**加权后贡献值**；
- 记录 `w_lp/w_gan/w_edge/w_flat`；
- 固定 3 张典型验证图：高纹理、低纹理、强 defocus，做 epoch-by-epoch 可视化。

---

如果你愿意，下一步我可以再给你一份“直接可复制的超参数 preset（v10）”，把上面的阶段和权重写成一套完整配置（含建议 early stop 规则与阈值）。
