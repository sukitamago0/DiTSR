# v8/v9 分离训练 vs 统一自动化训练（v10 launcher）

## 新增内容
- 新增统一入口：`experiments/train_4090_auto_v10_unified.py`
  - Phase-1 自动跑 v8（按步数停止）
  - 自动复制 handover checkpoint 到 v9 路径
  - Phase-2 自动跑 v9（按步数停止）
- v8/v9 edge 脚本新增 `MAX_TRAIN_STEPS` 环境变量，支持稳定的“到步切换”。

## 两种范式具体区别

### 1) 调度一致性
- **分离手动**：切换点由人工决定，常见“epoch取整误差”“忘记同步路径/参数”。
- **统一自动**：切换由 `phase1_steps` 确定，handover 可复现。

量化影响（工程面）：
- 人工切换决策点：`>=2`（停v8、启v9） -> `0`
- 路径/ckpt 手工操作步骤：`>=2`（拷贝/改配置） -> `0`

### 2) 可复现性
- **分离手动**：实验复现需要口头补充“何时切、拷贝了哪个 last.pth”。
- **统一自动**：命令行完整描述实验过程（phase1/phase2 步数 + 脚本 + ckpt路径）。

量化影响（复现实验所需显式参数）：
- 手动流程通常分散在日志/命令历史中（不稳定）
- 统一流程收敛为 1 条命令（固定参数集合）

### 3) 科研严谨性风险
- **分离手动**主要风险：
  - 切换点漂移（不同人不同时机切）
  - handover checkpoint 不一致
- **统一自动**主要收益：
  - 切换点固定
  - handover 机制固定

量化影响（偏差来源个数，按流程项计）：
- 手动：3 类高频偏差源（切换时机、ckpt路径、命令参数）
- 自动：1 类主要偏差源（超参本身）

## 注意事项
- 统一 launcher 解决的是“流程可复现与自动化”，不是替代算法本身。
- v8->v9 的 GAN warmup 仍按 v9 脚本内部逻辑执行（已支持 handover-local 处理）。

## 推荐使用方式
```bash
python experiments/train_4090_auto_v10_unified.py \
  --phase1-steps 12000 \
  --phase2-steps 8000 \
  --v8-script experiments/train_4090_auto_v8_edge_gtmask.py \
  --v9-script experiments/train_4090_auto_v9_gan_edge_gtmask.py
```
