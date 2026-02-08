# SR 实验计划（对齐评测 + v4 / cfdrop 对比）

本文件对应 `scripts/run_sr_eval_plan.sh`，用于完成你目前最需要的三类实验：

1. **identity / bicubic 基线**：定位退化难度与插值下限。
2. **v4 vs cfdrop 对比**：验证 LR-consistency + adapter CF-drop 是否带来稳定提升。
3. **text+adapter vs adapter_only**：判断文本条件是否干扰 SR 指标。

## 为什么要测 v4 与 v4 cfdrop

- v4 与 cfdrop 的训练目标明显不同，cfdrop 带 **LR-consistency** 和 **adapter CF-drop**，理论上更贴合 SR 指标。
- 如果只测一个，你无法判断指标低是来自“训练方式”还是“评测方式/推理方式”。
- 建议优先测 v4（作为参考），再测 cfdrop（验证改进是否有效）。

## 使用方式

```bash
PACK_DIR=/path/to/valpack \
V4_CKPT=/path/to/v4_last.pth \
CFDROP_CKPT=/path/to/cfdrop_last.pth \
OUT_ROOT=experiments_results/sr_eval_plan \
MAX_N=50 \
CFG=3.0 \
STEPS=50 \
./scripts/run_sr_eval_plan.sh
```

### 仅测 v4（不测 cfdrop）

```bash
PACK_DIR=/path/to/valpack \
V4_CKPT=/path/to/v4_last.pth \
OUT_ROOT=experiments_results/sr_eval_plan \
./scripts/run_sr_eval_plan.sh
```

## 输出结构

```
experiments_results/sr_eval_plan/
  baseline_identity.json
  baseline_bicubic.json
  v4/
    text+adapter/
      summary.json
      metrics.json
    adapter_only/
      summary.json
      metrics.json
  cfdrop/
    text+adapter/
      summary.json
      metrics.json
    adapter_only/
      summary.json
      metrics.json
```

## 如何解读

- **identity**：越高说明退化越轻；过低表示任务很难。
- **bicubic**：你模型要超越它，论文才有意义。
- **v4 vs cfdrop**：若 cfdrop > v4，说明 LR-consistency/CF-drop 是有效方向。
- **adapter_only > text+adapter**：说明文本干扰，应弱化文本条件或采用 dropout。
