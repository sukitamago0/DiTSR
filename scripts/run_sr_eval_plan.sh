#!/usr/bin/env bash
set -euo pipefail

PACK_DIR="${PACK_DIR:-}"
V4_CKPT="${V4_CKPT:-}"
CFDROP_CKPT="${CFDROP_CKPT:-}"
OUT_ROOT="${OUT_ROOT:-experiments_results/sr_eval_plan}"
MAX_N="${MAX_N:-50}"
CFG="${CFG:-3.0}"
STEPS="${STEPS:-50}"
DEVICE="${DEVICE:-cuda}"
TRAIN_MODULE_V4="${TRAIN_MODULE_V4:-train_4090_auto_v4}"
TRAIN_MODULE_CFDROP="${TRAIN_MODULE_CFDROP:-train_4090_auto_v4_cfdrop_hrtarget}"

if [[ -z "${PACK_DIR}" ]]; then
  echo "ERROR: PACK_DIR is required."
  echo "Example:"
  echo "  PACK_DIR=/path/to/valpack V4_CKPT=/path/to/v4_last.pth \\"
  echo "  CFDROP_CKPT=/path/to/cfdrop_last.pth ./scripts/run_sr_eval_plan.sh"
  exit 1
fi

mkdir -p "${OUT_ROOT}"

echo "==> Baselines (identity / bicubic)"
python experiments/eval_valpack_metrics_gt512.py \
  --pack_dir "${PACK_DIR}" \
  --method identity \
  --max_n "${MAX_N}" \
  --device "${DEVICE}" \
  > "${OUT_ROOT}/baseline_identity.json"

python experiments/eval_valpack_metrics_gt512.py \
  --pack_dir "${PACK_DIR}" \
  --method bicubic \
  --max_n "${MAX_N}" \
  --device "${DEVICE}" \
  > "${OUT_ROOT}/baseline_bicubic.json"

run_model_eval() {
  local tag="$1"
  local ckpt="$2"
  local train_module="$3"

  if [[ -z "${ckpt}" ]]; then
    echo "==> Skip ${tag}: ckpt not provided."
    return 0
  fi

  echo "==> Eval ${tag} (ckpt=${ckpt})"

  for mode in text+adapter adapter_only; do
    local out_dir="${OUT_ROOT}/${tag}/${mode}"
    mkdir -p "${out_dir}"

    python experiments/eval_valpack_ditsr.py \
      --train_module "${train_module}" \
      --ckpt "${ckpt}" \
      --pack_dir "${PACK_DIR}" \
      --mode "${mode}" \
      --cfg "${CFG}" \
      --steps "${STEPS}" \
      --max_n "${MAX_N}" \
      --out_dir "${out_dir}"

    local pred_dir
    pred_dir="$(python - "${out_dir}" <<'PY'
import json
import sys
from pathlib import Path

summary = Path(sys.argv[1]) / "summary.json"
data = json.loads(summary.read_text(encoding="utf-8"))
print(data["pred_dir"])
PY
    )"

    python experiments/eval_valpack_metrics_gt512.py \
      --pack_dir "${PACK_DIR}" \
      --method folder \
      --pred_dir "${pred_dir}" \
      --max_n "${MAX_N}" \
      --device "${DEVICE}" \
      > "${out_dir}/metrics.json"
  done
}

run_model_eval "v4" "${V4_CKPT}" "${TRAIN_MODULE_V4}"
run_model_eval "cfdrop" "${CFDROP_CKPT}" "${TRAIN_MODULE_CFDROP}"

echo "==> Done. Results in ${OUT_ROOT}"
