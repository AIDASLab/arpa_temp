#!/bin/bash

set -euo pipefail

# TRAIN_KEYS=${1:-ADNI_VQA,ADNI_VQA_HISTORY}
TRAIN_KEYS=${1:-ADNI_VQA_AGE}
EVAL_KEYS=${2:-$TRAIN_KEYS}
FINETUNE_MODE=${3:-${FINETUNE_MODE:-lora}}
GENERATION_EVAL=${GENERATION_EVAL:-true}
MASTER_PORT=${MASTER_PORT:-29500}
EVAL_MASTER_PORT=${EVAL_MASTER_PORT:-$((MASTER_PORT+1))}
OUTPUT_DIR=${OUTPUT_DIR:-output/adni_vqa_age_lora_128}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}
EVAL_LORA_PATH=${EVAL_LORA_PATH:-$OUTPUT_DIR}
NPROC=${NPROC:-8}
PRECISION=${PRECISION:-auto}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

TRAIN_CMD=(
  torchrun --nproc_per_node=8 --master_port="${MASTER_PORT}"
  src/train/train_adni_vqa_lora.py
  --model_name MagicXin/Med3DVLM-Qwen-2.5-7B
  --train_dataset_keys "$TRAIN_KEYS"
  --eval_dataset_keys "$EVAL_KEYS"
  --output_dir "$OUTPUT_DIR"
  --batch_size 4
  --grad_accum 1
  --epochs 100
  --lr 1e-4
  --max_length 1024
  --finetune_mode "$FINETUNE_MODE"
  --precision "$PRECISION"
  --max_grad_norm "$MAX_GRAD_NORM"
  --trust_remote_code
)

if [[ "${GENERATION_EVAL}" == "true" ]]; then
  TRAIN_CMD+=(--generation_eval_per_epoch)
else
  TRAIN_CMD+=(--no_generation_eval_per_epoch)
fi

if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  TRAIN_CMD+=(--resume_from_checkpoint "${RESUME_CHECKPOINT}")
fi

"${TRAIN_CMD[@]}"

run_eval() {
  local dataset_json=$1
  local label=$2
  torchrun --nproc_per_node="${NPROC}" --master_port="${EVAL_MASTER_PORT}" \
    src/train/eval_adni_vqa_lora.py \
    --model_name MagicXin/Med3DVLM-Qwen-2.5-7B \
    --lora_path "$EVAL_LORA_PATH" \
    --dataset_json "$dataset_json" \
    --dataset_label "$label" \
    --trust_remote_code
}

resolve_eval_datasets() {
  python - "$1" <<'PY'
import sys
from pathlib import Path
from src.train.train_adni_vqa_lora import DATASET_REGISTRY, PROJECT_ROOT

raw_keys = sys.argv[1]
keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
if not keys:
    raise SystemExit("No evaluation dataset keys provided.")

for key in keys:
    if key not in DATASET_REGISTRY:
        raise SystemExit(f"Unknown dataset key '{key}'. Available: {list(DATASET_REGISTRY)}")
    rel_path = DATASET_REGISTRY[key].get("eval")
    if not rel_path:
        raise SystemExit(f"Dataset '{key}' does not define an eval split.")
    path = PROJECT_ROOT / rel_path
    if not path.exists():
        raise SystemExit(f"Eval dataset for key '{key}' not found: {path}")
    print(f"{key}|{rel_path}")
PY
}

mapfile -t EVAL_DATASETS < <(resolve_eval_datasets "$EVAL_KEYS")
for entry in "${EVAL_DATASETS[@]}"; do
  IFS='|' read -r eval_key eval_path <<< "$entry"
  run_eval "$eval_path" "$eval_key"
done
