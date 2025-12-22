#!/bin/bash

set -euo pipefail

LORA_PATH=${1:-${LORA_PATH:-output/adni_vqa_age_lora_128_1}}
DATASET_JSON=${DATASET_JSON:-data/ADNI_VQA_age_test.json}
DATASET_LABEL=${DATASET_LABEL:-adni_vqa_age}
MODEL_NAME=${MODEL_NAME:-MagicXin/Med3DVLM-Qwen-2.5-7B}
MASTER_PORT=${MASTER_PORT:-29560}
NPROC=${NPROC:-}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
USE_WANDB=${USE_WANDB:-true}
WANDB_PROJECT=${WANDB_PROJECT:-Med3DVLM}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
GPU_IDS="$CUDA_VISIBLE_DEVICES"

IFS=',' read -ra __EVAL_GPU_IDS <<< "$GPU_IDS"
GPU_COUNT=${#__EVAL_GPU_IDS[@]}
if [[ -z "$GPU_COUNT" || "$GPU_COUNT" -eq 0 ]]; then
  GPU_COUNT=1
fi

if [[ -z "$NPROC" ]]; then
  NPROC=$GPU_COUNT
elif (( NPROC <= 0 )); then
  NPROC=$GPU_COUNT
fi

echo "Using GPUs: $GPU_IDS (nproc=$NPROC)"

CMD=(
  torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}"
  src/train/eval_adni_vqa_age.py
  --model_name "$MODEL_NAME"
  --lora_path "$LORA_PATH"
  --dataset_json "$DATASET_JSON"
  --dataset_label "$DATASET_LABEL"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --trust_remote_code
  --wandb_project "$WANDB_PROJECT"
)

if [[ -n "$WANDB_ENTITY" ]]; then
  CMD+=(--wandb_entity "$WANDB_ENTITY")
fi

if [[ -n "$WANDB_RUN_NAME" ]]; then
  CMD+=(--wandb_run_name "$WANDB_RUN_NAME")
fi

if [[ "${USE_WANDB}" != "true" ]]; then
  CMD+=(--no_wandb)
fi
echo "Running: ${CMD[*]}"
"${CMD[@]}"
