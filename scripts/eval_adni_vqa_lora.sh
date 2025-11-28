#!/bin/bash

set -euo pipefail

TRAIN_KEYS=${1:-ADNI_VQA,ADNI_VQA_HISTORY}
EVAL_KEYS=${2:-$TRAIN_KEYS}
MASTER_PORT=${MASTER_PORT:-29500}
EVAL_MASTER_PORT=${EVAL_MASTER_PORT:-$((MASTER_PORT+1))}
OUTPUT_DIR=${OUTPUT_DIR:-output/adni_vqa_lora_128}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}
EVAL_LORA_PATH=${EVAL_LORA_PATH:-$OUTPUT_DIR}
EVAL_DATASET_NO_HISTORY=${EVAL_DATASET_NO_HISTORY:-data/ADNI_VQA_test.json}
EVAL_DATASET_HISTORY=${EVAL_DATASET_HISTORY:-data/ADNI_VQA_with_history_test.json}
NPROC=${NPROC:-8}
WANDB_RUN_PREFIX=${WANDB_RUN_PREFIX:-eval_adni_vqa}

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


resolve_lora_path() {
  local path="$1"
  if [[ -z "$path" ]]; then
    echo ""
    return
  fi
  if [[ -f "$path/adapter_config.json" ]]; then
    echo "$path"
    return
  fi
  local latest_dir=""
  local latest_step=-1
  shopt -s nullglob
  for ckpt_dir in "$path"/checkpoint-*; do
    if [[ -d "$ckpt_dir" && -f "$ckpt_dir/adapter_config.json" ]]; then
      local basename="${ckpt_dir##*-}"
      if [[ "$basename" =~ ^[0-9]+$ && "${BASH_REMATCH[0]}" -gt $latest_step ]]; then
        latest_step="${BASH_REMATCH[0]}"
        latest_dir="$ckpt_dir"
      fi
    fi
  done
  shopt -u nullglob
  if [[ -n "$latest_dir" ]]; then
    echo "$latest_dir"
  else
    echo ""
  fi
}

# run_eval() {
#   local dataset_json=$1
#   local label=$2
#   torchrun --nproc_per_node="${NPROC}" --master_port="${EVAL_MASTER_PORT}" \
#     src/train/eval_adni_vqa_lora.py \
#     --model_name MagicXin/Med3DVLM-Qwen-2.5-7B \
#     --lora_path "/home/arpa/Med3DVLM/output/adni_vqa_lora/checkpoint-26108" \
#     --dataset_json "$dataset_json" \
#     --dataset_label "$label" \
#     --trust_remote_code
# }

# # run_eval "$EVAL_DATASET_NO_HISTORY" "no_history"
# run_eval "$EVAL_DATASET_HISTORY" "history"


run_eval() {
  local dataset_json=$1
  local label=$2
  local lora_path=$3
  local ckpt_name
  ckpt_name=$(basename "$lora_path")
  local timestamp
  timestamp=$(date +"%Y%m%d_%H%M%S")
  local wandb_run_name="${WANDB_RUN_PREFIX}_${label}_${ckpt_name}_${timestamp}"

  torchrun --nproc_per_node="${NPROC}" --master_port="${EVAL_MASTER_PORT}" \
    src/train/eval_adni_vqa_lora.py \
    --model_name MagicXin/Med3DVLM-Qwen-2.5-7B \
    --lora_path "$lora_path" \
    --dataset_json "$dataset_json" \
    --dataset_label "$label" \
    --wandb_run_name "$wandb_run_name" \
    --trust_remote_code
}

DEFAULT_LORA_CHECKPOINTS=(
  "/home/arpa/Med3DVLM/output/adni_vqa_lora_128/checkpoint-31244"
)

if [[ -n "${EVAL_LORA_PATHS:-}" ]]; then
  IFS=',' read -r -a LORA_CHECKPOINTS <<< "$EVAL_LORA_PATHS"
elif [[ -n "${EVAL_LORA_PATH:-}" ]]; then
  LORA_CHECKPOINTS=("$EVAL_LORA_PATH")
else
  LORA_CHECKPOINTS=("${DEFAULT_LORA_CHECKPOINTS[@]}")
fi

RESOLVED_LORA_CHECKPOINTS=()
for lora_path in "${LORA_CHECKPOINTS[@]}"; do
  resolved_path=$(resolve_lora_path "$lora_path")
  if [[ -z "$resolved_path" ]]; then
    echo "[eval] Warning: could not find adapter_config.json under '$lora_path'; skipping." >&2
    continue
  fi
  RESOLVED_LORA_CHECKPOINTS+=("$resolved_path")
done

if [[ ${#RESOLVED_LORA_CHECKPOINTS[@]} -eq 0 ]]; then
  echo "[eval] Error: no valid LoRA checkpoints were found. Please set EVAL_LORA_PATH or EVAL_LORA_PATHS." >&2
  exit 1
fi

for lora_ckpt in "${RESOLVED_LORA_CHECKPOINTS[@]}"; do
  run_eval "$EVAL_DATASET_NO_HISTORY" "history" "$lora_ckpt"
done
