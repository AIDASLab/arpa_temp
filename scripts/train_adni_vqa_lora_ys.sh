#!/bin/bash

set -euo pipefail

# TRAIN_KEYS=${1:-ADNI_VQA,ADNI_VQA_HISTORY,ADNI_VQA_HISTORY_UPDATED}
# TRAIN_KEYS=${1:-ADNI_VQA_AGQ,ADNI_VQA_AGE_HISTORY,ADNI_VQA_AGE_HISTORY_UPDATED}
TRAIN_KEYS=${1:-ADNI_VQA_v2}
EVAL_KEYS=${2:-$TRAIN_KEYS}
FINETUNE_MODE=${3:-${FINETUNE_MODE:-lora}}
GENERATION_EVAL=${GENERATION_EVAL:-true}
MASTER_PORT=${MASTER_PORT:-29500}
EVAL_MASTER_PORT=${EVAL_MASTER_PORT:-$((MASTER_PORT+1))}
OUTPUT_DIR_OVERRIDE=${OUTPUT_DIR:-}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}
# Path to best checkpoint for evaluation
BASE_BEST_LORA_PATH=${BEST_EVAL_LORA_PATH:-}
NPROC=${NPROC:-8}
PRECISION=${PRECISION:-auto}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}

REGRESSION_LOSS_TYPE=${REGRESSION_LOSS_TYPE:-gaussian}      # gaussian | mae | huber
REGRESSION_MLE_VARIANCE=${REGRESSION_MLE_VARIANCE:-1.0}     # used when gaussian
REGRESSION_HUBER_BETA=${REGRESSION_HUBER_BETA:-1.0}         # used when huber
NORMALIZED=${NORMALIZED:-false}

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Set your wandb API key here
export WANDB_API_KEY=9c2e7fba1823550f9d2f28b6d6e141982aeb7e3b

run_eval() {
  local dataset_json=$1
  local label=$2
  local lora_path=$3
  torchrun --nproc_per_node="${NPROC}" --master_port="${EVAL_MASTER_PORT}" \
    src/train/eval_adni_vqa_lora_ys.py \
    --model_name MagicXin/Med3DVLM-Qwen-2.5-7B \
    --lora_path "$lora_path" \
    --dataset_json "$dataset_json" \
    --dataset_label "$label" \
    --trust_remote_code
}

resolve_eval_datasets() {
  python - "$1" <<'PY'
import sys
from pathlib import Path
from src.train.train_adni_vqa_lora_ys import DATASET_REGISTRY, PROJECT_ROOT

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

IFS=',' read -ra TRAIN_KEYS_ARRAY <<< "$TRAIN_KEYS"
IFS=',' read -ra EVAL_KEYS_ARRAY <<< "$EVAL_KEYS"
TOTAL_RUNS=${#TRAIN_KEYS_ARRAY[@]}

if [[ ${#EVAL_KEYS_ARRAY[@]} -ne ${#TRAIN_KEYS_ARRAY[@]} ]]; then
  USE_SHARED_EVAL_KEYS=true
else
  USE_SHARED_EVAL_KEYS=false
fi

build_regression_head_suffix() {
  local safe_loss
  safe_loss=$(echo "${REGRESSION_LOSS_TYPE:-gaussian}" | tr '[:upper:]' '[:lower:]')
  local suffix
  case "$safe_loss" in
    gaussian)
      local var_sanitized
      var_sanitized=$(echo "${REGRESSION_MLE_VARIANCE:-1.0}" | tr '.' 'p')
      suffix="loss_gauss_var${var_sanitized}"
      ;;
    mae)
      suffix="loss_mae"
      ;;
    huber)
      local beta_sanitized
      beta_sanitized=$(echo "${REGRESSION_HUBER_BETA:-1.0}" | tr '.' 'p')
      suffix="loss_huber_beta${beta_sanitized}"
      ;;
    *)
      suffix="loss_${safe_loss}"
      ;;
  esac
  if [[ "${NORMALIZED}" == true ]]; then
    suffix="${suffix}_norm"
  fi
  printf '%s' "$suffix"
}

age_output_dir_with_variant() {
  local variant="$1"
  local head_suffix
  head_suffix=$(build_regression_head_suffix)
  echo "output/ys_adni_vqa_age${variant}_lora_128_head_${head_suffix}"
}

default_output_dir_for_key() {
  case "$1" in
    ADNI_VQA)
      echo "output/ys_adni_vqa_lora_128_300epochs"
      ;;
    ADNI_VQA_HISTORY)
      echo "output/ys_adni_vqa_history_combined_lora_128"
      ;;
    ADNI_VQA_HISTORY_UPDATED)
      echo "output/ys_adni_vqa_history_updated_combined_lora_128"
      ;;
    ADNI_VQA_v2)
      echo "output/ys_adni_vqa_lora_128_v2"
      ;;
    ADNI_VQA_AGE)
      age_output_dir_with_variant ""
      ;;
    ADNI_VQA_AGE_HISTORY)
      age_output_dir_with_variant "_history"
      ;;
    ADNI_VQA_AGE_HISTORY_UPDATED)
      age_output_dir_with_variant "_history_updated"
      ;;
    *)
      local safe
      safe=$(echo "$1" | tr '[:upper:]' '[:lower:]' | tr -cs '[:alnum:]' '_')
      echo "output/ys_${safe}_lora"
      ;;
  esac
}

for idx in "${!TRAIN_KEYS_ARRAY[@]}"; do
  train_key=$(echo "${TRAIN_KEYS_ARRAY[idx]}" | xargs)
  [[ -z "$train_key" ]] && continue
  safe_key=$(echo "$train_key" | tr '[:upper:]' '[:lower:]' | tr -cs '[:alnum:]' '_')

  if [[ -n "$OUTPUT_DIR_OVERRIDE" ]]; then
    if (( TOTAL_RUNS > 1 )); then
      current_output_dir="${OUTPUT_DIR_OVERRIDE%/}_${safe_key}"
    else
      current_output_dir="$OUTPUT_DIR_OVERRIDE"
    fi
  else
    current_output_dir=$(default_output_dir_for_key "$train_key")
  fi

  default_best_dir="${current_output_dir}/best_checkpoints/${safe_key}"

  if [[ -n "$BASE_BEST_LORA_PATH" ]]; then
    if (( TOTAL_RUNS > 1 )); then
      current_best_lora_path="${BASE_BEST_LORA_PATH%/}_${safe_key}"
    else
      current_best_lora_path="$BASE_BEST_LORA_PATH"
    fi
  else
    current_best_lora_path="$default_best_dir"
  fi

  if [[ ! -d "$current_best_lora_path" ]]; then
    echo "[warn] Best checkpoint not found at '$current_best_lora_path'. Falling back to '$current_output_dir'."
    current_best_lora_path="$current_output_dir"
  fi

  if [[ "$USE_SHARED_EVAL_KEYS" == true ]]; then
    current_eval_keys="$EVAL_KEYS"
  else
    current_eval_keys=$(echo "${EVAL_KEYS_ARRAY[idx]}" | xargs)
  fi

  TRAIN_CMD=(
    torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}"
    src/train/train_adni_vqa_lora_ys.py
    --model_name MagicXin/Med3DVLM-Qwen-2.5-7B
    --train_dataset_keys "$train_key"
    --eval_dataset_keys "$current_eval_keys"
    --output_dir "$current_output_dir"
    --batch_size 4
    --grad_accum 1
    --epochs 100
    --lr 1e-4
    --max_length 1024
    --finetune_mode "$FINETUNE_MODE"
    --precision "$PRECISION"
    --max_grad_norm "$MAX_GRAD_NORM"
    --regression_loss_type "$REGRESSION_LOSS_TYPE"
    --regression_mle_variance "$REGRESSION_MLE_VARIANCE"
    --regression_huber_beta "$REGRESSION_HUBER_BETA"
    --trust_remote_code
  )
  if [[ "${NORMALIZED}" == true ]]; then
    TRAIN_CMD+=(--normalized)
  fi

  if [[ "${GENERATION_EVAL}" == "true" ]]; then
    TRAIN_CMD+=(--generation_eval_per_epoch)
  else
    TRAIN_CMD+=(--no_generation_eval_per_epoch)
  fi

  if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    TRAIN_CMD+=(--resume_from_checkpoint "${RESUME_CHECKPOINT}")
  fi

  echo "[train] Starting run for dataset key '${train_key}' with output dir '${current_output_dir}'"
  "${TRAIN_CMD[@]}"

  if [[ -z "$BASE_BEST_LORA_PATH" ]]; then
    if [[ -d "$default_best_dir" ]]; then
      current_best_lora_path="$default_best_dir"
    else
      current_best_lora_path="$current_output_dir"
    fi
  fi

  mapfile -t EVAL_DATASETS < <(resolve_eval_datasets "$current_eval_keys")
  for entry in "${EVAL_DATASETS[@]}"; do
    IFS='|' read -r eval_key eval_path <<< "$entry"
    run_eval "$eval_path" "$eval_key" "$current_best_lora_path"
  done
done
