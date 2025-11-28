#!/usr/bin/env bash
# Convenience wrapper to run the Med3DVLM caption demo with optional LoRA.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

MODEL_PATH="${MODEL_PATH:-./models/Med3DVLM-Qwen-2.5-7B}"
DATA_ROOT="${DATA_ROOT:-./data/demo}"
SAMPLE_ID="${SAMPLE_ID:-024421}"
IMAGE_PATH="${IMAGE_PATH:-}"
QUESTION="${QUESTION:-Describe the findings of the medical image you see.}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
PROJ_OUT_NUM="${PROJ_OUT_NUM:-256}"
SEED="${SEED:-42}"
LORA_PATH="${LORA_PATH:-}"

# Optional first positional argument (not starting with '-') overrides LORA_PATH.
if [[ $# -gt 0 && "${1#--}" == "$1" ]]; then
  LORA_PATH="$1"
  shift
fi

ARGS=(
  --model_path "$MODEL_PATH"
  --question "$QUESTION"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --dtype "$DTYPE"
  --device "$DEVICE"
  --proj_out_num "$PROJ_OUT_NUM"
  --seed "$SEED"
)

if [[ -n "$IMAGE_PATH" ]]; then
  ARGS+=(--image_path "$IMAGE_PATH")
else
  ARGS+=(--data_root "$DATA_ROOT" --sample_id "$SAMPLE_ID")
fi

if [[ -n "$LORA_PATH" ]]; then
  ARGS+=(--lora_path "$LORA_PATH")
fi

python src/demo/caption_test.py "${ARGS[@]}" "$@"
