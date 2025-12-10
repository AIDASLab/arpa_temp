#!/bin/bash

DATASET_JSON="./data/ADNI_VQA_CVRF_GDS_AGE_test.json"
LORA_PATH="./output/ys_adni_vqa_cvrf_gds_age_lora_128/best_checkpoints/adni_vqa_cvrf_gds_age"

MODEL_NAME="${MODEL_NAME:-MagicXin/Med3DVLM-Qwen-2.5-7B}"
NPROC="${NPROC:-8}"
MASTER_PORT="${MASTER_PORT:-29511}"

# torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
#   src/train/eval_adni_vqa_lora_ys.py \
#   --model_name "${MODEL_NAME}" \
#   --lora_path "${LORA_PATH}" \
#   --dataset_json "${DATASET_JSON}" \
#   --dataset_label "ADNI_VQA_CVRF_GDS_AGE" \
#   --no_wandb \
#   --trust_remote_code

python src/train/eval_adni_vqa_lora_ys.py \
  --model_name "${MODEL_NAME}" \
  --lora_path "${LORA_PATH}" \
  --dataset_json "${DATASET_JSON}" \
  --dataset_label "ADNI_VQA_CVRF_GDS_AGE" \
  --no_wandb \
  --trust_remote_code
