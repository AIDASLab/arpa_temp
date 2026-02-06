#!/bin/bash

DATASET_JSON="./data/ADNI_VQA_recon_up_AGE_test.json"
LORA_PATH="./age_best/best_checkpoints/adni_vqa_cvrf_gds_age"

MODEL_NAME="${MODEL_NAME:-MagicXin/Med3DVLM-Qwen-2.5-7B}"
NPROC="${NPROC:-8}"
MASTER_PORT="${MASTER_PORT:-29511}"
ZERO_MRI="${ZERO_MRI:-1}"
ZERO_MRI_SIZE="${ZERO_MRI_SIZE:-}"

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
  --trust_remote_code \
  $( [[ "${ZERO_MRI}" == "1" ]] && echo --zero_mri ) \
  $( [[ -n "${ZERO_MRI_SIZE}" ]] && echo --zero_mri_size "${ZERO_MRI_SIZE}" )
