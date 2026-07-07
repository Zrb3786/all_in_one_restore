#!/usr/bin/env bash
set -euo pipefail
GPU_ID=${GPU_ID:-1}
REFINER_CKPT=${REFINER_CKPT:?set REFINER_CKPT}
OUT_ROOT=${OUT_ROOT:-/data1/zhangruibo/runs/diffuir_clean310_moce_refiner_v1_val}
VAL_ROOT=/data1/zhangruibo/datasets/CVPR26_LoViF_AIO_split_seed42/val
BASE_ROOT=/data1/zhangruibo/runs/diffuir_clean_model310_cache_seed42/val
for task in Blur Lowlight Haze Rain Snow; do
  echo "==== infer refiner val $task ===="
  CUDA_VISIBLE_DEVICES=$GPU_ID python infer_refiner_clean310_moce.py \
    --input_dir "$VAL_ROOT/$task/LQ" \
    --base_dir "$BASE_ROOT/$task" \
    --output_dir "$OUT_ROOT/$task" \
    --refiner_ckpt "$REFINER_CKPT" \
    --use_ema
done
