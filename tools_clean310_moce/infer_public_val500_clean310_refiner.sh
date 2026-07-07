#!/usr/bin/env bash
set -euo pipefail
GPU_ID=${GPU_ID:-1}
REFINER_CKPT=${REFINER_CKPT:?set REFINER_CKPT}
CLEAN_CKPT=/data1/zhangruibo/runs/diffuir_lovif_clean_official_from300_seed42/ckpt
LQ_DIR=/data1/zhangruibo/datasets/CVPR26_LoViF_AIO/Val_AIO/LQ
BASE_DIR=/data1/zhangruibo/runs/diffuir_lovif_clean_official_model310_val500/aio
OUT_DIR=/data1/zhangruibo/runs/diffuir_clean310_moce_refiner_v1_val500/aio
cd /data1/zhangruibo/projects/DiffUIR
mkdir -p "$BASE_DIR" "$OUT_DIR"
CUDA_VISIBLE_DEVICES=$GPU_ID env -u PYTORCH_CUDA_ALLOC_CONF python infer_lovif_flat.py \
  --input_dir "$LQ_DIR" \
  --output_dir "$BASE_DIR" \
  --ckpt_dir "$CLEAN_CKPT" \
  --milestone 310 \
  --sampling_timesteps 3
CUDA_VISIBLE_DEVICES=$GPU_ID python infer_refiner_clean310_moce.py \
  --input_dir "$LQ_DIR" \
  --base_dir "$BASE_DIR" \
  --output_dir "$OUT_DIR" \
  --refiner_ckpt "$REFINER_CKPT" \
  --use_ema
