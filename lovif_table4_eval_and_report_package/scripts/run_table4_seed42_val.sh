#!/usr/bin/env bash
set -euo pipefail

DIFFUIR_ROOT=${DIFFUIR_ROOT:-/data1/zhangruibo/projects/DiffUIR}
VAL_ROOT=${VAL_ROOT:-/data1/zhangruibo/datasets/CVPR26_LoViF_AIO_split_seed42/val}
OUT_ROOT=${OUT_ROOT:-/data1/zhangruibo/runs/lovif_table4_seed42_eval}
GPU_ID=${GPU_ID:-1}

BASE300_CKPT_DIR=${BASE300_CKPT_DIR:-/data1/zhangruibo/projects/DiffUIR/ckpt_universal/diffuir}
CLEAN310_CKPT_DIR=${CLEAN310_CKPT_DIR:-/data1/zhangruibo/runs/diffuir_lovif_clean_official_from300_seed42/ckpt}
CLEAN310_CACHE_VAL=${CLEAN310_CACHE_VAL:-/data1/zhangruibo/runs/diffuir_clean_model310_cache_seed42/val}
FINAL_REFINER_CKPT=${FINAL_REFINER_CKPT:-/data1/zhangruibo/runs/diffuir_clean310_moce_refiner_v2_weather_ft100k/ckpt/refiner_step150000.pt}

SAMPLING_TIMESTEPS=${SAMPLING_TIMESTEPS:-3}
TASKS=(Blur Lowlight Haze Rain Snow)

mkdir -p "$OUT_ROOT/preds"
cd "$DIFFUIR_ROOT"

echo "[INFO] DIFFUIR_ROOT=$DIFFUIR_ROOT"
echo "[INFO] VAL_ROOT=$VAL_ROOT"
echo "[INFO] OUT_ROOT=$OUT_ROOT"
echo "[INFO] GPU_ID=$GPU_ID"
echo "[INFO] FINAL_REFINER_CKPT=$FINAL_REFINER_CKPT"

for task in "${TASKS[@]}"; do
  echo "==== [Table4] infer base-300 val $task ===="
  CUDA_VISIBLE_DEVICES=$GPU_ID env -u PYTORCH_CUDA_ALLOC_CONF \
  python infer_lovif_flat.py \
    --input_dir "$VAL_ROOT/$task/LQ" \
    --output_dir "$OUT_ROOT/preds/base300/$task" \
    --ckpt_dir "$BASE300_CKPT_DIR" \
    --milestone 300 \
    --sampling_timesteps "$SAMPLING_TIMESTEPS"
done

for task in "${TASKS[@]}"; do
  echo "==== [Table4] infer clean-310 val $task ===="
  CUDA_VISIBLE_DEVICES=$GPU_ID env -u PYTORCH_CUDA_ALLOC_CONF \
  python infer_lovif_flat.py \
    --input_dir "$VAL_ROOT/$task/LQ" \
    --output_dir "$OUT_ROOT/preds/clean310/$task" \
    --ckpt_dir "$CLEAN310_CKPT_DIR" \
    --milestone 310 \
    --sampling_timesteps "$SAMPLING_TIMESTEPS"
done

for task in "${TASKS[@]}"; do
  echo "==== [Table4] infer final refiner-v2-150k val $task ===="
  BASE_DIR="$CLEAN310_CACHE_VAL/$task"
  if [ ! -d "$BASE_DIR" ]; then
    echo "[WARN] clean310 cache missing for $task, fallback to newly generated clean310 predictions"
    BASE_DIR="$OUT_ROOT/preds/clean310/$task"
  fi

  CUDA_VISIBLE_DEVICES=$GPU_ID python infer_refiner_clean310_moce.py \
    --input_dir "$VAL_ROOT/$task/LQ" \
    --base_dir "$BASE_DIR" \
    --output_dir "$OUT_ROOT/preds/final_refiner_v2_150k/$task" \
    --refiner_ckpt "$FINAL_REFINER_CKPT" \
    --use_ema
done

python lovif_table4_eval_and_report_package/scripts/eval_table4_metrics.py \
  --gt_root "$VAL_ROOT" \
  --pred_root "$OUT_ROOT/preds" \
  --out_csv "$OUT_ROOT/table4_metrics.csv" \
  --out_json "$OUT_ROOT/table4_metrics.json" \
  --out_tex "$OUT_ROOT/table4_latex.tex"

python lovif_table4_eval_and_report_package/scripts/make_report_tables.py \
  --table4_csv "$OUT_ROOT/table4_metrics.csv" \
  --out_dir "$OUT_ROOT" \
  --final_score 28.47 \
  --final_psnr 22.65 \
  --final_ssim 0.77 \
  --final_lpips 0.38

echo "[OK] Table4 outputs saved in: $OUT_ROOT"
echo "[OK] Table4 LaTeX: $OUT_ROOT/table4_latex.tex"
echo "[OK] Final score LaTeX: $OUT_ROOT/final_score_latex.tex"
echo "[OK] Combined report tables: $OUT_ROOT/report_ready_tables.tex"
