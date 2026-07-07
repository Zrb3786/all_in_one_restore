# LoViF All-in-One Final Submission Toolkit

This toolkit packages and runs one all-in-one restoration model:

1. DiffUIR clean-310 backbone
2. MoCEPromptRefiner v2-150k

Both are stored inside one `.pt` file and executed by one inference script. The inference script saves JPEG images with `quality=96, subsampling=0, optimize=True`.

## Expected server paths

```bash
DIFFUIR_ROOT=/data1/zhangruibo/projects/DiffUIR
BASE_CKPT=/data1/zhangruibo/runs/diffuir_lovif_clean_official_from300_seed42/ckpt/model-310.pt
REFINER_CKPT=/data1/zhangruibo/runs/diffuir_clean310_moce_refiner_v2_weather_ft100k/ckpt/refiner_step150000.pt
```

Adjust `REFINER_CKPT` if your final v2-150k checkpoint is under a different run directory.

## 1. Pack one checkpoint

```bash
python scripts/pack_single_ckpt_clean310_refiner.py \
  --base_ckpt "$BASE_CKPT" \
  --refiner_ckpt "$REFINER_CKPT" \
  --out /data1/zhangruibo/runs/lovif_final_submission_pkg/checkpoints/lovif_clean310_refiner_v2_150k_single.pt \
  --base_milestone 310
```

## 2. Download final test set

```bash
bash scripts/download_final_test.sh /data1/zhangruibo/datasets/LoViF_AIO_Final_Test 1TBwosX7DOK7Aocbj_srKEwW4xUrucDsY
```

Then inspect extracted folders and set `INPUT_DIR` to the flat folder containing test images.

## 3. Run final inference

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/infer_single_ckpt_jpeg96.py \
  --single_ckpt /data1/zhangruibo/runs/lovif_final_submission_pkg/checkpoints/lovif_clean310_refiner_v2_150k_single.pt \
  --input_dir "$INPUT_DIR" \
  --output_dir /data1/zhangruibo/runs/lovif_final_submission_pkg/results_jpeg96 \
  --code_root /data1/zhangruibo/projects/DiffUIR \
  --sampling_timesteps 3 \
  --image_size 256
```

## 4. Verify outputs

```bash
python scripts/verify_jpeg_results.py \
  --input_dir "$INPUT_DIR" \
  --output_dir /data1/zhangruibo/runs/lovif_final_submission_pkg/results_jpeg96
```

## 5. Zip results

```bash
cd /data1/zhangruibo/runs/lovif_final_submission_pkg/results_jpeg96
zip -r ../lovif_final_results_jpeg96.zip .
```

## 6. Build source package

```bash
bash scripts/build_source_package.sh \
  /data1/zhangruibo/projects/DiffUIR \
  /data1/zhangruibo/runs/lovif_final_submission_pkg \
  /data1/zhangruibo/runs/lovif_final_submission_pkg/checkpoints/lovif_clean310_refiner_v2_150k_single.pt
```

This creates:

```text
/data1/zhangruibo/runs/lovif_final_submission_pkg/lovif_final_source_code_with_single_ckpt.tar.gz
```

## Included training code

The source package keeps the useful training/inference code:

- DiffUIR official `src/`
- LoViF paired dataset loader
- clean DiffUIR resume training script
- MoCEPromptRefiner model
- refiner dataset/loss/training/inference scripts
- final single-checkpoint pack/inference/verification scripts

