# LoViF final submission fix: readme.txt + timed JPEG96 inference

This toolkit fixes the Codabench submission format by writing the required `readme.txt` into the result directory and into the final `submission.zip`.

Required `readme.txt` format:

```text
runtime per img [s] : <measured_runtime>
CPU[1] / GPU[0] : 0
Extra Data [1] / No Extra Data [0] : 1
Other description : LoViF
```

The inference script loads a single packed checkpoint and internally executes:

```text
LQ -> DiffUIR clean310 -> MoCEPromptRefiner -> JPEG quality=96 output
```

It avoids `from src.model import ...` and dynamically imports code from `--code_root`.

## Deployment

Copy this folder into DiffUIR root, for example:

```bash
cd /data1/zhangruibo/projects/DiffUIR
unzip -o /path/to/lovif_readme_submission_fix.zip -d /data1/zhangruibo/projects/DiffUIR/
```

## Inference

```bash
INPUT_DIR=/data1/zhangruibo/datasets/LoViF_AIO_Final_Test/TestLQ_extracted/TestLQ
OUT_DIR=/data1/zhangruibo/runs/lovif_final_submission_pkg/results_jpeg96_testlq_readme
SINGLE_CKPT=/data1/zhangruibo/runs/lovif_final_submission_pkg/checkpoints/lovif_clean310_refiner_v2_150k_single.pt
CUDA_VISIBLE_DEVICES=1 python lovif_readme_submission_fix/scripts/infer_single_ckpt_jpeg96_timed.py \
  --single_ckpt "$SINGLE_CKPT" \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUT_DIR" \
  --code_root /data1/zhangruibo/projects/DiffUIR \
  --sampling_timesteps 3 \
  --image_size 256 \
  --device_type 0 \
  --extra_data 1 \
  --description LoViF
```

## Package submission

```bash
python lovif_readme_submission_fix/scripts/make_submission_zip_with_readme.py \
  --result_dir "$OUT_DIR" \
  --zip_path /data1/zhangruibo/runs/lovif_final_submission_pkg/submission.zip \
  --expect_count 500
```

Upload `submission.zip` to Codabench.
