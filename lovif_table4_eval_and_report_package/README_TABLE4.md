# LoViF Table 4 Evaluation + Report Completion Toolkit

This toolkit reproduces the internal seed42 validation comparison for the factsheet Table 4
**without running the refiner-100k row**.

Rows:
1. DiffUIR base-300
2. DiffUIR clean-310
3. Clean-310 + MoCE-Prompt Refiner v2-150k (final model)

Default server paths:
- DiffUIR project: `/data1/zhangruibo/projects/DiffUIR`
- seed42 val split: `/data1/zhangruibo/datasets/CVPR26_LoViF_AIO_split_seed42/val`
- clean310 cache: `/data1/zhangruibo/runs/diffuir_clean_model310_cache_seed42/val`
- final refiner checkpoint: `/data1/zhangruibo/runs/diffuir_clean310_moce_refiner_v2_weather_ft100k/ckpt/refiner_step150000.pt`

## Run

```bash
cd /data1/zhangruibo/projects/DiffUIR
conda activate /data1/zhangruibo/conda_envs/diffuir
unzip -o /path/to/lovif_table4_eval_and_report_package.zip -d .
bash lovif_table4_eval_and_report_package/scripts/run_table4_seed42_val.sh
```

Outputs:
```text
/data1/zhangruibo/runs/lovif_table4_seed42_eval/
  preds/
  table4_metrics.csv
  table4_metrics.json
  table4_latex.tex
  final_score_latex.tex
  report_ready_tables.tex
```

If LPIPS is unavailable in the environment, PSNR(Y) and SSIM(Y) are still computed and LPIPS is marked as `N/A`.
To enable LPIPS:

```bash
pip install lpips
```

Known final Codabench result:
- Final Score: 28.47
- PSNR(Y): 22.65
- SSIM(Y): 0.77
- LPIPS: 0.38
