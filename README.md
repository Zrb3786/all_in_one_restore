# LoViF Final Source Code Only

This package contains source code only. It does not include model checkpoints, restored images, datasets, or submission zip files.

Final method:
1. DiffUIR clean310 backbone.
2. Lightweight MoCE-Prompt residual refiner.
3. Both stages are packed into a single checkpoint during final inference.

Expected checkpoint, provided separately:
checkpoints/lovif_clean310_refiner_v2_150k_single.pt

Codabench result zip format:
- 0001.jpg ... 0500.jpg
- readme.txt

JPEG saving format:
format="JPEG", quality=96, subsampling=0, optimize=True

Measured runtime on one A800 GPU:
runtime per img [s] : 0.7263

Important inference scripts:
- lovif_readme_submission_fix/scripts/infer_single_ckpt_jpeg96_timed.py
- lovif_readme_submission_fix/scripts/make_submission_zip_with_readme.py
- lovif_readme_submission_fix/scripts/verify_submission_with_readme.py

Important model files:
- src/model.py
- models/moce_prompt_refiner.py
- losses/refiner_losses.py
- data/refiner_dataset.py

