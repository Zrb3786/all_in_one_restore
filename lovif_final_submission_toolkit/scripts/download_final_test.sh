#!/usr/bin/env bash
set -euo pipefail
OUT_ROOT=${1:-/data1/zhangruibo/datasets/LoViF_AIO_Final_Test}
FILE_ID=${2:-1TBwosX7DOK7Aocbj_srKEwW4xUrucDsY}
mkdir -p "$OUT_ROOT"
cd "$OUT_ROOT"
python -m pip show gdown >/dev/null 2>&1 || python -m pip install -q gdown
if [ ! -f final_test.zip ]; then
  gdown "$FILE_ID" -O final_test.zip
fi
mkdir -p extracted
unzip -o final_test.zip -d extracted
find "$OUT_ROOT/extracted" -maxdepth 3 -type f \( -iname '*.jpg' -o -iname '*.png' -o -iname '*.jpeg' \) | head -20
printf '\n[INFO] If images are flat, use the directory printed above. If nested, set INPUT_DIR manually.\n'
