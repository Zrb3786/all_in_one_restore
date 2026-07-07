#!/usr/bin/env bash
set -euo pipefail

DIFFUIR_ROOT=${1:-/data1/zhangruibo/projects/DiffUIR}
OUT_ROOT=${2:-/data1/zhangruibo/runs/lovif_final_submission_pkg}
SINGLE_CKPT=${3:-/data1/zhangruibo/runs/lovif_final_submission_pkg/checkpoints/lovif_clean310_refiner_v2_150k_single.pt}

SOURCE_DIR="$OUT_ROOT/source"
CKPT_DIR="$OUT_ROOT/checkpoints"

mkdir -p "$OUT_ROOT" "$CKPT_DIR"
rm -rf "$SOURCE_DIR"
mkdir -p "$SOURCE_DIR"

cd "$DIFFUIR_ROOT"

echo "[INFO] DIFFUIR_ROOT=$DIFFUIR_ROOT"
echo "[INFO] OUT_ROOT=$OUT_ROOT"
echo "[INFO] SOURCE_DIR=$SOURCE_DIR"
echo "[INFO] CKPT_DIR=$CKPT_DIR"
echo "[INFO] SINGLE_CKPT=$SINGLE_CKPT"

# Copy necessary DiffUIR code and our LoViF additions.
# Avoid heavy caches/runs/datasets/checkpoints.
copy_item() {
  local item="$1"
  if [ -e "$item" ]; then
    rsync -a \
      --exclude '.git' \
      --exclude '__pycache__' \
      --exclude '*.pyc' \
      --exclude 'runs' \
      --exclude 'wandb' \
      --exclude 'ckpt_universal' \
      --exclude 'datasets' \
      --exclude 'outputs' \
      "$item" "$SOURCE_DIR/"
    echo "[OK] copied $item"
  else
    echo "[WARN] missing item, skipped: $item"
  fi
}

copy_item src
copy_item data
copy_item models
copy_item losses
copy_item tools_clean310_moce

copy_item infer_lovif_flat.py
copy_item infer_refiner_clean310_moce.py
copy_item train_lovif_clean_resume.py
copy_item train_refiner_clean310_moce.py

# Add final submission scripts.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cp "$SCRIPT_DIR"/*.py "$SOURCE_DIR/" 2>/dev/null || true
cp "$SCRIPT_DIR"/*.sh "$SOURCE_DIR/" 2>/dev/null || true

# Add toolkit README if exists.
if [ -f "$SCRIPT_DIR/../README_SUBMISSION.md" ]; then
  cp "$SCRIPT_DIR/../README_SUBMISSION.md" "$SOURCE_DIR/README_SUBMISSION.md"
fi

# Add requirements.
if [ -f "$SCRIPT_DIR/../requirements.txt" ]; then
  cp "$SCRIPT_DIR/../requirements.txt" "$SOURCE_DIR/requirements.txt"
elif [ -f "$DIFFUIR_ROOT/requirements.txt" ]; then
  cp "$DIFFUIR_ROOT/requirements.txt" "$SOURCE_DIR/requirements.txt"
else
  cat > "$SOURCE_DIR/requirements.txt" <<'REQ'
torch
torchvision
numpy
Pillow
opencv-python
einops
accelerate
tqdm
scikit-image
REQ
fi

# Copy single checkpoint if needed.
if [ -f "$SINGLE_CKPT" ]; then
  TARGET_CKPT="$CKPT_DIR/$(basename "$SINGLE_CKPT")"

  SRC_REAL="$(readlink -f "$SINGLE_CKPT")"
  TGT_REAL="$(readlink -f "$TARGET_CKPT" 2>/dev/null || true)"

  if [ "$SRC_REAL" = "$TGT_REAL" ]; then
    echo "[INFO] single ckpt already exists at target: $TARGET_CKPT"
  else
    cp "$SINGLE_CKPT" "$TARGET_CKPT"
    echo "[OK] copied single ckpt to $TARGET_CKPT"
  fi
else
  echo "[WARN] single checkpoint not found yet: $SINGLE_CKPT"
fi

# Add a small manifest.
cat > "$SOURCE_DIR/PACKAGE_MANIFEST.txt" <<EOF
LoViF 2026 final source package

Included:
- DiffUIR necessary source code
- LoViF data loaders and training scripts
- Clean310 + MoCE refiner inference/training code
- Single-checkpoint inference script
- requirements.txt
- trained single model checkpoint under checkpoints/

Single checkpoint:
$(basename "$SINGLE_CKPT")
EOF

tar -czf "$OUT_ROOT/lovif_final_source_code_with_single_ckpt.tar.gz" -C "$OUT_ROOT" source checkpoints

printf '[OK] source package: %s\n' "$OUT_ROOT/lovif_final_source_code_with_single_ckpt.tar.gz"
