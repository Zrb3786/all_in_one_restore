import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

TASKS = ["Blur", "Lowlight", "Haze", "Rain", "Snow"]

def read_rgb(p):
    return np.array(Image.open(p).convert("RGB")).astype(np.float32) / 255.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_root", required=True)
    ap.add_argument("--gt_root", default="/data1/zhangruibo/datasets/CVPR26_LoViF_AIO_split_seed42/val")
    args = ap.parse_args()
    pred_root = Path(args.pred_root); gt_root = Path(args.gt_root)
    all_psnr, all_ssim = [], []
    for task in TASKS:
        pred_dir = pred_root / task
        gt_dir = gt_root / task / "GT"
        psnrs, ssims = [], []
        for pred in sorted(pred_dir.glob("*.png")):
            gt = gt_dir / (pred.stem + ".jpg")
            if not gt.exists(): gt = gt_dir / (pred.stem + ".png")
            if not gt.exists():
                print("[WARN] missing GT", task, pred.name); continue
            a, b = read_rgb(pred), read_rgb(gt)
            if a.shape != b.shape:
                b = np.array(Image.open(gt).convert("RGB").resize((a.shape[1], a.shape[0]), Image.BICUBIC)).astype(np.float32)/255.0
            psnrs.append(peak_signal_noise_ratio(b, a, data_range=1.0))
            ssims.append(structural_similarity(b, a, data_range=1.0, channel_axis=-1))
        print(f"{task:9s} n={len(psnrs):4d} PSNR={np.mean(psnrs):.4f} SSIM={np.mean(ssims):.4f}")
        all_psnr += psnrs; all_ssim += ssims
    print("Overall")
    print(f"n={len(all_psnr)} PSNR={np.mean(all_psnr):.4f} SSIM={np.mean(all_ssim):.4f}")
if __name__ == "__main__":
    main()
