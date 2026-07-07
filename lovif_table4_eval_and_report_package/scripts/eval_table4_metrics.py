#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

TASKS = ["Blur", "Lowlight", "Haze", "Rain", "Snow"]
DEFAULT_MODELS = [
    ("base300", "DiffUIR base-300"),
    ("clean310", "DiffUIR clean-310"),
    ("final_refiner_v2_150k", "Clean-310 + MoCE-Prompt Refiner"),
]

def read_rgb(path: Path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img).astype(np.float32) / 255.0

def rgb_to_y(x):
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]

def find_pred(pred_dir: Path, stem: str):
    for ext in [".png", ".jpg", ".jpeg", ".JPG", ".PNG", ".JPEG"]:
        p = pred_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None

def compute_psnr_ssim_y(gt, pred):
    if pred.shape != gt.shape:
        pred_img = Image.fromarray((pred * 255).clip(0, 255).astype(np.uint8)).resize((gt.shape[1], gt.shape[0]), Image.BICUBIC)
        pred = np.asarray(pred_img).astype(np.float32) / 255.0
    gy = rgb_to_y(gt)
    py = rgb_to_y(pred)
    psnr = peak_signal_noise_ratio(gy, py, data_range=1.0)
    ssim = structural_similarity(gy, py, data_range=1.0)
    return float(psnr), float(ssim)

def try_lpips_init(device="cuda"):
    try:
        import torch
        import lpips
        model = lpips.LPIPS(net="alex").to(device).eval()
        return model, torch
    except Exception as e:
        print(f"[WARN] LPIPS unavailable, will write N/A: {repr(e)}")
        return None, None

def compute_lpips(lpips_model, torch_mod, gt, pred, device="cuda"):
    if lpips_model is None:
        return None
    if pred.shape != gt.shape:
        pred_img = Image.fromarray((pred * 255).clip(0, 255).astype(np.uint8)).resize((gt.shape[1], gt.shape[0]), Image.BICUBIC)
        pred = np.asarray(pred_img).astype(np.float32) / 255.0
    with torch_mod.no_grad():
        gt_t = torch_mod.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).float().to(device) * 2 - 1
        pr_t = torch_mod.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).float().to(device) * 2 - 1
        return float(lpips_model(pr_t, gt_t).item())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_root", required=True)
    ap.add_argument("--pred_root", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_tex", required=True)
    ap.add_argument("--models", nargs="*", default=[m[0] for m in DEFAULT_MODELS])
    ap.add_argument("--no_lpips", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    gt_root = Path(args.gt_root)
    pred_root = Path(args.pred_root)

    name_map = dict(DEFAULT_MODELS)
    lpips_model, torch_mod = (None, None) if args.no_lpips else try_lpips_init(args.device)

    rows = []
    summary = {}

    for model_key in args.models:
        display = name_map.get(model_key, model_key)
        summary[model_key] = {"display": display, "tasks": {}}
        overall_psnr, overall_ssim, overall_lpips = [], [], []

        for task in TASKS:
            gt_dir = gt_root / task / "GT"
            pred_dir = pred_root / model_key / task
            if not pred_dir.exists():
                print(f"[WARN] missing pred dir: {pred_dir}")
                continue

            psnrs, ssims, lpipss = [], [], []
            gt_files = sorted([p for p in gt_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
            for gt_path in gt_files:
                pred_path = find_pred(pred_dir, gt_path.stem)
                if pred_path is None:
                    print(f"[WARN] missing pred for {model_key}/{task}/{gt_path.name}")
                    continue
                gt = read_rgb(gt_path)
                pred = read_rgb(pred_path)
                psnr, ssim = compute_psnr_ssim_y(gt, pred)
                psnrs.append(psnr)
                ssims.append(ssim)
                if lpips_model is not None:
                    lp = compute_lpips(lpips_model, torch_mod, gt, pred, args.device)
                    if lp is not None:
                        lpipss.append(lp)

            task_row = {
                "model_key": model_key,
                "model": display,
                "task": task,
                "n": len(psnrs),
                "psnr_y": float(np.mean(psnrs)) if psnrs else None,
                "ssim_y": float(np.mean(ssims)) if ssims else None,
                "lpips": float(np.mean(lpipss)) if lpipss else None,
            }
            rows.append(task_row)
            summary[model_key]["tasks"][task] = task_row
            overall_psnr += psnrs
            overall_ssim += ssims
            overall_lpips += lpipss

        overall = {
            "model_key": model_key,
            "model": display,
            "task": "Overall",
            "n": len(overall_psnr),
            "psnr_y": float(np.mean(overall_psnr)) if overall_psnr else None,
            "ssim_y": float(np.mean(overall_ssim)) if overall_ssim else None,
            "lpips": float(np.mean(overall_lpips)) if overall_lpips else None,
        }
        rows.append(overall)
        summary[model_key]["overall"] = overall

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model_key", "model", "task", "n", "psnr_y", "ssim_y", "lpips"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    Path(args.out_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    def fmt(x):
        return "N/A" if x is None else f"{x:.3f}"

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Internal seed-42 validation metrics. Each cell reports PSNR(Y) / SSIM(Y) / LPIPS. The refiner-100k row is intentionally omitted.}",
        r"\label{tab:val_metrics_filled}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Blur} & \textbf{Low-light} & \textbf{Haze} & \textbf{Rain} & \textbf{Snow} & \textbf{Overall}\\",
        r"\midrule",
    ]
    for model_key in args.models:
        display = summary[model_key]["display"]
        cells = []
        for task in TASKS + ["Overall"]:
            r = summary[model_key]["overall"] if task == "Overall" else summary[model_key]["tasks"].get(task)
            if not r:
                cells.append("N/A")
            else:
                cells.append(f"{fmt(r['psnr_y'])}/{fmt(r['ssim_y'])}/{fmt(r['lpips'])}")
        lines.append(display.replace("_", r"\_") + " & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    Path(args.out_tex).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] CSV: {args.out_csv}")
    print(f"[OK] JSON: {args.out_json}")
    print(f"[OK] LaTeX: {args.out_tex}")

if __name__ == "__main__":
    main()
