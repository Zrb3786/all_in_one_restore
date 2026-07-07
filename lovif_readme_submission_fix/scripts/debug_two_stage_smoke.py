#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import shutil
import tempfile
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
import torchvision.transforms.functional as TF

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_timed_module(code_root):
    p = Path(code_root) / "lovif_readme_submission_fix/scripts/infer_single_ckpt_jpeg96_timed.py"
    if not p.exists():
        raise FileNotFoundError(p)
    spec = importlib.util.spec_from_file_location("infer_timed", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tensor_to_pil(x):
    x = x.detach().cpu().squeeze(0).clamp(0, 1)
    return TF.to_pil_image(x)


def save_png(x, path, size=None):
    img = tensor_to_pil(x)
    if size is not None and img.size != size:
        img = img.resize(size, Image.BICUBIC)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def make_diff_heatmap(base, final):
    # base/final: [1,3,H,W], range [0,1]
    diff = (final - base).abs().mean(dim=1, keepdim=True).clamp(0, 1)
    # 放大差异，红色显示
    d = (diff * 8.0).clamp(0, 1)
    heat = torch.cat([d, torch.zeros_like(d), torch.zeros_like(d)], dim=1)
    return heat


def fit_image(img, size=(260, 260)):
    img = img.convert("RGB")
    img.thumbnail(size, Image.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(img, ((size[0] - img.width)//2, (size[1] - img.height)//2))
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single_ckpt", required=True)
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--code_root", default="/data1/zhangruibo/projects/DiffUIR")
    ap.add_argument("--sampling_timesteps", type=int, default=3)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--names", type=str, default="", help="comma separated stems, e.g. 0001,0101,0225,0401")
    ap.add_argument("--use_ema", action="store_true", default=True)
    args = ap.parse_args()

    code_root = Path(args.code_root).resolve()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timed = load_timed_module(code_root)

    single = torch.load(args.single_ckpt, map_location="cpu")
    base_ckpt, ref_ckpt = timed.get_single_keys(single)
    milestone = int(single.get("base_milestone", 310))

    model_mod = timed.import_module_dynamic(code_root, "src.model", "src/model.py")
    ref_mod = timed.import_module_dynamic(code_root, "models.moce_prompt_refiner", "models/moce_prompt_refiner.py")

    tmp = Path(tempfile.mkdtemp(prefix="lovif_debug_two_stage_"))
    try:
        torch.save(base_ckpt, tmp / f"model-{milestone}.pt")
        trainer = timed.build_diffuir_trainer(model_mod, tmp, milestone, args.sampling_timesteps, args.image_size)
        device = trainer.device

        refiner, cfg = timed.build_refiner(ref_mod, ref_ckpt, use_ema=args.use_ema)
        refiner = refiner.to(device).eval()

        all_paths = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
        if args.names.strip():
            want = {x.strip() for x in args.names.split(",") if x.strip()}
            paths = [p for p in all_paths if p.stem in want]
        else:
            paths = all_paths[:args.limit]

        if not paths:
            raise RuntimeError("No input images selected")

        print("[INFO] selected:", [p.name for p in paths])
        print("[INFO] refiner cfg:", cfg)

        subdirs = {
            "lq": out_dir / "00_lq",
            "base": out_dir / "01_stage1_clean310",
            "final": out_dir / "02_stage2_final",
            "diff": out_dir / "03_absdiff_final_minus_base",
            "compare": out_dir / "compare",
        }
        for d in subdirs.values():
            d.mkdir(parents=True, exist_ok=True)

        rows = []
        with torch.no_grad():
            for p in tqdm(paths, desc="two-stage-debug", dynamic_ncols=True):
                lq, size = timed.read_tensor(p)
                lq = lq.unsqueeze(0).to(device)

                base_samples = list(trainer.ema.ema_model.sample(lq, batch_size=1, last=True, task=str(p)))
                base = base_samples[-1].clamp(0, 1)

                final = refiner(lq, base).clamp(0, 1)
                heat = make_diff_heatmap(base, final)

                mae = float((final - base).abs().mean().detach().cpu())
                maxdiff = float((final - base).abs().max().detach().cpu())
                rows.append([p.name, mae, maxdiff])

                save_png(lq, subdirs["lq"] / f"{p.stem}.png", size=size)
                save_png(base, subdirs["base"] / f"{p.stem}.png", size=size)
                save_png(final, subdirs["final"] / f"{p.stem}.png", size=size)
                save_png(heat, subdirs["diff"] / f"{p.stem}.png", size=size)

        csv_path = out_dir / "two_stage_diff_stats.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filename", "final_vs_base_mae", "final_vs_base_maxdiff"])
            w.writerows(rows)

        print("[OK] stats:", csv_path)
        for r in rows:
            print(f"{r[0]}  final-base MAE={r[1]:.8f}  maxdiff={r[2]:.8f}")

        # HTML
        side_paths = []
        thumb = (260, 260)
        label_h = 48
        gap = 12
        cols = ["LQ", "Stage1 clean310", "Stage2 final", "|final-base| x8"]
        for name, mae, maxdiff in rows:
            stem = Path(name).stem
            imgs = [
                Image.open(subdirs["lq"] / f"{stem}.png"),
                Image.open(subdirs["base"] / f"{stem}.png"),
                Image.open(subdirs["final"] / f"{stem}.png"),
                Image.open(subdirs["diff"] / f"{stem}.png"),
            ]
            canvas = Image.new("RGB", (thumb[0]*4 + gap*3, thumb[1]+label_h), "white")
            draw = ImageDraw.Draw(canvas)
            for i, (img, lab) in enumerate(zip(imgs, cols)):
                x = i * (thumb[0] + gap)
                draw.text((x+4, 5), lab, fill=(0,0,0))
                canvas.paste(fit_image(img, thumb), (x, label_h))
            draw.text((4, label_h-18), f"{name}  MAE={mae:.6f}  Max={maxdiff:.6f}", fill=(0,0,0))
            out = subdirs["compare"] / f"compare_{stem}.jpg"
            canvas.save(out, quality=95)
            side_paths.append(out)

        html = subdirs["compare"] / "index.html"
        html.write_text(
            "<html><head><meta charset='utf-8'><title>Two-stage debug</title>"
            "<style>body{font-family:Arial;} img{max-width:100%;}.case{margin:22px 0;border-bottom:1px solid #ddd;}</style>"
            "</head><body>"
            "<h2>Two-stage inference debug: LQ | Stage1 clean310 | Stage2 final | Difference</h2>"
            "<p>If final-base MAE is non-zero, the refiner stage is active. Difference map is amplified by 8x in red.</p>"
            + "\n".join([f"<div class='case'><h3>{p.name}</h3><img src='{p.name}'></div>" for p in side_paths])
            + "</body></html>",
            encoding="utf-8"
        )
        print("[OK] html:", html)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
