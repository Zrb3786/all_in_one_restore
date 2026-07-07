from pathlib import Path
from PIL import Image, ImageDraw
import math

cols = [
    ("LQ", Path("/data1/zhangruibo/datasets/CVPR26_LoViF_AIO/Val_AIO_task5_smoke25/LQ")),
    ("base-300", Path("/data1/zhangruibo/runs/diffuir_lovif_base300_smoke25/aio")),
    ("clean-310", Path("/data1/zhangruibo/runs/diffuir_lovif_clean_official_model310_smoke25/aio")),
    ("v1-305", Path("/data1/zhangruibo/runs/diffuir_lovif_loss_v1_model305_smoke25/aio")),
    ("v2-308", Path("/data1/zhangruibo/runs/diffuir_lovif_loss_v2_model308_smoke25/aio")),
    ("clean310-refiner", Path("/data1/zhangruibo/runs/diffuir_clean310_moce_refiner_v1_smoke25/aio")),
]
compare_dir = Path("/data1/zhangruibo/runs/diffuir_clean310_moce_refiner_v1_compare_smoke25/compare")
compare_dir.mkdir(parents=True, exist_ok=True)

def img_files(d):
    if not d.exists():
        print("[WARN] missing dir:", d)
        return []
    return sorted([p for p in d.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])

existing = []
for name, d in cols:
    fs = img_files(d)
    print(f"{name:18s} count={len(fs)} {d}")
    if fs:
        existing.append((name, d))
if not existing:
    raise SystemExit("no dirs")

lq_dir = existing[0][1]
lq_imgs = img_files(lq_dir)
maps = [(name, {p.stem: p for p in img_files(d)}) for name, d in existing[1:]]
rows = []
for lq in lq_imgs:
    row = [(existing[0][0], lq)]
    ok = True
    for name, mp in maps:
        p = mp.get(lq.stem)
        if p is None:
            print("[WARN] missing", name, "for", lq.name)
            ok = False
            break
        row.append((name, p))
    if ok:
        rows.append(row)
print("matched rows:", len(rows))
if not rows:
    raise SystemExit("no matched rows")

thumb_w, thumb_h = 230, 230
label_h, gap = 58, 10
ncols = len(existing)
item_w = thumb_w * ncols + gap * (ncols - 1)
item_h = thumb_h + label_h

def fit_image(path):
    img = Image.open(path).convert("RGB")
    img.thumbnail((thumb_w, thumb_h), Image.LANCZOS)
    canvas = Image.new("RGB", (thumb_w, thumb_h), "white")
    canvas.paste(img, ((thumb_w - img.width)//2, (thumb_h - img.height)//2))
    return canvas

def group(stem):
    try:
        idx = int(stem)
    except Exception:
        return ""
    if 1 <= idx <= 100: return "group-1"
    if 101 <= idx <= 200: return "group-2"
    if 201 <= idx <= 300: return "group-3"
    if 301 <= idx <= 400: return "group-4"
    if 401 <= idx <= 500: return "group-5"
    return ""

side_paths = []
for row in rows:
    canvas = Image.new("RGB", (item_w, item_h), "white")
    draw = ImageDraw.Draw(canvas)
    stem = row[0][1].stem
    for j, (label, path) in enumerate(row):
        x = j * (thumb_w + gap)
        text = f"LQ: {path.name} {group(stem)}" if j == 0 else label
        draw.text((x + 4, 5), text, fill=(0,0,0))
        canvas.paste(fit_image(path), (x, label_h))
    out = compare_dir / f"compare_{stem}.jpg"
    canvas.save(out, quality=95)
    side_paths.append(out)

sheet_w = item_w
sheet_h = len(side_paths) * item_h + (len(side_paths)-1)*18
sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
for i, p in enumerate(side_paths):
    sheet.paste(Image.open(p).convert("RGB"), (0, i * (item_h + 18)))
sheet_path = compare_dir / "contact_sheet.jpg"
sheet.save(sheet_path, quality=95)

html_path = compare_dir / "index.html"
html_path.write_text(
    "<html><head><meta charset='utf-8'><title>clean310 moce refiner smoke25</title>"
    "<style>body{font-family:Arial;} img{max-width:100%;}.case{margin:24px 0;border-bottom:1px solid #ddd;}</style>"
    "</head><body>"
    "<h2>Clean-310 MoCE Refiner Smoke25 Compare</h2>"
    "<p>Columns: " + " | ".join([c[0] for c in existing]) + "</p>"
    "<p>Goal: not worse than clean-310; improve lowlight/weather; preserve blur and structure.</p>"
    f"<img src='{sheet_path.name}'>"
    "<hr>" + "\n".join([f"<div class='case'><h3>{p.name}</h3><img src='{p.name}'></div>" for p in side_paths]) +
    "</body></html>", encoding="utf-8")
print("[OK] html:", html_path)
