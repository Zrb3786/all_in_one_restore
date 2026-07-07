import argparse
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as TF
from torchvision.utils import save_image

from models.moce_prompt_refiner import build_moce_prompt_refiner

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--base_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--refiner_ckpt", required=True)
    p.add_argument("--use_ema", action="store_true")
    p.add_argument("--suffix", default=".png")
    return p.parse_args()


def read_img(path):
    return TF.to_tensor(Image.open(path).convert("RGB")).float()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.refiner_ckpt, map_location="cpu")
    cfg = ckpt.get("config", {})
    model = build_moce_prompt_refiner(
        width=cfg.get("width", 48),
        num_blocks=cfg.get("num_blocks", cfg.get("blocks", 8)),
        residual_scale=cfg.get("residual_scale", 0.15),
    ).to(device)
    key = "ema" if args.use_ema and "ema" in ckpt else "model"
    model.load_state_dict(ckpt[key], strict=True)
    model.eval()

    input_dir = Path(args.input_dir)
    base_dir = Path(args.base_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
    base_map = {p.stem: p for p in base_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS}
    print("[INFO] inputs:", len(imgs), "base:", len(base_map), "ckpt:", args.refiner_ckpt, "key:", key)

    with torch.no_grad():
        for p in tqdm(imgs, dynamic_ncols=True):
            base_path = base_map.get(p.stem)
            if base_path is None:
                print("[WARN] missing base for", p.name)
                continue
            lq = read_img(p).unsqueeze(0).to(device)
            base = read_img(base_path).unsqueeze(0).to(device)
            pred = model(lq, base).clamp(0, 1)
            save_image(pred, str(out_dir / (p.stem + args.suffix)))
    print("[OK] saved to", out_dir)


if __name__ == "__main__":
    main()
