import random
from pathlib import Path
from typing import Dict, Tuple

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

TASKS = ["Blur", "Lowlight", "Haze", "Rain", "Snow"]
TASK_TO_IDX = {t: i for i, t in enumerate(TASKS)}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class LoViFRefinerDataset(Dataset):
    """LoViF paired dataset for training a post-DiffUIR refiner.

    Expected:
      dataroot/{Task}/LQ, dataroot/{Task}/GT
      base_root/{Task}/{same_stem}.png or .jpg
    """
    def __init__(self, dataroot: str, base_root: str, image_size: int = 256, augment: bool = True, crop: bool = True):
        self.dataroot = Path(dataroot)
        self.base_root = Path(base_root)
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.crop = bool(crop)
        self.samples = []

        for task in TASKS:
            lq_dir = self.dataroot / task / "LQ"
            gt_dir = self.dataroot / task / "GT"
            base_dir = self.base_root / task
            if not lq_dir.exists():
                raise FileNotFoundError(f"Missing LQ dir: {lq_dir}")
            if not gt_dir.exists():
                raise FileNotFoundError(f"Missing GT dir: {gt_dir}")
            if not base_dir.exists():
                raise FileNotFoundError(f"Missing base cache dir: {base_dir}")

            lq_files = sorted([p for p in lq_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
            gt_map = {p.stem: p for p in gt_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS}
            base_map = {p.stem: p for p in base_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS}
            missing = []
            for lq in lq_files:
                gt = gt_map.get(lq.stem)
                base = base_map.get(lq.stem)
                if gt is None or base is None:
                    missing.append(lq.name)
                    continue
                self.samples.append({"task": task, "lq": lq, "gt": gt, "base": base, "name": lq.name})
            if missing:
                print(f"[WARN] {task}: skipped {len(missing)} files due to missing GT/base, sample={missing[:5]}")
            print(f"[DATA] {task}: {len(lq_files) - len(missing)} paired samples")

        if not self.samples:
            raise RuntimeError("No samples found for LoViFRefinerDataset")
        print(f"[DATA] total: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def _read(self, path: Path):
        return Image.open(path).convert("RGB")

    def _resize_if_needed(self, imgs):
        s = self.image_size
        w, h = imgs[0].size
        if w < s or h < s:
            nw, nh = max(w, s), max(h, s)
            imgs = [im.resize((nw, nh), Image.BICUBIC) for im in imgs]
        return imgs

    def _paired_crop(self, imgs):
        s = self.image_size
        imgs = self._resize_if_needed(imgs)
        w, h = imgs[0].size
        if self.crop:
            left = random.randint(0, w - s)
            top = random.randint(0, h - s)
            imgs = [im.crop((left, top, left + s, top + s)) for im in imgs]
        else:
            imgs = [im.resize((s, s), Image.BICUBIC) for im in imgs]
        return imgs

    def __getitem__(self, index):
        item = self.samples[index % len(self.samples)]
        lq = self._read(item["lq"])
        gt = self._read(item["gt"])
        base = self._read(item["base"])
        lq, gt, base = self._paired_crop([lq, gt, base])

        if self.augment and random.random() < 0.5:
            lq = TF.hflip(lq); gt = TF.hflip(gt); base = TF.hflip(base)
        if self.augment and random.random() < 0.5:
            lq = TF.vflip(lq); gt = TF.vflip(gt); base = TF.vflip(base)

        return {
            "lq": TF.to_tensor(lq).float(),
            "gt": TF.to_tensor(gt).float(),
            "base": TF.to_tensor(base).float(),
            "task_idx": TASK_TO_IDX[item["task"]],
            "task": item["task"],
            "name": item["name"],
            "lq_path": str(item["lq"]),
            "base_path": str(item["base"]),
        }
