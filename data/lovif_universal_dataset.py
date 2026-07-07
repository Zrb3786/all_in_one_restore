import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

TASK_MAP = {
    "fog":        ("Haze/LQ",     "Haze/GT"),
    "light_only": ("Lowlight/LQ", "Lowlight/GT"),
    "rain":       ("Rain/LQ",    "Rain/GT"),
    "snow":       ("Snow/LQ",    "Snow/GT"),
    "blur":       ("Blur/LQ",    "Blur/GT"),
}


class LoViFAlignedDataset(Dataset):
    """
    LoViF paired dataset for DiffUIR finetuning.

    dataroot can be either:
      /data1/.../CVPR26_LoViF_AIO
    or:
      /data1/.../CVPR26_LoViF_AIO_split_seed42/train
      /data1/.../CVPR26_LoViF_AIO_split_seed42/val

    Expected:
      dataroot/Blur/LQ, dataroot/Blur/GT
      dataroot/Lowlight/LQ, dataroot/Lowlight/GT
      dataroot/Rain/LQ, dataroot/Rain/GT
      dataroot/Snow/LQ, dataroot/Snow/GT
      dataroot/Haze/LQ, dataroot/Haze/GT
    """
    def __init__(
        self,
        dataroot,
        task,
        image_size=256,
        augment_flip=True,
        crop_patch=True,
        equalizeHist=True,
    ):
        self.dataroot = Path(dataroot)
        self.task = task
        self.image_size = image_size
        self.augment_flip = augment_flip
        self.crop_patch = crop_patch
        self.equalizeHist = equalizeHist

        if task not in TASK_MAP:
            raise ValueError(f"Unknown task {task}; available={list(TASK_MAP)}")

        lq_rel, gt_rel = TASK_MAP[task]
        self.lq_dir = self.dataroot / lq_rel
        self.gt_dir = self.dataroot / gt_rel

        if not self.lq_dir.exists():
            raise FileNotFoundError(f"LQ dir not found: {self.lq_dir}")
        if not self.gt_dir.exists():
            raise FileNotFoundError(f"GT dir not found: {self.gt_dir}")

        lq_names = sorted([
            p.name for p in self.lq_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        ])
        gt_names = {
            p.name for p in self.gt_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        }

        missing = [n for n in lq_names if n not in gt_names]
        if missing:
            raise RuntimeError(f"{task}: missing GT for {len(missing)} files, sample={missing[:10]}")

        self.names = lq_names
        print(f"[LoViFAlignedDataset] task={task}, len={len(self.names)}, lq={self.lq_dir}, gt={self.gt_dir}")

    def __len__(self):
        return len(self.names)

    def _read_rgb(self, path):
        return Image.open(path).convert("RGB")

    def _equalize_lowlight_input(self, img):
        # Official DiffUIR applies cv2.equalizeHist on LOL condition/input.
        arr = np.asarray(img)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        b, g, r = cv2.split(bgr)
        b = cv2.equalizeHist(b)
        g = cv2.equalizeHist(g)
        r = cv2.equalizeHist(r)
        out = cv2.merge((b, g, r))
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        return Image.fromarray(out)

    def _resize_if_needed(self, a, b):
        w, h = a.size
        s = self.image_size
        if w < s or h < s:
            nw = max(w, s)
            nh = max(h, s)
            a = a.resize((nw, nh), Image.BICUBIC)
            b = b.resize((nw, nh), Image.BICUBIC)
        return a, b

    def _paired_crop(self, a, b):
        a, b = self._resize_if_needed(a, b)
        w, h = a.size
        s = self.image_size

        if self.crop_patch:
            left = random.randint(0, w - s)
            top = random.randint(0, h - s)
            a = a.crop((left, top, left + s, top + s))
            b = b.crop((left, top, left + s, top + s))
        else:
            a = a.resize((s, s), Image.BICUBIC)
            b = b.resize((s, s), Image.BICUBIC)

        return a, b

    def __getitem__(self, index):
        name = self.names[index % len(self.names)]
        A_path = self.lq_dir / name
        B_path = self.gt_dir / name

        condition = self._read_rgb(A_path)
        gt = self._read_rgb(B_path)

        # Keep DiffUIR official low-light behavior.
        # It equalizes only input/condition for LOL-style low-light.
        if self.task == "light_only" and self.equalizeHist:
            condition = self._equalize_lowlight_input(condition)

        condition, gt = self._paired_crop(condition, gt)

        if self.augment_flip and random.random() < 0.5:
            condition = TF.hflip(condition)
            gt = TF.hflip(gt)

        condition = TF.to_tensor(condition).float()
        gt = TF.to_tensor(gt).float()

        return {
            "adap": condition,
            "gt": gt,
            "A_paths": str(A_path),
            "B_paths": str(B_path),
        }
