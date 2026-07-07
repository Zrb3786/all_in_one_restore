import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

TASK_MAP = {
    'fog':        ('Haze',     'Haze/LQ',     'Haze/GT'),
    'light_only': ('Lowlight', 'Lowlight/LQ', 'Lowlight/GT'),
    'rain':       ('Rain',     'Rain/LQ',     'Rain/GT'),
    'snow':       ('Snow',     'Snow/LQ',     'Snow/GT'),
    'blur':       ('Blur',     'Blur/LQ',     'Blur/GT'),
}


class LoViFDistillDataset(Dataset):
    """
    LoViF paired dataset with optional teacher cache for DiffUIR distillation.

    dataroot:
      /data1/.../CVPR26_LoViF_AIO_split_seed42/train
      /data1/.../CVPR26_LoViF_AIO_split_seed42/val

    teacher_root:
      /data1/.../runs/diffuir_teacher_cache_seed42/train
        Blur/*.png
        Lowlight/*.png
        Haze/*.png
        Rain/*.png
        Snow/*.png

    Returns keys compatible with a patched DiffUIR Trainer:
      adap, gt, teacher, A_paths, B_paths, teacher_paths
    """
    def __init__(
        self,
        dataroot,
        task,
        image_size=256,
        augment_flip=True,
        crop_patch=True,
        equalizeHist=False,
        teacher_root=None,
        use_teacher=True,
        teacher_suffix='.png',
    ):
        self.dataroot = Path(dataroot)
        self.task = task
        self.image_size = image_size
        self.augment_flip = augment_flip
        self.crop_patch = crop_patch
        self.equalizeHist = equalizeHist
        self.teacher_root = Path(teacher_root) if teacher_root else None
        self.use_teacher = use_teacher and self.teacher_root is not None
        self.teacher_suffix = teacher_suffix

        if task not in TASK_MAP:
            raise ValueError(f'Unknown task {task}; available={list(TASK_MAP)}')

        self.task_name, lq_rel, gt_rel = TASK_MAP[task]
        self.lq_dir = self.dataroot / lq_rel
        self.gt_dir = self.dataroot / gt_rel
        self.teacher_dir = self.teacher_root / self.task_name if self.use_teacher else None

        if not self.lq_dir.exists():
            raise FileNotFoundError(f'LQ dir not found: {self.lq_dir}')
        if not self.gt_dir.exists():
            raise FileNotFoundError(f'GT dir not found: {self.gt_dir}')
        if self.use_teacher and not self.teacher_dir.exists():
            raise FileNotFoundError(f'Teacher dir not found: {self.teacher_dir}')

        lq_names = sorted([
            p.name for p in self.lq_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        ])
        gt_names = {
            p.name for p in self.gt_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        }

        missing_gt = [n for n in lq_names if n not in gt_names]
        if missing_gt:
            raise RuntimeError(f'{task}: missing GT for {len(missing_gt)} files, sample={missing_gt[:10]}')

        if self.use_teacher:
            teacher_stems = {
                p.stem for p in self.teacher_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMG_EXTS
            }
            missing_teacher = [n for n in lq_names if Path(n).stem not in teacher_stems]
            if missing_teacher:
                raise RuntimeError(
                    f'{task}: missing teacher for {len(missing_teacher)} files, sample={missing_teacher[:10]}, '
                    f'teacher_dir={self.teacher_dir}'
                )

        self.names = lq_names
        print(
            f'[LoViFDistillDataset] task={task}, task_name={self.task_name}, len={len(self.names)}, '
            f'lq={self.lq_dir}, gt={self.gt_dir}, teacher={self.teacher_dir}'
        )

    def __len__(self):
        return len(self.names)

    def _read_rgb(self, path):
        return Image.open(path).convert('RGB')

    def _equalize_lowlight_input(self, img):
        arr = np.asarray(img)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        b, g, r = cv2.split(bgr)
        b = cv2.equalizeHist(b)
        g = cv2.equalizeHist(g)
        r = cv2.equalizeHist(r)
        out = cv2.merge((b, g, r))
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        return Image.fromarray(out)

    def _resize_if_needed(self, imgs):
        s = self.image_size
        w, h = imgs[0].size
        if w < s or h < s:
            nw = max(w, s)
            nh = max(h, s)
            imgs = [img.resize((nw, nh), Image.BICUBIC) for img in imgs]
        return imgs

    def _paired_crop(self, imgs):
        imgs = self._resize_if_needed(imgs)
        w, h = imgs[0].size
        s = self.image_size
        if self.crop_patch:
            left = random.randint(0, w - s)
            top = random.randint(0, h - s)
            imgs = [img.crop((left, top, left + s, top + s)) for img in imgs]
        else:
            imgs = [img.resize((s, s), Image.BICUBIC) for img in imgs]
        return imgs

    def __getitem__(self, index):
        name = self.names[index % len(self.names)]
        stem = Path(name).stem
        A_path = self.lq_dir / name
        B_path = self.gt_dir / name

        condition = self._read_rgb(A_path)
        gt = self._read_rgb(B_path)

        if self.task == 'light_only' and self.equalizeHist:
            condition = self._equalize_lowlight_input(condition)

        teacher_path = ''
        if self.use_teacher:
            # Prefer .png cache, but fall back to common extensions.
            candidates = [self.teacher_dir / f'{stem}{self.teacher_suffix}']
            candidates += [self.teacher_dir / f'{stem}{ext}' for ext in ['.png', '.jpg', '.jpeg']]
            teacher_path_obj = next((p for p in candidates if p.exists()), None)
            if teacher_path_obj is None:
                raise FileNotFoundError(f'Missing teacher for {A_path}, tried={candidates[:3]}')
            teacher_path = str(teacher_path_obj)
            teacher = self._read_rgb(teacher_path_obj)
            imgs = [condition, gt, teacher]
        else:
            teacher = gt.copy()
            imgs = [condition, gt, teacher]

        condition, gt, teacher = self._paired_crop(imgs)

        if self.augment_flip and random.random() < 0.5:
            condition = TF.hflip(condition)
            gt = TF.hflip(gt)
            teacher = TF.hflip(teacher)

        return {
            'adap': TF.to_tensor(condition).float(),
            'gt': TF.to_tensor(gt).float(),
            'teacher': TF.to_tensor(teacher).float(),
            'A_paths': str(A_path),
            'B_paths': str(B_path),
            'teacher_paths': teacher_path,
        }
