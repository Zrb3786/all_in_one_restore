#!/usr/bin/env python3
"""Final LoViF inference from ONE packed checkpoint.

It performs two sequential internal stages from one .pt file:
  1) DiffUIR clean checkpoint inference
  2) MoCEPromptRefiner inference

Outputs are saved as JPEG quality=96, subsampling=0, optimize=True, with the
same stem/name expected by LoViF/Codabench. This script intentionally avoids
`from src.model import ...`; it uses importlib with an explicit code_root.
"""
import argparse
import importlib
import os
from pathlib import Path
import shutil
import sys
import tempfile

import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as TF

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def import_module_from_code_root(code_root: Path, module_name: str):
    code_root = Path(code_root).resolve()
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    return importlib.import_module(module_name)


class DummyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 1
    def __getitem__(self, idx):
        z = torch.zeros(3, 256, 256)
        return {"adap": z, "gt": z, "A_paths": "dummy.jpg", "B_paths": "dummy.jpg"}


def build_diffuir_trainer(model_mod, ckpt_dir: Path, milestone: int, sampling_timesteps: int, image_size: int):
    condition = True
    num_unet = 1
    objective = "pred_res"
    test_res_or_noise = "res"
    sum_scale = 0.01
    delta_end = 1.8e-3
    ddim_sampling_eta = 0.0

    model = model_mod.UnetRes(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        num_unet=num_unet,
        condition=condition,
        objective=objective,
        test_res_or_noise=test_res_or_noise,
    )
    diffusion = model_mod.ResidualDiffusion(
        model,
        image_size=image_size,
        timesteps=1000,
        delta_end=delta_end,
        sampling_timesteps=sampling_timesteps,
        ddim_sampling_eta=ddim_sampling_eta,
        objective=objective,
        loss_type="l1",
        condition=condition,
        sum_scale=sum_scale,
        test_res_or_noise=test_res_or_noise,
    )

    class Opt:
        phase = "test"
        max_dataset_size = float("inf")
        load_size = image_size
        crop_size = image_size
        direction = "AtoB"
        preprocess = "none"
        no_flip = True
        bsize = 1

    trainer = model_mod.Trainer(
        diffusion,
        DummyDataset(),
        Opt(),
        train_batch_size=1,
        num_samples=1,
        train_lr=2e-4,
        train_num_steps=1,
        gradient_accumulate_every=1,
        ema_decay=0.995,
        amp=False,
        convert_image_to="RGB",
        results_folder=str(ckpt_dir),
        condition=condition,
        save_and_sample_every=1000,
        num_unet=num_unet,
    )
    trainer.load(milestone)
    trainer.ema.ema_model.init()
    trainer.ema.to(trainer.device)
    trainer.ema.ema_model.eval()
    return trainer


def read_rgb_tensor(path: Path):
    img = Image.open(path).convert("RGB")
    return TF.to_tensor(img).float(), img.size


def tensor_to_pil(x: torch.Tensor):
    x = x.detach().cpu().squeeze(0).clamp(0, 1)
    return TF.to_pil_image(x)


def save_jpeg96(img: Image.Image, out_path: Path, target_size=None):
    img.load()
    if img.mode != "RGB":
        img = img.convert("RGB")
    if target_size is not None and img.size != target_size:
        img = img.resize(target_size, Image.BICUBIC)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = dict(format="JPEG", quality=96, subsampling=0, optimize=True)
    img.save(out_path, **save_kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single_ckpt", required=True)
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--code_root", default=".", help="Root containing src/model.py and models/moce_prompt_refiner.py")
    ap.add_argument("--gpu", default="0", help="Set CUDA_VISIBLE_DEVICES before launching if preferred; this is only logged.")
    ap.add_argument("--sampling_timesteps", type=int, default=3)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--use_ema", action="store_true", default=True)
    ap.add_argument("--limit", type=int, default=-1)
    args = ap.parse_args()

    code_root = Path(args.code_root).resolve()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(input_dir)

    print("[INFO] code_root:", code_root)
    print("[INFO] single_ckpt:", args.single_ckpt)
    print("[INFO] input_dir:", input_dir)
    print("[INFO] output_dir:", output_dir)
    print("[INFO] visible CUDA devices should be set by caller; gpu arg=", args.gpu)

    single = torch.load(args.single_ckpt, map_location="cpu")
    if "base_ckpt" not in single or "refiner_ckpt" not in single:
        raise KeyError("single checkpoint must contain keys: base_ckpt, refiner_ckpt")
    milestone = int(single.get("base_milestone", 310))

    model_mod = import_module_from_code_root(code_root, "src.model")
    ref_mod = import_module_from_code_root(code_root, "models.moce_prompt_refiner")

    tmpdir = Path(tempfile.mkdtemp(prefix="lovif_single_ckpt_"))
    try:
        base_tmp = tmpdir / f"model-{milestone}.pt"
        torch.save(single["base_ckpt"], str(base_tmp))
        trainer = build_diffuir_trainer(model_mod, tmpdir, milestone, args.sampling_timesteps, args.image_size)

        device = trainer.device
        ref_ckpt = single["refiner_ckpt"]
        cfg = ref_ckpt.get("config", {})
        refiner = ref_mod.build_moce_prompt_refiner(
            width=cfg.get("width", 32),
            num_blocks=cfg.get("num_blocks", cfg.get("blocks", 4)),
            residual_scale=cfg.get("residual_scale", 0.15),
        ).to(device)
        key = "ema" if args.use_ema and "ema" in ref_ckpt else "model"
        refiner.load_state_dict(ref_ckpt[key], strict=True)
        refiner.eval()
        print("[INFO] refiner key:", key, "config:", cfg)

        imgs = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
        if args.limit > 0:
            imgs = imgs[:args.limit]
        output_dir.mkdir(parents=True, exist_ok=True)
        print("[INFO] num images:", len(imgs))

        with torch.no_grad():
            for p in tqdm(imgs, dynamic_ncols=True):
                lq, original_size = read_rgb_tensor(p)
                lq = lq.unsqueeze(0).to(device)

                # Stage 1: DiffUIR clean backbone.
                sampled = list(trainer.ema.ema_model.sample(lq, batch_size=1, last=True, task=str(p)))
                base = sampled[-1].clamp(0, 1)

                # Stage 2: MoCE refiner.
                pred = refiner(lq, base).clamp(0, 1)
                out_img = tensor_to_pil(pred)
                # final test input is expected to be jpg; keep same filename for 0001.jpg -> 0001.jpg.
                out_name = p.name if p.suffix.lower() in {".jpg", ".jpeg"} else p.stem + ".jpg"
                save_jpeg96(out_img, output_dir / out_name, target_size=original_size)
        print("[OK] saved JPEG quality=96 outputs to", output_dir)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
