#!/usr/bin/env python3
"""Final LoViF inference from ONE packed checkpoint, with timed readme.txt.

This script loads one checkpoint containing:
  - DiffUIR clean310 checkpoint
  - MoCEPromptRefiner checkpoint

It runs two internal stages and saves output images as JPEG quality=96 with the
same filename and size as input. It also writes readme.txt required by Codabench.

It intentionally avoids `from src.model import ...`; DiffUIR code is dynamically
imported from --code_root.
"""
import argparse
import importlib
import importlib.util
import shutil
import sys
import tempfile
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as TF

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def import_module_dynamic(code_root: Path, module_name: str, file_rel: str = None):
    code_root = Path(code_root).resolve()
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        if file_rel is None:
            raise
        path = code_root / file_rel
        if not path.exists():
            raise FileNotFoundError(path) from e
        spec = importlib.util.spec_from_file_location(module_name.replace('.', '_'), str(path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod


class DummyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 1
    def __getitem__(self, idx):
        z = torch.zeros(3, 256, 256)
        return {"adap": z, "gt": z, "A_paths": "dummy.jpg", "B_paths": "dummy.jpg"}


def get_single_keys(single):
    base = None
    ref = None
    for k in ["base_ckpt", "base_checkpoint", "base_diffuir", "diffuir_ckpt", "base_model"]:
        if k in single:
            base = single[k]
            break
    for k in ["refiner_ckpt", "refiner_checkpoint", "refiner", "moce_refiner", "refiner_model"]:
        if k in single:
            ref = single[k]
            break
    if base is None:
        raise KeyError(f"Cannot find base DiffUIR checkpoint in single ckpt keys={list(single.keys())}")
    if ref is None:
        raise KeyError(f"Cannot find refiner checkpoint in single ckpt keys={list(single.keys())}")
    return base, ref


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


def build_refiner(ref_mod, ref_ckpt, use_ema=True):
    cfg = ref_ckpt.get("config", {}) if isinstance(ref_ckpt, dict) else {}

    # Support several naming variants from our previous scripts.
    width = int(cfg.get("width", cfg.get("model_width", 32)))
    blocks = int(cfg.get("blocks", cfg.get("num_blocks", cfg.get("depth", 4))))
    residual_scale = float(cfg.get("residual_scale", 0.15))

    if hasattr(ref_mod, "build_moce_prompt_refiner"):
        model = ref_mod.build_moce_prompt_refiner(width=width, num_blocks=blocks, residual_scale=residual_scale)
    elif hasattr(ref_mod, "MoCEPromptRefiner"):
        cls = ref_mod.MoCEPromptRefiner
        # Try robust constructor variants.
        try:
            model = cls(width=width, num_blocks=blocks, residual_scale=residual_scale)
        except TypeError:
            try:
                model = cls(width=width, depth=blocks, residual_scale=residual_scale)
            except TypeError:
                model = cls(width=width, blocks=blocks, residual_scale=residual_scale)
    else:
        raise AttributeError("models.moce_prompt_refiner must define build_moce_prompt_refiner or MoCEPromptRefiner")

    if isinstance(ref_ckpt, dict):
        if use_ema and "ema" in ref_ckpt:
            state = ref_ckpt["ema"]
            key = "ema"
        elif "model" in ref_ckpt:
            state = ref_ckpt["model"]
            key = "model"
        elif "state_dict" in ref_ckpt:
            state = ref_ckpt["state_dict"]
            key = "state_dict"
        else:
            # Maybe checkpoint itself is state_dict.
            state = ref_ckpt
            key = "raw_dict"
    else:
        raise TypeError("refiner checkpoint must be a dict")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[INFO] loaded refiner key={key}, missing={len(missing)}, unexpected={len(unexpected)}")
    if missing[:5]:
        print("[WARN] missing sample:", missing[:5])
    if unexpected[:5]:
        print("[WARN] unexpected sample:", unexpected[:5])
    return model, cfg


def read_tensor(path: Path):
    img = Image.open(path).convert("RGB")
    return TF.to_tensor(img).float(), img.size


def save_jpeg96_tensor(x: torch.Tensor, out_path: Path, size):
    img = TF.to_pil_image(x.detach().cpu().clamp(0, 1))
    if img.mode != "RGB":
        img = img.convert("RGB")
    if img.size != size:
        img = img.resize(size, Image.BICUBIC)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="JPEG", quality=96, subsampling=0, optimize=True)


def write_readme(out_dir: Path, runtime_per_img: float, device_type: int, extra_data: int, desc: str):
    text = (
        f"runtime per img [s] : {runtime_per_img:.4f}\n"
        f"CPU[1] / GPU[0] : {int(device_type)}\n"
        f"Extra Data [1] / No Extra Data [0] : {int(extra_data)}\n"
        f"Other description : {desc}\n"
    )
    (out_dir / "readme.txt").write_text(text, encoding="utf-8")
    print("[OK] wrote readme.txt:\n" + text)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--single_ckpt", required=True)
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--code_root", default=".")
    p.add_argument("--sampling_timesteps", type=int, default=3)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--use_ema", action="store_true", default=True)
    p.add_argument("--limit", type=int, default=-1)
    p.add_argument("--device_type", type=int, default=0, help="CPU[1] / GPU[0]")
    p.add_argument("--extra_data", type=int, default=1, help="Extra Data [1] / No Extra Data [0]")
    p.add_argument("--description", default="LoViF")
    return p.parse_args()


def main():
    args = parse_args()
    code_root = Path(args.code_root).resolve()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(input_dir)
    if out_dir.exists():
        # Keep safe: do not rm automatically; overwrite individual outputs.
        pass
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] code_root:", code_root)
    print("[INFO] input_dir:", input_dir)
    print("[INFO] output_dir:", out_dir)
    print("[INFO] single_ckpt:", args.single_ckpt)

    single = torch.load(args.single_ckpt, map_location="cpu")
    base_ckpt, ref_ckpt = get_single_keys(single)
    milestone = int(single.get("base_milestone", 310))
    print("[INFO] base milestone:", milestone)

    model_mod = import_module_dynamic(code_root, "src.model", "src/model.py")
    ref_mod = import_module_dynamic(code_root, "models.moce_prompt_refiner", "models/moce_prompt_refiner.py")

    tmp = Path(tempfile.mkdtemp(prefix="lovif_final_base_"))
    try:
        torch.save(base_ckpt, tmp / f"model-{milestone}.pt")
        trainer = build_diffuir_trainer(model_mod, tmp, milestone, args.sampling_timesteps, args.image_size)
        device = trainer.device
        refiner, cfg = build_refiner(ref_mod, ref_ckpt, use_ema=args.use_ema)
        refiner = refiner.to(device).eval()
        print("[INFO] refiner cfg:", cfg)

        paths = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
        if args.limit > 0:
            paths = paths[:args.limit]
        if not paths:
            raise RuntimeError(f"No images found in {input_dir}")
        print("[INFO] num images:", len(paths))

        # Measure full per-image processing after model loading: read + DiffUIR + refiner + save.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for p in tqdm(paths, dynamic_ncols=True):
                lq, size = read_tensor(p)
                lq = lq.unsqueeze(0).to(device)
                base_samples = list(trainer.ema.ema_model.sample(lq, batch_size=1, last=True, task=str(p)))
                base = base_samples[-1].clamp(0, 1)
                pred = refiner(lq, base).clamp(0, 1)
                # Keep the same name. If input is non-jpg, still use stem.jpg to satisfy JPEG requirement.
                out_name = p.name if p.suffix.lower() in {".jpg", ".jpeg"} else p.stem + ".jpg"
                if out_name.lower().endswith(".jpeg"):
                    out_name = p.stem + ".jpg"
                save_jpeg96_tensor(pred[0], out_dir / out_name, size)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        runtime = (t1 - t0) / len(paths)
        write_readme(out_dir, runtime, args.device_type, args.extra_data, args.description)
        print(f"[OK] avg runtime per image: {runtime:.4f}s")
        print("[OK] saved outputs to", out_dir)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
