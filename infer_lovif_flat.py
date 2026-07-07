import os
import argparse
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
from torchvision import utils as vutils

from src.model import ResidualDiffusion, Trainer, UnetRes, set_seed


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class FlatImageDataset(torch.utils.data.Dataset):
    """
    DiffUIR-style minimal dataset for LoViF Val_LQ/Test_LQ.
    It returns keys compatible with DiffUIR Trainer, but we use our own loop.
    """
    def __init__(self, input_dir):
        self.input_dir = Path(input_dir)
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input dir not found: {self.input_dir}")

        self.paths = sorted([
            p for p in self.input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        ])

        if not self.paths:
            raise RuntimeError(f"No image files found in {self.input_dir}")

        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        x = self.to_tensor(img).float()
        return {
            "adap": x,
            "gt": x,  # dummy, not used for metric
            "A_paths": str(p),
            "B_paths": str(p),
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing LQ images, e.g. Val_AIO/LQ")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save restored images")
    parser.add_argument("--ckpt_dir", type=str,
                        default="/data1/zhangruibo/projects/DiffUIR/ckpt_universal/diffuir",
                        help="Directory containing model-300.pt")
    parser.add_argument("--milestone", type=int, default=300)
    parser.add_argument("--sampling_timesteps", type=int, default=3)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--suffix", type=str, default=".png",
                        help="Output image suffix, default .png")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Only infer first N images when >0")
    return parser.parse_args()


def build_trainer(dataset, args):
    # Match official DiffUIR test.py core settings.
    condition = True
    train_batch_size = 1
    num_samples = 1
    train_num_steps = 100000
    save_and_sample_every = 1000

    num_unet = 1
    objective = "pred_res"
    test_res_or_noise = "res"
    sum_scale = 0.01
    ddim_sampling_eta = 0.0
    delta_end = 1.8e-3

    model = UnetRes(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        num_unet=num_unet,
        condition=condition,
        objective=objective,
        test_res_or_noise=test_res_or_noise,
    )

    diffusion = ResidualDiffusion(
        model,
        image_size=args.image_size,
        timesteps=1000,
        delta_end=delta_end,
        sampling_timesteps=args.sampling_timesteps,
        ddim_sampling_eta=ddim_sampling_eta,
        objective=objective,
        loss_type="l1",
        condition=condition,
        sum_scale=sum_scale,
        test_res_or_noise=test_res_or_noise,
    )

    # Minimal opts object for Trainer.
    class Opt:
        phase = "test"

    trainer = Trainer(
        diffusion,
        dataset,
        Opt(),
        train_batch_size=train_batch_size,
        num_samples=num_samples,
        train_lr=2e-4,
        train_num_steps=train_num_steps,
        gradient_accumulate_every=2,
        ema_decay=0.995,
        amp=False,
        convert_image_to="RGB",
        results_folder=args.ckpt_dir,
        condition=condition,
        save_and_sample_every=save_and_sample_every,
        num_unet=num_unet,
    )
    return trainer


def main():
    args = parse_args()
    set_seed(10)

    ckpt_path = Path(args.ckpt_dir) / f"model-{args.milestone}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = FlatImageDataset(args.input_dir)
    if args.limit and args.limit > 0:
        dataset.paths = dataset.paths[:args.limit]

    print(f"[INFO] input_dir : {args.input_dir}")
    print(f"[INFO] output_dir: {args.output_dir}")
    print(f"[INFO] ckpt     : {ckpt_path}")
    print(f"[INFO] num imgs : {len(dataset)}")
    print(f"[INFO] sampling_timesteps: {args.sampling_timesteps}")

    trainer = build_trainer(dataset, args)
    trainer.load(args.milestone)

    trainer.ema.ema_model.init()
    trainer.ema.to(trainer.device)
    trainer.ema.ema_model.eval()

    for item in tqdm(dataset, total=len(dataset), desc="DiffUIR LoViF infer"):
        in_path = Path(item["A_paths"])
        x = item["adap"].unsqueeze(0).to(trainer.device)

        with torch.no_grad():
            sampled = list(
                trainer.ema.ema_model.sample(
                    x,
                    batch_size=1,
                    last=True,
                    task=str(in_path),
                )
            )

        # Official Trainer.test saves the last sample.
        restored = sampled[-1]
        save_name = in_path.stem + args.suffix
        save_path = out_dir / save_name
        vutils.save_image(restored, str(save_path), nrow=1)

    print(f"[OK] saved {len(dataset)} images to {out_dir}")


if __name__ == "__main__":
    main()
