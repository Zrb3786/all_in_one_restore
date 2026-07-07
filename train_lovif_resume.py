import argparse
from pathlib import Path

from data.lovif_distill_dataset import LoViFDistillDataset
from src.model import ResidualDiffusion, Trainer, UnetRes, set_seed


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataroot", type=str, required=True)
    parser.add_argument("--phase", type=str, default="train")
    parser.add_argument("--max_dataset_size", type=int, default=float("inf"))

    parser.add_argument("--load_size", type=int, default=268)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--direction", type=str, default="AtoB")
    parser.add_argument("--preprocess", type=str, default="crop")
    parser.add_argument("--no_flip", action="store_true")

    parser.add_argument("--results_folder", type=str, required=True)
    parser.add_argument("--resume_milestone", type=int, default=300)
    parser.add_argument("--train_num_steps", type=int, default=301000)
    parser.add_argument("--save_every", type=int, default=1000)

    parser.add_argument("--bsize", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)

    parser.add_argument("--sampling_timesteps", type=int, default=3)
    parser.add_argument("--no_lowlight_equalize", action="store_true",
                        help="Disable DiffUIR/LOL-style equalizeHist for LoViF lowlight.")
    parser.add_argument("--teacher_root", type=str, default=None, help="Teacher cache root for distillation.")
    parser.add_argument("--no_teacher", action="store_true", help="Disable teacher distillation even if teacher_root is set.")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_workers_note", type=int, default=0)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(10)

    condition = True
    num_unet = 1
    objective = "pred_res"
    test_res_or_noise = "res"
    sum_scale = 0.01
    delta_end = 1.8e-3
    ddim_sampling_eta = 0.0
    num_samples = 1

    print("[INFO] dataroot:", args.dataroot)
    print("[INFO] results_folder:", args.results_folder)
    print("[INFO] resume_milestone:", args.resume_milestone)
    print("[INFO] train_num_steps:", args.train_num_steps)
    print("[INFO] batch_size:", args.bsize)
    print("[INFO] lr:", args.lr)
    print("[INFO] sampling_timesteps:", args.sampling_timesteps)

    datasets = []
    for task in ["fog", "light_only", "rain", "snow", "blur"]:
        ds = LoViFDistillDataset(
            dataroot=args.dataroot,
            task=task,
            image_size=args.image_size,
            augment_flip=True,
            crop_patch=True,
            equalizeHist=(not args.no_lowlight_equalize),
            teacher_root=args.teacher_root,
            use_teacher=(args.teacher_root is not None and not args.no_teacher),
        )
        print(f"[DATA] {task}: {len(ds)}")
        datasets.append(ds)

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

    trainer = Trainer(
        diffusion,
        datasets,
        args,
        train_batch_size=args.bsize,
        num_samples=num_samples,
        train_lr=args.lr,
        train_num_steps=args.train_num_steps,
        gradient_accumulate_every=args.grad_accum,
        ema_decay=0.995,
        amp=False,
        convert_image_to="RGB",
        results_folder=args.results_folder,
        condition=condition,
        save_and_sample_every=args.save_every,
        num_unet=num_unet,
    )

    if args.resume_milestone is not None and args.resume_milestone >= 0:
        print(f"[INFO] loading milestone {args.resume_milestone}")
        trainer.load(args.resume_milestone)

        # ZRB: after loading optimizer state, force lr to current args.lr.
        # Otherwise resume checkpoint may keep the previous learning rate.
        for opt_name in ["opt0", "opt1", "opt"]:
            opt_obj = getattr(trainer, opt_name, None)
            if opt_obj is not None:
                for group in opt_obj.param_groups:
                    group["lr"] = args.lr
                print(f"[INFO] reset {opt_name} lr to {args.lr}")

    trainer.train()


if __name__ == "__main__":
    main()
