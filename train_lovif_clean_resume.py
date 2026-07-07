import argparse

from data.lovif_universal_dataset import LoViFAlignedDataset
from src.model import ResidualDiffusion, Trainer, UnetRes, set_seed


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataroot", type=str, required=True)
    parser.add_argument("--results_folder", type=str, required=True)
    # Compatibility fields expected by official DiffUIR Trainer / dataset-style opts.
    parser.add_argument("--phase", type=str, default="train")
    parser.add_argument("--max_dataset_size", type=float, default=float("inf"))
    parser.add_argument("--load_size", type=int, default=268)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--direction", type=str, default="AtoB")
    parser.add_argument("--preprocess", type=str, default="crop")
    parser.add_argument("--no_flip", action="store_true")


    parser.add_argument("--resume_milestone", type=int, default=300)
    parser.add_argument("--train_num_steps", type=int, default=310000)
    parser.add_argument("--save_every", type=int, default=1000)

    parser.add_argument("--bsize", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)

    parser.add_argument("--sampling_timesteps", type=int, default=3)
    parser.add_argument("--image_size", type=int, default=256)

    parser.add_argument("--no_lowlight_equalize", action="store_true")
    parser.add_argument("--seed", type=int, default=10)

    return parser.parse_args()


def main():
    args = parse_args()

    # Defensive defaults for official DiffUIR Trainer.
    # Official Trainer accesses opts.phase directly.
    if not hasattr(args, "phase"):
        args.phase = "train"
    if not hasattr(args, "max_dataset_size"):
        args.max_dataset_size = float("inf")
    if not hasattr(args, "load_size"):
        args.load_size = 268
    if not hasattr(args, "crop_size"):
        args.crop_size = args.image_size
    if not hasattr(args, "direction"):
        args.direction = "AtoB"
    if not hasattr(args, "preprocess"):
        args.preprocess = "crop"
    if not hasattr(args, "no_flip"):
        args.no_flip = False

    set_seed(args.seed)

    condition = True
    num_unet = 1
    objective = "pred_res"
    test_res_or_noise = "res"
    sum_scale = 0.01
    delta_end = 1.8e-3
    ddim_sampling_eta = 0.0
    num_samples = 1

    print("[INFO] clean DiffUIR official-loss LoViF finetune")
    print("[INFO] dataroot:", args.dataroot)
    print("[INFO] results_folder:", args.results_folder)
    print("[INFO] resume_milestone:", args.resume_milestone)
    print("[INFO] train_num_steps:", args.train_num_steps)
    print("[INFO] bsize:", args.bsize)
    print("[INFO] grad_accum:", args.grad_accum)
    print("[INFO] lr:", args.lr)
    print("[INFO] sampling_timesteps:", args.sampling_timesteps)
    print("[INFO] no_lowlight_equalize:", args.no_lowlight_equalize)

    datasets = []
    for task in ["fog", "light_only", "rain", "snow", "blur"]:
        ds = LoViFAlignedDataset(
            dataroot=args.dataroot,
            task=task,
            image_size=args.image_size,
            augment_flip=True,
            crop_patch=True,
            equalizeHist=(not args.no_lowlight_equalize),
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

        # 官方 ckpt 会带 optimizer 状态，这里强制使用当前命令行 lr。
        for opt_name in ["opt0", "opt1", "opt"]:
            opt_obj = getattr(trainer, opt_name, None)
            if opt_obj is not None:
                for group in opt_obj.param_groups:
                    group["lr"] = args.lr
                print(f"[INFO] reset {opt_name} lr to {args.lr}")

    trainer.train()


if __name__ == "__main__":
    main()
