import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.refiner_dataset import LoViFRefinerDataset
from losses.refiner_losses import Clean310RefinerLoss
from models.moce_prompt_refiner import build_moce_prompt_refiner


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataroot", type=str, required=True)
    p.add_argument("--base_root", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--width", type=int, default=48)
    p.add_argument("--blocks", type=int, default=8)
    p.add_argument("--residual_scale", type=float, default=0.15)
    p.add_argument("--aux_scale", type=float, default=0.7)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def update_ema(model, ema_model, decay):
    for p, ep in zip(model.parameters(), ema_model.parameters()):
        ep.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
    for b, eb in zip(model.buffers(), ema_model.buffers()):
        eb.copy_(b)


def save_ckpt(path, model, ema_model, opt, step, args):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "ema": ema_model.state_dict(),
        "opt": opt.state_dict(),
        "config": {
            "width": args.width,
            "num_blocks": args.blocks,
            "residual_scale": args.residual_scale,
            "aux_scale": args.aux_scale,
            "image_size": args.image_size,
        },
    }, str(path))
    print(f"[SAVE] {path}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    (out_dir / "ckpt").mkdir(parents=True, exist_ok=True)

    ds = LoViFRefinerDataset(args.dataroot, args.base_root, image_size=args.image_size, augment=True, crop=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    model = build_moce_prompt_refiner(width=args.width, num_blocks=args.blocks, residual_scale=args.residual_scale).to(device)
    ema_model = build_moce_prompt_refiner(width=args.width, num_blocks=args.blocks, residual_scale=args.residual_scale).to(device)
    ema_model.load_state_dict(model.state_dict())
    ema_model.eval()

    criterion = Clean310RefinerLoss(aux_scale=args.aux_scale).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        if "ema" in ckpt:
            ema_model.load_state_dict(ckpt["ema"], strict=True)
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
            for g in opt.param_groups:
                g["lr"] = args.lr
        start_step = int(ckpt.get("step", 0))
        print(f"[RESUME] {args.resume}, start_step={start_step}, reset lr={args.lr}")

    print("[INFO] device:", device)
    print("[INFO] steps:", args.steps, "batch_size:", args.batch_size, "lr:", args.lr)
    print("[INFO] residual_scale:", args.residual_scale, "aux_scale:", args.aux_scale)

    data_iter = iter(dl)
    pbar = tqdm(range(start_step + 1, args.steps + 1), dynamic_ncols=True)
    model.train()
    for step in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            batch = next(data_iter)

        lq = batch["lq"].to(device, non_blocking=True)
        base = batch["base"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)
        task_idx = batch["task_idx"].to(device, non_blocking=True)

        pred, aux = model(lq, base, return_aux=True)
        loss, log = criterion(pred, gt, lq, base, task_idx, aux)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        update_ema(model, ema_model, args.ema_decay)

        if step % 20 == 0:
            pbar.set_description(f"step={step} loss={loss.item():.4f} char={log['loss_char'].item():.4f} change={log['loss_change'].item():.4f} base={log['loss_base'].item():.4f}")

        if step % args.save_every == 0 or step == args.steps:
            save_ckpt(out_dir / "ckpt" / f"refiner_step{step}.pt", model, ema_model, opt, step, args)

    save_ckpt(out_dir / "ckpt" / f"refiner_final_step{args.steps}.pt", model, ema_model, opt, args.steps, args)
    print("[DONE]")


if __name__ == "__main__":
    main()
