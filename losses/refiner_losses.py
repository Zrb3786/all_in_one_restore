import torch
import torch.nn as nn
import torch.nn.functional as F

TASKS = ["Blur", "Lowlight", "Haze", "Rain", "Snow"]
TASK_TO_IDX = {t: i for i, t in enumerate(TASKS)}


def flat_mean(x):
    return x.flatten(1).mean(dim=1)


def rgb_to_y(x):
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def charbonnier(pred, target, eps=1e-3):
    return flat_mean(torch.sqrt((pred - target) ** 2 + eps ** 2))


def grad_loss(pred, target):
    px = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    tx = target[:, :, :, 1:] - target[:, :, :, :-1]
    py = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    ty = target[:, :, 1:, :] - target[:, :, :-1, :]
    return torch.abs(px - tx).flatten(1).mean(dim=1) + torch.abs(py - ty).flatten(1).mean(dim=1)


def lap_loss(pred, target):
    kernel = torch.tensor(
        [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
        device=pred.device,
        dtype=pred.dtype
    )
    kernel = kernel.view(1, 1, 3, 3).repeat(pred.shape[1], 1, 1, 1)
    p = F.conv2d(pred, kernel, padding=1, groups=pred.shape[1])
    t = F.conv2d(target, kernel, padding=1, groups=target.shape[1])
    return torch.abs(p - t).flatten(1).mean(dim=1)


def fft_loss(pred, target):
    pf = torch.fft.rfft2(pred.float(), norm="ortho")
    tf = torch.fft.rfft2(target.float(), norm="ortho")
    pa = torch.log1p(torch.abs(pf))
    ta = torch.log1p(torch.abs(tf))
    return torch.abs(pa - ta).flatten(1).mean(dim=1)


def y_loss(pred, target):
    return torch.abs(rgb_to_y(pred) - rgb_to_y(target)).flatten(1).mean(dim=1)


def exposure_loss(pred, target, pool=16):
    yp = F.avg_pool2d(rgb_to_y(pred), pool, stride=pool)
    yt = F.avg_pool2d(rgb_to_y(target), pool, stride=pool)
    return torch.abs(yp - yt).flatten(1).mean(dim=1)


def color_loss(pred, target):
    return torch.abs(pred.flatten(2).mean(dim=2) - target.flatten(2).mean(dim=2)).mean(dim=1)


def local_contrast(x, k=9):
    y = rgb_to_y(x)
    mean = F.avg_pool2d(y, k, stride=1, padding=k // 2)
    return torch.abs(y - mean)


def contrast_loss(pred, target):
    return torch.abs(local_contrast(pred) - local_contrast(target)).flatten(1).mean(dim=1)


def edge_weighted_l1(pred, target, alpha=2.8, eps=1e-6):
    diff = torch.abs(pred - target).mean(dim=1, keepdim=True)
    y = rgb_to_y(target)
    dx = torch.abs(y[:, :, :, 1:] - y[:, :, :, :-1])
    dy = torch.abs(y[:, :, 1:, :] - y[:, :, :-1, :])
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    edge = dx + dy
    edge = edge / (edge.flatten(1).mean(dim=1).view(-1, 1, 1, 1) + eps)
    edge = edge.clamp(0, 5)
    return (diff * (1.0 + alpha * edge)).flatten(1).mean(dim=1)


def change_weighted_l1(pred, target, lq, alpha=4.5, eps=1e-6):
    diff = torch.abs(pred - target).mean(dim=1, keepdim=True)
    change = 0.5 * torch.abs(lq - target).mean(dim=1, keepdim=True) + 0.5 * torch.abs(rgb_to_y(lq) - rgb_to_y(target))
    change = change / (change.flatten(1).mean(dim=1).view(-1, 1, 1, 1) + eps)
    change = change.clamp(0, 7)
    return (diff * (1.0 + alpha * change)).flatten(1).mean(dim=1)


def snow_mask_l1(pred, target, lq, eps=1e-6):
    """Pseudo snow cue from LQ: bright + low saturation + local high contrast.

    This is only a training loss. It does not require explicit task labels at inference.
    """
    maxc = lq.max(dim=1, keepdim=True).values
    minc = lq.min(dim=1, keepdim=True).values
    sat = maxc - minc
    y = rgb_to_y(lq)
    white = torch.sigmoid((y - 0.62) * 12.0) * torch.sigmoid((0.18 - sat) * 18.0)

    # add local contrast cue so snow streak/dot regions receive larger gradients
    lc = local_contrast(lq)
    lc = lc / (lc.flatten(1).mean(dim=1).view(-1, 1, 1, 1) + eps)
    mask = (white * (1.0 + 0.5 * lc)).clamp(0, 3)

    diff = torch.abs(pred - target).mean(dim=1, keepdim=True)
    return (diff * (1.0 + 3.0 * mask)).flatten(1).mean(dim=1)


def base_keep(pred, base):
    return torch.abs(pred - base).flatten(1).mean(dim=1)


def residual_mag(pred, base):
    return torch.abs(pred - base).flatten(1).mean(dim=1)


def mask_sparsity(mask):
    return mask.flatten(1).mean(dim=1)


class Clean310RefinerLoss(nn.Module):
    """Weather-v2 loss for clean-310 post-refiner.

    Compared with v1:
      - stronger Haze/Snow change/contrast/snow-mask
      - stronger Blur base keep to protect blur
      - Rain remains conservative
    """
    def __init__(self, aux_scale=1.0):
        super().__init__()
        self.aux_scale = float(aux_scale)

    def forward(self, pred, gt, lq, base, task_idx, aux=None):
        pred = pred.clamp(0, 1)
        gt = gt.clamp(0, 1)
        lq = lq.clamp(0, 1)
        base = base.clamp(0, 1)
        task_idx = task_idx.to(pred.device).long()

        losses = {
            "char": charbonnier(pred, gt),
            "grad": grad_loss(pred, gt),
            "lap": lap_loss(pred, gt),
            "fft": fft_loss(pred, gt),
            "y": y_loss(pred, gt),
            "exposure": exposure_loss(pred, gt),
            "color": color_loss(pred, gt),
            "edge": edge_weighted_l1(pred, gt),
            "change": change_weighted_l1(pred, gt, lq),
            "contrast": contrast_loss(pred, gt),
            "snow": snow_mask_l1(pred, gt, lq),
            "base": base_keep(pred, base),
            "resmag": residual_mag(pred, base),
        }

        if aux is not None and "mask" in aux:
            losses["mask"] = mask_sparsity(aux["mask"])
        else:
            losses["mask"] = torch.zeros_like(losses["char"])

        # columns:
        # char grad lap fft y exposure color edge change contrast snow base resmag mask
        B = pred.shape[0]
        W = torch.zeros(B, 14, device=pred.device)

        for i in range(B):
            task = TASKS[int(task_idx[i].item())]

            if task == "Blur":
                # Strongly protect clean310/base behavior; do not allow weather-style residual drift.
                vals = [0.35, 0.20, 0.10, 0.02, 0.00, 0.00, 0.00, 0.10, 0.01, 0.00, 0.00, 3.20, 1.10, 0.12]

            elif task == "Lowlight":
                # Keep lowlight improvement; modest base keep.
                vals = [0.95, 0.18, 0.04, 0.01, 0.85, 0.55, 0.12, 0.08, 0.14, 0.04, 0.00, 0.25, 0.08, 0.04]

            elif task == "Haze":
                # More aggressive haze removal: change + contrast, less base lock.
                vals = [1.05, 0.42, 0.18, 0.02, 0.20, 0.08, 0.03, 0.58, 1.75, 0.70, 0.00, 0.25, 0.14, 0.05]

            elif task == "Rain":
                # Rain is already good; keep structure and avoid over-editing.
                vals = [0.90, 0.36, 0.12, 0.02, 0.06, 0.02, 0.02, 0.40, 0.70, 0.08, 0.00, 0.48, 0.16, 0.06]

            elif task == "Snow":
                # Stronger snow-specific correction. Reduce base lock, increase change/snow cue.
                vals = [1.05, 0.42, 0.18, 0.02, 0.10, 0.04, 0.02, 0.55, 1.95, 0.25, 1.10, 0.24, 0.14, 0.05]

            else:
                vals = [1.00, 0.30, 0.10, 0.02, 0.10, 0.05, 0.02, 0.20, 0.30, 0.05, 0.00, 0.50, 0.10, 0.05]

            W[i] = torch.tensor(vals, device=pred.device)

        keys = ["char", "grad", "lap", "fft", "y", "exposure", "color", "edge", "change", "contrast", "snow", "base", "resmag", "mask"]
        stack = torch.stack([losses[k] for k in keys], dim=1)
        total = (W * stack).sum(dim=1).mean() * self.aux_scale

        log = {f"loss_{k}": losses[k].mean().detach() for k in keys}
        log["loss_total"] = total.detach()
        return total, log
