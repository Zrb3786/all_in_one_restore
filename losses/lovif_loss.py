import os
import torch
import torch.nn.functional as F


def _reduce_per_sample(x):
    return x.flatten(1).mean(dim=1)


def charbonnier_per_sample(pred, target, eps=1e-3):
    return _reduce_per_sample(torch.sqrt((pred - target) ** 2 + eps ** 2))


def rgb_to_y(x):
    r = x[:, 0:1]
    g = x[:, 1:2]
    b = x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def y_l1_per_sample(pred, target):
    return _reduce_per_sample(torch.abs(rgb_to_y(pred) - rgb_to_y(target)))


def grad_l1_per_sample(pred, target):
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    tgt_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    tgt_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

    loss_x = torch.abs(pred_dx - tgt_dx).flatten(1).mean(dim=1)
    loss_y = torch.abs(pred_dy - tgt_dy).flatten(1).mean(dim=1)
    return loss_x + loss_y


def fft_l1_per_sample(pred, target):
    pred_f = torch.fft.rfft2(pred.float(), norm="ortho")
    tgt_f = torch.fft.rfft2(target.float(), norm="ortho")
    pred_amp = torch.log1p(torch.abs(pred_f))
    tgt_amp = torch.log1p(torch.abs(tgt_f))
    return torch.abs(pred_amp - tgt_amp).flatten(1).mean(dim=1)


def pcc_loss_per_sample(pred, target, eps=1e-6):
    x = pred.flatten(1).float()
    y = target.flatten(1).float()
    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
    corr = (x * y).sum(dim=1) / (
        torch.sqrt((x ** 2).sum(dim=1) + eps) *
        torch.sqrt((y ** 2).sum(dim=1) + eps)
    )
    return 1.0 - corr.clamp(-1.0, 1.0)


def laplacian_l1_per_sample(pred, target):
    kernel = torch.tensor(
        [[0., 1., 0.],
         [1., -4., 1.],
         [0., 1., 0.]],
        device=pred.device,
        dtype=pred.dtype
    ).view(1, 1, 3, 3)

    kernel = kernel.repeat(pred.shape[1], 1, 1, 1)

    pred_lap = F.conv2d(pred, kernel, padding=1, groups=pred.shape[1])
    tgt_lap = F.conv2d(target, kernel, padding=1, groups=target.shape[1])

    return torch.abs(pred_lap - tgt_lap).flatten(1).mean(dim=1)


def edge_weighted_l1_per_sample(pred, target, alpha=3.0, eps=1e-6):
    diff = torch.abs(pred - target).mean(dim=1, keepdim=True)

    y = rgb_to_y(target)
    dx = torch.abs(y[:, :, :, 1:] - y[:, :, :, :-1])
    dy = torch.abs(y[:, :, 1:, :] - y[:, :, :-1, :])

    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))

    edge = dx + dy
    norm = edge.flatten(1).mean(dim=1).view(-1, 1, 1, 1) + eps
    edge = (edge / norm).clamp(0.0, 5.0)

    weighted = diff * (1.0 + alpha * edge)
    return weighted.flatten(1).mean(dim=1)


def change_weighted_l1_per_sample(pred, target, inp=None, alpha=4.0, eps=1e-6):
    """
    强化 LQ 和 GT 差异大的区域。
    对 snow/rain/haze 特别重要：避免模型把雪/烟/雨当环境跳过。
    """
    if inp is None:
        return torch.zeros(pred.shape[0], device=pred.device, dtype=pred.dtype)

    diff = torch.abs(pred - target).mean(dim=1, keepdim=True)

    change_rgb = torch.abs(inp - target).mean(dim=1, keepdim=True)
    change_y = torch.abs(rgb_to_y(inp) - rgb_to_y(target))
    change = 0.5 * change_rgb + 0.5 * change_y

    norm = change.flatten(1).mean(dim=1).view(-1, 1, 1, 1) + eps
    change = (change / norm).clamp(0.0, 6.0)

    weighted = diff * (1.0 + alpha * change)
    return weighted.flatten(1).mean(dim=1)


def _infer_task(path):
    p = str(path).lower()
    if "lowlight" in p or "/lol/" in p or "light" in p:
        return "light"
    if "blur" in p or "deblur" in p:
        return "blur"
    if "haze" in p or "fog" in p or "reside" in p:
        return "haze"
    if "rain" in p:
        return "rain"
    if "snow" in p:
        return "snow"
    return "default"


def _weights(task_paths, batch, device):
    if isinstance(task_paths, str):
        task_paths = [task_paths] * batch
    elif task_paths is None:
        task_paths = [""] * batch
    else:
        task_paths = list(task_paths)
        if len(task_paths) == 1 and batch > 1:
            task_paths = task_paths * batch

    w_char = torch.full((batch,), 0.010, device=device)
    w_grad = torch.full((batch,), 0.012, device=device)
    w_fft  = torch.full((batch,), 0.001, device=device)
    w_y    = torch.full((batch,), 0.002, device=device)
    w_pcc  = torch.full((batch,), 0.000, device=device)
    w_edge = torch.full((batch,), 0.004, device=device)
    w_lap  = torch.full((batch,), 0.004, device=device)
    w_chg  = torch.full((batch,), 0.004, device=device)

    for i, path in enumerate(task_paths[:batch]):
        t = _infer_task(path)

        if t == "blur":
            # 从 300 重训时保护 deblur 能力：强化梯度/拉普拉斯/频域，但不过度改亮度。
            w_char[i] = 0.006
            w_grad[i] = 0.070
            w_fft[i]  = 0.012
            w_y[i]    = 0.000
            w_pcc[i]  = 0.000
            w_edge[i] = 0.018
            w_lap[i]  = 0.035
            w_chg[i]  = 0.010

        elif t == "light":
            # 保留 v1/v2 中有效的 lowlight 提亮，但稍微收敛，避免过曝。
            w_char[i] = 0.014
            w_grad[i] = 0.010
            w_fft[i]  = 0.001
            w_y[i]    = 0.055
            w_pcc[i]  = 0.001
            w_edge[i] = 0.004
            w_lap[i]  = 0.003
            w_chg[i]  = 0.006

        elif t == "haze":
            # 整体烟雾笼罩：需要更多全局变化，但保护结构。
            w_char[i] = 0.015
            w_grad[i] = 0.040
            w_fft[i]  = 0.003
            w_y[i]    = 0.012
            w_pcc[i]  = 0.001
            w_edge[i] = 0.020
            w_lap[i]  = 0.012
            w_chg[i]  = 0.055

        elif t == "rain":
            # rain 目前效果不错，别过度处理，主要保护结构。
            w_char[i] = 0.012
            w_grad[i] = 0.035
            w_fft[i]  = 0.003
            w_y[i]    = 0.004
            w_pcc[i]  = 0.000
            w_edge[i] = 0.018
            w_lap[i]  = 0.010
            w_chg[i]  = 0.035

        elif t == "snow":
            # snow 容易被当环境跳过：强化 LQ-GT 差异区域，同时保边。
            w_char[i] = 0.013
            w_grad[i] = 0.040
            w_fft[i]  = 0.003
            w_y[i]    = 0.006
            w_pcc[i]  = 0.000
            w_edge[i] = 0.018
            w_lap[i]  = 0.012
            w_chg[i]  = 0.065

    return w_char, w_grad, w_fft, w_y, w_pcc, w_edge, w_lap, w_chg


def lovif_aux_loss(pred01, gt01, input01=None, task_paths=None):
    pred01 = pred01.clamp(0, 1)
    gt01 = gt01.clamp(0, 1)
    if input01 is not None:
        input01 = input01.clamp(0, 1)

    b = pred01.shape[0]
    device = pred01.device

    w_char, w_grad, w_fft, w_y, w_pcc, w_edge, w_lap, w_chg = _weights(task_paths, b, device)

    l_char = charbonnier_per_sample(pred01, gt01)
    l_grad = grad_l1_per_sample(pred01, gt01)
    l_fft = fft_l1_per_sample(pred01, gt01)
    l_y = y_l1_per_sample(pred01, gt01)
    l_pcc = pcc_loss_per_sample(pred01, gt01)
    l_edge = edge_weighted_l1_per_sample(pred01, gt01)
    l_lap = laplacian_l1_per_sample(pred01, gt01)
    l_chg = change_weighted_l1_per_sample(pred01, gt01, input01)

    loss = (
        w_char * l_char +
        w_grad * l_grad +
        w_fft  * l_fft +
        w_y    * l_y +
        w_pcc  * l_pcc +
        w_edge * l_edge +
        w_lap  * l_lap +
        w_chg  * l_chg
    ).mean()

    scale = float(os.environ.get("LOVIF_AUX_SCALE", "1.0"))
    return loss * scale
