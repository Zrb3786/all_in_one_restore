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
    return torch.abs(pred_dx - tgt_dx).flatten(1).mean(dim=1) + torch.abs(pred_dy - tgt_dy).flatten(1).mean(dim=1)


def fft_l1_per_sample(pred, target):
    pred_f = torch.fft.rfft2(pred.float(), norm='ortho')
    tgt_f = torch.fft.rfft2(target.float(), norm='ortho')
    pred_amp = torch.log1p(torch.abs(pred_f))
    tgt_amp = torch.log1p(torch.abs(tgt_f))
    return torch.abs(pred_amp - tgt_amp).flatten(1).mean(dim=1)


def laplacian_l1_per_sample(pred, target):
    kernel = torch.tensor(
        [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
        device=pred.device,
        dtype=pred.dtype,
    ).view(1, 1, 3, 3).repeat(pred.shape[1], 1, 1, 1)
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
    return (diff * (1.0 + alpha * edge)).flatten(1).mean(dim=1)


def change_weighted_l1_per_sample(pred, target, inp=None, alpha=4.0, eps=1e-6):
    if inp is None:
        return torch.zeros(pred.shape[0], device=pred.device, dtype=pred.dtype)
    diff = torch.abs(pred - target).mean(dim=1, keepdim=True)
    change_rgb = torch.abs(inp - target).mean(dim=1, keepdim=True)
    change_y = torch.abs(rgb_to_y(inp) - rgb_to_y(target))
    change = 0.5 * change_rgb + 0.5 * change_y
    norm = change.flatten(1).mean(dim=1).view(-1, 1, 1, 1) + eps
    change = (change / norm).clamp(0.0, 6.0)
    return (diff * (1.0 + alpha * change)).flatten(1).mean(dim=1)


def _infer_task(path):
    p = str(path).lower()
    if 'lowlight' in p or '/lol/' in p or 'light' in p:
        return 'light'
    if 'blur' in p or 'deblur' in p:
        return 'blur'
    if 'haze' in p or 'fog' in p or 'reside' in p:
        return 'haze'
    if 'rain' in p:
        return 'rain'
    if 'snow' in p:
        return 'snow'
    return 'default'


def _weights(task_paths, batch, device):
    if isinstance(task_paths, str):
        task_paths = [task_paths] * batch
    elif task_paths is None:
        task_paths = [''] * batch
    else:
        task_paths = list(task_paths)
        if len(task_paths) == 1 and batch > 1:
            task_paths = task_paths * batch

    # Base weights. These are deliberately conservative because DiffUIR's own diffusion loss is still primary.
    w_gt_char = torch.full((batch,), 0.010, device=device)
    w_gt_grad = torch.full((batch,), 0.012, device=device)
    w_gt_lap  = torch.full((batch,), 0.004, device=device)
    w_gt_fft  = torch.full((batch,), 0.001, device=device)
    w_gt_y    = torch.full((batch,), 0.002, device=device)
    w_gt_edge = torch.full((batch,), 0.004, device=device)
    w_gt_chg  = torch.full((batch,), 0.004, device=device)

    w_t_l1    = torch.full((batch,), 0.010, device=device)
    w_t_grad  = torch.full((batch,), 0.004, device=device)
    w_t_lap   = torch.full((batch,), 0.002, device=device)
    w_t_y     = torch.full((batch,), 0.001, device=device)

    for i, path in enumerate(task_paths[:batch]):
        t = _infer_task(path)
        if t == 'blur':
            # Teacher is base model-300, strongest blur checkpoint. Keep student close to it.
            w_gt_char[i] = 0.004
            w_gt_grad[i] = 0.030
            w_gt_lap[i]  = 0.020
            w_gt_fft[i]  = 0.006
            w_gt_y[i]    = 0.000
            w_gt_edge[i] = 0.010
            w_gt_chg[i]  = 0.003
            w_t_l1[i]    = 0.050
            w_t_grad[i]  = 0.030
            w_t_lap[i]   = 0.025
            w_t_y[i]     = 0.000
        elif t == 'light':
            # Teacher is v1-best. Preserve light enhancement and color/exposure behavior.
            w_gt_char[i] = 0.012
            w_gt_grad[i] = 0.008
            w_gt_lap[i]  = 0.002
            w_gt_fft[i]  = 0.0005
            w_gt_y[i]    = 0.035
            w_gt_edge[i] = 0.002
            w_gt_chg[i]  = 0.004
            w_t_l1[i]    = 0.030
            w_t_grad[i]  = 0.004
            w_t_lap[i]   = 0.001
            w_t_y[i]     = 0.020
        elif t == 'rain':
            # Teacher is v2-308. Rain is already relatively good; don't overprocess.
            w_gt_char[i] = 0.010
            w_gt_grad[i] = 0.025
            w_gt_lap[i]  = 0.006
            w_gt_fft[i]  = 0.002
            w_gt_y[i]    = 0.002
            w_gt_edge[i] = 0.010
            w_gt_chg[i]  = 0.020
            w_t_l1[i]    = 0.030
            w_t_grad[i]  = 0.010
            w_t_lap[i]   = 0.004
            w_t_y[i]     = 0.000
        elif t == 'haze':
            # Haze needs stronger GT change mask to avoid local-only dehaze.
            w_gt_char[i] = 0.012
            w_gt_grad[i] = 0.030
            w_gt_lap[i]  = 0.008
            w_gt_fft[i]  = 0.002
            w_gt_y[i]    = 0.008
            w_gt_edge[i] = 0.012
            w_gt_chg[i]  = 0.050
            w_t_l1[i]    = 0.025
            w_t_grad[i]  = 0.008
            w_t_lap[i]   = 0.004
            w_t_y[i]     = 0.002
        elif t == 'snow':
            # Snow also needs change mask, but avoid erasing structural white regions by keeping edge/teacher.
            w_gt_char[i] = 0.012
            w_gt_grad[i] = 0.030
            w_gt_lap[i]  = 0.008
            w_gt_fft[i]  = 0.002
            w_gt_y[i]    = 0.004
            w_gt_edge[i] = 0.012
            w_gt_chg[i]  = 0.055
            w_t_l1[i]    = 0.025
            w_t_grad[i]  = 0.008
            w_t_lap[i]   = 0.004
            w_t_y[i]     = 0.000

    return w_gt_char, w_gt_grad, w_gt_lap, w_gt_fft, w_gt_y, w_gt_edge, w_gt_chg, w_t_l1, w_t_grad, w_t_lap, w_t_y


def _auto_to_01(x):
    if x is None:
        return None
    # Accept either [0,1] or [-1,1] tensors. This makes the model patch robust
    # to where DiffUIR normalizes tensors in the training path.
    x_det = x.detach()
    if x_det.min() < -0.05 or x_det.max() > 1.05:
        x = (x + 1.) * 0.5
    return x.clamp(0, 1)


def lovif_aux_loss(pred01, gt01, input01=None, teacher01=None, task_paths=None):
    pred01 = _auto_to_01(pred01)
    gt01 = _auto_to_01(gt01)
    input01 = _auto_to_01(input01)
    teacher01 = _auto_to_01(teacher01)

    b = pred01.shape[0]
    device = pred01.device
    weights = _weights(task_paths, b, device)
    w_gt_char, w_gt_grad, w_gt_lap, w_gt_fft, w_gt_y, w_gt_edge, w_gt_chg, w_t_l1, w_t_grad, w_t_lap, w_t_y = weights

    l_gt_char = charbonnier_per_sample(pred01, gt01)
    l_gt_grad = grad_l1_per_sample(pred01, gt01)
    l_gt_lap  = laplacian_l1_per_sample(pred01, gt01)
    l_gt_fft  = fft_l1_per_sample(pred01, gt01)
    l_gt_y    = y_l1_per_sample(pred01, gt01)
    l_gt_edge = edge_weighted_l1_per_sample(pred01, gt01)
    l_gt_chg  = change_weighted_l1_per_sample(pred01, gt01, input01)

    loss = (
        w_gt_char * l_gt_char +
        w_gt_grad * l_gt_grad +
        w_gt_lap  * l_gt_lap +
        w_gt_fft  * l_gt_fft +
        w_gt_y    * l_gt_y +
        w_gt_edge * l_gt_edge +
        w_gt_chg  * l_gt_chg
    )

    if teacher01 is not None:
        l_t_l1   = charbonnier_per_sample(pred01, teacher01)
        l_t_grad = grad_l1_per_sample(pred01, teacher01)
        l_t_lap  = laplacian_l1_per_sample(pred01, teacher01)
        l_t_y    = y_l1_per_sample(pred01, teacher01)
        loss = loss + w_t_l1 * l_t_l1 + w_t_grad * l_t_grad + w_t_lap * l_t_lap + w_t_y * l_t_y

    scale = float(os.environ.get('LOVIF_AUX_SCALE', '1.0'))
    return loss.mean() * scale
