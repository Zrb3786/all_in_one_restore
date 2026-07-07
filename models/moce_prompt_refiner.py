import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        var, mean = torch.var_mean(x, dim=1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFExpert(nn.Module):
    """NAFNet-style expert block, lightweight and activation-free in the core."""
    def __init__(self, channels: int, dilation: int = 1, expansion: int = 2):
        super().__init__()
        hidden = channels * expansion
        self.norm1 = LayerNorm2d(channels)
        self.pw1 = nn.Conv2d(channels, hidden * 2, 1)
        self.dw = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=dilation, dilation=dilation, groups=hidden * 2)
        self.sg = SimpleGate()
        self.pw2 = nn.Conv2d(hidden, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

        self.norm2 = LayerNorm2d(channels)
        self.ffn1 = nn.Conv2d(channels, hidden * 2, 1)
        self.ffn2 = nn.Conv2d(hidden, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        y = self.norm1(x)
        y = self.pw1(y)
        y = self.dw(y)
        y = self.sg(y)
        y = self.pw2(y)
        x = x + self.beta * y

        y = self.norm2(x)
        y = self.ffn1(y)
        y = self.sg(y)
        y = self.ffn2(y)
        x = x + self.gamma * y
        return x


class PromptRouter(nn.Module):
    """Implicit degradation router. No external task label is used at inference."""
    def __init__(self, channels: int, num_experts: int = 4, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_experts),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


class MoCEBlock(nn.Module):
    """Mixture-of-complexity experts block.

    Expert 0 is identity/protective, and other experts use increasing dilation.
    This gives the refiner a low-risk path for blur/structure and larger-context
    paths for haze/snow/rain residual correction.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.router = PromptRouter(channels, num_experts=4)
        self.experts = nn.ModuleList([
            nn.Identity(),
            NAFExpert(channels, dilation=1),
            NAFExpert(channels, dilation=2),
            NAFExpert(channels, dilation=3),
        ])
        self.proj = nn.Conv2d(channels, channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x, return_gate: bool = False):
        gates = self.router(x)
        outs = torch.stack([expert(x) for expert in self.experts], dim=1)  # B,E,C,H,W
        y = (outs * gates[:, :, None, None, None]).sum(dim=1)
        y = self.proj(y)
        out = x + self.alpha * y
        if return_gate:
            return out, gates
        return out


class MoCEPromptRefiner(nn.Module):
    """Conservative all-in-one residual refiner after clean DiffUIR-310.

    Inputs in [0,1]:
      lq: original degraded input
      base: clean-310 output

    Output starts exactly at base because residual and mask heads are zero/small-init:
      out = base + residual_scale * sigmoid(mask) * tanh(residual)
    """
    def __init__(self, width: int = 48, num_blocks: int = 8, residual_scale: float = 0.15):
        super().__init__()
        self.width = int(width)
        self.num_blocks = int(num_blocks)
        self.residual_scale = float(residual_scale)
        in_ch = 9  # LQ, base, base-LQ

        self.intro = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
        )
        self.blocks = nn.ModuleList([MoCEBlock(width) for _ in range(num_blocks)])
        self.norm = LayerNorm2d(width)
        self.res_head = nn.Conv2d(width, 3, 3, padding=1)
        self.mask_head = nn.Conv2d(width, 1, 3, padding=1)

        nn.init.zeros_(self.res_head.weight)
        nn.init.zeros_(self.res_head.bias)
        nn.init.zeros_(self.mask_head.weight)
        nn.init.constant_(self.mask_head.bias, -3.0)  # very conservative initial update

    def forward(self, lq, base, return_aux: bool = False):
        x = torch.cat([lq, base, base - lq], dim=1)
        feat = self.intro(x)
        gates = []
        for block in self.blocks:
            feat, g = block(feat, return_gate=True)
            gates.append(g)
        feat = self.norm(feat)
        residual = torch.tanh(self.res_head(feat))
        mask = torch.sigmoid(self.mask_head(feat))
        out = (base + self.residual_scale * mask * residual).clamp(0, 1)
        if return_aux:
            gate_mean = torch.stack(gates, dim=1).mean(dim=1) if gates else None
            return out, {"residual": residual, "mask": mask, "gate": gate_mean}
        return out


def build_moce_prompt_refiner(width=48, num_blocks=8, residual_scale=0.15):
    return MoCEPromptRefiner(width=width, num_blocks=num_blocks, residual_scale=residual_scale)
