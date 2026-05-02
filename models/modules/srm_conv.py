from __future__ import annotations

import torch
from torch import nn


def _base_srm_kernels() -> torch.Tensor:
    kernels = []

    k1 = torch.tensor(
        [
            [0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    ) / 4.0
    kernels.append(k1)

    k2 = torch.tensor(
        [
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1],
        ],
        dtype=torch.float32,
    ) / 12.0
    kernels.append(k2)

    k3 = torch.tensor(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, -2, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    ) / 2.0
    kernels.append(k3)

    k4 = k3.t().contiguous()
    kernels.append(k4)

    k5 = torch.tensor(
        [
            [0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    kernels.append(k5)

    k6 = torch.tensor(
        [
            [0, 0, 0, 0, 0],
            [0, 1, -2, 1, 0],
            [0, -2, 4, -2, 0],
            [0, 1, -2, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    ) / 4.0
    kernels.append(k6)

    return torch.stack(kernels, dim=0)


class SRMConv2d(nn.Module):
    """SRM-style high-pass stem initialized with 5x5 residual filters."""

    def __init__(self, out_channels: int = 30, trainable: bool = True):
        super().__init__()
        if out_channels < 6:
            raise ValueError("SRMConv2d out_channels must be at least 6.")
        self.conv = nn.Conv2d(3, out_channels, kernel_size=5, padding=2, bias=False)
        self.reset_parameters()
        for p in self.parameters():
            p.requires_grad = trainable

    def reset_parameters(self) -> None:
        base = _base_srm_kernels()
        repeats = (self.conv.out_channels + base.shape[0] - 1) // base.shape[0]
        kernels = base.repeat(repeats, 1, 1)[: self.conv.out_channels]
        weight = kernels[:, None, :, :].repeat(1, 3, 1, 1) / 3.0
        with torch.no_grad():
            self.conv.weight.copy_(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
