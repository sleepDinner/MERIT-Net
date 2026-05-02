from __future__ import annotations

from typing import List

import torch
from torch import nn

from models.modules.srm_conv import SRMConv2d


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(min(32, out_ch), out_ch)
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(min(32, out_ch), out_ch),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.act(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return self.act(out + identity)


class ResidualEncoder(nn.Module):
    """Lightweight ResNet18-like residual encoder fed by an SRM high-pass stem."""

    def __init__(self, srm_trainable: bool = True, srm_channels: int = 30, channels: List[int] | None = None):
        super().__init__()
        channels = channels or [64, 128, 256, 512]
        self.channels = channels
        self.srm = SRMConv2d(out_channels=srm_channels, trainable=srm_trainable)
        self.stem = nn.Sequential(
            nn.Conv2d(srm_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(32, channels[0]), channels[0]),
            nn.GELU(),
        )
        self.layer1 = nn.Sequential(BasicBlock(channels[0], channels[0]), BasicBlock(channels[0], channels[0]))
        self.layer2 = nn.Sequential(BasicBlock(channels[0], channels[1], stride=2), BasicBlock(channels[1], channels[1]))
        self.layer3 = nn.Sequential(BasicBlock(channels[1], channels[2], stride=2), BasicBlock(channels[2], channels[2]))
        self.layer4 = nn.Sequential(BasicBlock(channels[2], channels[3], stride=2), BasicBlock(channels[3], channels[3]))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.srm(x)
        f1 = self.layer1(self.stem(x))
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f1, f2, f3, f4]
