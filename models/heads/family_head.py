from __future__ import annotations

import torch
from torch import nn


class FamilyHead(nn.Module):
    def __init__(self, in_channels: int, num_families: int = 5):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, max(64, in_channels // 2)),
            nn.GELU(),
            nn.Linear(max(64, in_channels // 2), num_families),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(feature))
