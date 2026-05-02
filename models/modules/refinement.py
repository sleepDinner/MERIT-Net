from __future__ import annotations

import torch
from torch import nn


class MaskGuidedRefinement(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels + 1, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(32, in_channels), in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(32, in_channels), in_channels),
            nn.GELU(),
        )
        self.mask_head = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, feature: torch.Tensor, coarse_logits_low: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coarse_prob = torch.sigmoid(coarse_logits_low)
        refined_feature = self.block(torch.cat([feature, coarse_prob], dim=1))
        final_low = self.mask_head(refined_feature)
        return final_low, refined_feature
