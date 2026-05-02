from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from losses.dice_loss import DiceLoss


def masked_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, valid_region: torch.Tensor | None = None) -> torch.Tensor:
    target = target.float()
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if valid_region is None:
        return loss.mean()
    valid_region = valid_region.float()
    return (loss * valid_region).sum() / valid_region.sum().clamp_min(1.0)


class SegmentationLoss(nn.Module):
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor, valid_region: torch.Tensor | None = None) -> torch.Tensor:
        return self.bce_weight * masked_bce_with_logits(logits, target, valid_region) + self.dice_weight * self.dice(
            logits, target, valid_region
        )
