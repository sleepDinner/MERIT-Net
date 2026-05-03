from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from losses.dice_loss import DiceLoss, TverskyLoss


def masked_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, valid_region: torch.Tensor | None = None) -> torch.Tensor:
    target = target.float()
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if valid_region is None:
        return loss.mean()
    valid_region = valid_region.float()
    return (loss * valid_region).sum() / valid_region.sum().clamp_min(1.0)


class SegmentationLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        focal_weight: float = 0.0,
        tversky_weight: float = 0.0,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
    ):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.focal_weight = float(focal_weight)
        self.tversky_weight = float(tversky_weight)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        self.dice = DiceLoss()
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)

    def focal_loss(self, logits: torch.Tensor, target: torch.Tensor, valid_region: torch.Tensor | None = None) -> torch.Tensor:
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        prob = torch.sigmoid(logits)
        p_t = prob * target + (1.0 - prob) * (1.0 - target)
        alpha_t = self.focal_alpha * target + (1.0 - self.focal_alpha) * (1.0 - target)
        loss = alpha_t * torch.pow((1.0 - p_t).clamp_min(1e-6), self.focal_gamma) * bce
        if valid_region is None:
            return loss.mean()
        valid_region = valid_region.float()
        return (loss * valid_region).sum() / valid_region.sum().clamp_min(1.0)

    def forward(self, logits: torch.Tensor, target: torch.Tensor, valid_region: torch.Tensor | None = None) -> torch.Tensor:
        loss = logits.new_tensor(0.0)
        if self.bce_weight > 0:
            loss = loss + self.bce_weight * masked_bce_with_logits(logits, target, valid_region)
        if self.dice_weight > 0:
            loss = loss + self.dice_weight * self.dice(logits, target, valid_region)
        if self.focal_weight > 0:
            loss = loss + self.focal_weight * self.focal_loss(logits, target, valid_region)
        if self.tversky_weight > 0:
            loss = loss + self.tversky_weight * self.tversky(logits, target, valid_region)
        return loss
