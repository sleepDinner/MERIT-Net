from __future__ import annotations

import torch
from torch import nn


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor, valid_region: torch.Tensor | None = None) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        target = target.float()
        if valid_region is None:
            valid_region = torch.ones_like(target)
        valid_region = valid_region.float()
        dims = (1, 2, 3)
        prob = prob * valid_region
        target = target * valid_region
        intersection = (prob * target).sum(dim=dims)
        denominator = prob.sum(dim=dims) + target.sum(dim=dims)
        dice = (2.0 * intersection + self.eps) / (denominator + self.eps)
        return (1.0 - dice).mean()
