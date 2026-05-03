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


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, eps: float = 1e-6):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor, valid_region: torch.Tensor | None = None) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        target = target.float()
        if valid_region is None:
            valid_region = torch.ones_like(target)
        valid_region = valid_region.float()
        prob = prob * valid_region
        target = target * valid_region
        dims = (1, 2, 3)
        tp = (prob * target).sum(dim=dims)
        fp = (prob * (1.0 - target) * valid_region).sum(dim=dims)
        fn = ((1.0 - prob) * target).sum(dim=dims)
        score = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        return (1.0 - score).mean()
