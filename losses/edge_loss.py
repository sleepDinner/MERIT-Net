from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def mask_to_edge(mask: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    pad = kernel_size // 2
    dilated = F.max_pool2d(mask.float(), kernel_size=kernel_size, stride=1, padding=pad)
    eroded = 1.0 - F.max_pool2d(1.0 - mask.float(), kernel_size=kernel_size, stride=1, padding=pad)
    return (dilated - eroded).clamp(0, 1)


class EdgeLoss(nn.Module):
    def __init__(self, kernel_size: int = 5):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, logits: torch.Tensor, target: torch.Tensor, valid_region: torch.Tensor | None = None) -> torch.Tensor:
        prob_edge = mask_to_edge(torch.sigmoid(logits.float()), self.kernel_size)
        target_edge = mask_to_edge(target.float(), self.kernel_size)
        loss = torch.abs(prob_edge - target_edge)
        if valid_region is None:
            return loss.mean()
        valid_region = valid_region.float()
        safe_valid = 1.0 - F.max_pool2d(1.0 - valid_region, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2)
        safe_valid = safe_valid.clamp(0.0, 1.0)
        if safe_valid.sum() < 1:
            safe_valid = valid_region
        return (loss * safe_valid).sum() / safe_valid.sum().clamp_min(1.0)
