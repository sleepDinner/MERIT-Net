from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConfidenceLoss(nn.Module):
    def __init__(self, mode: str = "l1"):
        super().__init__()
        if mode not in {"l1", "bce"}:
            raise ValueError("ConfidenceLoss mode must be 'l1' or 'bce'.")
        self.mode = mode

    def forward(
        self,
        confidence_logits: torch.Tensor,
        final_mask_logits: torch.Tensor,
        target: torch.Tensor,
        valid_region: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            p = torch.sigmoid(final_mask_logits.detach())
            y = target.float()
            conf_target = y * p + (1.0 - y) * (1.0 - p)
        if self.mode == "bce":
            loss = F.binary_cross_entropy_with_logits(confidence_logits, conf_target, reduction="none")
        else:
            loss = torch.abs(torch.sigmoid(confidence_logits) - conf_target)
        if valid_region is None:
            return loss.mean()
        valid_region = valid_region.float()
        return (loss * valid_region).sum() / valid_region.sum().clamp_min(1.0)
