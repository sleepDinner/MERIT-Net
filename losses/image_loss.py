from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ImageLevelLoss(nn.Module):
    def forward(self, image_logits: torch.Tensor, image_labels: torch.Tensor) -> torch.Tensor:
        labels = image_labels.float().view(-1, 1)
        return F.binary_cross_entropy_with_logits(image_logits, labels)
