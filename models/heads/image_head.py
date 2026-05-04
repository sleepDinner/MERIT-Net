from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ImageLevelHead(nn.Module):
    def __init__(self, hidden_dim: int = 32, topk_ratio: float = 0.01):
        super().__init__()
        self.topk_ratio = topk_ratio
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        final_mask_logits: torch.Tensor,
        confidence_logits: torch.Tensor | None = None,
        valid_region: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask_prob = torch.sigmoid(final_mask_logits)
        if confidence_logits is None:
            confidence_prob = torch.ones_like(mask_prob)
        else:
            confidence_prob = torch.sigmoid(confidence_logits)
        if valid_region is None:
            valid_region = torch.ones_like(mask_prob)
        if valid_region.shape[-2:] != mask_prob.shape[-2:]:
            valid_region = F.interpolate(valid_region.float(), size=mask_prob.shape[-2:], mode="nearest")
        valid_region = (valid_region > 0.5).float()

        features = []
        bsz = mask_prob.shape[0]
        for b in range(bsz):
            valid = valid_region[b, 0] > 0.5
            values = mask_prob[b, 0][valid]
            conf = confidence_prob[b, 0][valid]
            if values.numel() == 0:
                features.append(mask_prob.new_zeros(4))
                continue
            mean_conf = (values * conf).mean()
            max_val = values.max()
            k = max(1, int(values.numel() * self.topk_ratio))
            topk_mean = torch.topk(values, k=k, largest=True).values.mean()
            area_ratio = (values > 0.5).float().mean()
            features.append(torch.stack([mean_conf, max_val, topk_mean, area_ratio]))
        pooled = torch.stack(features, dim=0)
        return self.mlp(pooled)
