from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class LightweightDecoder(nn.Module):
    def __init__(self, in_channels: List[int], embed_dim: int = 256, use_aux_outputs: bool = False):
        super().__init__()
        self.use_aux_outputs = use_aux_outputs
        self.proj = nn.ModuleList([nn.Conv2d(ch, embed_dim, kernel_size=1) for ch in in_channels])
        self.aux_heads = nn.ModuleList([nn.Conv2d(embed_dim, 1, kernel_size=1) for _ in in_channels]) if use_aux_outputs else None
        fusion_ch = embed_dim * len(in_channels)
        self.fuse = nn.Sequential(
            nn.Conv2d(fusion_ch, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(32, embed_dim), embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(32, embed_dim), embed_dim),
            nn.GELU(),
        )
        self.mask_head = nn.Conv2d(embed_dim, 1, kernel_size=1)

    def forward(
        self,
        features: List[torch.Tensor],
        input_size: Tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        target_size = features[0].shape[-2:]
        resized = []
        aux_outputs: List[torch.Tensor] = []
        for idx, (feat, proj) in enumerate(zip(features, self.proj)):
            feat = proj(feat)
            if self.aux_heads is not None:
                aux = self.aux_heads[idx](feat)
                aux_outputs.append(F.interpolate(aux, size=input_size, mode="bilinear", align_corners=False))
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
            resized.append(feat)
        fused_feature = self.fuse(torch.cat(resized, dim=1))
        coarse_low = self.mask_head(fused_feature)
        coarse_full = F.interpolate(coarse_low, size=input_size, mode="bilinear", align_corners=False)
        return coarse_full, fused_feature, coarse_low, aux_outputs
