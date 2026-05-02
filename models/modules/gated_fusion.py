from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn


class _FusionUnit(nn.Module):
    def __init__(self, global_ch: int, local_ch: int, out_ch: int, use_gated_fusion: bool = True):
        super().__init__()
        self.use_gated_fusion = use_gated_fusion
        self.global_proj = nn.Conv2d(global_ch, out_ch, kernel_size=1)
        self.local_proj = nn.Conv2d(local_ch, out_ch, kernel_size=1)
        if use_gated_fusion:
            hidden = max(out_ch // 4, 16)
            self.gate = nn.Sequential(
                nn.Conv2d(out_ch * 2, hidden, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(min(16, hidden), hidden),
                nn.GELU(),
                nn.Conv2d(hidden, 2, kernel_size=1),
            )
        else:
            self.fuse = nn.Sequential(
                nn.Conv2d(out_ch * 2, out_ch, kernel_size=1, bias=False),
                nn.GroupNorm(min(32, out_ch), out_ch),
                nn.GELU(),
            )

    def forward(self, g: torch.Tensor, l: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        g = self.global_proj(g)
        l = self.local_proj(l)
        if g.shape[-2:] != l.shape[-2:]:
            l = torch.nn.functional.interpolate(l, size=g.shape[-2:], mode="bilinear", align_corners=False)
        cat = torch.cat([g, l], dim=1)
        if not self.use_gated_fusion:
            return self.fuse(cat), None
        weights = torch.softmax(self.gate(cat), dim=1)
        fused = weights[:, 0:1] * g + weights[:, 1:2] * l
        return fused, weights


class ScaleWiseGatedFusion(nn.Module):
    def __init__(
        self,
        global_channels: List[int],
        local_channels: List[int],
        out_channels: List[int],
        use_gated_fusion: bool = True,
    ):
        super().__init__()
        self.use_gated_fusion = use_gated_fusion
        self.units = nn.ModuleList(
            [
                _FusionUnit(g_ch, l_ch, out_ch, use_gated_fusion=use_gated_fusion)
                for g_ch, l_ch, out_ch in zip(global_channels, local_channels, out_channels)
            ]
        )
        self.last_gate_weights: List[torch.Tensor] = []

    def forward(self, global_feats: List[torch.Tensor], local_feats: List[torch.Tensor]) -> List[torch.Tensor]:
        outputs = []
        self.last_gate_weights = []
        for unit, g, l in zip(self.units, global_feats, local_feats):
            fused, weights = unit(g, l)
            outputs.append(fused)
            if weights is not None:
                self.last_gate_weights.append(weights.detach())
        return outputs

    def gate_statistics(self) -> Dict[str, float]:
        stats: Dict[str, float] = {}
        for idx, weight in enumerate(self.last_gate_weights, start=1):
            stats[f"gate_s{idx}_global"] = float(weight[:, 0].mean().item())
            stats[f"gate_s{idx}_local"] = float(weight[:, 1].mean().item())
        return stats
