from __future__ import annotations

from typing import List

import torch
from torch import nn


class ConvNormAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch),
            nn.GELU(),
        )


class FallbackGlobalEncoder(nn.Module):
    """Small CNN fallback used when timm or the requested backbone is unavailable."""

    def __init__(self, in_chans: int = 3, channels: List[int] | None = None):
        super().__init__()
        channels = channels or [64, 128, 320, 512]
        self.channels = channels
        self.stem = nn.Sequential(
            ConvNormAct(in_chans, 32, stride=2),
            ConvNormAct(32, channels[0], stride=2),
        )
        self.stage2 = nn.Sequential(ConvNormAct(channels[0], channels[1], stride=2), ConvNormAct(channels[1], channels[1]))
        self.stage3 = nn.Sequential(ConvNormAct(channels[1], channels[2], stride=2), ConvNormAct(channels[2], channels[2]))
        self.stage4 = nn.Sequential(ConvNormAct(channels[2], channels[3], stride=2), ConvNormAct(channels[3], channels[3]))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        f1 = self.stem(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return [f1, f2, f3, f4]


class GlobalEncoder(nn.Module):
    def __init__(
        self,
        backbone_name: str = "pvt_v2_b1",
        pretrained: bool = True,
        gradient_checkpointing: bool = False,
        allow_fallback: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.allow_fallback = allow_fallback
        self.uses_timm = False
        self.fallback_reason = ""
        self.model: nn.Module
        self.channels: List[int]

        if backbone_name.lower() in {"fallback", "fallback_cnn", "small_cnn"}:
            self.model = FallbackGlobalEncoder()
            self.channels = self.model.channels
            self.uses_timm = False
            self.fallback_reason = "explicit fallback backbone requested"
            return

        try:
            import timm

            self.model = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=(0, 1, 2, 3),
            )
            self.channels = list(self.model.feature_info.channels())
            self.uses_timm = True
            if gradient_checkpointing and hasattr(self.model, "set_grad_checkpointing"):
                self.model.set_grad_checkpointing(True)
        except Exception as exc:
            if not allow_fallback:
                raise RuntimeError(
                    "Failed to initialize the requested global backbone "
                    f"'{backbone_name}' with pretrained={pretrained}. Install timm and make sure "
                    "the pretrained weights are available, or set model.allow_global_fallback=true "
                    "or model.global_backbone=fallback_cnn explicitly if you really want the small CNN fallback. "
                    f"Original error: {type(exc).__name__}: {exc}"
                ) from exc
            self.model = FallbackGlobalEncoder()
            self.channels = self.model.channels
            self.uses_timm = False
            self.fallback_reason = str(exc)

    def summary(self) -> str:
        if self.uses_timm:
            return (
                f"GlobalEncoder: uses_timm=True backbone={self.backbone_name} "
                f"pretrained={self.pretrained} channels={self.channels}"
            )
        return (
            f"GlobalEncoder: uses_timm=False backbone=fallback_cnn "
            f"channels={self.channels} reason={self.fallback_reason}"
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = self.model(x)
        if not isinstance(feats, (list, tuple)):
            raise RuntimeError("Global encoder must return a list/tuple of feature maps.")
        feats = list(feats)
        if len(feats) < 4:
            raise RuntimeError(f"Global encoder returned {len(feats)} features; expected at least 4.")
        return feats[:4]
