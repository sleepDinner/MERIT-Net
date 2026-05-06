from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn

from models.modules.lora import LoRAInjectionStats, inject_lora_linear


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
        backbone_name: str = "pvt_v2_b2",
        pretrained: bool = True,
        gradient_checkpointing: bool = False,
        allow_fallback: bool = False,
        lora_config: Dict | None = None,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.allow_fallback = allow_fallback
        self.lora_config = lora_config or {}
        self.uses_timm = False
        self.fallback_reason = ""
        self.lora_stats: LoRAInjectionStats | None = None
        self.model: nn.Module
        self.channels: List[int]

        if backbone_name.lower() in {"fallback", "fallback_cnn", "small_cnn"}:
            self.model = FallbackGlobalEncoder()
            self.channels = self.model.channels
            self.uses_timm = False
            self.fallback_reason = "explicit fallback backbone requested"
            return

        timm_version = "not imported"
        try:
            import timm

            timm_version = getattr(timm, "__version__", "unknown")
            self.model = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=(0, 1, 2, 3),
            )
            self.channels = list(self.model.feature_info.channels())
            self.uses_timm = True
            if bool(self.lora_config.get("enabled", False)):
                self._inject_lora()
            if gradient_checkpointing and hasattr(self.model, "set_grad_checkpointing"):
                self.model.set_grad_checkpointing(True)
        except Exception as exc:
            if not allow_fallback:
                raise RuntimeError(
                    "Failed to initialize the requested global backbone "
                    f"'{backbone_name}' with pretrained={pretrained}. Install timm>=1.0.26 "
                    f"(current timm version: {timm_version}) and make sure "
                    "the pretrained weights are available, or set model.allow_global_fallback=true "
                    "or model.global_backbone=fallback_cnn explicitly if you really want the small CNN fallback. "
                    f"Original error: {type(exc).__name__}: {exc}"
                ) from exc
            self.model = FallbackGlobalEncoder()
            self.channels = self.model.channels
            self.uses_timm = False
            self.fallback_reason = str(exc)

    def _inject_lora(self) -> None:
        target_modules = self.lora_config.get(
            "target_modules",
            ["attn.q", "attn.kv", "attn.proj", "mlp.fc1", "mlp.fc2"],
        )
        freeze_base = bool(self.lora_config.get("freeze_base", True))
        if freeze_base:
            for param in self.model.parameters():
                param.requires_grad_(False)
        self.lora_stats = inject_lora_linear(
            self.model,
            target_modules=target_modules,
            rank=int(self.lora_config.get("rank", 8)),
            alpha=float(self.lora_config.get("alpha", 16.0)),
            dropout=float(self.lora_config.get("dropout", 0.05)),
            freeze_base=freeze_base,
        )
        if self.lora_stats.applied_modules <= 0:
            raise RuntimeError(
                "LoRA is enabled but no target Linear modules were found in "
                f"backbone='{self.backbone_name}'. target_modules={list(target_modules)}"
            )

    def summary(self) -> str:
        if self.uses_timm:
            lora_text = ""
            if self.lora_stats is not None:
                lora_text = (
                    f" lora=enabled rank={self.lora_stats.rank} alpha={self.lora_stats.alpha:g} "
                    f"dropout={self.lora_stats.dropout:g} modules={self.lora_stats.applied_modules} "
                    f"trainable_lora_params={self.lora_stats.lora_parameters} "
                    f"freeze_base={self.lora_stats.freeze_base}"
                )
            return (
                f"GlobalEncoder: uses_timm=True backbone={self.backbone_name} "
                f"pretrained={self.pretrained} channels={self.channels}{lora_text}"
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
