from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn
import torch.nn.functional as F

from models.backbones.global_encoder import GlobalEncoder
from models.backbones.residual_encoder import ResidualEncoder
from models.heads.confidence_head import ConfidenceHead
from models.heads.family_head import FamilyHead
from models.heads.image_head import ImageLevelHead
from models.modules.decoder import LightweightDecoder
from models.modules.gated_fusion import ScaleWiseGatedFusion
from models.modules.logit_calibration import LogitCalibration
from models.modules.refinement import MaskGuidedRefinement


class _FeatureProjector(nn.Module):
    def __init__(self, in_channels: List[int], out_channels: List[int]):
        super().__init__()
        self.proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                    nn.GroupNorm(min(32, out_ch), out_ch),
                    nn.GELU(),
                )
                for in_ch, out_ch in zip(in_channels, out_channels)
            ]
        )

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        return [proj(feat) for proj, feat in zip(self.proj, feats)]


class MERITNet(nn.Module):
    def __init__(self, config: Dict | None = None, **kwargs):
        super().__init__()
        cfg = {}
        if config:
            cfg.update(config)
        cfg.update(kwargs)

        self.use_residual_branch = bool(cfg.get("use_residual_branch", True))
        self.use_transformer_branch = bool(cfg.get("use_transformer_branch", True))
        if not self.use_residual_branch and not self.use_transformer_branch:
            raise ValueError("At least one branch must be enabled: residual or transformer/global.")

        self.use_gated_fusion = bool(cfg.get("use_gated_fusion", True))
        self.use_refinement = bool(cfg.get("use_refinement", True))
        self.use_confidence_head = bool(cfg.get("use_confidence_head", True))
        self.use_image_head = bool(cfg.get("use_image_head", True))
        self.use_family_head = bool(cfg.get("use_family_head", False))
        self.use_aux_outputs = bool(cfg.get("use_aux_outputs", False))
        self.use_logit_calibration = bool(cfg.get("use_logit_calibration", False))
        self.gradient_checkpointing = bool(cfg.get("gradient_checkpointing", False))

        fusion_channels = cfg.get("fusion_channels", [64, 128, 256, 512])
        embed_dim = int(cfg.get("decoder_embed_dim", 256))

        if self.use_transformer_branch:
            self.global_encoder = GlobalEncoder(
                backbone_name=cfg.get("global_backbone", "pvt_v2_b2"),
                pretrained=bool(cfg.get("global_pretrained", True)),
                gradient_checkpointing=self.gradient_checkpointing,
                allow_fallback=bool(cfg.get("allow_global_fallback", False)),
                lora_config={
                    "enabled": bool(cfg.get("use_lora", False)),
                    "rank": int(cfg.get("lora_rank", 8)),
                    "alpha": float(cfg.get("lora_alpha", 16.0)),
                    "dropout": float(cfg.get("lora_dropout", 0.05)),
                    "freeze_base": bool(cfg.get("lora_freeze_base", True)),
                    "target_modules": cfg.get(
                        "lora_target_modules",
                        ["attn.q", "attn.kv", "attn.proj", "mlp.fc1", "mlp.fc2"],
                    ),
                },
            )
            global_channels = self.global_encoder.channels
        else:
            self.global_encoder = None
            global_channels = fusion_channels

        if self.use_residual_branch:
            self.residual_encoder = ResidualEncoder(srm_trainable=bool(cfg.get("srm_trainable", True)))
            local_channels = self.residual_encoder.channels
        else:
            self.residual_encoder = None
            local_channels = fusion_channels

        self.global_projector = None
        self.local_projector = None
        self.fusion = None
        if self.use_transformer_branch and self.use_residual_branch:
            self.fusion = ScaleWiseGatedFusion(
                global_channels=global_channels,
                local_channels=local_channels,
                out_channels=fusion_channels,
                use_gated_fusion=self.use_gated_fusion,
            )
        elif self.use_transformer_branch:
            self.global_projector = _FeatureProjector(global_channels, fusion_channels)
        else:
            self.local_projector = _FeatureProjector(local_channels, fusion_channels)

        self.decoder = LightweightDecoder(fusion_channels, embed_dim=embed_dim, use_aux_outputs=self.use_aux_outputs)
        self.refinement = MaskGuidedRefinement(embed_dim) if self.use_refinement else None
        self.logit_calibration = (
            LogitCalibration(
                init_scale=float(cfg.get("calibration_init_scale", 1.0)),
                init_bias=float(cfg.get("calibration_init_bias", 0.0)),
                min_scale=float(cfg.get("calibration_min_scale", 0.2)),
                max_scale=float(cfg.get("calibration_max_scale", 5.0)),
            )
            if self.use_logit_calibration
            else None
        )
        self.confidence_head = ConfidenceHead(embed_dim) if self.use_confidence_head else None
        self.image_head = ImageLevelHead() if self.use_image_head else None
        self.family_head = FamilyHead(embed_dim, int(cfg.get("num_families", 5))) if self.use_family_head else None

    def _encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        global_feats = self.global_encoder(x) if self.global_encoder is not None else None
        local_feats = self.residual_encoder(x) if self.residual_encoder is not None else None
        if global_feats is not None and local_feats is not None:
            return self.fusion(global_feats, local_feats)
        if global_feats is not None:
            return self.global_projector(global_feats)
        if local_feats is not None:
            return self.local_projector(local_feats)
        raise RuntimeError("No enabled encoder branch.")

    def forward(self, x: torch.Tensor, valid_region: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        input_size = x.shape[-2:]
        features = self._encode(x)
        coarse_full, decoder_feature, coarse_low, aux_outputs = self.decoder(features, input_size=input_size)

        if self.refinement is not None:
            final_low, final_feature = self.refinement(decoder_feature, coarse_low)
            final_full = F.interpolate(final_low, size=input_size, mode="bilinear", align_corners=False)
        else:
            final_feature = decoder_feature
            final_full = coarse_full
        raw_final_full = final_full
        if self.logit_calibration is not None:
            final_full = self.logit_calibration(raw_final_full)

        if self.confidence_head is not None:
            confidence_low = self.confidence_head(final_feature)
            confidence_full = F.interpolate(confidence_low, size=input_size, mode="bilinear", align_corners=False)
        else:
            confidence_full = torch.zeros_like(final_full)

        if self.image_head is not None:
            image_logits = self.image_head(
                final_mask_logits=final_full,
                confidence_logits=confidence_full if self.use_confidence_head else None,
                valid_region=valid_region,
            )
        else:
            image_logits = final_full.new_zeros((final_full.shape[0], 1))

        family_logits = None
        if self.family_head is not None:
            family_logits = self.family_head(final_feature)

        output = {
            "coarse_mask_logits": coarse_full,
            "raw_final_mask_logits": raw_final_full,
            "final_mask_logits": final_full,
            "confidence_logits": confidence_full,
            "image_logits": image_logits,
        }
        if family_logits is not None:
            output["family_logits"] = family_logits
        for idx, aux in enumerate(aux_outputs, start=1):
            output[f"aux_mask_logits_s{idx}"] = aux
        if self.fusion is not None and self.use_gated_fusion:
            output["gate_weights"] = self.fusion.last_gate_weights
        return output

    def gate_statistics(self) -> Dict[str, float]:
        if self.fusion is None:
            return {}
        return self.fusion.gate_statistics()

    def encoder_summary(self) -> Dict[str, str]:
        if self.global_encoder is None:
            return {"global_encoder": "disabled"}
        return {"global_encoder": self.global_encoder.summary()}

    def calibration_statistics(self) -> Dict[str, float]:
        if self.logit_calibration is None:
            return {}
        return self.logit_calibration.values()
