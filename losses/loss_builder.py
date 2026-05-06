from __future__ import annotations

from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F

from losses.confidence_loss import ConfidenceLoss
from losses.edge_loss import EdgeLoss
from losses.image_loss import ImageLevelLoss
from losses.seg_loss import SegmentationLoss


class MERITLoss(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        loss_cfg = config.get("loss", config)
        model_cfg = config.get("model", {})
        self.weights = {
            "final_seg": float(loss_cfg.get("final_seg", 1.0)),
            "coarse_seg": float(loss_cfg.get("coarse_seg", 0.4)),
            "aux_seg": float(loss_cfg.get("aux_seg", 0.0)),
            "edge": float(loss_cfg.get("edge", 0.2)) if model_cfg.get("use_edge_loss", True) else 0.0,
            "confidence": float(loss_cfg.get("confidence", 0.3)),
            "image": float(loss_cfg.get("image", 0.5)),
            "family": float(loss_cfg.get("family", 0.0)),
        }
        if not model_cfg.get("use_confidence_head", True):
            self.weights["confidence"] = 0.0
        if not model_cfg.get("use_image_head", True):
            self.weights["image"] = 0.0
        if not model_cfg.get("use_family_head", False):
            self.weights["family"] = 0.0
        self.confidence_warmup_epochs = int(loss_cfg.get("confidence_warmup_epochs", 5))
        self.seg_loss = SegmentationLoss(
            bce_weight=float(loss_cfg.get("seg_bce", 1.0)),
            dice_weight=float(loss_cfg.get("seg_dice", 1.0)),
            focal_weight=float(loss_cfg.get("seg_focal", 0.0)),
            tversky_weight=float(loss_cfg.get("seg_tversky", 0.0)),
            focal_alpha=float(loss_cfg.get("focal_alpha", 0.75)),
            focal_gamma=float(loss_cfg.get("focal_gamma", 2.0)),
            tversky_alpha=float(loss_cfg.get("tversky_alpha", 0.3)),
            tversky_beta=float(loss_cfg.get("tversky_beta", 0.7)),
            bce_positive_weight=float(loss_cfg.get("seg_bce_pos_weight", 1.0)),
        )
        self.edge_loss = EdgeLoss()
        self.confidence_loss = ConfidenceLoss(mode=loss_cfg.get("confidence_mode", "l1"))
        self.image_loss = ImageLevelLoss()

    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict, epoch: int = 0) -> tuple[torch.Tensor, Dict[str, float]]:
        target = batch["mask"].float()
        valid = batch["valid_region"].float()
        total = target.new_tensor(0.0)
        logs: Dict[str, torch.Tensor] = {}

        final_seg = self.seg_loss(outputs["final_mask_logits"], target, valid)
        coarse_seg = self.seg_loss(outputs["coarse_mask_logits"], target, valid)
        total = total + self.weights["final_seg"] * final_seg + self.weights["coarse_seg"] * coarse_seg
        logs["loss_final_seg"] = final_seg.detach()
        logs["loss_coarse_seg"] = coarse_seg.detach()

        aux_losses = []
        if self.weights["aux_seg"] > 0:
            for key, value in outputs.items():
                if key.startswith("aux_mask_logits_s"):
                    aux_losses.append(self.seg_loss(value, target, valid))
            if aux_losses:
                aux_seg = torch.stack(aux_losses).mean()
                total = total + self.weights["aux_seg"] * aux_seg
                logs["loss_aux_seg"] = aux_seg.detach()
            else:
                logs["loss_aux_seg"] = target.new_tensor(0.0)
        else:
            logs["loss_aux_seg"] = target.new_tensor(0.0)

        if self.weights["edge"] > 0:
            edge = self.edge_loss(outputs["final_mask_logits"], target, valid)
            total = total + self.weights["edge"] * edge
            logs["loss_edge"] = edge.detach()
        else:
            logs["loss_edge"] = target.new_tensor(0.0)

        if self.weights["confidence"] > 0 and epoch >= self.confidence_warmup_epochs:
            conf = self.confidence_loss(outputs["confidence_logits"], outputs["final_mask_logits"], target, valid)
            total = total + self.weights["confidence"] * conf
            logs["loss_confidence"] = conf.detach()
        else:
            logs["loss_confidence"] = target.new_tensor(0.0)

        if self.weights["image"] > 0:
            img = self.image_loss(outputs["image_logits"], batch["image_level_label"].to(outputs["image_logits"].device))
            total = total + self.weights["image"] * img
            logs["loss_image"] = img.detach()
        else:
            logs["loss_image"] = target.new_tensor(0.0)

        if self.weights["family"] > 0 and "family_logits" in outputs and "family_label" in batch:
            family_label = batch["family_label"].to(outputs["family_logits"].device)
            valid_family = family_label >= 0
            if valid_family.any():
                fam = F.cross_entropy(outputs["family_logits"][valid_family], family_label[valid_family])
                total = total + self.weights["family"] * fam
                logs["loss_family"] = fam.detach()
            else:
                logs["loss_family"] = target.new_tensor(0.0)
        else:
            logs["loss_family"] = target.new_tensor(0.0)

        logs["loss_total"] = total.detach()
        return total, {k: float(v.item()) for k, v in logs.items()}
