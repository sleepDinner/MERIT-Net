from __future__ import annotations

import math
from typing import Dict

import torch


def _match_lr_multiplier(name: str, multipliers: Dict[str, float]) -> tuple[str, float]:
    clean_name = name[7:] if name.startswith("module.") else name
    best_prefix = "default"
    best_multiplier = 1.0
    best_len = -1
    for prefix, multiplier in multipliers.items():
        prefix = str(prefix).strip()
        if not prefix:
            continue
        clean_prefix = prefix[7:] if prefix.startswith("module.") else prefix
        if clean_name == clean_prefix or clean_name.startswith(clean_prefix + "."):
            if len(clean_prefix) > best_len:
                best_prefix = clean_prefix
                best_multiplier = float(multiplier)
                best_len = len(clean_prefix)
    return best_prefix, best_multiplier


def build_optimizer(model: torch.nn.Module, train_cfg: Dict) -> torch.optim.Optimizer:
    name = str(train_cfg.get("optimizer", "AdamW")).lower()
    lr = float(train_cfg.get("lr", 6e-5))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    lr_multipliers = train_cfg.get("param_lr_multipliers", {}) or {}
    if lr_multipliers:
        grouped = {}
        for param_name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            group_name, multiplier = _match_lr_multiplier(param_name, lr_multipliers)
            group_key = (group_name, multiplier)
            grouped.setdefault(group_key, []).append(param)
        params = [
            {
                "params": group_params,
                "lr": lr * multiplier,
                "weight_decay": weight_decay,
                "name": group_name,
                "lr_multiplier": multiplier,
            }
            for (group_name, multiplier), group_params in grouped.items()
            if group_params
        ]
    else:
        params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found for optimizer.")
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {train_cfg.get('optimizer')}")


def build_scheduler(optimizer: torch.optim.Optimizer, train_cfg: Dict):
    scheduler_name = str(train_cfg.get("scheduler", "cosine")).lower()
    epochs = int(train_cfg.get("epochs", 80))
    warmup_epochs = int(train_cfg.get("warmup_epochs", 3))
    lr = float(train_cfg.get("lr", 6e-5))
    min_lr = float(train_cfg.get("min_lr", 1e-6))
    min_ratio = min_lr / max(lr, 1e-12)

    if scheduler_name != "cosine":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    def lr_lambda(epoch_index: int) -> float:
        if epoch_index < warmup_epochs:
            return max(min_ratio, float(epoch_index + 1) / max(1, warmup_epochs))
        progress = (epoch_index - warmup_epochs) / max(1, epochs - warmup_epochs)
        return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
