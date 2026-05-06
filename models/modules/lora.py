from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Low-rank adapter for an existing Linear layer.

    The wrapped base layer keeps its original forward path. LoRA adds a small
    trainable residual branch initialized to zero output, so enabling it starts
    from the exact pretrained backbone behavior.
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
        freeze_base: bool = True,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}.")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.lora_down = nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_up = nn.Linear(self.rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)
        if freeze_base:
            for param in self.base.parameters():
                param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_up(self.dropout(self.lora_down(x))) * self.scaling


@dataclass
class LoRAInjectionStats:
    applied_modules: int = 0
    lora_parameters: int = 0
    target_patterns: tuple[str, ...] = ()
    rank: int = 0
    alpha: float = 0.0
    dropout: float = 0.0
    freeze_base: bool = True

    def as_dict(self) -> dict:
        return {
            "applied_modules": self.applied_modules,
            "lora_parameters": self.lora_parameters,
            "target_patterns": list(self.target_patterns),
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "freeze_base": self.freeze_base,
        }


def _matches_target(module_name: str, patterns: Iterable[str]) -> bool:
    return any(module_name == pattern or module_name.endswith(pattern) or pattern in module_name for pattern in patterns)


def _parent_and_child(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent._modules[part]
    return parent, parts[-1]


def inject_lora_linear(
    root: nn.Module,
    target_modules: Iterable[str],
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
    freeze_base: bool = True,
) -> LoRAInjectionStats:
    patterns = tuple(str(item).strip() for item in target_modules if str(item).strip())
    if not patterns:
        raise ValueError("LoRA target_modules is empty.")

    matches = [
        (name, module)
        for name, module in root.named_modules()
        if name and isinstance(module, nn.Linear) and not isinstance(module, LoRALinear) and _matches_target(name, patterns)
    ]

    stats = LoRAInjectionStats(
        applied_modules=0,
        lora_parameters=0,
        target_patterns=patterns,
        rank=int(rank),
        alpha=float(alpha),
        dropout=float(dropout),
        freeze_base=bool(freeze_base),
    )
    for name, module in matches:
        parent, child = _parent_and_child(root, name)
        wrapped = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout, freeze_base=freeze_base)
        parent._modules[child] = wrapped
        stats.applied_modules += 1
        stats.lora_parameters += sum(
            param.numel()
            for param_name, param in wrapped.named_parameters()
            if param_name.startswith("lora_") and param.requires_grad
        )
    return stats
