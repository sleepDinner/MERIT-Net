from __future__ import annotations

import torch
from torch import nn


class LogitCalibration(nn.Module):
    """A lightweight scalar calibration layer for mask logits.

    It starts as identity: calibrated_logits = logits. During stage2/stage3
    fine-tuning it can learn a global scale and bias, which is useful when
    localization is reasonable but mask probabilities are poorly calibrated.
    """

    def __init__(
        self,
        init_scale: float = 1.0,
        init_bias: float = 0.0,
        min_scale: float = 0.2,
        max_scale: float = 5.0,
    ):
        super().__init__()
        init_scale = max(float(init_scale), 1e-6)
        self.log_scale = nn.Parameter(torch.tensor(float(init_scale)).log())
        self.bias = nn.Parameter(torch.tensor(float(init_bias)))
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        scale = self.log_scale.exp().clamp(self.min_scale, self.max_scale)
        return logits * scale + self.bias

    def values(self) -> dict[str, float]:
        scale = float(self.log_scale.detach().exp().clamp(self.min_scale, self.max_scale).item())
        bias = float(self.bias.detach().item())
        return {"logit_calibration_scale": scale, "logit_calibration_bias": bias}
