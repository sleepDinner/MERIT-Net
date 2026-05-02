from __future__ import annotations

from collections import defaultdict
from typing import Dict

import torch

from engines.progress import progress_message


def _to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def train_one_epoch(
    model: torch.nn.Module,
    data_loader,
    criterion,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    epoch: int,
    total_epochs: int = 1,
    accumulate_grad_batches: int = 1,
    amp: bool = True,
    logger=None,
    log_interval: int = 20,
) -> Dict[str, float]:
    model.train()
    logs = defaultdict(float)
    count = 0
    import time

    start_time = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    accumulate_grad_batches = max(1, int(accumulate_grad_batches))
    total_steps = len(data_loader)

    for step, batch in enumerate(data_loader, start=1):
        batch = _to_device(batch, device)
        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            outputs = model(batch["image"], valid_region=batch["valid_region"])
            loss, loss_logs = criterion(outputs, batch, epoch=epoch)
            loss_to_backward = loss / accumulate_grad_batches

        scaler.scale(loss_to_backward).backward()
        if step % accumulate_grad_batches == 0 or step == len(data_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        for k, v in loss_logs.items():
            logs[k] += float(v)
        raw_model = model.module if hasattr(model, "module") else model
        if hasattr(raw_model, "gate_statistics"):
            for k, v in raw_model.gate_statistics().items():
                logs[k] += float(v)
        count += 1
        if logger is not None and (step == 1 or step % max(1, log_interval) == 0 or step == total_steps):
            logger.info(
                progress_message(
                    "train",
                    epoch,
                    total_epochs,
                    step,
                    total_steps,
                    start_time,
                    {
                        "loss": logs["loss_total"] / max(1, count),
                        "final_seg": logs["loss_final_seg"] / max(1, count),
                        "coarse_seg": logs["loss_coarse_seg"] / max(1, count),
                    },
                )
            )

    return {k: v / max(1, count) for k, v in logs.items()}
