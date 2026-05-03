from __future__ import annotations

from collections import defaultdict
from typing import Dict

import torch
import torch.distributed as dist

from eval.metrics import MetricAccumulator
from engines.progress import ProgressLine, progress_message


def _to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def _autocast_context(device: torch.device, enabled: bool):
    enabled = bool(enabled and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


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
    threshold: float = 0.5,
    compute_metrics: bool = True,
) -> Dict[str, float]:
    model.train()
    logs = defaultdict(float)
    metric_acc = MetricAccumulator(threshold=threshold) if compute_metrics else None
    count = 0
    import time

    start_time = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    accumulate_grad_batches = max(1, int(accumulate_grad_batches))
    total_steps = len(data_loader)
    progress = ProgressLine() if logger is not None else None

    for step, batch in enumerate(data_loader, start=1):
        batch = _to_device(batch, device)
        with _autocast_context(device, amp):
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
        if metric_acc is not None:
            metric_acc.update(
                mask_prob=torch.sigmoid(outputs["final_mask_logits"]).detach(),
                gt_mask=batch["mask"].detach(),
                valid_region=batch["valid_region"].detach(),
                image_logits=outputs.get("image_logits").detach() if torch.is_tensor(outputs.get("image_logits")) else None,
            )
        count += 1
        if progress is not None and (step == 1 or step % max(1, log_interval) == 0 or step == total_steps):
            progress.update(
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

    if progress is not None:
        progress.finish()
    if dist.is_available() and dist.is_initialized():
        keys = sorted(logs.keys())
        tensor = torch.tensor([count] + [logs[k] for k in keys], dtype=torch.float64, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        total_count = int(tensor[0].item())
        result = {k: float(tensor[i + 1].item()) / max(1, total_count) for i, k in enumerate(keys)}
        if metric_acc is not None:
            gathered = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered, metric_acc.state_dict())
            merged = MetricAccumulator(threshold=threshold)
            for state in gathered:
                merged.merge(MetricAccumulator.from_state_dict(state))
            metric_acc = merged
    else:
        result = {k: v / max(1, count) for k, v in logs.items()}
    if metric_acc is not None:
        metrics = metric_acc.compute()
        result.update({k: v for k, v in metrics.items() if k in {"pixel_f1", "pixel_auc", "iou", "pixel_precision", "pixel_recall"}})
    return result
