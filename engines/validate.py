from __future__ import annotations

from collections import defaultdict
from typing import Dict

import torch
import torch.distributed as dist

from eval.metrics import MetricAccumulator
from engines.progress import progress_message
from engines.train_one_epoch import _autocast_context, _to_device


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    data_loader,
    criterion,
    device: torch.device,
    epoch: int,
    total_epochs: int = 1,
    amp: bool = True,
    threshold: float = 0.5,
    prefix: str = "val",
    logger=None,
    log_interval: int = 20,
) -> Dict[str, float]:
    model.eval()
    accumulator = MetricAccumulator(threshold=threshold)
    loss_logs = defaultdict(float)
    count = 0
    import time

    start_time = time.perf_counter()
    total_steps = len(data_loader)

    for step, batch in enumerate(data_loader, start=1):
        batch = _to_device(batch, device)
        with _autocast_context(device, amp):
            outputs = model(batch["image"], valid_region=batch["valid_region"])
            loss, logs = criterion(outputs, batch, epoch=epoch)
        for k, v in logs.items():
            loss_logs[k] += float(v)
        accumulator.update(
            mask_prob=torch.sigmoid(outputs["final_mask_logits"]),
            gt_mask=batch["mask"],
            valid_region=batch["valid_region"],
            image_logits=outputs.get("image_logits"),
        )
        count += 1
        if logger is not None and (step == 1 or step % max(1, log_interval) == 0 or step == total_steps):
            logger.info(
                progress_message(
                    prefix,
                    epoch,
                    total_epochs,
                    step,
                    total_steps,
                    start_time,
                    {"loss": loss_logs["loss_total"] / max(1, count)},
                )
            )

    if dist.is_available() and dist.is_initialized():
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, accumulator.state_dict())
        merged = MetricAccumulator(threshold=threshold)
        for state in gathered:
            merged.merge(MetricAccumulator.from_state_dict(state))
        accumulator = merged
        tensor = torch.tensor([count] + [loss_logs[k] for k in sorted(loss_logs)], dtype=torch.float64, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        total_count = int(tensor[0].item())
        reduced_loss = {k: float(tensor[i + 1].item()) / max(1, total_count) for i, k in enumerate(sorted(loss_logs))}
    else:
        reduced_loss = {k: v / max(1, count) for k, v in loss_logs.items()}

    metrics = accumulator.compute(prefix=prefix)
    metrics.update({f"{prefix}_{k}": v for k, v in reduced_loss.items()})
    if logger is not None:
        logger.info(
            f"Epoch {epoch}/{total_epochs} {prefix} metrics | "
            f"pixel_f1={metrics.get(prefix + '_pixel_f1', float('nan')):.5f} "
            f"iou={metrics.get(prefix + '_iou', float('nan')):.5f} "
            f"pixel_auc={metrics.get(prefix + '_pixel_auc', float('nan')):.5f} "
            f"image_auc={metrics.get(prefix + '_image_auc', float('nan')):.5f} "
            f"balanced_acc={metrics.get(prefix + '_balanced_accuracy', float('nan')):.5f} "
            f"fpr={metrics.get(prefix + '_fpr', float('nan')):.5f}"
        )
    return metrics
