from __future__ import annotations

from collections import defaultdict
from typing import Dict

import torch
import torch.distributed as dist

from eval.metrics import MetricAccumulator
from engines.progress import ProgressLine, progress_message
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
    display_prefix: str | None = None,
    logger=None,
    log_interval: int = 20,
    max_batches: int | None = None,
    max_pixel_auc_samples: int = 2_000_000,
    pixel_auc_samples_per_image: int = 4096,
    pixel_auc_seed: int = 12345,
    report_thresholds: list[float] | None = None,
) -> Dict[str, float]:
    model.eval()
    accumulator = MetricAccumulator(
        threshold=threshold,
        max_pixel_auc_samples=max_pixel_auc_samples,
        pixel_auc_samples_per_image=pixel_auc_samples_per_image,
        pixel_auc_seed=pixel_auc_seed,
        report_thresholds=report_thresholds,
    )
    loss_logs = defaultdict(float)
    count = 0
    import time

    start_time = time.perf_counter()
    batch_limit = int(max_batches) if max_batches is not None and int(max_batches) > 0 else None
    total_steps = len(data_loader)
    if batch_limit is not None:
        total_steps = min(total_steps, batch_limit)
    progress = ProgressLine() if logger is not None else None
    progress_prefix = display_prefix or prefix

    for step, batch in enumerate(data_loader, start=1):
        if batch_limit is not None and step > batch_limit:
            break
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
        if progress is not None and (step == 1 or step % max(1, log_interval) == 0 or step == total_steps):
            progress.update(
                progress_message(
                    progress_prefix,
                    epoch,
                    total_epochs,
                    step,
                    total_steps,
                    start_time,
                    {"loss": loss_logs["loss_total"] / max(1, count)},
                )
            )

    if progress is not None:
        progress.finish()
    if dist.is_available() and dist.is_initialized():
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, accumulator.state_dict())
        merged = MetricAccumulator(threshold=threshold, report_thresholds=report_thresholds)
        for state in gathered:
            merged.merge(MetricAccumulator.from_state_dict(state))
        accumulator = merged
        loss_states = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(loss_states, {"count": count, "loss_logs": dict(loss_logs)})
        total_count = sum(int(state["count"]) for state in loss_states if state is not None)
        keys = sorted({k for state in loss_states if state is not None for k in state["loss_logs"]})
        reduced_loss = {
            k: sum(float(state["loss_logs"].get(k, 0.0)) for state in loss_states if state is not None) / max(1, total_count)
            for k in keys
        }
    else:
        reduced_loss = {k: v / max(1, count) for k, v in loss_logs.items()}

    metrics = accumulator.compute(prefix=prefix)
    metrics.update({f"{prefix}_{k}": v for k, v in reduced_loss.items()})
    if logger is not None:
        logger.info(
            f"Epoch {epoch}/{total_epochs} {progress_prefix} metrics | "
            f"pixel_f1={metrics.get(prefix + '_pixel_f1', float('nan')):.5f} "
            f"best_f1={metrics.get(prefix + '_best_pixel_f1', float('nan')):.5f} "
            f"best_thr={metrics.get(prefix + '_best_threshold', float('nan')):.3f} "
            f"recall={metrics.get(prefix + '_pixel_recall', float('nan')):.5f} "
            f"iou={metrics.get(prefix + '_iou', float('nan')):.5f} "
            f"pixel_auc={metrics.get(prefix + '_pixel_auc', float('nan')):.5f} "
            f"image_auc={metrics.get(prefix + '_image_auc', float('nan')):.5f} "
            f"balanced_acc={metrics.get(prefix + '_balanced_accuracy', float('nan')):.5f} "
            f"fpr={metrics.get(prefix + '_fpr', float('nan')):.5f}"
        )
    return metrics
