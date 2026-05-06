from __future__ import annotations

import os
import random
import csv
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

from datasets.dataset_scanner import scan_and_split_from_config, should_resplit_from_config
from datasets.tamper_dataset import TamperDataset
from engines.logger import CSVMetricLogger, setup_logger
from engines.optim_scheduler import build_optimizer, build_scheduler
from engines.plot_curves import plot_training_curves
from engines.progress import format_seconds
from engines.train_one_epoch import train_one_epoch
from engines.trainer_ddp import distributed_barrier, init_distributed, is_rank0
from engines.validate import validate
from losses.loss_builder import MERITLoss
from models.merit_net import MERITNet


class DistributedEvalSampler(Sampler[int]):
    """Partition eval indices across ranks without padding or duplicated samples."""

    def __init__(self, dataset, num_replicas: int | None = None, rank: int | None = None):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self) -> int:
        if self.rank >= len(self.dataset):
            return 0
        return (len(self.dataset) - 1 - self.rank) // self.num_replicas + 1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def _worker_loader_kwargs(num_workers: int, persistent_workers: bool, prefetch_factor: int | None) -> Dict:
    if num_workers <= 0:
        return {}
    kwargs = {"persistent_workers": persistent_workers}
    if prefetch_factor is not None:
        kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


def _read_info_path(path: Path, key: str) -> Optional[str]:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{key}:"):
            return line.split(":", 1)[1].strip()
    return None


def _save_info(path: Path, lines: Dict[str, str | int | float]) -> None:
    path.write_text("\n".join(f"{k}: {v}" for k, v in lines.items()) + "\n", encoding="utf-8")


def _format_metric(value: object) -> str:
    try:
        metric = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(metric):
        return "nan"
    return f"{metric:.6f}"


def _checkpoint_metric_info(
    prefix: str,
    train_logs: Dict[str, float],
    val_logs: Dict[str, float],
    monitor: str,
) -> Dict[str, str]:
    return {
        f"{prefix}_monitor_name": monitor,
        f"{prefix}_monitor_value": _format_metric(val_logs.get(monitor)),
        f"{prefix}_train_loss_total": _format_metric(train_logs.get("loss_total")),
        f"{prefix}_val_loss_total": _format_metric(val_logs.get("val_loss_total")),
        f"{prefix}_val_loss_final_seg": _format_metric(val_logs.get("val_loss_final_seg")),
        f"{prefix}_val_loss_coarse_seg": _format_metric(val_logs.get("val_loss_coarse_seg")),
        f"{prefix}_val_pixel_f1": _format_metric(val_logs.get("val_pixel_f1")),
        f"{prefix}_val_best_pixel_f1": _format_metric(val_logs.get("val_best_pixel_f1")),
        f"{prefix}_val_best_threshold": _format_metric(val_logs.get("val_best_threshold")),
        f"{prefix}_val_pixel_auc": _format_metric(val_logs.get("val_pixel_auc")),
        f"{prefix}_val_boundary_f1": _format_metric(val_logs.get("val_boundary_f1")),
        f"{prefix}_val_image_auc": _format_metric(val_logs.get("val_image_auc")),
        f"{prefix}_val_best_score": _format_metric(val_logs.get("val_best_score")),
        f"{prefix}_val_best_score_raw": _format_metric(val_logs.get("val_best_score_raw", val_logs.get("val_best_score"))),
        f"{prefix}_loss_guard_penalty": _format_metric(val_logs.get("val_loss_guard_penalty", 0.0)),
        f"{prefix}_logit_calibration_scale": _format_metric(val_logs.get("logit_calibration_scale")),
        f"{prefix}_logit_calibration_bias": _format_metric(val_logs.get("logit_calibration_bias")),
    }


def _finite_metric(value: object, default: float = 0.0) -> float:
    try:
        metric = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(metric):
        return default
    return metric


def _add_best_score(metrics: Dict[str, float], eval_cfg: Dict) -> None:
    score_cfg = eval_cfg.get("best_score", {})
    weights = score_cfg.get(
        "weights",
        {
            "val_pixel_f1": 0.4,
            "val_best_pixel_f1": 0.3,
            "val_boundary_f1": 0.2,
            "val_pixel_auc": 0.1,
        },
    )
    total_weight = 0.0
    score = 0.0
    for key, weight in weights.items():
        weight = float(weight)
        total_weight += abs(weight)
        score += weight * _finite_metric(metrics.get(key), default=0.0)
    if total_weight <= 0:
        score = _finite_metric(metrics.get("val_pixel_f1"), default=0.0)
    metrics["val_best_score"] = float(score)


def _apply_loss_guard(metrics: Dict[str, float], eval_cfg: Dict, best_loss_ref: float) -> float:
    guard_cfg = eval_cfg.get("best_score", {}).get("loss_guard", {})
    if not guard_cfg.get("enabled", False):
        return best_loss_ref

    loss_key = guard_cfg.get("loss_key", "val_loss_total")
    current_loss = _finite_metric(metrics.get(loss_key), default=float("inf"))
    if not np.isfinite(current_loss):
        return best_loss_ref

    best_loss_ref = min(best_loss_ref, current_loss)
    denom = max(abs(best_loss_ref), 1e-6)
    tolerance = float(guard_cfg.get("relative_tolerance", 0.15))
    penalty_weight = float(guard_cfg.get("penalty_weight", 0.5))
    relative_over = max(0.0, (current_loss - best_loss_ref) / denom - tolerance)
    penalty = penalty_weight * relative_over

    metrics["val_best_score_raw"] = _finite_metric(metrics.get("val_best_score"), default=0.0)
    metrics["val_loss_guard_ref"] = float(best_loss_ref)
    metrics["val_loss_guard_over"] = float(relative_over)
    metrics["val_loss_guard_penalty"] = float(penalty)
    metrics["val_best_score"] = float(metrics["val_best_score_raw"] - penalty)
    return best_loss_ref


def _is_improved(current: float, best: float, mode: str, min_delta: float = 0.0) -> bool:
    if mode == "min":
        return current < best - min_delta
    return current > best + min_delta


def _finite_history_values(history: list[Dict[str, float]], key: str, window: int | None = None) -> list[float]:
    rows = history[-window:] if window is not None and window > 0 else history
    values = []
    for row in rows:
        value = _finite_metric(row.get(key), default=float("nan"))
        if np.isfinite(value):
            values.append(value)
    return values


def _early_stop_reason(
    history: list[Dict[str, float]],
    monitor_cfg: Dict,
    epoch: int,
    no_improve: int,
    best_metric: float,
    mode: str,
) -> str | None:
    min_epochs = int(monitor_cfg.get("min_epochs", 0))
    if epoch < min_epochs:
        return None

    patience = int(monitor_cfg.get("patience", 10))
    if no_improve >= patience:
        return f"monitor did not improve for {patience} epochs"

    loss_patience = int(monitor_cfg.get("loss_patience", 0))
    if loss_patience > 0 and no_improve >= loss_patience:
        loss_key = str(monitor_cfg.get("loss_key", "val_loss_final_seg"))
        recent_losses = _finite_history_values(history, loss_key, loss_patience)
        all_losses = _finite_history_values(history, loss_key)
        if len(recent_losses) >= loss_patience and all_losses:
            best_loss = min(all_losses)
            tolerance = float(monitor_cfg.get("loss_relative_tolerance", 0.25))
            limit = best_loss * (1.0 + tolerance)
            if all(loss > limit for loss in recent_losses):
                recent_mean = float(np.mean(recent_losses))
                return (
                    f"{loss_key} stayed above best loss by more than "
                    f"{tolerance:.1%} for {loss_patience} epochs "
                    f"(recent_mean={recent_mean:.5f}, best={best_loss:.5f})"
                )

    score_drop_patience = int(monitor_cfg.get("score_drop_patience", 0))
    if score_drop_patience > 0 and no_improve >= score_drop_patience and np.isfinite(best_metric):
        monitor = str(monitor_cfg.get("monitor", "val_pixel_f1"))
        recent_scores = _finite_history_values(history, monitor, score_drop_patience)
        if len(recent_scores) >= score_drop_patience:
            recent_mean = float(np.mean(recent_scores))
            drop = float(monitor_cfg.get("score_drop_tolerance", 0.04))
            if mode == "min":
                dropped = recent_mean >= best_metric + drop
            else:
                dropped = recent_mean <= best_metric - drop
            if dropped:
                return (
                    f"{monitor} stayed worse than best by more than {drop:.4f} "
                    f"for {score_drop_patience} epochs "
                    f"(recent_mean={recent_mean:.5f}, best={best_metric:.5f})"
                )

    auc_patience = int(monitor_cfg.get("auc_drop_patience", 0))
    if auc_patience > 0:
        auc_key = str(monitor_cfg.get("auc_key", "val_pixel_auc"))
        recent_auc = _finite_history_values(history, auc_key, auc_patience)
        all_auc = _finite_history_values(history, auc_key)
        if len(recent_auc) >= auc_patience and all_auc:
            best_auc = max(all_auc)
            recent_mean = float(np.mean(recent_auc))
            drop = float(monitor_cfg.get("auc_drop_tolerance", 0.06))
            if recent_mean <= best_auc - drop and no_improve >= min(auc_patience, patience):
                return (
                    f"{auc_key} stayed below best by more than {drop:.4f} "
                    f"for {auc_patience} epochs "
                    f"(recent_mean={recent_mean:.5f}, best={best_auc:.5f})"
                )

    return None


def save_checkpoint(
    ckpt_dir: Path,
    filename_template: str,
    epoch: int,
    model,
    optimizer,
    scheduler,
    scaler,
    best_metric: float,
    config: Dict,
    loss_guard_best_loss: float | None = None,
) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    raw_model = model.module if hasattr(model, "module") else model
    filename = filename_template.format(epoch=epoch)
    path = ckpt_dir / filename
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "best_metric": best_metric,
            "loss_guard_best_loss": loss_guard_best_loss,
            "config": config,
        },
        path,
    )
    return path


def _resolve_resume_path(output_dir: Path, resume: str | None, latest_name: str) -> Optional[Path]:
    if not resume:
        return None
    if resume.lower() in {"latest", "auto"}:
        latest_txt = output_dir / "checkpoints" / latest_name
        p = _read_info_path(latest_txt, "latest_checkpoint_path")
        return Path(p) if p else None
    return Path(resume)


def _load_checkpoint(path: Path, model, optimizer, scheduler, scaler, map_location):
    ckpt = torch.load(path, map_location=map_location)
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(ckpt["model_state_dict"], strict=True)
    if optimizer is not None and ckpt.get("optimizer_state_dict"):
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt


def _load_pretrained_model(path: Path, model, map_location, logger=None) -> None:
    ckpt = torch.load(path, map_location=map_location)
    state = ckpt.get("model_state_dict", ckpt)
    raw_model = model.module if hasattr(model, "module") else model
    current = raw_model.state_dict()
    matched = {}
    skipped = []
    for key, value in state.items():
        if key in current and current[key].shape == value.shape:
            matched[key] = value
        else:
            skipped.append(key)
    current.update(matched)
    raw_model.load_state_dict(current, strict=True)
    message = f"Loaded pretrained model weights from {path}. matched={len(matched)} skipped={len(skipped)}"
    if logger is not None:
        logger.info(message)
        if skipped:
            logger.info(f"Skipped pretrained keys due to missing/shape mismatch: {skipped[:20]}")
    else:
        print(message)


def _apply_freeze_config(model: torch.nn.Module, train_cfg: Dict, logger=None) -> None:
    freeze_modules = list(train_cfg.get("freeze_modules", []))
    if bool(train_cfg.get("freeze_encoders", False)):
        freeze_modules.extend(["global_encoder", "residual_encoder"])
    freeze_modules = [str(item).strip() for item in freeze_modules if str(item).strip()]
    if not freeze_modules:
        return

    total_params = 0
    frozen_params = 0
    frozen_tensors = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        should_freeze = any(name == prefix or name.startswith(prefix + ".") for prefix in freeze_modules)
        if should_freeze:
            param.requires_grad_(False)
            frozen_params += param.numel()
            frozen_tensors += 1
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    if trainable_params <= 0:
        raise RuntimeError(f"freeze_modules={freeze_modules} froze all model parameters.")
    message = (
        f"Freeze config applied: freeze_modules={freeze_modules} "
        f"frozen_tensors={frozen_tensors} frozen_params={frozen_params} "
        f"trainable_params={trainable_params} total_params={total_params}"
    )
    if logger is not None:
        logger.info(message)
    else:
        print(message)


def _disable_family_if_unreliable(config: Dict, scan_output_dir: Path, logger) -> None:
    model_cfg = config.setdefault("model", {})
    if not model_cfg.get("use_family_head", False):
        return
    valid_csv = scan_output_dir / "valid_samples.csv"
    if not valid_csv.exists():
        model_cfg["use_family_head"] = False
        config.setdefault("loss", {})["family"] = 0.0
        logger.info("No reliable manipulation family labels found. Family head is disabled.")
        return
    total = 0
    known = 0
    with valid_csv.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            try:
                known += int(int(row.get("family_label", -1)) >= 0)
            except ValueError:
                pass
    if total == 0 or known / max(1, total) < 0.5:
        model_cfg["use_family_head"] = False
        config.setdefault("loss", {})["family"] = 0.0
        logger.info("No reliable manipulation family labels found. Family head is disabled.")


def train(config: Dict, resume: str | None = None, pretrained: str | None = None, debug: bool = False) -> None:
    distributed, rank, local_rank, _ = init_distributed(config)
    seed = int(config.get("seed", 42)) + rank
    set_seed(seed)

    output_dir = Path(config.get("output_dir", "outputs/merit_net_s_512"))
    if is_rank0():
        output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir, rank=rank)

    data_cfg = config.get("data", {})
    scan_output_dir = Path(data_cfg.get("scan_output_dir", "outputs"))
    train_split = scan_output_dir / "splits" / "train.txt"
    val_split = scan_output_dir / "splits" / "val.txt"
    needs_resplit, resplit_reason = should_resplit_from_config(config, train_split, val_split)
    if is_rank0() and needs_resplit:
        logger.info(f"Scanning dataset and creating train/val splits. reason={resplit_reason}")
        scan_and_split_from_config(config, log_fn=logger.info)
    if distributed:
        distributed_barrier(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    train_cfg = config.get("train", {})
    train_dataset = TamperDataset(
        train_split,
        img_size=int(data_cfg.get("img_size", 512)),
        is_train=True,
        augmentation=config.get("augmentation", {}),
        augmentation_schedule=config.get("augmentation_schedule", {}),
        seed=int(config.get("seed", 42)),
        debug=debug,
        crop_config=data_cfg,
    )
    train_metrics_mode = str(train_cfg.get("train_metrics_mode", "loss_only"))
    compute_epoch_train_metrics = bool(train_cfg.get("compute_train_metrics", True)) and train_metrics_mode in {"epoch_end", "epoch_end_sample"}
    train_eval_max_batches = int(train_cfg.get("train_eval_max_batches", 300)) if train_metrics_mode == "epoch_end_sample" else None
    train_eval_dataset = (
        TamperDataset(
            train_split,
            img_size=int(data_cfg.get("img_size", 512)),
            is_train=False,
            seed=int(config.get("seed", 42)),
            debug=debug,
            crop_config=data_cfg,
        )
        if compute_epoch_train_metrics
        else None
    )
    val_dataset = TamperDataset(
        val_split,
        img_size=int(data_cfg.get("img_size", 512)),
        is_train=False,
        seed=int(config.get("seed", 42)),
        debug=debug,
        crop_config=data_cfg,
    )
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError(f"Empty train/val dataset. train={len(train_dataset)}, val={len(val_dataset)}")
    _disable_family_if_unreliable(config, scan_output_dir, logger)

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    train_eval_sampler = DistributedEvalSampler(train_eval_dataset) if distributed and train_eval_dataset is not None else None
    val_sampler = DistributedEvalSampler(val_dataset) if distributed else None
    train_batch_size = int(train_cfg.get("batch_size_per_gpu", 4))
    train_eval_batch_size = int(train_cfg.get("train_eval_batch_size_per_gpu", train_batch_size))
    val_batch_size = int(config.get("eval", {}).get("batch_size_per_gpu", train_eval_batch_size))
    train_num_workers = int(data_cfg.get("num_workers", 8))
    eval_num_workers = int(data_cfg.get("eval_num_workers", min(train_num_workers, 8)))
    train_persistent_workers = bool(data_cfg.get("persistent_workers", False)) and train_num_workers > 0
    eval_persistent_workers = bool(data_cfg.get("eval_persistent_workers", False)) and eval_num_workers > 0
    train_prefetch_factor = int(data_cfg.get("prefetch_factor", 2)) if train_num_workers > 0 else None
    eval_prefetch_factor = int(data_cfg.get("eval_prefetch_factor", data_cfg.get("prefetch_factor", 2))) if eval_num_workers > 0 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=train_num_workers,
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        drop_last=False,
        **_worker_loader_kwargs(train_num_workers, train_persistent_workers, train_prefetch_factor),
    )
    train_eval_loader = (
        DataLoader(
            train_eval_dataset,
            batch_size=train_eval_batch_size,
            shuffle=False,
            sampler=train_eval_sampler,
            num_workers=eval_num_workers,
            pin_memory=bool(data_cfg.get("pin_memory", True)),
            drop_last=False,
            **_worker_loader_kwargs(eval_num_workers, eval_persistent_workers, eval_prefetch_factor),
        )
        if train_eval_dataset is not None
        else None
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=eval_num_workers,
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        drop_last=False,
        **_worker_loader_kwargs(eval_num_workers, eval_persistent_workers, eval_prefetch_factor),
    )

    model = MERITNet(config.get("model", {})).to(device)
    if is_rank0():
        for summary in model.encoder_summary().values():
            logger.info(summary)

    if pretrained:
        pretrained_path = Path(pretrained)
        if not pretrained_path.exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")
        _load_pretrained_model(pretrained_path, model, map_location=device, logger=logger if is_rank0() else None)

    _apply_freeze_config(model, train_cfg, logger=logger if is_rank0() else None)

    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=bool(config.get("ddp", {}).get("find_unused_parameters", False)),
        )

    criterion = MERITLoss(config)
    optimizer = build_optimizer(model, config.get("train", {}))
    if is_rank0():
        group_text = []
        for idx, group in enumerate(optimizer.param_groups):
            group_params = sum(param.numel() for param in group.get("params", []))
            group_text.append(
                f"{idx}:{group.get('name', 'default')} lr={group.get('lr', float('nan')):.8f} params={group_params}"
            )
        logger.info("Optimizer parameter groups: " + "; ".join(group_text))
    scheduler = build_scheduler(optimizer, config.get("train", {}))
    amp = bool(config.get("train", {}).get("amp", True)) and device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=amp)
        except TypeError:
            scaler = torch.amp.GradScaler(enabled=amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp)

    ckpt_cfg = config.get("checkpoint", {})
    ckpt_dir = output_dir / "checkpoints"
    best_info_name = ckpt_cfg.get("best_info_txt", "best_checkpoint.txt")
    latest_info_name = ckpt_cfg.get("latest_info_txt", "latest_checkpoint.txt")
    filename_template = ckpt_cfg.get("filename_template", "epoch{epoch}.pth")
    monitor_cfg = config.get("train", {}).get("early_stopping", {})
    monitor = monitor_cfg.get("monitor", "val_pixel_f1")
    mode = monitor_cfg.get("mode", "max")
    min_delta = float(monitor_cfg.get("min_delta", 0.0))
    best_metric = -float("inf") if mode == "max" else float("inf")
    loss_guard_best_loss = float("inf")
    start_epoch = 1

    resume_path = _resolve_resume_path(output_dir, resume, latest_info_name)
    if resume_path:
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        ckpt = _load_checkpoint(resume_path, model, optimizer, scheduler, scaler, map_location=device)
        start_epoch = int(ckpt["epoch"]) + 1
        best_metric = float(ckpt.get("best_metric", best_metric))
        loss_guard_best_loss = float(ckpt.get("loss_guard_best_loss", loss_guard_best_loss))
        logger.info(f"Resumed from {resume_path} at epoch {start_epoch}.")

    metric_logger = CSVMetricLogger(output_dir / "metrics.csv")
    epochs = int(train_cfg.get("epochs", 80))
    eval_cfg = config.get("eval", {})
    threshold = float(eval_cfg.get("threshold", 0.5))
    max_pixel_auc_samples = int(eval_cfg.get("max_pixel_auc_samples", 2_000_000))
    pixel_auc_samples_per_image = int(eval_cfg.get("pixel_auc_samples_per_image", 4096))
    pixel_auc_seed = int(eval_cfg.get("pixel_auc_seed", config.get("seed", 42)))
    report_thresholds = [float(t) for t in eval_cfg.get("report_thresholds", [0.001, 0.01, 0.05, 0.1, 0.5])]
    log_interval = int(train_cfg.get("log_interval", 20))
    train_eval_interval = max(1, int(train_cfg.get("train_eval_interval", 1)))
    train_eval_first_epoch = bool(train_cfg.get("train_eval_first_epoch", True))
    no_improve = 0
    val_history: list[Dict[str, float]] = []

    for epoch in range(start_epoch, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_dataset.set_epoch(epoch, total_epochs=epochs)
        if is_rank0():
            logger.info(f"Epoch {epoch}/{epochs} started.")
            logger.info(f"Epoch {epoch}/{epochs} augmentation: {train_dataset.augmentation_log_state()}")
        epoch_start_time = time.perf_counter()

        train_logs = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            epoch=epoch,
            total_epochs=epochs,
            accumulate_grad_batches=int(train_cfg.get("accumulate_grad_batches", 1)),
            amp=amp,
            logger=logger if is_rank0() else None,
            log_interval=log_interval,
        )
        train_metric_logs = {}
        run_train_eval = train_eval_loader is not None and (
            epoch % train_eval_interval == 0 or (train_eval_first_epoch and epoch == start_epoch)
        )
        if run_train_eval:
            train_metric_logs = validate(
                model,
                train_eval_loader,
                criterion,
                device,
                epoch=epoch,
                total_epochs=epochs,
                amp=amp,
                threshold=threshold,
                prefix="train",
                display_prefix="train_eval",
                logger=logger if is_rank0() else None,
                log_interval=log_interval,
                max_batches=train_eval_max_batches,
                max_pixel_auc_samples=max_pixel_auc_samples,
                pixel_auc_samples_per_image=pixel_auc_samples_per_image,
                pixel_auc_seed=pixel_auc_seed,
                report_thresholds=report_thresholds,
            )
        val_logs = validate(
            model,
            val_loader,
            criterion,
            device,
            epoch=epoch,
            total_epochs=epochs,
            amp=amp,
            threshold=threshold,
            prefix="val",
            logger=logger if is_rank0() else None,
            log_interval=log_interval,
            max_pixel_auc_samples=max_pixel_auc_samples,
            pixel_auc_samples_per_image=pixel_auc_samples_per_image,
            pixel_auc_seed=pixel_auc_seed,
            report_thresholds=report_thresholds,
        )
        _add_best_score(val_logs, config.get("eval", {}))
        loss_guard_best_loss = _apply_loss_guard(val_logs, config.get("eval", {}), loss_guard_best_loss)
        raw_model = model.module if hasattr(model, "module") else model
        if hasattr(raw_model, "calibration_statistics"):
            val_logs.update(raw_model.calibration_statistics())
        scheduler.step()
        val_history.append({"epoch": float(epoch), **val_logs})

        current_metric = _finite_metric(val_logs.get(monitor), default=-float("inf") if mode == "max" else float("inf"))
        improved = _is_improved(current_metric, best_metric, mode, min_delta=min_delta)
        if improved:
            best_metric = current_metric
            no_improve = 0
        else:
            no_improve += 1

        if is_rank0():
            ckpt_path = save_checkpoint(
                ckpt_dir,
                filename_template,
                epoch,
                model,
                optimizer,
                scheduler,
                scaler,
                best_metric,
                config,
                loss_guard_best_loss=loss_guard_best_loss,
            )
            _save_info(
                ckpt_dir / latest_info_name,
                {
                    "latest_epoch": epoch,
                    "latest_checkpoint_file": ckpt_path.name,
                    "latest_checkpoint_path": str(ckpt_path),
                    **_checkpoint_metric_info("latest", train_logs, val_logs, monitor),
                },
            )
            if improved:
                best_info = {
                    "best_epoch": epoch,
                    "best_metric_name": monitor,
                    "best_metric_value": f"{best_metric:.6f}",
                    "best_checkpoint_file": ckpt_path.name,
                    "best_checkpoint_path": str(ckpt_path),
                    **_checkpoint_metric_info("best", train_logs, val_logs, monitor),
                }
                if "val_best_score_raw" in val_logs:
                    best_info["best_score_raw"] = f"{val_logs.get('val_best_score_raw', float('nan')):.6f}"
                    best_info["loss_guard_best_loss"] = f"{val_logs.get('val_loss_guard_ref', float('nan')):.6f}"
                    best_info["loss_guard_penalty"] = f"{val_logs.get('val_loss_guard_penalty', float('nan')):.6f}"
                _save_info(ckpt_dir / best_info_name, best_info)
            row = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                **{f"train_optim_{k}": v for k, v in train_logs.items()},
                "train_loss_total": train_logs.get("loss_total", float("nan")),
                **train_metric_logs,
                **val_logs,
            }
            metric_logger.append(row)
            try:
                plot_training_curves(output_dir / "metrics.csv", output_dir)
                if train_metrics_mode == "loss_only":
                    for stale_curve in ("train_f1.png", "train_auc.png"):
                        stale_path = output_dir / "curves" / stale_curve
                        if stale_path.exists():
                            stale_path.unlink()
            except Exception as exc:
                logger.warning(f"Failed to plot training curves: {exc}")
            epoch_time = format_seconds(time.perf_counter() - epoch_start_time)
            train_eval_loss = train_metric_logs.get("train_loss_total", float("nan"))
            train_eval_f1 = train_metric_logs.get("train_pixel_f1", float("nan"))
            train_eval_auc = train_metric_logs.get("train_pixel_auc", float("nan"))
            train_eval_text = (
                f"train_eval_loss={train_eval_loss:.5f} "
                f"train_eval_pixel_f1={train_eval_f1:.5f} "
                f"train_eval_pixel_auc={train_eval_auc:.5f} "
                if train_metric_logs
                else ""
            )
            logger.info(
                f"Epoch {epoch}/{epochs} done | time={epoch_time} "
                f"train_loss={train_logs.get('loss_total', float('nan')):.5f} "
                f"{train_eval_text}"
                f"val_loss={val_logs.get('val_loss_total', float('nan')):.5f} "
                f"val_pixel_f1={val_logs.get('val_pixel_f1', float('nan')):.5f} "
                f"val_best_f1={val_logs.get('val_best_pixel_f1', float('nan')):.5f} "
                f"val_pixel_auc={val_logs.get('val_pixel_auc', float('nan')):.5f} "
                f"val_boundary_f1={val_logs.get('val_boundary_f1', float('nan')):.5f} "
                f"val_best_score={val_logs.get('val_best_score', float('nan')):.5f} "
                f"loss_guard_penalty={val_logs.get('val_loss_guard_penalty', 0.0):.5f} "
                f"val_image_auc={val_logs.get('val_image_auc', float('nan')):.5f} "
                f"calib_scale={val_logs.get('logit_calibration_scale', float('nan')):.4f} "
                f"calib_bias={val_logs.get('logit_calibration_bias', float('nan')):.4f} "
                f"best_{monitor}={best_metric:.5f} ckpt={ckpt_path}"
            )

        stop_reason = _early_stop_reason(
            val_history,
            monitor_cfg,
            epoch=epoch,
            no_improve=no_improve,
            best_metric=best_metric,
            mode=mode,
        )
        if distributed:
            distributed_barrier(local_rank)
        if stop_reason:
            if is_rank0():
                logger.info(f"Early stopping triggered: {stop_reason}.")
            break
