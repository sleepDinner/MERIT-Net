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

from datasets.dataset_scanner import scan_and_split_from_config
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


def _read_info_path(path: Path, key: str) -> Optional[str]:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{key}:"):
            return line.split(":", 1)[1].strip()
    return None


def _save_info(path: Path, lines: Dict[str, str | int | float]) -> None:
    path.write_text("\n".join(f"{k}: {v}" for k, v in lines.items()) + "\n", encoding="utf-8")


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
    if is_rank0() and (not train_split.exists() or not val_split.exists()):
        logger.info("Split files not found; scanning dataset and creating train/val splits.")
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
        persistent_workers=train_persistent_workers,
        prefetch_factor=train_prefetch_factor,
        drop_last=False,
    )
    train_eval_loader = (
        DataLoader(
            train_eval_dataset,
            batch_size=train_eval_batch_size,
            shuffle=False,
            sampler=train_eval_sampler,
            num_workers=eval_num_workers,
            pin_memory=bool(data_cfg.get("pin_memory", True)),
            persistent_workers=eval_persistent_workers,
            prefetch_factor=eval_prefetch_factor,
            drop_last=False,
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
        persistent_workers=eval_persistent_workers,
        prefetch_factor=eval_prefetch_factor,
        drop_last=False,
    )

    model = MERITNet(config.get("model", {})).to(device)
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=bool(config.get("ddp", {}).get("find_unused_parameters", False)),
        )

    if pretrained:
        pretrained_path = Path(pretrained)
        if not pretrained_path.exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")
        _load_pretrained_model(pretrained_path, model, map_location=device, logger=logger if is_rank0() else None)

    criterion = MERITLoss(config)
    optimizer = build_optimizer(model, config.get("train", {}))
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
    patience = int(monitor_cfg.get("patience", 10))
    best_metric = -float("inf") if mode == "max" else float("inf")
    start_epoch = 1

    resume_path = _resolve_resume_path(output_dir, resume, latest_info_name)
    if resume_path:
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        ckpt = _load_checkpoint(resume_path, model, optimizer, scheduler, scaler, map_location=device)
        start_epoch = int(ckpt["epoch"]) + 1
        best_metric = float(ckpt.get("best_metric", best_metric))
        logger.info(f"Resumed from {resume_path} at epoch {start_epoch}.")

    metric_logger = CSVMetricLogger(output_dir / "metrics.csv")
    epochs = int(train_cfg.get("epochs", 80))
    threshold = float(config.get("eval", {}).get("threshold", 0.5))
    log_interval = int(train_cfg.get("log_interval", 20))
    train_eval_interval = max(1, int(train_cfg.get("train_eval_interval", 1)))
    train_eval_first_epoch = bool(train_cfg.get("train_eval_first_epoch", True))
    no_improve = 0

    for epoch in range(start_epoch, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_dataset.set_epoch(epoch)
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
        )
        _add_best_score(val_logs, config.get("eval", {}))
        scheduler.step()

        current_metric = _finite_metric(val_logs.get(monitor), default=-float("inf") if mode == "max" else float("inf"))
        improved = (current_metric > best_metric) if mode == "max" else (current_metric < best_metric)
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
            )
            _save_info(
                ckpt_dir / latest_info_name,
                {
                    "latest_epoch": epoch,
                    "latest_checkpoint_file": ckpt_path.name,
                    "latest_checkpoint_path": str(ckpt_path),
                },
            )
            if improved:
                _save_info(
                    ckpt_dir / best_info_name,
                    {
                        "best_epoch": epoch,
                        "best_metric_name": monitor,
                        "best_metric_value": f"{best_metric:.6f}",
                        "best_checkpoint_file": ckpt_path.name,
                        "best_checkpoint_path": str(ckpt_path),
                    },
                )
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
                f"val_image_auc={val_logs.get('val_image_auc', float('nan')):.5f} "
                f"best_{monitor}={best_metric:.5f} ckpt={ckpt_path}"
            )

        if distributed:
            distributed_barrier(local_rank)
        if no_improve >= patience:
            if is_rank0():
                logger.info(f"Early stopping triggered after {patience} epochs without improvement.")
            break
