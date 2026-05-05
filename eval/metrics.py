from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def _edge_np(mask: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(mask.astype(np.float32))[None, None]
    dil = F.max_pool2d(t, 5, stride=1, padding=2)
    ero = 1.0 - F.max_pool2d(1.0 - t, 5, stride=1, padding=2)
    return ((dil - ero).squeeze().numpy() > 0.5).astype(np.uint8)


def _safe_valid_np(valid: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    t = torch.from_numpy(valid.astype(np.float32))[None, None]
    unsafe = F.max_pool2d(1.0 - t, kernel_size, stride=1, padding=kernel_size // 2)
    safe = (unsafe.squeeze().numpy() <= 0.0)
    return safe if safe.any() else valid.astype(bool)


class MetricAccumulator:
    def __init__(
        self,
        threshold: float = 0.5,
        max_pixel_auc_samples: int = 2_000_000,
        pixel_auc_samples_per_image: int = 4096,
        pixel_auc_seed: int = 12345,
    ):
        self.threshold = float(threshold)
        self.max_pixel_auc_samples = int(max_pixel_auc_samples)
        self.pixel_auc_samples_per_image = int(pixel_auc_samples_per_image)
        self.pixel_auc_seed = int(pixel_auc_seed)
        self._pixel_auc_rng = np.random.default_rng(self.pixel_auc_seed)
        self._pixel_auc_count = 0
        self._pixel_auc_seen = 0
        self._pixel_scores_buf = None
        self._pixel_labels_buf = None
        self.sweep_thresholds = [
            0.0001,
            0.0005,
            0.001,
            0.002,
            0.005,
            0.01,
            0.02,
            0.05,
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.85,
            0.90,
            0.95,
        ]
        self.sweep_stats = {str(t): {"tp": 0, "fp": 0, "fn": 0} for t in self.sweep_thresholds}
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.boundary_tp = 0
        self.boundary_fp = 0
        self.boundary_fn = 0
        self.small_tp = 0
        self.small_fp = 0
        self.small_fn = 0
        self.image_scores: List[float] = []
        self.image_labels: List[int] = []
        self.authentic_fp = 0
        self.authentic_total = 0
        self.pixel_scores: List[np.ndarray] = []
        self.pixel_labels: List[np.ndarray] = []

    def _ensure_pixel_auc_buffers(self) -> None:
        if self.max_pixel_auc_samples <= 0 or self._pixel_scores_buf is not None:
            return
        self._pixel_scores_buf = np.empty(self.max_pixel_auc_samples, dtype=np.float32)
        self._pixel_labels_buf = np.empty(self.max_pixel_auc_samples, dtype=np.uint8)

    def _add_pixel_auc_samples(self, prob: np.ndarray, gt: np.ndarray) -> None:
        if self.max_pixel_auc_samples <= 0 or prob.size == 0:
            return
        sample_limit = max(1, self.pixel_auc_samples_per_image)
        sample_count = min(prob.size, sample_limit)
        if prob.size > sample_count:
            idx = self._pixel_auc_rng.choice(prob.size, size=sample_count, replace=False)
            scores = prob[idx].astype(np.float32, copy=False)
            labels = gt[idx].astype(np.uint8, copy=False)
        else:
            scores = prob.astype(np.float32, copy=False)
            labels = gt.astype(np.uint8, copy=False)
        self._add_pixel_auc_candidates(scores, labels)

    def _add_pixel_auc_candidates(self, scores: np.ndarray, labels: np.ndarray) -> None:
        if self.max_pixel_auc_samples <= 0 or scores.size == 0:
            return
        self._ensure_pixel_auc_buffers()
        if self._pixel_scores_buf is None or self._pixel_labels_buf is None:
            return

        n = int(scores.size)
        if self._pixel_auc_count < self.max_pixel_auc_samples:
            take = min(n, self.max_pixel_auc_samples - self._pixel_auc_count)
            start = self._pixel_auc_count
            end = start + take
            self._pixel_scores_buf[start:end] = scores[:take]
            self._pixel_labels_buf[start:end] = labels[:take]
            self._pixel_auc_count = end
            self._pixel_auc_seen += take
            scores = scores[take:]
            labels = labels[take:]
            n -= take
        if n <= 0:
            return

        stream_high = np.arange(self._pixel_auc_seen + 1, self._pixel_auc_seen + n + 1, dtype=np.int64)
        replace_pos = self._pixel_auc_rng.integers(0, stream_high)
        keep = replace_pos < self.max_pixel_auc_samples
        if keep.any():
            pos = replace_pos[keep]
            self._pixel_scores_buf[pos] = scores[keep]
            self._pixel_labels_buf[pos] = labels[keep]
        self._pixel_auc_seen += n

    def update(
        self,
        mask_prob: torch.Tensor,
        gt_mask: torch.Tensor,
        valid_region: torch.Tensor,
        image_logits: torch.Tensor | None = None,
    ) -> None:
        mask_prob = mask_prob.detach().float().cpu()
        gt_mask = gt_mask.detach().float().cpu()
        valid_region = valid_region.detach().float().cpu()
        if image_logits is not None:
            image_score_tensor = torch.sigmoid(image_logits.detach().float().cpu()).view(-1)
        else:
            image_score_tensor = None

        batch = mask_prob.shape[0]
        for b in range(batch):
            valid = valid_region[b, 0] > 0.5
            if valid.sum().item() == 0:
                continue
            prob = mask_prob[b, 0][valid].numpy()
            gt = (gt_mask[b, 0][valid].numpy() > 0.5).astype(np.uint8)
            pred = (prob >= self.threshold).astype(np.uint8)

            self.tp += int(((pred == 1) & (gt == 1)).sum())
            self.fp += int(((pred == 1) & (gt == 0)).sum())
            self.fn += int(((pred == 0) & (gt == 1)).sum())
            self.tn += int(((pred == 0) & (gt == 0)).sum())
            positives = gt == 1
            negatives = ~positives
            for threshold in self.sweep_thresholds:
                sweep_pred = prob >= threshold
                stat = self.sweep_stats[str(threshold)]
                stat["tp"] += int((sweep_pred & positives).sum())
                stat["fp"] += int((sweep_pred & negatives).sum())
                stat["fn"] += int((~sweep_pred & positives).sum())

            self._add_pixel_auc_samples(prob, gt)

            full_prob = mask_prob[b, 0].numpy()
            full_gt = (gt_mask[b, 0].numpy() > 0.5).astype(np.uint8)
            full_valid = (valid_region[b, 0].numpy() > 0.5)
            edge_valid = _safe_valid_np(full_valid)
            edge_gt = _edge_np(full_gt) & edge_valid
            edge_pred = _edge_np((full_prob >= self.threshold).astype(np.uint8)) & edge_valid
            self.boundary_tp += int((edge_pred & edge_gt).sum())
            self.boundary_fp += int((edge_pred & ~edge_gt & edge_valid).sum())
            self.boundary_fn += int((~edge_pred & edge_gt).sum())

            gt_area_ratio = float(gt.mean())
            if 0.0 < gt_area_ratio <= 0.01:
                self.small_tp += int(((pred == 1) & (gt == 1)).sum())
                self.small_fp += int(((pred == 1) & (gt == 0)).sum())
                self.small_fn += int(((pred == 0) & (gt == 1)).sum())

            image_label = int(gt.any())
            if image_score_tensor is not None:
                image_score = float(image_score_tensor[b].item())
            else:
                image_score = float(prob.max())
            self.image_scores.append(image_score)
            self.image_labels.append(image_label)
            if image_label == 0:
                self.authentic_total += 1
                self.authentic_fp += int(image_score >= self.threshold)

    def merge(self, other: "MetricAccumulator") -> None:
        for attr in (
            "tp",
            "fp",
            "fn",
            "tn",
            "boundary_tp",
            "boundary_fp",
            "boundary_fn",
            "small_tp",
            "small_fp",
            "small_fn",
            "authentic_fp",
            "authentic_total",
        ):
            setattr(self, attr, getattr(self, attr) + getattr(other, attr))
        self.image_scores.extend(other.image_scores)
        self.image_labels.extend(other.image_labels)
        if other._pixel_auc_count > 0 and other._pixel_scores_buf is not None and other._pixel_labels_buf is not None:
            self._add_pixel_auc_candidates(
                other._pixel_scores_buf[: other._pixel_auc_count],
                other._pixel_labels_buf[: other._pixel_auc_count],
            )
        for key, stat in other.sweep_stats.items():
            if key not in self.sweep_stats:
                self.sweep_stats[key] = {"tp": 0, "fp": 0, "fn": 0}
            self.sweep_stats[key]["tp"] += int(stat["tp"])
            self.sweep_stats[key]["fp"] += int(stat["fp"])
            self.sweep_stats[key]["fn"] += int(stat["fn"])

    def state_dict(self) -> Dict:
        state = self.__dict__.copy()
        state["_pixel_auc_rng_state"] = self._pixel_auc_rng.bit_generator.state
        state.pop("_pixel_auc_rng", None)
        return state

    @classmethod
    def from_state_dict(cls, state: Dict) -> "MetricAccumulator":
        obj = cls(
            threshold=state.get("threshold", 0.5),
            max_pixel_auc_samples=state.get("max_pixel_auc_samples", 2_000_000),
            pixel_auc_samples_per_image=state.get("pixel_auc_samples_per_image", 4096),
            pixel_auc_seed=state.get("pixel_auc_seed", 12345),
        )
        rng_state = state.pop("_pixel_auc_rng_state", None)
        obj.__dict__.update(state)
        obj._pixel_auc_rng = np.random.default_rng(obj.pixel_auc_seed)
        if rng_state is not None:
            obj._pixel_auc_rng.bit_generator.state = rng_state
        return obj

    @staticmethod
    def _f1(tp: int, fp: int, fn: int) -> float:
        denom = 2 * tp + fp + fn
        return float(2 * tp / denom) if denom > 0 else 0.0

    @staticmethod
    def _precision(tp: int, fp: int) -> float:
        return float(tp / max(1, tp + fp))

    @staticmethod
    def _recall(tp: int, fn: int) -> float:
        return float(tp / max(1, tp + fn))

    def _best_threshold_metrics(self) -> Dict[str, float]:
        best = {"f1": -1.0, "threshold": 0.5, "precision": 0.0, "recall": 0.0}
        for key, stat in self.sweep_stats.items():
            threshold = float(key)
            tp = int(stat["tp"])
            fp = int(stat["fp"])
            fn = int(stat["fn"])
            f1 = self._f1(tp, fp, fn)
            if f1 > best["f1"]:
                best = {
                    "f1": f1,
                    "threshold": float(threshold),
                    "precision": self._precision(tp, fp),
                    "recall": self._recall(tp, fn),
                }
        return {
            "best_pixel_f1": best["f1"],
            "best_threshold": best["threshold"],
            "best_pixel_precision": best["precision"],
            "best_pixel_recall": best["recall"],
        }

    def compute(self, prefix: str = "") -> Dict[str, float]:
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        pixel_f1 = self._f1(tp, fp, fn)
        pixel_precision = self._precision(tp, fp)
        pixel_recall = self._recall(tp, fn)
        iou = float(tp / max(1, tp + fp + fn))
        tpr = float(tp / max(1, tp + fn))
        tnr = float(tn / max(1, tn + fp))
        fpr = float(fp / max(1, fp + tn))
        balanced_accuracy = 0.5 * (tpr + tnr)
        authentic_fpr = float(self.authentic_fp / max(1, self.authentic_total))
        small_f1 = self._f1(self.small_tp, self.small_fp, self.small_fn)
        boundary_f1 = self._f1(self.boundary_tp, self.boundary_fp, self.boundary_fn)

        if self._pixel_auc_count > 0 and self._pixel_scores_buf is not None and self._pixel_labels_buf is not None:
            pixel_scores = self._pixel_scores_buf[: self._pixel_auc_count]
            pixel_labels = self._pixel_labels_buf[: self._pixel_auc_count]
        elif self.pixel_scores:
            pixel_scores = np.concatenate(self.pixel_scores)
            pixel_labels = np.concatenate(self.pixel_labels)
        else:
            pixel_scores = None
            pixel_labels = None
        if pixel_scores is not None and pixel_labels is not None:
            pixel_auc = _safe_auc(pixel_labels, pixel_scores)
        else:
            pixel_auc = float("nan")
        image_auc = _safe_auc(np.asarray(self.image_labels, dtype=np.uint8), np.asarray(self.image_scores, dtype=np.float32))

        metrics = {
            "pixel_f1": pixel_f1,
            "pixel_precision": pixel_precision,
            "pixel_recall": pixel_recall,
            "iou": iou,
            "pixel_auc": pixel_auc,
            "image_auc": image_auc,
            "balanced_accuracy": balanced_accuracy,
            "fpr": fpr,
            "authentic_image_fpr": authentic_fpr,
            "small_region_f1": small_f1,
            "boundary_f1": boundary_f1,
        }
        metrics.update(self._best_threshold_metrics())
        if prefix:
            return {f"{prefix}_{k}": v for k, v in metrics.items()}
        return metrics
