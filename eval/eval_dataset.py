from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.tamper_dataset import TamperDataset
from eval.metrics import MetricAccumulator
from models.merit_net import MERITNet


def load_model_from_checkpoint(config: Dict, ckpt_path: str | Path, device: torch.device) -> MERITNet:
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_config = ckpt.get("config") or {}
    model_cfg = ckpt_config.get("model", config.get("model", {}))
    model = MERITNet(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model


@torch.no_grad()
def evaluate_split(
    config: Dict,
    split_file: str | Path,
    ckpt_path: str | Path,
    output_csv: str | Path | None = None,
    batch_size: int = 1,
    num_workers: int = 4,
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(config, ckpt_path, device)
    dataset = TamperDataset(
        split_file,
        img_size=int(config.get("data", {}).get("img_size", config.get("model", {}).get("img_size", 512))),
        is_train=False,
        seed=int(config.get("seed", 42)),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    acc = MetricAccumulator(threshold=float(config.get("eval", {}).get("threshold", 0.5)))
    per_image_rows = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        valid = batch["valid_region"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        outputs = model(images, valid_region=valid)
        probs = torch.sigmoid(outputs["final_mask_logits"])
        acc.update(probs, masks, valid, outputs.get("image_logits"))
        image_scores = torch.sigmoid(outputs["image_logits"]).detach().cpu().view(-1).tolist()
        for path, score, label in zip(batch["image_path"], image_scores, batch["image_level_label"].tolist()):
            per_image_rows.append({"image_path": path, "image_score": score, "image_level_label": int(label)})
    metrics = acc.compute()
    if output_csv:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            writer.writeheader()
            writer.writerow(metrics)
        per_image_csv = output_csv.with_name(output_csv.stem + "_per_image.csv")
        with per_image_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image_path", "image_score", "image_level_label"])
            writer.writeheader()
            writer.writerows(per_image_rows)
    return metrics
