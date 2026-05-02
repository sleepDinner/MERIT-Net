from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset

from datasets.transforms import EvalTransform, TrainTransform


@dataclass
class SplitSample:
    image_path: str
    mask_path: str
    label: int
    family_label: int = -1


def read_split_file(path: str | Path) -> List[SplitSample]:
    path = Path(path)
    samples: List[SplitSample] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            family = int(parts[3]) if len(parts) > 3 and parts[3].lstrip("-").isdigit() else -1
            samples.append(SplitSample(parts[0], parts[1], int(parts[2]), family))
    family_csv_candidates = [path.parent.parent / "valid_samples.csv", path.parent / "valid_samples.csv"]
    family_map = {}
    for csv_path in family_csv_candidates:
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    family_map[row.get("image_path", "")] = int(row.get("family_label", -1))
                except ValueError:
                    continue
        break
    if family_map:
        for sample in samples:
            if sample.family_label < 0:
                sample.family_label = family_map.get(sample.image_path, -1)
    return samples


class TamperDataset(Dataset):
    def __init__(
        self,
        split_file: str | Path,
        img_size: int = 512,
        is_train: bool = True,
        augmentation: Dict | None = None,
        augmentation_schedule: Dict | None = None,
        seed: int = 42,
        debug: bool = False,
        debug_limit: int = 16,
    ):
        self.split_file = Path(split_file)
        self.samples = read_split_file(split_file)
        if debug:
            self.samples = self.samples[:debug_limit]
        self.img_size = int(img_size)
        self.is_train = is_train
        self.seed = int(seed)
        self.epoch = 0
        if is_train:
            self.transform = TrainTransform(img_size, augmentation or {}, augmentation_schedule or {}, epoch=0)
        else:
            self.transform = EvalTransform(img_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        if hasattr(self.transform, "set_epoch"):
            self.transform.set_epoch(epoch)

    def augmentation_log_state(self) -> Dict[str, float]:
        if hasattr(self.transform, "log_state"):
            return self.transform.log_state()
        return {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        rng = random.Random(self.seed + self.epoch * 1000003 + index)
        with Image.open(sample.image_path) as img:
            image = img.convert("RGB")
        with Image.open(sample.mask_path) as m:
            mask = m.convert("L").point(lambda p: 255 if p > 0 else 0)

        image_t, mask_t, valid_t = self.transform(image, mask, rng)
        image_level_label = int(mask_t.sum().item() > 0)
        return {
            "image": image_t,
            "mask": mask_t,
            "valid_region": valid_t,
            "image_level_label": torch.tensor(image_level_label, dtype=torch.float32),
            "family_label": torch.tensor(sample.family_label, dtype=torch.long),
            "image_path": sample.image_path,
            "mask_path": sample.mask_path,
        }
