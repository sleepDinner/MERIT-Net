from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.dataset_scanner import scan_and_split_from_config
from datasets.tamper_dataset import read_split_file


def parse_args():
    parser = argparse.ArgumentParser(description="Scan training data and build train/val splits.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    train_file, val_file = scan_and_split_from_config(config)
    train_samples = read_split_file(train_file)
    val_samples = read_split_file(val_file)
    print(f"train_split: {train_file} ({len(train_samples)} samples)")
    print(f"val_split: {val_file} ({len(val_samples)} samples)")


if __name__ == "__main__":
    main()
