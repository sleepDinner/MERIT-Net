from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.dataset_scanner import scan_dataset, write_split
from eval.eval_dataset import evaluate_split


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate one split or one dataset root.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--input_root", default=None)
    parser.add_argument("--output_csv", default="outputs/test_metrics.csv")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    split = args.split
    if split is None:
        if args.input_root is None:
            raise ValueError("Provide either --split or --input_root.")
        samples, _ = scan_dataset(args.input_root, output_dir="outputs", skipped_csv_name="test_skipped_samples.csv", valid_csv_name="test_valid_samples.csv")
        split = Path("outputs/test_splits/single_test.txt")
        write_split(split, samples)
    metrics = evaluate_split(
        config,
        split,
        args.ckpt,
        output_csv=args.output_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(metrics)


if __name__ == "__main__":
    main()
