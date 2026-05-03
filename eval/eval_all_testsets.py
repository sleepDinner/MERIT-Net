from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.dataset_scanner import SkippedRecord, scan_dataset, write_split
from eval.eval_dataset import evaluate_split


TESTSETS = {
    "Casiav1": "/data0/zh/VM-UNet/data/isic17/CAISA 1/",
    "Columbia": "/data0/zh/VM-UNet/data/isic17/Columbia/",
    "NIST16": "/data0/zh/VM-UNet/data/isic17/NIST16/",
    "IMD2020": "/data0/zh/VM-UNet/data/isic17/IMD2020/",
    "DSO-1": "/data0/zh/VM-UNet/data/isic17/DSO/",
    "Korus": "/data0/lzb-change-vmunet/FinalTest_Korus/",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MERIT-Net on all configured test sets.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=None, help="Override eval.threshold from config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if args.threshold is not None:
        config.setdefault("eval", {})["threshold"] = args.threshold
    output_root = Path("outputs")
    split_dir = output_root / "test_splits"
    result_dir = output_root / "test_results"
    split_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    all_skipped = []
    for name, root in TESTSETS.items():
        samples, skipped = scan_dataset(
            root,
            output_dir=output_root,
            skipped_csv_name=f"test_{name}_skipped_samples.csv",
            valid_csv_name=f"test_{name}_valid_samples.csv",
            auto_black_for_authentic=True,
        )
        for item in skipped:
            all_skipped.append({"dataset": name, "image_path": item.image_path, "mask_path": item.mask_path, "reason": item.reason})
        split_file = split_dir / f"{name}.txt"
        write_split(split_file, samples)
        if not samples:
            row = {"dataset": name, "num_samples": 0}
            summary_rows.append(row)
            print(f"{name}: no valid samples")
            continue
        metrics = evaluate_split(
            config,
            split_file,
            args.ckpt,
            output_csv=result_dir / f"{name}_metrics.csv",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        row = {"dataset": name, "num_samples": len(samples), **metrics}
        summary_rows.append(row)
        print(f"{name}: {metrics}")

    skipped_csv = output_root / "test_skipped_samples.csv"
    with skipped_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "image_path", "mask_path", "reason"])
        writer.writeheader()
        writer.writerows(all_skipped)

    summary_csv = output_root / "test_results_summary.csv"
    fieldnames = sorted({key for row in summary_rows for key in row.keys()})
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"summary: {summary_csv}")


if __name__ == "__main__":
    main()
