from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.dataset_scanner import scan_dataset, write_split
from eval.eval_dataset import evaluate_split


TESTSETS = {
    "Casiav1": "/data0/zh/VM-UNet/data/isic17/CAISA 1/",
    "Columbia": "/data0/zh/VM-UNet/data/isic17/Columbia/",
    "NIST16": "/data0/zh/VM-UNet/data/isic17/NIST16/",
    "IMD2020": "/data0/zh/VM-UNet/data/isic17/IMD2020/",
    "DSO-1": "/data0/zh/VM-UNet/data/isic17/DSO/",
    "Korus": "/data0/lzb-change-vmunet/FinalTest_Korus/",
}


DEFAULT_STAGE_CONFIGS = [
    Path("configs/stage1_512.yaml"),
    Path("configs/stage2_512.yaml"),
    Path("configs/stage3_768.yaml"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MERIT-Net on all configured test sets.")
    parser.add_argument("--config", default=None, help="Config for explicit single-checkpoint evaluation; requires --ckpt.")
    parser.add_argument("--ckpt", default=None, help="Checkpoint path. If omitted, best pth of staged configs is evaluated.")
    parser.add_argument("--pipeline", default=None, help="Pipeline yaml. If set without --ckpt, evaluates every stage best checkpoint.")
    parser.add_argument(
        "--stage_configs",
        nargs="+",
        default=None,
        help="Stage configs to evaluate when --ckpt is omitted. Defaults to stage1/2/3 pvt configs.",
    )
    parser.add_argument("--batch_size", type=int, default=None, help="Override eval batch size for every stage.")
    parser.add_argument("--num_workers", type=int, default=None, help="Override eval DataLoader workers.")
    parser.add_argument("--threshold", type=float, default=None, help="Override eval.threshold from config.")
    return parser.parse_args()


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_info_path(path: Path, key: str) -> str | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith(f"{key}:"):
            return line.split(":", 1)[1].strip()
    return None


def _best_checkpoint_from_config(config: dict[str, Any]) -> Path:
    output_dir = Path(config.get("output_dir", "outputs/merit_net"))
    best_name = config.get("checkpoint", {}).get("best_info_txt", "best_checkpoint.txt")
    best_txt = output_dir / "checkpoints" / best_name
    path = _read_info_path(best_txt, "best_checkpoint_path")
    if not path:
        raise RuntimeError(f"Could not read best_checkpoint_path from {best_txt}")
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Best checkpoint listed in {best_txt} does not exist: {ckpt_path}")
    return ckpt_path


def _stage_name_from_config(config_path: Path, config: dict[str, Any]) -> str:
    output_name = Path(config.get("output_dir", "")).name
    match = re.search(r"(stage\d+)", output_name)
    if match:
        parts = [match.group(1)]
        lower_name = output_name.lower()
        if "recall" in lower_name:
            parts.append("recall")
        if "calib" in lower_name or "calibration" in lower_name:
            parts.append("calib")
        backbone = str(config.get("model", {}).get("global_backbone", "")).lower()
        if "pvtv2b2" in lower_name or "pvt_v2_b2" in backbone:
            parts.append("pvtv2b2")
        elif "pvtv2b1" in lower_name or "pvt_v2_b1" in backbone:
            parts.append("pvtv2b1")
        elif "pvt" in lower_name or "pvt" in backbone:
            parts.append("pvt")
        if "lora" in lower_name or bool(config.get("model", {}).get("use_lora", False)):
            parts.append("lora")
        return "_".join(parts)
    return config_path.stem


def _stage_config_paths(args: argparse.Namespace) -> list[Path]:
    if args.pipeline:
        pipeline = _load_yaml(args.pipeline)
        stages = pipeline.get("stages", [])
        if not stages:
            raise ValueError("Pipeline yaml must contain a non-empty 'stages' list.")
        return [Path(stage["config"]) for stage in stages]
    if args.stage_configs:
        return [Path(path) for path in args.stage_configs]
    return DEFAULT_STAGE_CONFIGS


def _build_eval_targets(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.ckpt:
        if not args.config:
            raise ValueError("--config is required when --ckpt is provided.")
        config_path = Path(args.config)
        config = _load_yaml(config_path)
        if args.threshold is not None:
            config.setdefault("eval", {})["threshold"] = args.threshold
        return [
            {
                "stage": _stage_name_from_config(config_path, config),
                "config_path": config_path,
                "config": config,
                "ckpt": Path(args.ckpt),
                "single": True,
            }
        ]

    targets = []
    for config_path in _stage_config_paths(args):
        config = _load_yaml(config_path)
        if args.threshold is not None:
            config.setdefault("eval", {})["threshold"] = args.threshold
        targets.append(
            {
                "stage": _stage_name_from_config(config_path, config),
                "config_path": config_path,
                "config": config,
                "ckpt": _best_checkpoint_from_config(config),
                "single": False,
            }
        )
    return targets


def _scan_testsets(config: dict[str, Any], output_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    split_dir = output_root / "test_splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    split_rows = []
    all_skipped = []
    for name, root in TESTSETS.items():
        samples, skipped = scan_dataset(
            root,
            output_dir=output_root,
            skipped_csv_name=f"test_{name}_skipped_samples.csv",
            valid_csv_name=f"test_{name}_valid_samples.csv",
            auto_black_for_authentic=True,
            mask_threshold=float(config.get("data", {}).get("mask_threshold", 127.0)),
        )
        for item in skipped:
            all_skipped.append({"dataset": name, "image_path": item.image_path, "mask_path": item.mask_path, "reason": item.reason})
        split_file = split_dir / f"{name}.txt"
        write_split(split_file, samples)
        split_rows.append({"dataset": name, "split_file": split_file, "num_samples": len(samples)})
    return split_rows, all_skipped


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.config and not args.ckpt and not args.pipeline and not args.stage_configs:
        print("--config is ignored when --ckpt is omitted; evaluating default stage1/2/3 best checkpoints.")
    targets = _build_eval_targets(args)
    output_root = Path("outputs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    staged = len(targets) > 1 or not targets[0].get("single", False)
    result_root = output_root / "test_results"
    run_result_dir = result_root / f"staged_{timestamp}" if staged else result_root
    run_result_dir.mkdir(parents=True, exist_ok=True)

    split_rows, all_skipped = _scan_testsets(targets[0]["config"], output_root)

    skipped_csv = output_root / "test_skipped_samples.csv"
    with skipped_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "image_path", "mask_path", "reason"])
        writer.writeheader()
        writer.writerows(all_skipped)

    all_summary_rows = []
    for target_idx, target in enumerate(targets, start=1):
        stage = target["stage"]
        config = target["config"]
        ckpt = target["ckpt"]
        stage_result_dir = run_result_dir / stage if staged else run_result_dir
        stage_result_dir.mkdir(parents=True, exist_ok=True)
        batch_size = args.batch_size or int(config.get("eval", {}).get("batch_size_per_gpu", 64))
        num_workers = args.num_workers
        if num_workers is None:
            num_workers = int(config.get("data", {}).get("eval_num_workers", config.get("data", {}).get("num_workers", 4)))
        print(f"\n========== Eval {target_idx}/{len(targets)}: {stage} ==========")
        print(f"config: {target['config_path']}")
        print(f"checkpoint: {ckpt}")
        print(f"batch_size: {batch_size}, num_workers: {num_workers}")
        stage_rows = []
        for split_info in split_rows:
            name = split_info["dataset"]
            split_file = split_info["split_file"]
            num_samples = split_info["num_samples"]
            if not num_samples:
                row = {
                    "stage": stage,
                    "stage_config": str(target["config_path"]),
                    "checkpoint": str(ckpt),
                    "dataset": name,
                    "num_samples": 0,
                }
                stage_rows.append(row)
                all_summary_rows.append(row)
                print(f"{stage}/{name}: no valid samples")
                continue
            metrics = evaluate_split(
                config,
                split_file,
                ckpt,
                output_csv=stage_result_dir / f"{name}_metrics.csv",
                batch_size=batch_size,
                num_workers=num_workers,
                progress_name=f"{stage}-{name}",
            )
            row = {
                "stage": stage,
                "stage_config": str(target["config_path"]),
                "checkpoint": str(ckpt),
                "dataset": name,
                "num_samples": num_samples,
                **metrics,
            }
            stage_rows.append(row)
            all_summary_rows.append(row)
            print(f"{stage}/{name}: {metrics}")
        stage_summary_csv = output_root / f"test_results_summary_{stage}_{timestamp}.csv"
        _write_summary(stage_summary_csv, stage_rows)
        _write_summary(output_root / f"test_results_summary_{stage}_latest.csv", stage_rows)
        print(f"{stage} summary: {stage_summary_csv}")

    if staged:
        summary_csv = output_root / f"test_results_summary_all_stages_{timestamp}.csv"
        latest_csv = output_root / "test_results_summary_all_stages_latest.csv"
    else:
        summary_csv = output_root / f"test_results_summary_{timestamp}.csv"
        latest_csv = output_root / "test_results_summary_latest.csv"
    _write_summary(summary_csv, all_summary_rows)
    _write_summary(latest_csv, all_summary_rows)
    if staged:
        _write_summary(output_root / "test_results_summary_latest.csv", all_summary_rows)
    print(f"summary: {summary_csv}")
    print(f"summary_latest: {latest_csv}")
    if staged:
        print(f"summary_latest_compat: {output_root / 'test_results_summary_latest.csv'}")
    print(f"per-dataset results: {run_result_dir}")


if __name__ == "__main__":
    main()
