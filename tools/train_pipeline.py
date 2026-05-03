from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engines.trainer import train
from engines.trainer_ddp import cleanup_distributed, distributed_barrier, init_distributed, is_rank0


def _read_info_path(path: Path, key: str) -> str | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith(f"{key}:"):
            return line.split(":", 1)[1].strip()
    return None


def _best_checkpoint_from_config(config: dict) -> str:
    output_dir = Path(config.get("output_dir", "outputs/merit_net"))
    best_name = config.get("checkpoint", {}).get("best_info_txt", "best_checkpoint.txt")
    best_txt = output_dir / "checkpoints" / best_name
    path = _read_info_path(best_txt, "best_checkpoint_path")
    if not path:
        raise RuntimeError(f"Could not read best_checkpoint_path from {best_txt}")
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Run staged MERIT-Net training from one pipeline yaml.")
    parser.add_argument("--pipeline", required=True, help="Path to pipeline yaml.")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    distributed, _, local_rank, _ = init_distributed({})
    with open(args.pipeline, "r", encoding="utf-8") as f:
        pipeline = yaml.safe_load(f)
    stages = pipeline.get("stages", [])
    if not stages:
        raise ValueError("Pipeline yaml must contain a non-empty 'stages' list.")

    previous_best: str | None = None
    try:
        for idx, stage in enumerate(stages, start=1):
            config_path = Path(stage["config"])
            with config_path.open("r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            pretrained = stage.get("pretrained")
            if pretrained == "previous_best":
                pretrained = previous_best
            if is_rank0():
                print(f"\n========== Stage {idx}/{len(stages)}: {stage.get('name', config_path.stem)} ==========")
                print(f"config: {config_path}")
                print(f"pretrained: {pretrained or 'none'}")
            train(config, resume=stage.get("resume"), pretrained=pretrained, debug=args.debug)
            if distributed:
                distributed_barrier(local_rank)
            previous_best = _best_checkpoint_from_config(config)
            if is_rank0():
                print(f"stage best: {previous_best}")
            if distributed:
                distributed_barrier(local_rank)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
