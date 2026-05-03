from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engines.trainer import train
from engines.trainer_ddp import cleanup_distributed


def parse_args():
    parser = argparse.ArgumentParser(description="Train MERIT-Net.")
    parser.add_argument("--config", required=True, help="Path to yaml config.")
    parser.add_argument("--resume", nargs="?", const="latest", default=None, help="Checkpoint path, or 'latest'.")
    parser.add_argument("--pretrained", default=None, help="Load model weights only; optimizer/scheduler are not restored.")
    parser.add_argument("--debug", action="store_true", help="Use a small subset to smoke-test one epoch quickly.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if args.debug:
        config.setdefault("train", {})["epochs"] = min(int(config.get("train", {}).get("epochs", 1)), 1)
        config.setdefault("data", {})["num_workers"] = 0
    try:
        train(config, resume=args.resume, pretrained=args.pretrained, debug=args.debug)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
