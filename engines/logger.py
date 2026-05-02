from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Dict


def setup_logger(output_dir: str | Path, rank: int = 0) -> logging.Logger:
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("MERIT-Net")
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    logger.handlers.clear()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if rank == 0:
        file_handler = logging.FileHandler(log_dir / "train.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


class CSVMetricLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: Dict[str, float | int | str]) -> None:
        exists = self.path.exists()
        fieldnames = list(row.keys())
        if exists:
            with self.path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                try:
                    fieldnames = next(reader)
                except StopIteration:
                    exists = False
        with self.path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in fieldnames})
