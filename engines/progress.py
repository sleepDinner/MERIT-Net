from __future__ import annotations

import sys
import time


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def progress_bar(current: int, total: int, width: int = 24) -> str:
    total = max(1, int(total))
    current = min(max(0, int(current)), total)
    filled = int(round(width * current / total))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def progress_message(
    phase: str,
    epoch: int,
    total_epochs: int,
    step: int,
    total_steps: int,
    start_time: float,
    metrics: dict[str, float] | None = None,
) -> str:
    elapsed = time.perf_counter() - start_time
    pct = 100.0 * step / max(1, total_steps)
    eta = elapsed / max(1, step) * max(0, total_steps - step)
    metric_text = ""
    if metrics:
        parts = []
        for key, value in metrics.items():
            if value is None:
                continue
            parts.append(f"{key}={value:.5f}")
        if parts:
            metric_text = " | " + " ".join(parts)
    return (
        f"Epoch {epoch}/{total_epochs} {phase} "
        f"{progress_bar(step, total_steps)} {step}/{total_steps} {pct:5.1f}% "
        f"elapsed={format_seconds(elapsed)} eta={format_seconds(eta)}{metric_text}"
    )


class ProgressLine:
    """In-place stdout progress line for tail-friendly long training runs."""

    def __init__(self):
        self._last_len = 0

    def update(self, message: str) -> None:
        padding = " " * max(0, self._last_len - len(message))
        sys.stdout.write("\r" + message + padding)
        sys.stdout.flush()
        self._last_len = len(message)

    def finish(self, message: str | None = None) -> None:
        if message is not None:
            self.update(message)
        sys.stdout.write("\n")
        sys.stdout.flush()
        self._last_len = 0
