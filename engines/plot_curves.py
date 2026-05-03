from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


def _read_float(row: dict, key: str):
    value = row.get(key, "")
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _plot_one(epochs: list[int], values: list[float], title: str, ylabel: str, path: Path) -> None:
    from PIL import Image, ImageDraw

    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 900, 560
    margin_l, margin_r, margin_t, margin_b = 80, 30, 55, 70
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    x0, y0 = margin_l, margin_t + plot_h
    draw.rectangle((margin_l, margin_t, margin_l + plot_w, margin_t + plot_h), outline=(180, 180, 180))
    draw.text((margin_l, 20), title, fill=(0, 0, 0))
    draw.text((margin_l, height - 35), "epoch", fill=(0, 0, 0))
    draw.text((10, margin_t), ylabel, fill=(0, 0, 0))

    if len(values) == 1:
        vmin = min(0.0, values[0])
        vmax = max(1.0, values[0])
    else:
        vmin, vmax = min(values), max(values)
        if abs(vmax - vmin) < 1e-12:
            vmax = vmin + 1.0
    emin, emax = min(epochs), max(epochs)
    if emin == emax:
        emax = emin + 1

    for i in range(6):
        frac = i / 5
        y = margin_t + int(plot_h * frac)
        draw.line((margin_l, y, margin_l + plot_w, y), fill=(235, 235, 235))
        value = vmax - frac * (vmax - vmin)
        draw.text((8, y - 7), f"{value:.3g}", fill=(80, 80, 80))
    for i in range(6):
        frac = i / 5
        x = margin_l + int(plot_w * frac)
        draw.line((x, margin_t, x, margin_t + plot_h), fill=(245, 245, 245))
        epoch = emin + frac * (emax - emin)
        draw.text((x - 12, y0 + 8), f"{epoch:.0f}", fill=(80, 80, 80))

    points = []
    for epoch, value in zip(epochs, values):
        x = margin_l + int((epoch - emin) / (emax - emin) * plot_w)
        y = margin_t + int((vmax - value) / (vmax - vmin) * plot_h)
        points.append((x, y))
    if len(points) == 1:
        x, y = points[0]
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(30, 90, 200))
    else:
        draw.line(points, fill=(30, 90, 200), width=3)
        for x, y in points:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(30, 90, 200))
    img.save(path)


def plot_training_curves(metrics_csv: str | Path, output_dir: str | Path) -> bool:
    metrics_csv = Path(metrics_csv)
    output_dir = Path(output_dir) / "curves"
    if not metrics_csv.exists():
        return False

    rows = []
    with metrics_csv.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return False

    specs = [
        ("train_loss_total", "train_loss.png", "Train Mean Loss", "loss"),
        ("val_loss_total", "val_loss.png", "Val Mean Loss", "loss"),
        ("train_pixel_f1", "train_f1.png", "Train Pixel F1", "F1"),
        ("val_pixel_f1", "val_f1.png", "Val Pixel F1", "F1"),
        ("train_pixel_auc", "train_auc.png", "Train Pixel AUC", "AUC"),
        ("val_pixel_auc", "val_auc.png", "Val Pixel AUC", "AUC"),
    ]
    for key, filename, title, ylabel in specs:
        epochs = []
        values = []
        for row in rows:
            value = _read_float(row, key)
            epoch = _read_float(row, "epoch")
            if value is None or epoch is None:
                continue
            epochs.append(int(epoch))
            values.append(value)
        if values:
            _plot_one(epochs, values, title, ylabel, output_dir / filename)
    return True
