from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.dataset_scanner import IMAGE_EXTS, _choose_mask, _mask_lookup, _normalize_stem, is_mask_candidate
from datasets.transforms import EvalTransform
from eval.eval_dataset import load_model_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize MERIT-Net predictions.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default="outputs/visualizations")
    parser.add_argument("--save_gates", action="store_true")
    return parser.parse_args()


def _panel(title: str, image: Image.Image, size: int) -> Image.Image:
    image = image.convert("RGB").resize((size, size))
    canvas = Image.new("RGB", (size, size + 24), (255, 255, 255))
    canvas.paste(image, (0, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), title, fill=(0, 0, 0))
    return canvas


def _heat(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 1)
    rgb = np.zeros((*arr.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (arr * 255).astype(np.uint8)
    rgb[..., 1] = ((1.0 - np.abs(arr - 0.5) * 2.0) * 180).astype(np.uint8)
    rgb[..., 2] = ((1.0 - arr) * 255).astype(np.uint8)
    return Image.fromarray(rgb)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(config, args.ckpt, device)
    img_size = int(config.get("data", {}).get("img_size", 512))
    transform = EvalTransform(img_size, preprocess_config=config.get("data", {}))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    root = Path(args.input_dir)
    all_images = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    mask_paths = [p for p in all_images if is_mask_candidate(p)]
    image_paths = [p for p in all_images if not is_mask_candidate(p)]
    mask_lookup = _mask_lookup(mask_paths)

    for path in image_paths:
        with Image.open(path) as img:
            image = img.convert("RGB")
        mask_path = _choose_mask(path, mask_lookup.get(_normalize_stem(path.stem), []))
        if mask_path and mask_path.exists():
            with Image.open(mask_path) as m:
                gt = m.convert("L").point(lambda p: 255 if p > 0 else 0)
        else:
            gt = Image.new("L", image.size, 0)
        image_t, gt_t, valid = transform(image, gt, None)
        outputs = model(image_t[None].to(device), valid_region=valid[None].to(device))
        pred = torch.sigmoid(outputs["final_mask_logits"])[0, 0].detach().cpu().numpy()
        conf = torch.sigmoid(outputs["confidence_logits"])[0, 0].detach().cpu().numpy()
        pred_img = Image.fromarray(np.clip(pred * 255, 0, 255).astype(np.uint8))
        conf_img = _heat(conf)
        overlay = image.resize((img_size, img_size)).convert("RGBA")
        red = Image.new("RGBA", (img_size, img_size), (255, 0, 0, 0))
        alpha = Image.fromarray(np.clip(pred * 160, 0, 160).astype(np.uint8))
        red.putalpha(alpha)
        overlay = Image.alpha_composite(overlay, red).convert("RGB")

        panels = [
            _panel("image", image, img_size),
            _panel("gt", gt, img_size),
            _panel("pred", pred_img, img_size),
            _panel("confidence", conf_img, img_size),
            _panel("overlay", overlay, img_size),
        ]
        canvas = Image.new("RGB", (img_size * len(panels), img_size + 24), (255, 255, 255))
        for i, panel in enumerate(panels):
            canvas.paste(panel, (i * img_size, 0))
        rel = path.relative_to(root)
        out_path = (output_dir / rel).with_suffix(".viz.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path)

        if args.save_gates and "gate_weights" in outputs:
            for idx, weight in enumerate(outputs["gate_weights"], start=1):
                global_w = weight[0, 0].detach().cpu().numpy()
                _heat(global_w).resize((img_size, img_size)).save(out_path.with_suffix(f".gate_s{idx}.png"))
    print(f"Saved visualizations to {output_dir}")


if __name__ == "__main__":
    main()
