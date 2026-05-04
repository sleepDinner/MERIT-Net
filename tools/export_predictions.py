from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.dataset_scanner import IMAGE_EXTS
from datasets.transforms import EvalTransform
from eval.eval_dataset import load_model_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Export mask and confidence predictions for an image folder.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default="outputs/predictions")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(config, args.ckpt, device)
    transform = EvalTransform(int(config.get("data", {}).get("img_size", 512)), preprocess_config=config.get("data", {}))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = [p for p in Path(args.input_dir).rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    for path in image_paths:
        with Image.open(path) as img:
            image = img.convert("RGB")
        dummy_mask = Image.new("L", image.size, 0)
        image_t, _, valid = transform(image, dummy_mask, None)
        outputs = model(image_t[None].to(device), valid_region=valid[None].to(device))
        pred = torch.sigmoid(outputs["final_mask_logits"])[0, 0].detach().cpu().numpy()
        conf = torch.sigmoid(outputs["confidence_logits"])[0, 0].detach().cpu().numpy()
        pred_img = Image.fromarray(np.clip(pred * 255, 0, 255).astype(np.uint8))
        conf_img = Image.fromarray(np.clip(conf * 255, 0, 255).astype(np.uint8))
        rel = path.relative_to(args.input_dir)
        pred_path = output_dir / rel.with_suffix(".mask.png")
        conf_path = output_dir / rel.with_suffix(".confidence.png")
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_img.save(pred_path)
        conf_img.save(conf_path)
    print(f"Saved predictions to {output_dir}")


if __name__ == "__main__":
    main()
