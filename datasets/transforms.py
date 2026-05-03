from __future__ import annotations

import io
import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@dataclass
class AugmentationState:
    jpeg_prob: float
    jpeg_quality: Tuple[int, int]
    noise_prob: float
    noise_sigma: Tuple[float, float]
    copy_move_prob: float
    inpainting_prob: float


def current_augmentation_state(base_aug: Dict, schedule: Dict | None, epoch: int) -> AugmentationState:
    if not schedule or not schedule.get("enabled", False):
        return AugmentationState(
            jpeg_prob=float(base_aug.get("jpeg_prob", 0.0)),
            jpeg_quality=tuple(base_aug.get("jpeg_quality", [40, 95])),
            noise_prob=float(base_aug.get("noise_prob", 0.0)),
            noise_sigma=tuple(base_aug.get("noise_sigma", [0, 10])),
            copy_move_prob=float(base_aug.get("copy_move_prob", 0.0)),
            inpainting_prob=float(base_aug.get("inpainting_prob", 0.0)),
        )
    warmup = float(schedule.get("warmup_epochs", 10))
    max_epoch = float(schedule.get("max_strength_epoch", 50))
    denom = max(1.0, max_epoch - warmup)
    progress = min(1.0, max(0.0, (float(epoch) - warmup) / denom))
    start = schedule.get("start", {})
    end = schedule.get("end", {})

    def lerp(key: str, default_start: float, default_end: float) -> float:
        sv = float(start.get(key, default_start))
        ev = float(end.get(key, default_end))
        return sv + progress * (ev - sv)

    quality_min = int(round(lerp("jpeg_quality_min", 80, 40)))
    sigma_max = lerp("noise_sigma_max", 3, 10)
    return AugmentationState(
        jpeg_prob=lerp("jpeg_prob", 0.1, 0.4),
        jpeg_quality=(quality_min, int(base_aug.get("jpeg_quality", [40, 95])[1])),
        noise_prob=lerp("noise_prob", 0.05, 0.25),
        noise_sigma=(0.0, sigma_max),
        copy_move_prob=lerp("copy_move_prob", 0.05, 0.25),
        inpainting_prob=lerp("inpainting_prob", 0.05, 0.25),
    )


def augmentation_state_to_log(state: AugmentationState) -> Dict[str, float]:
    return {
        "current_jpeg_prob": state.jpeg_prob,
        "current_jpeg_quality_min": float(state.jpeg_quality[0]),
        "current_jpeg_quality_max": float(state.jpeg_quality[1]),
        "current_noise_prob": state.noise_prob,
        "current_noise_sigma_min": float(state.noise_sigma[0]),
        "current_noise_sigma_max": float(state.noise_sigma[1]),
        "current_copy_move_prob": state.copy_move_prob,
        "current_inpainting_prob": state.inpainting_prob,
    }


def _jpeg_compress(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _add_gaussian_noise(image: Image.Image, sigma: float, rng: random.Random) -> Image.Image:
    arr = np.asarray(image).astype(np.float32)
    np_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))
    noise = np_rng.normal(0.0, sigma, size=arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _color_jitter(image: Image.Image, rng: random.Random, strength: float = 0.15) -> Image.Image:
    brightness = 1.0 + rng.uniform(-strength, strength)
    contrast = 1.0 + rng.uniform(-strength, strength)
    color = 1.0 + rng.uniform(-strength, strength)
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    image = ImageEnhance.Color(image).enhance(color)
    return image


def _random_box(width: int, height: int, rng: random.Random, min_ratio: float = 0.08, max_ratio: float = 0.35):
    bw = max(4, int(width * rng.uniform(min_ratio, max_ratio)))
    bh = max(4, int(height * rng.uniform(min_ratio, max_ratio)))
    bw = min(bw, width)
    bh = min(bh, height)
    x = rng.randint(0, max(0, width - bw))
    y = rng.randint(0, max(0, height - bh))
    return x, y, bw, bh


def _copy_move(image: Image.Image, mask: Image.Image, rng: random.Random) -> Tuple[Image.Image, Image.Image]:
    width, height = image.size
    if width < 16 or height < 16:
        return image, mask
    sx, sy, bw, bh = _random_box(width, height, rng, 0.08, 0.28)
    dx, dy, _, _ = _random_box(width, height, rng, 0.08, 0.28)
    dx = min(dx, width - bw)
    dy = min(dy, height - bh)
    patch = image.crop((sx, sy, sx + bw, sy + bh))
    image = image.copy()
    image.paste(patch, (dx, dy))
    mask_arr = np.asarray(mask.convert("L")).copy()
    mask_arr[dy : dy + bh, dx : dx + bw] = 255
    return image, Image.fromarray(mask_arr, mode="L")


def _irregular_mask(width: int, height: int, rng: random.Random) -> Image.Image:
    region = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(region)
    x, y, bw, bh = _random_box(width, height, rng, 0.08, 0.30)
    if rng.random() < 0.5:
        draw.rectangle((x, y, x + bw, y + bh), fill=255)
    else:
        points = []
        cx, cy = x + bw / 2, y + bh / 2
        for i in range(rng.randint(7, 12)):
            angle = 2 * math.pi * i / 10.0 + rng.uniform(-0.4, 0.4)
            radius = rng.uniform(0.35, 0.65)
            points.append((cx + math.cos(angle) * bw * radius, cy + math.sin(angle) * bh * radius))
        draw.polygon(points, fill=255)
    return region.filter(ImageFilter.GaussianBlur(radius=0.5)).point(lambda p: 255 if p > 16 else 0)


def _inpaint_or_remove(image: Image.Image, mask: Image.Image, rng: random.Random) -> Tuple[Image.Image, Image.Image]:
    width, height = image.size
    if width < 16 or height < 16:
        return image, mask
    region = _irregular_mask(width, height, rng)
    region_arr = np.asarray(region) > 0
    img_arr = np.asarray(image).copy()
    mode = rng.choice(["mean", "blur", "zero"])
    if mode == "mean":
        mean_color = img_arr[~region_arr].mean(axis=0) if np.any(~region_arr) else np.array([0, 0, 0])
        img_arr[region_arr] = mean_color.astype(np.uint8)
    elif mode == "blur":
        blurred = np.asarray(image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(3, 9))))
        img_arr[region_arr] = blurred[region_arr]
    else:
        img_arr[region_arr] = rng.randint(0, 32)
    mask_arr = np.asarray(mask.convert("L")).copy()
    mask_arr[region_arr] = 255
    return Image.fromarray(img_arr, mode="RGB"), Image.fromarray(mask_arr, mode="L")


def _focused_random_crop(
    image: Image.Image,
    mask: Image.Image,
    crop_size: int,
    rng: random.Random,
    tamper_crop_prob: float = 0.7,
) -> tuple[Image.Image, Image.Image]:
    width, height = image.size
    crop_w = min(int(crop_size), width)
    crop_h = min(int(crop_size), height)
    if crop_w <= 0 or crop_h <= 0 or (crop_w == width and crop_h == height):
        return image, mask

    mask_arr = np.asarray(mask.convert("L"))
    ys, xs = np.where(mask_arr > 0)
    if len(xs) > 0 and rng.random() < tamper_crop_prob:
        idx = rng.randrange(len(xs))
        cx = int(xs[idx] + rng.randint(-crop_w // 4, crop_w // 4))
        cy = int(ys[idx] + rng.randint(-crop_h // 4, crop_h // 4))
        left = max(0, min(width - crop_w, cx - crop_w // 2))
        top = max(0, min(height - crop_h, cy - crop_h // 2))
    else:
        left = rng.randint(0, max(0, width - crop_w))
        top = rng.randint(0, max(0, height - crop_h))
    box = (left, top, left + crop_w, top + crop_h)
    return image.crop(box), mask.crop(box)


class TrainTransform:
    def __init__(
        self,
        img_size: int,
        augmentation: Dict,
        schedule: Dict | None = None,
        epoch: int = 0,
        crop_config: Dict | None = None,
    ):
        self.img_size = int(img_size)
        self.augmentation = augmentation or {}
        self.schedule = schedule or {}
        self.crop_config = crop_config or {}
        self.epoch = epoch
        self.state = current_augmentation_state(self.augmentation, self.schedule, epoch)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.state = current_augmentation_state(self.augmentation, self.schedule, epoch)

    def log_state(self) -> Dict[str, float]:
        return augmentation_state_to_log(self.state)

    def _geometric(self, image: Image.Image, mask: Image.Image, rng: random.Random):
        aug = self.augmentation
        if rng.random() < float(aug.get("hflip_prob", 0.0)):
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if rng.random() < float(aug.get("vflip_prob", 0.0)):
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        if rng.random() < float(aug.get("rotate_prob", 0.0)):
            k = rng.choice([0, 1, 2, 3])
            if k:
                image = image.rotate(90 * k, expand=True)
                mask = mask.rotate(90 * k, expand=True)
        return image, mask

    def __call__(self, image: Image.Image, mask: Image.Image, rng: random.Random):
        image = image.convert("RGB")
        mask = mask.convert("L").point(lambda p: 255 if p > 0 else 0)
        if self.crop_config.get("train_crop_mode", "none") == "random_crop":
            image, mask = _focused_random_crop(
                image,
                mask,
                int(self.crop_config.get("crop_size", self.img_size)),
                rng,
                float(self.crop_config.get("tamper_crop_prob", 0.7)),
            )
        image, mask = self._geometric(image, mask, rng)
        if rng.random() < self.state.copy_move_prob:
            image, mask = _copy_move(image, mask, rng)
        if rng.random() < self.state.inpainting_prob:
            image, mask = _inpaint_or_remove(image, mask, rng)
        if rng.random() < float(self.augmentation.get("color_jitter_prob", 0.0)):
            image = _color_jitter(image, rng, float(self.augmentation.get("color_jitter_strength", 0.15)))
        if rng.random() < float(self.augmentation.get("blur_prob", 0.0)):
            image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, float(self.augmentation.get("blur_radius", 1.2)))))
        if rng.random() < self.state.jpeg_prob:
            q = rng.randint(int(self.state.jpeg_quality[0]), int(self.state.jpeg_quality[1]))
            image = _jpeg_compress(image, q)
        if rng.random() < self.state.noise_prob:
            sigma = rng.uniform(float(self.state.noise_sigma[0]), float(self.state.noise_sigma[1]))
            image = _add_gaussian_noise(image, sigma, rng)
        return resize_pad_normalize(image, mask, self.img_size)


class EvalTransform:
    def __init__(self, img_size: int):
        self.img_size = int(img_size)

    def set_epoch(self, epoch: int) -> None:
        return None

    def log_state(self) -> Dict[str, float]:
        return {}

    def __call__(self, image: Image.Image, mask: Image.Image, rng: random.Random | None = None):
        return resize_pad_normalize(image.convert("RGB"), mask.convert("L"), self.img_size)


def resize_pad_normalize(image: Image.Image, mask: Image.Image, img_size: int):
    width, height = image.size
    scale = min(float(img_size) / max(width, 1), float(img_size) / max(height, 1), 1.0)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    if (new_w, new_h) != (width, height):
        image = image.resize((new_w, new_h), Image.BILINEAR)
        mask = mask.resize((new_w, new_h), Image.NEAREST)

    padded_img = Image.new("RGB", (img_size, img_size), (0, 0, 0))
    padded_mask = Image.new("L", (img_size, img_size), 0)
    offset_x = (img_size - new_w) // 2
    offset_y = (img_size - new_h) // 2
    padded_img.paste(image, (offset_x, offset_y))
    padded_mask.paste(mask, (offset_x, offset_y))

    valid = Image.new("L", (img_size, img_size), 0)
    draw = ImageDraw.Draw(valid)
    draw.rectangle((offset_x, offset_y, offset_x + new_w - 1, offset_y + new_h - 1), fill=255)

    img_arr = np.asarray(padded_img).astype(np.float32) / 255.0
    mask_arr = (np.asarray(padded_mask) > 0).astype(np.float32)
    valid_arr = (np.asarray(valid) > 0).astype(np.float32)

    image_t = torch.from_numpy(img_arr).permute(2, 0, 1)
    image_t = (image_t - IMAGENET_MEAN) / IMAGENET_STD
    mask_t = torch.from_numpy(mask_arr)[None, :, :]
    valid_t = torch.from_numpy(valid_arr)[None, :, :]
    mask_t = (mask_t > 0.5).float()
    return image_t.float(), mask_t.float(), valid_t.float()
