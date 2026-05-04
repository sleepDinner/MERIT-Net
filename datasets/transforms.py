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


def binarize_mask_array(mask: np.ndarray, threshold: float = 127.0) -> np.ndarray:
    """Convert common 0/255 or 0/1 masks into a clean binary float array."""

    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    arr = arr.astype(np.float32, copy=False)
    if arr.size == 0:
        return arr.astype(np.float32)
    if float(np.nanmax(arr)) <= 1.5:
        return (arr > 0.5).astype(np.float32)
    return (arr > float(threshold)).astype(np.float32)


@dataclass
class AugmentationState:
    phase: str
    degradation_prob: float
    max_degradations: int
    jpeg_prob: float
    jpeg_quality: Tuple[int, int]
    blur_prob: float
    blur_radius: Tuple[float, float]
    noise_prob: float
    noise_sigma: Tuple[float, float]
    downscale_prob: float
    downscale_scale: Tuple[float, float]
    color_jitter_prob: float
    color_jitter_strength: float
    scale_prob: float
    scale_range: Tuple[float, float]
    copy_move_prob: float
    inpainting_prob: float


def _range_tuple(value, default: Tuple[float, float], cast=float) -> Tuple:
    if value is None:
        value = default
    if isinstance(value, (int, float)):
        value = [value, value]
    if len(value) == 1:
        value = [value[0], value[0]]
    return cast(value[0]), cast(value[1])


def _state_from_cfg(cfg: Dict, base_aug: Dict, phase: str) -> AugmentationState:
    return AugmentationState(
        phase=phase,
        degradation_prob=float(cfg.get("degradation_prob", cfg.get("degradation_p", 0.0))),
        max_degradations=max(1, int(cfg.get("max_degradations", 1))),
        jpeg_prob=float(cfg.get("jpeg_prob", base_aug.get("jpeg_prob", 0.0))),
        jpeg_quality=_range_tuple(cfg.get("jpeg_quality", base_aug.get("jpeg_quality", [40, 95])), (40, 95), int),
        blur_prob=float(cfg.get("blur_prob", base_aug.get("blur_prob", 0.0))),
        blur_radius=_range_tuple(cfg.get("blur_radius", base_aug.get("blur_radius", [0.2, 1.2])), (0.2, 1.2), float),
        noise_prob=float(cfg.get("noise_prob", base_aug.get("noise_prob", 0.0))),
        noise_sigma=_range_tuple(cfg.get("noise_sigma", base_aug.get("noise_sigma", [0, 10])), (0.0, 10.0), float),
        downscale_prob=float(cfg.get("downscale_prob", base_aug.get("downscale_prob", 0.0))),
        downscale_scale=_range_tuple(cfg.get("downscale_scale", base_aug.get("downscale_scale", [0.75, 0.95])), (0.75, 0.95), float),
        color_jitter_prob=float(cfg.get("color_jitter_prob", base_aug.get("color_jitter_prob", 0.0))),
        color_jitter_strength=float(cfg.get("color_jitter_strength", base_aug.get("color_jitter_strength", 0.15))),
        scale_prob=float(cfg.get("scale_prob", base_aug.get("scale_prob", 0.0))),
        scale_range=_range_tuple(cfg.get("scale_range", base_aug.get("scale_range", [0.9, 1.1])), (0.9, 1.1), float),
        copy_move_prob=float(cfg.get("copy_move_prob", base_aug.get("copy_move_prob", 0.0))),
        inpainting_prob=float(cfg.get("inpainting_prob", base_aug.get("inpainting_prob", 0.0))),
    )


def _lerp(a: float, b: float, t: float) -> float:
    return float(a) + max(0.0, min(1.0, float(t))) * (float(b) - float(a))


def _interpolate_state(a: AugmentationState, b: AugmentationState, t: float, phase: str) -> AugmentationState:
    return AugmentationState(
        phase=phase,
        degradation_prob=_lerp(a.degradation_prob, b.degradation_prob, t),
        max_degradations=b.max_degradations if t >= 0.5 else a.max_degradations,
        jpeg_prob=_lerp(a.jpeg_prob, b.jpeg_prob, t),
        jpeg_quality=(int(round(_lerp(a.jpeg_quality[0], b.jpeg_quality[0], t))), int(round(_lerp(a.jpeg_quality[1], b.jpeg_quality[1], t)))),
        blur_prob=_lerp(a.blur_prob, b.blur_prob, t),
        blur_radius=(_lerp(a.blur_radius[0], b.blur_radius[0], t), _lerp(a.blur_radius[1], b.blur_radius[1], t)),
        noise_prob=_lerp(a.noise_prob, b.noise_prob, t),
        noise_sigma=(_lerp(a.noise_sigma[0], b.noise_sigma[0], t), _lerp(a.noise_sigma[1], b.noise_sigma[1], t)),
        downscale_prob=_lerp(a.downscale_prob, b.downscale_prob, t),
        downscale_scale=(_lerp(a.downscale_scale[0], b.downscale_scale[0], t), _lerp(a.downscale_scale[1], b.downscale_scale[1], t)),
        color_jitter_prob=_lerp(a.color_jitter_prob, b.color_jitter_prob, t),
        color_jitter_strength=_lerp(a.color_jitter_strength, b.color_jitter_strength, t),
        scale_prob=_lerp(a.scale_prob, b.scale_prob, t),
        scale_range=(_lerp(a.scale_range[0], b.scale_range[0], t), _lerp(a.scale_range[1], b.scale_range[1], t)),
        copy_move_prob=_lerp(a.copy_move_prob, b.copy_move_prob, t),
        inpainting_prob=_lerp(a.inpainting_prob, b.inpainting_prob, t),
    )


def _phased_augmentation_state(base_aug: Dict, schedule: Dict, epoch: int, total_epochs: int | None = None) -> AugmentationState:
    warmup_cfg = schedule.get("warmup", {})
    middle_cfg = schedule.get("middle", {})
    robust_cfg = schedule.get("robust", {})
    warmup_state = _state_from_cfg(warmup_cfg, base_aug, "warmup")
    middle_state = _state_from_cfg(middle_cfg, base_aug, "middle")
    robust_state = _state_from_cfg(robust_cfg, base_aug, "robust")

    total_epochs = int(total_epochs or schedule.get("total_epochs", 0) or 0)
    if total_epochs > 0:
        warmup_end = max(1, int(round(total_epochs * float(schedule.get("warmup_ratio", 0.15)))))
        robust_start = max(warmup_end + 1, int(round(total_epochs * float(schedule.get("robust_start_ratio", 0.50)))))
    else:
        warmup_end = max(1, int(schedule.get("warmup_epochs", 10)))
        robust_start = max(warmup_end + 1, int(schedule.get("strong_aug_start_epoch", schedule.get("max_strength_epoch", 40))))

    if epoch <= warmup_end:
        return warmup_state
    if epoch < robust_start:
        progress = (float(epoch) - warmup_end) / max(1.0, float(robust_start - warmup_end))
        return _interpolate_state(warmup_state, middle_state, progress, "middle")
    progress = (float(epoch) - robust_start) / max(1.0, float(max(total_epochs, robust_start + 1) - robust_start))
    return _interpolate_state(middle_state, robust_state, progress, "robust")


def current_augmentation_state(base_aug: Dict, schedule: Dict | None, epoch: int, total_epochs: int | None = None) -> AugmentationState:
    if not schedule or not schedule.get("enabled", False):
        return _state_from_cfg(
            {
                "degradation_prob": float(base_aug.get("degradation_prob", 1.0)),
                "max_degradations": int(base_aug.get("max_degradations", 5)),
            },
            base_aug,
            "fixed",
        )
    if str(schedule.get("mode", "progressive")).lower() in {"phased", "phase", "progressive_phased"}:
        return _phased_augmentation_state(base_aug, schedule, epoch, total_epochs=total_epochs)

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
    return _state_from_cfg(
        {
            "phase": "progressive",
            "degradation_prob": 1.0,
            "max_degradations": 5,
            "jpeg_prob": lerp("jpeg_prob", 0.1, 0.4),
            "jpeg_quality": (quality_min, int(base_aug.get("jpeg_quality", [40, 95])[1])),
            "noise_prob": lerp("noise_prob", 0.05, 0.25),
            "noise_sigma": (0.0, sigma_max),
            "copy_move_prob": lerp("copy_move_prob", 0.05, 0.25),
            "inpainting_prob": lerp("inpainting_prob", 0.05, 0.25),
            "blur_prob": float(base_aug.get("blur_prob", 0.0)),
            "color_jitter_prob": float(base_aug.get("color_jitter_prob", 0.0)),
            "downscale_prob": float(base_aug.get("downscale_prob", 0.0)),
        },
        base_aug,
        "progressive",
    )


def augmentation_state_to_log(state: AugmentationState) -> Dict[str, float | str]:
    return {
        "current_aug_phase": state.phase,
        "current_degradation_prob": state.degradation_prob,
        "current_max_degradations": float(state.max_degradations),
        "current_jpeg_prob": state.jpeg_prob,
        "current_jpeg_quality_min": float(state.jpeg_quality[0]),
        "current_jpeg_quality_max": float(state.jpeg_quality[1]),
        "current_blur_prob": state.blur_prob,
        "current_blur_radius_min": float(state.blur_radius[0]),
        "current_blur_radius_max": float(state.blur_radius[1]),
        "current_noise_prob": state.noise_prob,
        "current_noise_sigma_min": float(state.noise_sigma[0]),
        "current_noise_sigma_max": float(state.noise_sigma[1]),
        "current_downscale_prob": state.downscale_prob,
        "current_downscale_scale_min": float(state.downscale_scale[0]),
        "current_downscale_scale_max": float(state.downscale_scale[1]),
        "current_color_jitter_prob": state.color_jitter_prob,
        "current_color_jitter_strength": state.color_jitter_strength,
        "current_scale_prob": state.scale_prob,
        "current_scale_min": float(state.scale_range[0]),
        "current_scale_max": float(state.scale_range[1]),
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


def _downscale_upscale(image: Image.Image, rng: random.Random, scale_range: Tuple[float, float]) -> Image.Image:
    width, height = image.size
    if width < 16 or height < 16:
        return image
    lo, hi = sorted((float(scale_range[0]), float(scale_range[1])))
    scale = max(0.1, min(1.0, rng.uniform(lo, hi)))
    down_w = max(8, int(round(width * scale)))
    down_h = max(8, int(round(height * scale)))
    if (down_w, down_h) == (width, height):
        return image
    resample_down = rng.choice([Image.BILINEAR, Image.BICUBIC])
    small = image.resize((down_w, down_h), resample_down)
    return small.resize((width, height), Image.BILINEAR)


def _color_jitter(image: Image.Image, rng: random.Random, strength: float = 0.15) -> Image.Image:
    brightness = 1.0 + rng.uniform(-strength, strength)
    contrast = 1.0 + rng.uniform(-strength, strength)
    color = 1.0 + rng.uniform(-strength, strength)
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    image = ImageEnhance.Color(image).enhance(color)
    return image


def _random_scale_pair(image: Image.Image, mask: Image.Image, rng: random.Random, scale_range: Tuple[float, float]) -> Tuple[Image.Image, Image.Image]:
    width, height = image.size
    lo, hi = sorted((float(scale_range[0]), float(scale_range[1])))
    scale = max(0.2, rng.uniform(lo, hi))
    new_w = max(8, int(round(width * scale)))
    new_h = max(8, int(round(height * scale)))
    if (new_w, new_h) == (width, height):
        return image, mask
    return image.resize((new_w, new_h), Image.BILINEAR), mask.resize((new_w, new_h), Image.NEAREST)


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
    mask_threshold: float = 127.0,
) -> tuple[Image.Image, Image.Image]:
    width, height = image.size
    crop_w = min(int(crop_size), width)
    crop_h = min(int(crop_size), height)
    if crop_w <= 0 or crop_h <= 0 or (crop_w == width and crop_h == height):
        return image, mask

    mask_arr = binarize_mask_array(np.asarray(mask.convert("L")), threshold=mask_threshold)
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


def _weighted_sample_without_replacement(items: list[tuple[str, float]], k: int, rng: random.Random) -> list[str]:
    pool = [(name, max(0.0, float(weight))) for name, weight in items if float(weight) > 0]
    selected: list[str] = []
    for _ in range(min(k, len(pool))):
        total = sum(weight for _, weight in pool)
        if total <= 0:
            break
        pick = rng.random() * total
        acc = 0.0
        chosen_idx = 0
        for idx, (_, weight) in enumerate(pool):
            acc += weight
            if pick <= acc:
                chosen_idx = idx
                break
        selected.append(pool[chosen_idx][0])
        pool.pop(chosen_idx)
    return selected


def _apply_degradation_group(image: Image.Image, state: AugmentationState, rng: random.Random) -> Image.Image:
    if state.degradation_prob <= 0 or rng.random() >= state.degradation_prob:
        return image
    candidates = [
        ("jpeg", state.jpeg_prob),
        ("blur", state.blur_prob),
        ("noise", state.noise_prob),
        ("downscale", state.downscale_prob),
        ("color", state.color_jitter_prob),
    ]
    k = rng.randint(1, max(1, int(state.max_degradations)))
    selected = _weighted_sample_without_replacement(candidates, k, rng)
    for name in selected:
        if name == "jpeg":
            q_min, q_max = sorted((int(state.jpeg_quality[0]), int(state.jpeg_quality[1])))
            image = _jpeg_compress(image, rng.randint(q_min, q_max))
        elif name == "blur":
            lo, hi = sorted((float(state.blur_radius[0]), float(state.blur_radius[1])))
            image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(max(0.01, lo), max(0.01, hi))))
        elif name == "noise":
            sigma = rng.uniform(float(state.noise_sigma[0]), float(state.noise_sigma[1]))
            image = _add_gaussian_noise(image, sigma, rng)
        elif name == "downscale":
            image = _downscale_upscale(image, rng, state.downscale_scale)
        elif name == "color":
            image = _color_jitter(image, rng, state.color_jitter_strength)
    return image


class TrainTransform:
    def __init__(
        self,
        img_size: int,
        augmentation: Dict,
        schedule: Dict | None = None,
        epoch: int = 0,
        total_epochs: int | None = None,
        crop_config: Dict | None = None,
    ):
        self.img_size = int(img_size)
        self.augmentation = augmentation or {}
        self.schedule = schedule or {}
        self.crop_config = crop_config or {}
        self.mask_threshold = float(self.crop_config.get("mask_threshold", 127.0))
        self.pad_position = str(self.crop_config.get("pad_position", "top_left"))
        self.preprocess_mode = str(self.crop_config.get("preprocess_mode", "pad"))
        self.epoch = epoch
        self.total_epochs = int(total_epochs or self.schedule.get("total_epochs", 0) or 0)
        self.state = current_augmentation_state(self.augmentation, self.schedule, epoch, total_epochs=self.total_epochs)

    def set_epoch(self, epoch: int, total_epochs: int | None = None) -> None:
        self.epoch = epoch
        if total_epochs is not None:
            self.total_epochs = int(total_epochs)
        self.state = current_augmentation_state(self.augmentation, self.schedule, epoch, total_epochs=self.total_epochs)

    def log_state(self) -> Dict[str, float | str]:
        state = augmentation_state_to_log(self.state)
        crop_mode = str(self.crop_config.get("train_crop_mode", "none"))
        state["train_crop_mode"] = crop_mode
        state["preprocess_mode"] = self.preprocess_mode
        state["pad_position"] = self.pad_position
        state["mask_threshold"] = self.mask_threshold
        if crop_mode in {"mixed", "mixed_crop", "random_crop"}:
            state["current_crop_prob"] = float(self.crop_config.get("crop_prob", 1.0 if crop_mode == "random_crop" else 0.5))
        return state

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
        mask = mask.convert("L")
        if rng.random() < self.state.scale_prob:
            image, mask = _random_scale_pair(image, mask, rng, self.state.scale_range)
        crop_mode = str(self.crop_config.get("train_crop_mode", "none"))
        crop_prob = float(self.crop_config.get("crop_prob", 1.0 if crop_mode == "random_crop" else 0.5))
        use_crop = crop_mode == "random_crop" or (crop_mode in {"mixed", "mixed_crop"} and rng.random() < crop_prob)
        if use_crop:
            image, mask = _focused_random_crop(
                image,
                mask,
                int(self.crop_config.get("crop_size", self.img_size)),
                rng,
                float(self.crop_config.get("tamper_crop_prob", 0.7)),
                self.mask_threshold,
            )
        image, mask = self._geometric(image, mask, rng)
        if rng.random() < self.state.copy_move_prob:
            image, mask = _copy_move(image, mask, rng)
        if rng.random() < self.state.inpainting_prob:
            image, mask = _inpaint_or_remove(image, mask, rng)
        image = _apply_degradation_group(image, self.state, rng)
        return resize_pad_normalize(
            image,
            mask,
            self.img_size,
            pad_position=self.pad_position,
            mask_threshold=self.mask_threshold,
            preprocess_mode=self.preprocess_mode,
        )


class EvalTransform:
    def __init__(self, img_size: int, preprocess_config: Dict | None = None):
        self.img_size = int(img_size)
        self.preprocess_config = preprocess_config or {}
        self.mask_threshold = float(self.preprocess_config.get("mask_threshold", 127.0))
        self.pad_position = str(self.preprocess_config.get("pad_position", "top_left"))
        self.preprocess_mode = str(self.preprocess_config.get("preprocess_mode", "pad"))

    def set_epoch(self, epoch: int) -> None:
        return None

    def log_state(self) -> Dict[str, float | str]:
        return {
            "preprocess_mode": self.preprocess_mode,
            "pad_position": self.pad_position,
            "mask_threshold": self.mask_threshold,
        }

    def __call__(self, image: Image.Image, mask: Image.Image, rng: random.Random | None = None):
        return resize_pad_normalize(
            image.convert("RGB"),
            mask.convert("L"),
            self.img_size,
            pad_position=self.pad_position,
            mask_threshold=self.mask_threshold,
            preprocess_mode=self.preprocess_mode,
        )


def resize_pad_normalize(
    image: Image.Image,
    mask: Image.Image,
    img_size: int,
    pad_position: str = "top_left",
    mask_threshold: float = 127.0,
    preprocess_mode: str = "pad",
):
    width, height = image.size
    preprocess_mode = str(preprocess_mode).lower()
    if preprocess_mode not in {"pad", "resize"}:
        raise ValueError(f"Unsupported preprocess_mode={preprocess_mode!r}; expected 'pad' or 'resize'.")

    if preprocess_mode == "resize":
        image = image.resize((img_size, img_size), Image.BILINEAR)
        mask = mask.resize((img_size, img_size), Image.NEAREST)
        new_w = img_size
        new_h = img_size
        offset_x = 0
        offset_y = 0
    else:
        scale = min(float(img_size) / max(width, 1), float(img_size) / max(height, 1), 1.0)
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        if (new_w, new_h) != (width, height):
            image = image.resize((new_w, new_h), Image.BILINEAR)
            mask = mask.resize((new_w, new_h), Image.NEAREST)
        pad_position = str(pad_position).lower()
        if pad_position in {"center", "centre"}:
            offset_x = (img_size - new_w) // 2
            offset_y = (img_size - new_h) // 2
        elif pad_position in {"top_left", "topleft", "left_top"}:
            offset_x = 0
            offset_y = 0
        else:
            raise ValueError(f"Unsupported pad_position={pad_position!r}; expected 'top_left' or 'center'.")

    padded_img = Image.new("RGB", (img_size, img_size), (0, 0, 0))
    padded_mask = Image.new("L", (img_size, img_size), 0)
    padded_img.paste(image, (offset_x, offset_y))
    padded_mask.paste(mask, (offset_x, offset_y))

    valid = Image.new("L", (img_size, img_size), 0)
    draw = ImageDraw.Draw(valid)
    draw.rectangle((offset_x, offset_y, offset_x + new_w - 1, offset_y + new_h - 1), fill=255)

    img_arr = np.asarray(padded_img).astype(np.float32) / 255.0
    valid_arr = (np.asarray(valid) > 0).astype(np.float32)
    mask_arr = binarize_mask_array(np.asarray(padded_mask), threshold=mask_threshold)

    image_t = torch.from_numpy(img_arr).permute(2, 0, 1)
    image_t = (image_t - IMAGENET_MEAN) / IMAGENET_STD
    mask_t = torch.from_numpy(mask_arr)[None, :, :]
    valid_t = torch.from_numpy(valid_arr)[None, :, :]
    mask_t = (mask_t > 0.5).float()
    return image_t.float(), mask_t.float(), valid_t.float()
