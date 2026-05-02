from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_KEYWORDS = {
    "mask",
    "masks",
    "gt",
    "gts",
    "groundtruth",
    "ground_truth",
    "label",
    "labels",
    "annotation",
    "annotations",
    "seg",
    "binary",
}
AUTHENTIC_KEYWORDS = {"real", "authentic", "negative", "pristine", "original", "untampered", "clean"}
FAMILY_KEYWORDS = {
    "real": 0,
    "authentic": 0,
    "negative": 0,
    "splicing": 1,
    "splice": 1,
    "copy-move": 2,
    "copymove": 2,
    "copy_move": 2,
    "inpainting": 3,
    "removal": 3,
    "remove": 3,
    "generative-edit": 4,
    "generative_edit": 4,
    "genedit": 4,
    "ai-edit": 4,
}
FAMILY_FIELDS = {"family", "type", "manipulation", "category"}


@dataclass
class SampleRecord:
    image_path: str
    mask_path: str
    label: int
    family_label: int = -1
    auto_mask: int = 0


@dataclass
class SkippedRecord:
    image_path: str
    mask_path: str
    reason: str


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _path_tokens(path: Path) -> List[str]:
    text = " ".join(path.parts).lower()
    return [tok for tok in re.split(r"[^a-z0-9]+", text) if tok]


def is_mask_candidate(path: Path) -> bool:
    tokens = set(_path_tokens(path))
    stem = path.stem.lower()
    return bool(tokens & MASK_KEYWORDS) or any(stem.endswith(f"_{kw}") or stem.endswith(f"-{kw}") for kw in MASK_KEYWORDS)


def is_authentic_path(path: Path) -> bool:
    tokens = set(_path_tokens(path))
    return bool(tokens & AUTHENTIC_KEYWORDS)


def _normalize_stem(stem: str) -> str:
    s = stem.lower()
    for kw in sorted(MASK_KEYWORDS, key=len, reverse=True):
        s = re.sub(rf"(^|[_\-. ]){re.escape(kw)}($|[_\-. ])", " ", s)
        s = re.sub(rf"[_\-. ]?{re.escape(kw)}$", "", s)
        s = re.sub(rf"^{re.escape(kw)}[_\-. ]?", "", s)
    return re.sub(r"[^a-z0-9]+", "", s)


def _family_from_value(value: str) -> int:
    value = value.strip().lower()
    for key, idx in FAMILY_KEYWORDS.items():
        if key in value:
            return idx
    return -1


def family_from_path(path: Path) -> int:
    text = " ".join(path.parts).lower()
    for key, idx in FAMILY_KEYWORDS.items():
        pattern = re.escape(key).replace("\\-", "[-_ ]")
        if re.search(rf"(^|[^a-z0-9]){pattern}([^a-z0-9]|$)", text):
            return idx
    return -1


def load_family_metadata(root: str | Path) -> Dict[str, int]:
    root = Path(root)
    mapping: Dict[str, int] = {}
    names = {"metadata.csv", "labels.csv", "annotations.csv", "train.txt", "annotations.json"}
    for file in root.rglob("*"):
        if not file.is_file() or file.name.lower() not in names:
            continue
        try:
            if file.suffix.lower() == ".json":
                data = json.loads(file.read_text(encoding="utf-8"))
                rows = data if isinstance(data, list) else data.get("annotations", data.get("samples", []))
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    fam = -1
                    for key, value in row.items():
                        if key.lower() in FAMILY_FIELDS and isinstance(value, str):
                            fam = _family_from_value(value)
                            break
                    if fam < 0:
                        continue
                    for key in ("image", "image_path", "path", "file", "filename"):
                        if key in row:
                            p = str(row[key])
                            mapping[p] = fam
                            mapping[Path(p).name] = fam
                            break
            else:
                with file.open("r", encoding="utf-8", errors="ignore", newline="") as f:
                    sample = f.read(4096)
                    f.seek(0)
                    dialect = csv.Sniffer().sniff(sample, delimiters=",\t ") if sample.strip() else csv.excel
                    reader = csv.DictReader(f, dialect=dialect)
                    if not reader.fieldnames:
                        continue
                    lower_fields = {name.lower(): name for name in reader.fieldnames}
                    family_field = next((lower_fields[k] for k in FAMILY_FIELDS if k in lower_fields), None)
                    path_field = next(
                        (lower_fields[k] for k in ("image", "image_path", "path", "file", "filename") if k in lower_fields),
                        None,
                    )
                    if family_field is None or path_field is None:
                        continue
                    for row in reader:
                        fam = _family_from_value(row.get(family_field, ""))
                        if fam < 0:
                            continue
                        p = row.get(path_field, "")
                        mapping[p] = fam
                        mapping[Path(p).name] = fam
        except Exception:
            continue
    return mapping


def _open_image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as img:
        img.load()
        return img.size


def _read_binary_mask(path: Path) -> np.ndarray:
    with Image.open(path) as mask:
        arr = np.array(mask.convert("L"))
    return (arr > 0).astype(np.uint8)


def _write_csv(path: Path, rows: Iterable[dict], fieldnames: List[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mask_lookup(mask_paths: List[Path]) -> Dict[str, List[Path]]:
    lookup: Dict[str, List[Path]] = {}
    for mask in mask_paths:
        key = _normalize_stem(mask.stem)
        if key:
            lookup.setdefault(key, []).append(mask)
    return lookup


def _choose_mask(image_path: Path, candidates: List[Path]) -> Optional[Path]:
    if not candidates:
        return None

    image_parts = list(reversed(image_path.parent.parts))

    def score(mask: Path) -> Tuple[int, int, int]:
        mask_parts = list(reversed(mask.parent.parts))
        common = 0
        for a, b in zip(image_parts, mask_parts):
            if a == b:
                common += 1
            else:
                break
        same_stem = int(mask.stem.lower() == image_path.stem.lower())
        return (same_stem, common, -len(str(mask)))

    return sorted(candidates, key=score, reverse=True)[0]


def _auto_mask_path(output_dir: Path, image_path: Path) -> Path:
    digest = hashlib.sha1(str(image_path).encode("utf-8")).hexdigest()[:16]
    return output_dir / "auto_masks" / f"{image_path.stem}_{digest}.png"


def scan_dataset(
    root: str | Path,
    output_dir: str | Path = "outputs",
    skipped_csv_name: str = "skipped_samples.csv",
    valid_csv_name: str = "valid_samples.csv",
    auto_black_for_authentic: bool = True,
    log_fn: Optional[Callable[[str], None]] = None,
    progress_interval: int = 500,
) -> Tuple[List[SampleRecord], List[SkippedRecord]]:
    root = Path(root)
    output_dir = ensure_dir(output_dir)
    skipped: List[SkippedRecord] = []
    valid: List[SampleRecord] = []

    if not root.exists():
        skipped.append(SkippedRecord(str(root), "", "root_not_found"))
        _write_csv(output_dir / skipped_csv_name, [asdict(x) for x in skipped], ["image_path", "mask_path", "reason"])
        _write_csv(output_dir / valid_csv_name, [], ["image_path", "mask_path", "label", "family_label", "auto_mask"])
        return valid, skipped

    scan_start = time.perf_counter()
    if log_fn:
        log_fn(f"Scanning dataset root: {root}")
    all_files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    mask_paths = [p for p in all_files if is_mask_candidate(p)]
    image_paths = [p for p in all_files if not is_mask_candidate(p)]
    if log_fn:
        log_fn(f"Found {len(image_paths)} image candidates and {len(mask_paths)} mask candidates.")
    masks_by_key = _mask_lookup(mask_paths)
    family_meta = load_family_metadata(root)

    total_images = len(image_paths)
    for idx, image_path in enumerate(sorted(image_paths), start=1):
        mask_path: Optional[Path] = None
        auto_mask = 0
        try:
            if image_path.stat().st_size <= 0:
                skipped.append(SkippedRecord(str(image_path), "", "empty_image_file"))
                continue

            key = _normalize_stem(image_path.stem)
            mask_path = _choose_mask(image_path, masks_by_key.get(key, []))
            image_size = _open_image_size(image_path)

            if mask_path is None:
                if auto_black_for_authentic and is_authentic_path(image_path):
                    mask_path = _auto_mask_path(output_dir, image_path)
                    ensure_dir(mask_path.parent)
                    if not mask_path.exists():
                        Image.new("L", image_size, 0).save(mask_path)
                    auto_mask = 1
                else:
                    skipped.append(SkippedRecord(str(image_path), "", "mask_not_found"))
                    continue

            if mask_path.stat().st_size <= 0:
                skipped.append(SkippedRecord(str(image_path), str(mask_path), "empty_mask_file"))
                continue
            mask_size = _open_image_size(mask_path)
            if image_size != mask_size:
                skipped.append(SkippedRecord(str(image_path), str(mask_path), "image_mask_size_mismatch"))
                continue
            mask_arr = _read_binary_mask(mask_path)
            label = int(mask_arr.any())
            family = family_meta.get(str(image_path), family_meta.get(image_path.name, -1))
            if family < 0:
                family = family_from_path(image_path)
            valid.append(
                SampleRecord(
                    image_path=str(image_path),
                    mask_path=str(mask_path),
                    label=label,
                    family_label=family,
                    auto_mask=auto_mask,
                )
            )
        except Exception as exc:
            skipped.append(SkippedRecord(str(image_path), str(mask_path or ""), f"open_or_parse_failed:{exc}"))
        if log_fn and (idx == 1 or idx % max(1, progress_interval) == 0 or idx == total_images):
            elapsed = time.perf_counter() - scan_start
            log_fn(
                f"scan progress: {idx}/{total_images} "
                f"valid={len(valid)} skipped={len(skipped)} elapsed={elapsed:.1f}s"
            )

    known_family = sum(1 for s in valid if s.family_label >= 0)
    if valid and known_family / max(1, len(valid)) < 0.5:
        for s in valid:
            s.family_label = -1
        message = "No reliable manipulation family labels found. Family head is disabled."
        log_fn(message) if log_fn else print(message)

    _write_csv(output_dir / skipped_csv_name, [asdict(x) for x in skipped], ["image_path", "mask_path", "reason"])
    _write_csv(output_dir / valid_csv_name, [asdict(x) for x in valid], ["image_path", "mask_path", "label", "family_label", "auto_mask"])
    auto_count = sum(s.auto_mask for s in valid)
    if auto_count:
        message = f"Generated {auto_count} black masks for clearly authentic images under {output_dir / 'auto_masks'}."
        log_fn(message) if log_fn else print(message)
    if log_fn:
        pos = sum(1 for s in valid if s.label == 1)
        neg = sum(1 for s in valid if s.label == 0)
        log_fn(
            f"Scan finished: valid={len(valid)} positive={pos} negative={neg} "
            f"skipped={len(skipped)} output_dir={output_dir}"
        )
    return valid, skipped


def _split_class(samples: List[SampleRecord], ratio: float, rng: random.Random) -> Tuple[List[SampleRecord], List[SampleRecord]]:
    samples = list(samples)
    rng.shuffle(samples)
    n_train = int(round(len(samples) * ratio))
    n_train = min(max(n_train, 0), len(samples))
    return samples[:n_train], samples[n_train:]


def _balance(samples: List[SampleRecord], rng: random.Random) -> List[SampleRecord]:
    pos = [s for s in samples if s.label == 1]
    neg = [s for s in samples if s.label == 0]
    if not pos or not neg:
        rng.shuffle(samples)
        return samples
    n = min(len(pos), len(neg))
    rng.shuffle(pos)
    rng.shuffle(neg)
    balanced = pos[:n] + neg[:n]
    rng.shuffle(balanced)
    return balanced


def write_split(path: str | Path, samples: List[SampleRecord]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        for s in samples:
            f.write(f"{s.image_path},{s.mask_path},{s.label}\n")


def split_train_val(
    samples: List[SampleRecord],
    output_dir: str | Path = "outputs",
    split_ratio: float = 0.9,
    seed: int = 42,
    balance_pos_neg: bool = True,
) -> Tuple[Path, Path]:
    rng = random.Random(seed)
    pos = [s for s in samples if s.label == 1]
    neg = [s for s in samples if s.label == 0]
    train_pos, val_pos = _split_class(pos, split_ratio, rng)
    train_neg, val_neg = _split_class(neg, split_ratio, rng)
    train = train_pos + train_neg
    val = val_pos + val_neg
    if balance_pos_neg:
        train = _balance(train, rng)
        val = _balance(val, rng)
    else:
        rng.shuffle(train)
        rng.shuffle(val)

    split_dir = ensure_dir(Path(output_dir) / "splits")
    train_file = split_dir / "train.txt"
    val_file = split_dir / "val.txt"
    write_split(train_file, train)
    write_split(val_file, val)
    return train_file, val_file


def scan_and_split_from_config(config: dict, log_fn: Optional[Callable[[str], None]] = None) -> Tuple[Path, Path]:
    data_cfg = config.get("data", {})
    output_dir = data_cfg.get("scan_output_dir", "outputs")
    samples, _ = scan_dataset(
        root=data_cfg.get("train_root", "/data0/lzb-change-vmunet/FinalTrainData/"),
        output_dir=output_dir,
        log_fn=log_fn,
        progress_interval=int(data_cfg.get("scan_progress_interval", 500)),
    )
    train_file, val_file = split_train_val(
        samples,
        output_dir=output_dir,
        split_ratio=float(data_cfg.get("split_ratio", 0.9)),
        seed=int(config.get("seed", 42)),
        balance_pos_neg=bool(data_cfg.get("balance_pos_neg", True)),
    )
    if log_fn:
        train_count = sum(1 for _ in train_file.open("r", encoding="utf-8", errors="ignore"))
        val_count = sum(1 for _ in val_file.open("r", encoding="utf-8", errors="ignore"))
        log_fn(f"Split finished: train={train_count} val={val_count} train_file={train_file} val_file={val_file}")
    return train_file, val_file
