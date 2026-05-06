from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
from PIL import ImageFile

from datasets.transforms import binarize_mask_array

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", message="Corrupt EXIF data.*")

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
SOURCE_GROUP_IGNORE_DIRS = {
    "image",
    "images",
    "img",
    "imgs",
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


@dataclass
class SampleRecord:
    image_path: str
    mask_path: str
    label: int
    family_label: int = -1
    auto_mask: int = 0
    source_group: str = ""
    pair_hash: str = ""


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


def _read_binary_mask(path: Path, mask_threshold: float = 127.0) -> np.ndarray:
    with Image.open(path) as mask:
        arr = np.array(mask.convert("L"))
    return binarize_mask_array(arr, threshold=mask_threshold).astype(np.uint8)


def _pair_hash(image_path: str | Path, mask_path: str | Path) -> str:
    text = f"{Path(image_path)}||{Path(mask_path)}"
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def infer_source_group(
    image_path: str | Path,
    root: str | Path | None = None,
    depth: int = 2,
    ignore_dirs: Iterable[str] | None = None,
) -> str:
    image_path = Path(image_path)
    ignore = {x.lower() for x in (ignore_dirs or SOURCE_GROUP_IGNORE_DIRS)}
    try:
        rel = image_path.relative_to(Path(root)) if root is not None else image_path
    except ValueError:
        rel = image_path
    parent_parts = list(rel.parent.parts)
    kept = [part for part in parent_parts if part.lower() not in ignore]
    if not kept:
        kept = parent_parts[-1:] if parent_parts else [image_path.parent.name or "root"]
    depth = max(1, int(depth))
    return "/".join(kept[:depth]) if kept else "root"


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


def _choose_mask(
    image_path: Path,
    candidates: List[Path],
    image_size: Tuple[int, int] | None = None,
) -> Tuple[Optional[Path], List[Path]]:
    if not candidates:
        return None, []

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

    usable = list(candidates)
    if image_size is not None:
        size_matched = []
        for mask in candidates:
            try:
                if mask.stat().st_size > 0 and _open_image_size(mask) == image_size:
                    size_matched.append(mask)
            except Exception:
                continue
        if size_matched:
            usable = size_matched
    ranked = sorted(usable, key=score, reverse=True)
    return ranked[0], ranked[1:]


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
    mask_threshold: float = 127.0,
    source_group_depth: int = 2,
    source_group_ignore_dirs: Iterable[str] | None = None,
) -> Tuple[List[SampleRecord], List[SkippedRecord]]:
    root = Path(root)
    output_dir = ensure_dir(output_dir)
    skipped: List[SkippedRecord] = []
    valid: List[SampleRecord] = []
    ambiguous_pairs: List[dict] = []
    valid_fields = ["image_path", "mask_path", "label", "family_label", "auto_mask", "source_group", "pair_hash"]

    if not root.exists():
        skipped.append(SkippedRecord(str(root), "", "root_not_found"))
        _write_csv(output_dir / skipped_csv_name, [asdict(x) for x in skipped], ["image_path", "mask_path", "reason"])
        _write_csv(output_dir / valid_csv_name, [], valid_fields)
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

            image_size = _open_image_size(image_path)
            key = _normalize_stem(image_path.stem)
            mask_path, alternatives = _choose_mask(image_path, masks_by_key.get(key, []), image_size=image_size)
            if alternatives:
                ambiguous_pairs.append(
                    {
                        "image_path": str(image_path),
                        "chosen_mask_path": str(mask_path),
                        "alternative_mask_paths": "|".join(str(p) for p in alternatives[:20]),
                        "num_alternatives": len(alternatives),
                    }
                )

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
            mask_arr = _read_binary_mask(mask_path, mask_threshold=mask_threshold)
            label = int(mask_arr.any())
            family = family_meta.get(str(image_path), family_meta.get(image_path.name, -1))
            if family < 0:
                family = family_from_path(image_path)
            source_group = infer_source_group(
                image_path,
                root=root,
                depth=source_group_depth,
                ignore_dirs=source_group_ignore_dirs,
            )
            valid.append(
                SampleRecord(
                    image_path=str(image_path),
                    mask_path=str(mask_path),
                    label=label,
                    family_label=family,
                    auto_mask=auto_mask,
                    source_group=source_group,
                    pair_hash=_pair_hash(image_path, mask_path),
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
    _write_csv(output_dir / valid_csv_name, [asdict(x) for x in valid], valid_fields)
    _write_csv(
        output_dir / "ambiguous_pairs.csv",
        ambiguous_pairs,
        ["image_path", "chosen_mask_path", "alternative_mask_paths", "num_alternatives"],
    )
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


def _count_labels(samples: List[SampleRecord]) -> Dict[str, int]:
    pos = sum(1 for s in samples if int(s.label) == 1)
    neg = sum(1 for s in samples if int(s.label) == 0)
    return {"total": len(samples), "positive": pos, "negative": neg}


def _sample_source_group(
    sample: SampleRecord,
    train_root: str | Path | None,
    depth: int,
    ignore_dirs: Iterable[str] | None,
) -> str:
    if sample.source_group:
        return sample.source_group
    return infer_source_group(sample.image_path, root=train_root, depth=depth, ignore_dirs=ignore_dirs)


def _source_group_table(
    train: List[SampleRecord],
    val: List[SampleRecord],
    train_root: str | Path | None,
    depth: int,
    ignore_dirs: Iterable[str] | None,
) -> List[dict]:
    rows = []
    for split_name, rows_samples in (("train", train), ("val", val)):
        groups: Dict[str, List[SampleRecord]] = {}
        for sample in rows_samples:
            groups.setdefault(_sample_source_group(sample, train_root, depth, ignore_dirs), []).append(sample)
        for group, group_samples in sorted(groups.items()):
            counts = _count_labels(group_samples)
            rows.append({"split": split_name, "source_group": group, **counts})
    return rows


def _source_aware_split(
    samples: List[SampleRecord],
    split_ratio: float,
    seed: int,
    train_root: str | Path | None,
    source_group_depth: int,
    source_group_ignore_dirs: Iterable[str] | None,
    min_val_groups: int = 2,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Tuple[List[SampleRecord], List[SampleRecord]]:
    rng = random.Random(seed)
    groups: Dict[str, List[SampleRecord]] = {}
    for sample in samples:
        group = _sample_source_group(sample, train_root, source_group_depth, source_group_ignore_dirs)
        sample.source_group = group
        if not sample.pair_hash:
            sample.pair_hash = _pair_hash(sample.image_path, sample.mask_path)
        groups.setdefault(group, []).append(sample)

    if len(groups) < 2:
        if log_fn:
            log_fn("source_aware split requested but fewer than two source groups were found; falling back to random split.")
        pos = [s for s in samples if s.label == 1]
        neg = [s for s in samples if s.label == 0]
        train_pos, val_pos = _split_class(pos, split_ratio, rng)
        train_neg, val_neg = _split_class(neg, split_ratio, rng)
        return train_pos + train_neg, val_pos + val_neg

    total_counts = _count_labels(samples)
    val_ratio = max(0.0, min(1.0, 1.0 - float(split_ratio)))
    target_total = max(1, int(round(total_counts["total"] * val_ratio)))
    target_pos = max(1, int(round(total_counts["positive"] * val_ratio))) if total_counts["positive"] else 0
    target_neg = max(1, int(round(total_counts["negative"] * val_ratio))) if total_counts["negative"] else 0

    candidates = []
    for group, group_samples in groups.items():
        counts = _count_labels(group_samples)
        candidates.append({"group": group, "samples": group_samples, **counts})
    rng.shuffle(candidates)

    val_groups: set[str] = set()
    val_total = 0
    val_pos = 0
    val_neg = 0

    def split_score(total: int, pos: int, neg: int) -> float:
        score = abs(total - target_total) / max(1, target_total)
        if target_pos:
            score += abs(pos - target_pos) / max(1, target_pos)
        if target_neg:
            score += abs(neg - target_neg) / max(1, target_neg)
        if total > target_total:
            score += 0.5 * (total - target_total) / max(1, target_total)
        return score

    effective_min_val_groups = min(max(1, int(min_val_groups)), max(1, len(groups) - 1))
    max_val_groups = max(1, len(groups) - 1)
    while candidates and len(val_groups) < max_val_groups and (val_total < target_total or len(val_groups) < effective_min_val_groups):
        current_score = split_score(val_total, val_pos, val_neg)
        best_idx = 0
        best_score = float("inf")
        for idx, candidate in enumerate(candidates):
            new_score = split_score(
                val_total + int(candidate["total"]),
                val_pos + int(candidate["positive"]),
                val_neg + int(candidate["negative"]),
            )
            if new_score < best_score:
                best_score = new_score
                best_idx = idx
        candidate = candidates.pop(best_idx)
        if val_groups and val_total >= target_total and best_score > current_score:
            break
        val_groups.add(str(candidate["group"]))
        val_total += int(candidate["total"])
        val_pos += int(candidate["positive"])
        val_neg += int(candidate["negative"])

    train = []
    val = []
    for group, group_samples in groups.items():
        (val if group in val_groups else train).extend(group_samples)
    rng.shuffle(train)
    rng.shuffle(val)
    if log_fn:
        train_counts = _count_labels(train)
        val_counts = _count_labels(val)
        log_fn(
            "Source-aware split before balancing: "
            f"groups={len(groups)} val_groups={len(val_groups)} "
            f"train={train_counts} val={val_counts}"
        )
    return train, val


def _audit_split_samples(
    train: List[SampleRecord],
    val: List[SampleRecord],
    output_dir: str | Path,
    train_root: str | Path | None,
    source_group_depth: int,
    source_group_ignore_dirs: Iterable[str] | None,
    mask_threshold: float,
    strict: bool,
    require_source_disjoint: bool,
    summary: dict,
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    output_dir = Path(output_dir)
    split_dir = ensure_dir(output_dir / "splits")
    audit_rows = []
    mismatch_rows = []
    seen_images: Dict[str, str] = {}
    seen_masks: Dict[str, str] = {}
    source_sets = {"train": set(), "val": set()}

    def add_issue(split_name: str, sample: SampleRecord, reason: str, actual_label: int | str = "") -> None:
        row = {
            "split": split_name,
            "image_path": sample.image_path,
            "mask_path": sample.mask_path,
            "source_group": _sample_source_group(sample, train_root, source_group_depth, source_group_ignore_dirs),
            "expected_label": sample.label,
            "actual_label": actual_label,
            "status": "error",
            "reason": reason,
        }
        audit_rows.append(row)
        mismatch_rows.append(row)

    for split_name, rows_samples in (("train", train), ("val", val)):
        for sample in rows_samples:
            group = _sample_source_group(sample, train_root, source_group_depth, source_group_ignore_dirs)
            source_sets[split_name].add(group)
            image_key = str(Path(sample.image_path))
            mask_key = str(Path(sample.mask_path))
            if image_key in seen_images:
                add_issue(split_name, sample, f"duplicate_image_also_in_{seen_images[image_key]}")
                continue
            if mask_key in seen_masks:
                add_issue(split_name, sample, f"duplicate_mask_also_in_{seen_masks[mask_key]}")
                continue
            seen_images[image_key] = split_name
            seen_masks[mask_key] = split_name

            image_path = Path(sample.image_path)
            mask_path = Path(sample.mask_path)
            try:
                if not image_path.exists():
                    add_issue(split_name, sample, "image_missing")
                    continue
                if not mask_path.exists():
                    add_issue(split_name, sample, "mask_missing")
                    continue
                if image_path.stat().st_size <= 0:
                    add_issue(split_name, sample, "empty_image_file")
                    continue
                if mask_path.stat().st_size <= 0:
                    add_issue(split_name, sample, "empty_mask_file")
                    continue
                image_size = _open_image_size(image_path)
                mask_size = _open_image_size(mask_path)
                if image_size != mask_size:
                    add_issue(split_name, sample, f"image_mask_size_mismatch:image={image_size}:mask={mask_size}")
                    continue
                actual_label = int(_read_binary_mask(mask_path, mask_threshold=mask_threshold).any())
                if actual_label != int(sample.label):
                    add_issue(split_name, sample, "label_mismatch", actual_label=actual_label)
                    continue
                audit_rows.append(
                    {
                        "split": split_name,
                        "image_path": sample.image_path,
                        "mask_path": sample.mask_path,
                        "source_group": group,
                        "expected_label": sample.label,
                        "actual_label": actual_label,
                        "status": "ok",
                        "reason": "",
                    }
                )
            except Exception as exc:
                add_issue(split_name, sample, f"open_or_parse_failed:{exc}")

    source_overlap = sorted(source_sets["train"] & source_sets["val"])
    for group in source_overlap:
        row = {
            "split": "train_val",
            "image_path": "",
            "mask_path": "",
            "source_group": group,
            "expected_label": "",
            "actual_label": "",
            "status": "error",
            "reason": "source_group_overlap",
        }
        if not require_source_disjoint:
            row["status"] = "warning"
        audit_rows.append(row)
        if require_source_disjoint:
            mismatch_rows.append(row)

    source_rows = _source_group_table(train, val, train_root, source_group_depth, source_group_ignore_dirs)
    _write_csv(
        split_dir / "split_audit.csv",
        audit_rows,
        ["split", "image_path", "mask_path", "source_group", "expected_label", "actual_label", "status", "reason"],
    )
    _write_csv(
        split_dir / "pair_mismatches.csv",
        mismatch_rows,
        ["split", "image_path", "mask_path", "source_group", "expected_label", "actual_label", "status", "reason"],
    )
    _write_csv(split_dir / "source_groups.csv", source_rows, ["split", "source_group", "total", "positive", "negative"])

    summary.update(
        {
            "train": _count_labels(train),
            "val": _count_labels(val),
            "train_source_groups": len(source_sets["train"]),
            "val_source_groups": len(source_sets["val"]),
            "source_group_overlap_count": len(source_overlap),
            "audit_total_rows": len(audit_rows),
            "audit_error_rows": len(mismatch_rows),
        }
    )
    (split_dir / "split_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if log_fn:
        log_fn(
            f"Split audit finished: errors={len(mismatch_rows)} "
            f"train={summary['train']} val={summary['val']} "
            f"source_overlap={len(source_overlap)}"
        )
    if strict and mismatch_rows:
        raise RuntimeError(
            f"Split audit failed with {len(mismatch_rows)} errors. "
            f"See {split_dir / 'split_audit.csv'} and {split_dir / 'pair_mismatches.csv'}."
        )


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
    split_strategy: str = "random",
    train_root: str | Path | None = None,
    source_group_depth: int = 2,
    source_group_ignore_dirs: Iterable[str] | None = None,
    min_val_groups: int = 2,
    split_audit: bool = True,
    strict_pair_audit: bool = True,
    mask_threshold: float = 127.0,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Tuple[Path, Path]:
    rng = random.Random(seed)
    split_strategy = str(split_strategy or "random").lower()
    if split_strategy == "source_aware":
        train, val = _source_aware_split(
            samples,
            split_ratio=split_ratio,
            seed=seed,
            train_root=train_root,
            source_group_depth=source_group_depth,
            source_group_ignore_dirs=source_group_ignore_dirs,
            min_val_groups=min_val_groups,
            log_fn=log_fn,
        )
    else:
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
    summary = {
        "split_strategy": split_strategy,
        "split_ratio": float(split_ratio),
        "seed": int(seed),
        "balance_pos_neg": bool(balance_pos_neg),
        "source_group_depth": int(source_group_depth),
        "min_val_groups": int(min_val_groups),
        "train_root": str(train_root or ""),
        "num_input_samples": len(samples),
    }
    if split_audit:
        _audit_split_samples(
            train,
            val,
            output_dir=output_dir,
            train_root=train_root,
            source_group_depth=source_group_depth,
            source_group_ignore_dirs=source_group_ignore_dirs,
            mask_threshold=mask_threshold,
            strict=strict_pair_audit,
            require_source_disjoint=split_strategy == "source_aware",
            summary=summary,
            log_fn=log_fn,
        )
    else:
        summary.update({"train": _count_labels(train), "val": _count_labels(val)})
        (split_dir / "split_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return train_file, val_file


def scan_and_split_from_config(config: dict, log_fn: Optional[Callable[[str], None]] = None) -> Tuple[Path, Path]:
    data_cfg = config.get("data", {})
    output_dir = data_cfg.get("scan_output_dir", "outputs")
    train_root = data_cfg.get("train_root", "/data0/lzb-change-vmunet/FinalTrainData/")
    source_group_depth = int(data_cfg.get("source_group_depth", 2))
    source_group_ignore_dirs = data_cfg.get("source_group_ignore_dirs", list(SOURCE_GROUP_IGNORE_DIRS))
    mask_threshold = float(data_cfg.get("mask_threshold", 127.0))
    samples, _ = scan_dataset(
        root=train_root,
        output_dir=output_dir,
        log_fn=log_fn,
        progress_interval=int(data_cfg.get("scan_progress_interval", 500)),
        mask_threshold=mask_threshold,
        source_group_depth=source_group_depth,
        source_group_ignore_dirs=source_group_ignore_dirs,
    )
    train_file, val_file = split_train_val(
        samples,
        output_dir=output_dir,
        split_ratio=float(data_cfg.get("split_ratio", 0.9)),
        seed=int(config.get("seed", 42)),
        balance_pos_neg=bool(data_cfg.get("balance_pos_neg", True)),
        split_strategy=str(data_cfg.get("split_strategy", "random")),
        train_root=train_root,
        source_group_depth=source_group_depth,
        source_group_ignore_dirs=source_group_ignore_dirs,
        min_val_groups=int(data_cfg.get("min_val_groups", 2)),
        split_audit=bool(data_cfg.get("split_audit", True)),
        strict_pair_audit=bool(data_cfg.get("strict_pair_audit", True)),
        mask_threshold=mask_threshold,
        log_fn=log_fn,
    )
    if log_fn:
        train_count = sum(1 for _ in train_file.open("r", encoding="utf-8", errors="ignore"))
        val_count = sum(1 for _ in val_file.open("r", encoding="utf-8", errors="ignore"))
        log_fn(f"Split finished: train={train_count} val={val_count} train_file={train_file} val_file={val_file}")
    return train_file, val_file


def should_resplit_from_config(config: dict, train_file: str | Path, val_file: str | Path) -> Tuple[bool, str]:
    data_cfg = config.get("data", {})
    train_file = Path(train_file)
    val_file = Path(val_file)
    if bool(data_cfg.get("force_resplit", False)):
        return True, "force_resplit=true"
    if not train_file.exists() or not val_file.exists():
        return True, "split_files_missing"

    expected_strategy = str(data_cfg.get("split_strategy", "random")).lower()
    summary_path = Path(data_cfg.get("scan_output_dir", "outputs")) / "splits" / "split_summary.json"
    if expected_strategy == "source_aware":
        if not summary_path.exists():
            return True, "source_aware_requested_but_split_summary_missing"
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            return True, "split_summary_unreadable"
        if str(summary.get("split_strategy", "")).lower() != expected_strategy:
            return True, "split_strategy_changed"
        expected_depth = int(data_cfg.get("source_group_depth", 2))
        if int(summary.get("source_group_depth", -1)) != expected_depth:
            return True, "source_group_depth_changed"
        if int(summary.get("audit_error_rows", 0)) > 0 and bool(data_cfg.get("strict_pair_audit", True)):
            return True, "previous_split_audit_failed"
    return False, "existing_splits_ok"
