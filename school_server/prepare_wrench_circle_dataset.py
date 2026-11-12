#!/usr/bin/env python3
"""Prepare a leakage-safe one-class wrench_circle YOLO dataset.

The script requires a manifest with recording/workpiece groups. It never does
an image-level random split, because neighbouring video frames would make a
test score misleading.

Expected input layout:
    source/images/...\n    source/labels/...\n    manifest.csv (image, recording_sequence, workpiece_family, split)

The image value is relative to source/images. Split values are train, val or
validation, and test. Only class id 0 (wrench_circle) is accepted.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif"}
SPLIT_NAMES = {"train": "train", "val": "val", "validation": "val", "test": "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Roboflow export root")
    parser.add_argument("--manifest", type=Path, required=True, help="CSV with grouped split metadata")
    parser.add_argument("--output", type=Path, required=True, help="Prepared dataset directory")
    parser.add_argument("--force", action="store_true", help="Overwrite files inside output (never deletes output)")
    return parser.parse_args()


def normalise_split(value: str, row_number: int) -> str:
    split = SPLIT_NAMES.get(value.strip().lower())
    if split is None:
        raise ValueError(f"manifest row {row_number}: split must be train, val/validation, or test; got {value!r}")
    return split


def find_image(source_images: Path, relative_name: str) -> Path:
    relative = Path(relative_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"image path must be relative and stay below source/images: {relative_name!r}")
    candidate = (source_images / relative).resolve()
    if source_images.resolve() not in candidate.parents:
        raise ValueError(f"image path escapes source/images: {relative_name!r}")
    if not candidate.is_file() or candidate.suffix.lower() not in IMAGE_EXTENSIONS:
        raise FileNotFoundError(f"image not found or unsupported: {candidate}")
    return candidate


def label_path(source: Path, image: Path, source_images: Path) -> Path:
    relative = image.relative_to(source_images).with_suffix(".txt")
    labels_root = source / "labels"
    direct = labels_root / relative
    if direct.is_file():
        return direct
    matches = list(labels_root.rglob(relative.name)) if labels_root.exists() else []
    return matches[0] if len(matches) == 1 else direct


def validate_labels(path: Path) -> int:
    if not path.is_file():
        return 0
    count = 0
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 YOLO fields, got {len(fields)}")
        try:
            class_id = int(fields[0])
            values = [float(item) for item in fields[1:]]
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: non-numeric YOLO label") from exc
        if class_id != 0:
            raise ValueError(f"{path}:{line_number}: only class 0 wrench_circle is allowed, got {class_id}")
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError(f"{path}:{line_number}: box coordinates must be normalized to [0, 1]")
        if values[2] <= 0.0 or values[3] <= 0.0:
            raise ValueError(f"{path}:{line_number}: box width/height must be positive")
        count += 1
    return count


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    source_images = (source / "images").resolve()
    manifest_path = args.manifest.resolve()
    output = args.output.resolve()
    if not source_images.is_dir():
        raise SystemExit(f"missing input directory: {source_images}")
    if not manifest_path.is_file():
        raise SystemExit(f"missing manifest: {manifest_path}")
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    groups: dict[tuple[str, str], str] = {}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image", "recording_sequence", "workpiece_family", "split"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            missing = required - set(reader.fieldnames or [])
            raise SystemExit(f"manifest is missing columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, 2):
            image_name = (row.get("image") or "").strip()
            sequence = (row.get("recording_sequence") or "").strip()
            family = (row.get("workpiece_family") or "").strip()
            if not image_name or not sequence or not family:
                raise ValueError(f"manifest row {row_number}: image, recording_sequence and workpiece_family are required")
            split = normalise_split(row.get("split") or "", row_number)
            group = (sequence, family)
            previous = groups.setdefault(group, split)
            if previous != split:
                raise ValueError(f"group {group!r} appears in both {previous} and {split}; fix split leakage")
            image = find_image(source_images, image_name)
            label = label_path(source, image, source_images)
            object_count = validate_labels(label)
            rows.append({
                "image": image_name,
                "recording_sequence": sequence,
                "workpiece_family": family,
                "physical_wrench": (row.get("physical_wrench") or "").strip(),
                "camera_view": (row.get("camera_view") or "").strip(),
                "distance_group": (row.get("distance_group") or "").strip(),
                "lighting_group": (row.get("lighting_group") or "").strip(),
                "circle_visible": (row.get("circle_visible") or ("yes" if object_count else "no")).strip(),
                "split": split,
                "label_path": str(label.relative_to(source)) if label.exists() else "",
                "objects": str(object_count),
            })

    if not rows:
        raise SystemExit("manifest has no data rows")
    split_counts = Counter(row["split"] for row in rows)
    if set(split_counts) != {"train", "val", "test"}:
        raise SystemExit(f"manifest must contain all train/val/test splits; got {dict(split_counts)}")

    for row in rows:
        src_image = find_image(source_images, row["image"])
        relative = Path(row["image"])
        dst_image = output / row["split"] / "images" / relative
        dst_label = output / row["split"] / "labels" / relative.with_suffix(".txt")
        dst_image.parent.mkdir(parents=True, exist_ok=True)
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        if args.force or not dst_image.exists():
            shutil.copy2(src_image, dst_image)
        src_label = source / row["label_path"] if row["label_path"] else None
        if src_label and src_label.is_file():
            if args.force or not dst_label.exists():
                shutil.copy2(src_label, dst_label)
        elif args.force or not dst_label.exists():
            dst_label.write_text("", encoding="utf-8")

    prepared_manifest = output / "dataset_manifest.csv"
    with prepared_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    (output / "data.yaml").write_text(
        f"path: {output.as_posix()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "names:\n"
        "  0: wrench_circle\n",
        encoding="utf-8",
    )
    audit = {
        "class_names": ["wrench_circle"],
        "image_count": len(rows),
        "split_counts": dict(split_counts),
        "object_counts": {split: sum(int(row["objects"]) for row in rows if row["split"] == split) for split in sorted(split_counts)},
        "negative_image_counts": {split: sum(int(row["objects"]) == 0 for row in rows if row["split"] == split) for split in sorted(split_counts)},
        "group_count": len(groups),
        "groups": {f"{sequence}::{family}": split for (sequence, family), split in sorted(groups.items())},
        "source": str(source),
        "manifest": str(manifest_path),
    }
    (output / "dataset_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print(f"Prepared YOLO dataset: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
