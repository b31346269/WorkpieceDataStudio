#!/usr/bin/env python3
"""Validate a two-class Ultralytics YOLO segmentation dataset.

The existing WorkpieceDataStudio exports are detection datasets.  A YOLO
segmentation label must contain a class id followed by at least three
polygon points (six normalized coordinates); a five-value detection row is
not silently converted into a mask by this tool.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


EXPECTED_NAMES = ["workpiece", "wrench"]


def _resolve_split(root: Path, value: str | list[str] | None) -> Path | None:
    if value is None:
        return None
    value = value[0] if isinstance(value, list) else value
    path = Path(value)
    return path if path.is_absolute() else root / path


def _label_for_image(image: Path, images_root: Path, labels_root: Path) -> Path:
    relative = image.relative_to(images_root).with_suffix(".txt")
    return labels_root / relative


def validate_dataset(data_yaml: Path) -> dict[str, Any]:
    data_yaml = data_yaml.resolve()
    root = Path(data_yaml.parent)
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}

    names = data.get("names", [])
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names, key=lambda item: int(item))]
    names = [str(name) for name in names]

    report: dict[str, Any] = {
        "data_yaml": str(data_yaml),
        "expected_names": EXPECTED_NAMES,
        "names": names,
        "splits": {},
        "errors": [],
        "warnings": [],
    }

    if names != EXPECTED_NAMES:
        report["errors"].append(
            f"names must be exactly {EXPECTED_NAMES}, got {names!r}"
        )

    for split in ("train", "val", "test"):
        images_root = _resolve_split(root, data.get(split))
        if images_root is None or not images_root.exists():
            if split in ("train", "val"):
                report["errors"].append(f"missing {split} image directory: {images_root}")
            continue

        labels_root = images_root.parent.parent / "labels" / images_root.name
        if not labels_root.exists():
            # Standard YOLO layout is split/images + split/labels.
            labels_root = images_root.parent / "labels"

        images = sorted(
            p for p in images_root.rglob("*")
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        class_counts: Counter[str] = Counter()
        polygon_rows = 0
        detection_rows = 0
        empty_labels = 0
        missing_labels = 0
        invalid_rows: list[str] = []

        for image in images:
            label = _label_for_image(image, images_root, labels_root)
            if not label.exists():
                missing_labels += 1
                continue
            rows = [line.strip() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not rows:
                empty_labels += 1
            for line_number, row in enumerate(rows, start=1):
                fields = row.split()
                if len(fields) == 5:
                    detection_rows += 1
                    invalid_rows.append(f"{label}:{line_number} is a detection box, not a polygon")
                    continue
                if len(fields) < 7 or (len(fields) - 1) % 2 != 0:
                    invalid_rows.append(f"{label}:{line_number} has {len(fields)} fields; expected class + polygon pairs")
                    continue
                try:
                    class_id = int(fields[0])
                    coords = [float(value) for value in fields[1:]]
                except ValueError:
                    invalid_rows.append(f"{label}:{line_number} contains non-numeric values")
                    continue
                if class_id not in (0, 1):
                    invalid_rows.append(f"{label}:{line_number} uses class {class_id}; expected 0 or 1")
                if any(value < 0 or value > 1 for value in coords):
                    invalid_rows.append(f"{label}:{line_number} has coordinates outside [0, 1]")
                class_counts[str(class_id)] += 1
                polygon_rows += 1

        split_report = {
            "images": len(images),
            "polygon_rows": polygon_rows,
            "detection_rows": detection_rows,
            "empty_labels": empty_labels,
            "missing_labels": missing_labels,
            "class_counts": dict(class_counts),
            "invalid_row_count": len(invalid_rows),
            "invalid_rows_sample": invalid_rows[:20],
        }
        report["splits"][split] = split_report

        if detection_rows:
            report["errors"].append(
                f"{split} contains {detection_rows} five-value detection rows; polygon masks are required"
            )
        if missing_labels:
            report["warnings"].append(f"{split} has {missing_labels} images without label files")
        if invalid_rows:
            report["errors"].append(f"{split} has {len(invalid_rows)} invalid polygon rows")

    train = report["splits"].get("train", {})
    val = report["splits"].get("val", {})
    if train.get("polygon_rows", 0) == 0 or val.get("polygon_rows", 0) == 0:
        report["errors"].append("train and val must both contain polygon masks")

    report["ready_for_yolo_seg"] = not report["errors"]
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a two-class YOLO segmentation dataset.")
    parser.add_argument("--data", type=Path, required=True, help="YOLO segmentation data.yaml")
    parser.add_argument("--report", type=Path, help="Optional JSON report path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = validate_dataset(args.data)
    output = args.report or args.data.with_name("segmentation_dataset_report.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready_for_yolo_seg"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
