#!/usr/bin/env python3
"""Measure EfficientDet anchor-shape coverage for Pascal VOC annotations."""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


CLASSES = ("hole", "screw", "tool")
DEFAULT_RATIO_SETS = (
    (1.0,),
    (0.5, 1.0, 2.0),
    (0.33, 0.5, 1.0, 2.0, 3.0),
)
DEFAULT_SCALES = (0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0)


def shape_iou(box_width: float, box_height: float, anchor_width: float, anchor_height: float) -> float:
    intersection = min(box_width, anchor_width) * min(box_height, anchor_height)
    union = box_width * box_height + anchor_width * anchor_height - intersection
    return intersection / union if union else 0.0


def anchors(anchor_scale: float, aspect_ratios: tuple[float, ...]) -> list[tuple[float, float]]:
    result = []
    for level in range(3, 8):
        stride = 2**level
        for octave in range(3):
            size = anchor_scale * stride * (2 ** (octave / 3.0))
            for ratio in aspect_ratios:
                ratio_root = math.sqrt(ratio)
                result.append((size * ratio_root, size / ratio_root))
    return result


def load_boxes(dataset: Path, splits: tuple[str, ...]) -> dict[str, list[tuple[float, float]]]:
    boxes: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for split in splits:
        for xml_path in sorted((dataset / split).glob("*.xml")):
            root = ET.parse(xml_path).getroot()
            for obj in root.findall("object"):
                class_name = (obj.findtext("name") or "").strip()
                box = obj.find("bndbox")
                if class_name not in CLASSES or box is None:
                    continue
                xmin = float(box.findtext("xmin", "0"))
                ymin = float(box.findtext("ymin", "0"))
                xmax = float(box.findtext("xmax", "0"))
                ymax = float(box.findtext("ymax", "0"))
                if xmax > xmin and ymax > ymin:
                    boxes[class_name].append((xmax - xmin, ymax - ymin))
    return boxes


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "boxes": len(values),
        "mean_best_iou": sum(values) / len(values),
        "p10_best_iou": percentile(values, 0.10),
        "p50_best_iou": percentile(values, 0.50),
        "coverage_iou_0.50": sum(value >= 0.50 for value in values) / len(values),
        "coverage_iou_0.70": sum(value >= 0.70 for value in values) / len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--splits", nargs="+", default=("train", "valid"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    boxes = load_boxes(args.dataset.resolve(), tuple(args.splits))
    if not any(boxes.values()):
        raise RuntimeError("No supported Pascal VOC boxes found")

    experiments = []
    for anchor_scale in DEFAULT_SCALES:
        for ratios in DEFAULT_RATIO_SETS:
            anchor_shapes = anchors(anchor_scale, ratios)
            class_results = {}
            all_ious = []
            for class_name in CLASSES:
                best_ious = [
                    max(shape_iou(width, height, aw, ah) for aw, ah in anchor_shapes)
                    for width, height in boxes[class_name]
                ]
                class_results[class_name] = summarize(best_ious)
                all_ious.extend(best_ious)
            experiments.append(
                {
                    "anchor_scale": anchor_scale,
                    "aspect_ratios": list(ratios),
                    "overall": summarize(all_ious),
                    "classes": class_results,
                }
            )

    experiments.sort(
        key=lambda item: (
            item["overall"]["coverage_iou_0.70"],
            item["overall"]["mean_best_iou"],
        ),
        reverse=True,
    )
    payload = {
        "dataset": str(args.dataset.resolve()),
        "splits": list(args.splits),
        "ranking": experiments,
        "recommended": experiments[0],
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
