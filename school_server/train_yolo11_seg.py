#!/usr/bin/env python3
"""Train/export the two-class Workpiece + Wrench YOLO11 segmentation model.

This deliberately validates polygon labels before starting training.  It will
not turn detection boxes into fake masks.  The command is intended for the
school ``ping`` environment (physical GPUs 6/8), but also accepts device
``0`` for the local RTX 4060 fallback.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from segmentation_dataset import validate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO11n/s segmentation model.")
    parser.add_argument("--data", type=Path, required=True, help="Two-class segmentation data.yaml")
    parser.add_argument("--model", default="yolo11n-seg.pt", help="Ultralytics segmentation checkpoint")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--device",
        default="6",
        help="0 for local RTX 4060; 6/8 for school GPUs; 6,8 for multi-GPU",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="YOLO11n_workpiece_wrench_seg")
    parser.add_argument("--output", type=Path, default=Path("school_training/segmentation"))
    parser.add_argument("--export-imgsz", type=int, default=640)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    devices = {item.strip() for item in str(args.device).split(",") if item.strip()}
    allowed_devices = {"0", "6", "8", "cpu"}
    if not devices or not devices <= allowed_devices:
        raise SystemExit(
            "Device must be 0 (local RTX 4060), 6/8 (school GPUs), 6,8, or cpu."
        )

    report = validate_dataset(args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "dataset_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not report["ready_for_yolo_seg"]:
        raise SystemExit(
            "Segmentation training stopped: polygon labels are missing or invalid. "
            f"See {args.output / 'dataset_validation.json'}"
        )

    from ultralytics import YOLO

    model = YOLO(args.model)
    train_kwargs: dict[str, Any] = {
        "data": str(args.data.resolve()),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "seed": args.seed,
        "deterministic": True,
        "patience": 30,
        "cache": False,
        "project": str((args.output / "runs").resolve()),
        "name": args.name,
        "task": "segment",
        "plots": True,
    }
    model.train(**train_kwargs)

    best_path = Path(model.trainer.best)
    exported = YOLO(str(best_path)).export(
        format="onnx",
        imgsz=args.export_imgsz,
        dynamic=False,
        simplify=True,
        nms=False,
    )
    (args.output / "training_summary.json").write_text(
        json.dumps(
            {
                "best_pt": str(best_path),
                "onnx": str(exported),
                "classes": ["workpiece", "wrench"],
                "device": args.device,
                "imgsz": args.imgsz,
                "epochs": args.epochs,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"best.pt: {best_path}")
    print(f"Unity-compatible segmentation ONNX: {exported}")


if __name__ == "__main__":
    main()
