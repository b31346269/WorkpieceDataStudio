#!/usr/bin/env python3
"""Validate wrench_circle on an isolated test split and/or a video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--device", default="6")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--project", type=Path, default=Path("training_runs/wrench_circle_validation"))
    parser.add_argument("--name", default="test_v1")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.50)
    return parser.parse_args()


def safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def main() -> int:
    args = parse_args()
    if not args.model.is_file():
        raise SystemExit(f"model not found: {args.model}")
    if not args.data.is_file():
        raise SystemExit(f"data yaml not found: {args.data}")

    from ultralytics import YOLO

    model = YOLO(str(args.model))
    metrics = model.val(
        data=str(args.data), split="test", imgsz=args.imgsz, batch=args.batch,
        device=args.device, conf=args.conf, iou=args.iou, plots=True,
        project=str(args.project), name=args.name, exist_ok=False,
    )
    report = {
        "model": str(args.model.resolve()), "data": str(args.data.resolve()),
        "split": "test", "device": args.device,
        "metrics": safe(getattr(metrics, "results_dict", {})),
        "speed": safe(getattr(metrics, "speed", {})),
    }
    if args.video:
        if not args.video.is_file():
            raise SystemExit(f"video not found: {args.video}")
        model.predict(
            source=str(args.video), conf=args.conf, iou=args.iou,
            imgsz=args.imgsz, device=args.device, save=True,
            project=str(args.project), name=f"{args.name}_video", exist_ok=False,
        )
        report["video"] = str(args.video.resolve())
        report["video_prediction_note"] = "Qualitative output is saved by Ultralytics; measure update/display FPS separately in Unity."

    args.project.mkdir(parents=True, exist_ok=True)
    report_path = args.project / f"{args.name}_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
