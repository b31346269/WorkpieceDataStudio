#!/usr/bin/env python3
"""Train and optionally export the one-class wrench_circle detector.

This thin Ultralytics wrapper is intended for school-server GPU 6 or 8. It
records actual metrics/speed values and does not claim improvement by itself.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--project", type=Path, default=Path("training_runs/wrench_circle"))
    parser.add_argument("--name", default="wrench_circle_v1")
    parser.add_argument("--device", default="6", help="GPU id/list such as 6, 8, 6,8, or cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--export-onnx", action="store_true")
    return parser.parse_args()


def validate_device(value: str) -> str:
    value = value.strip().lower()
    if value != "cpu" and not re.fullmatch(r"\d+(,\d+)*", value):
        raise SystemExit("--device must be cpu or a GPU id/list such as 6, 8, or 6,8")
    return value


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def main() -> int:
    args = parse_args()
    data_path = args.data.resolve()
    if not data_path.is_file():
        raise SystemExit(f"data yaml not found: {data_path}")
    if args.epochs <= 0 or args.imgsz <= 0 or args.batch == 0:
        raise SystemExit("epochs/imgsz must be positive and batch cannot be zero")
    device = validate_device(args.device)

    from ultralytics import YOLO

    model = YOLO(args.model)
    train_args = dict(
        data=str(data_path), epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, workers=args.workers, patience=args.patience,
        device=device, project=str(args.project), name=args.name,
        exist_ok=False, pretrained=True, seed=42, deterministic=True,
        degrees=15.0, translate=0.10, scale=0.50, shear=5.0,
        perspective=0.0005, fliplr=0.50, hsv_h=0.015, hsv_s=0.50,
        hsv_v=0.35, mosaic=0.50, mixup=0.0, copy_paste=0.0,
        close_mosaic=10,
    )
    model.train(**train_args)
    save_dir = Path(getattr(getattr(model, "trainer", None), "save_dir", args.project / args.name))
    best = save_dir / "weights" / "best.pt"
    val_results = model.val(data=str(data_path), split="val", imgsz=args.imgsz, batch=args.batch, device=device, plots=True)

    exported = None
    if args.export_onnx:
        exported = model.export(format="onnx", dynamic=True, simplify=True, imgsz=args.imgsz)

    report = {
        "data": str(data_path), "model": args.model, "device": device,
        "train_args": train_args, "run_dir": str(save_dir), "best_pt": str(best),
        "exported_onnx": json_safe(exported),
        "val_results": json_safe(getattr(val_results, "results_dict", {})),
        "val_speed": json_safe(getattr(val_results, "speed", {})),
    }
    report_path = save_dir / "wrench_circle_training_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
