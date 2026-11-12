#!/usr/bin/env python3
"""Create workpiece/wrench candidate boxes for polygon annotation.

The project has two detectors whose class name is ``tool`` but whose meaning is
different: the main model's tool is the workpiece, while ``yolo11n_tool`` is the
wrench detector.  This script keeps those semantics separate and writes only
candidate boxes/overlays.  It never writes YOLO-seg polygons, because a box is
not a valid object mask.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prelabel segmentation candidates with two detector models.")
    parser.add_argument("--images", type=Path, required=True, help="Folder of images to inspect")
    parser.add_argument("--workpiece-model", type=Path, required=True, help="Main detector; its tool class means workpiece")
    parser.add_argument("--wrench-model", type=Path, required=True, help="Extra tool/wrench detector")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workpiece-conf", type=float, default=0.35)
    parser.add_argument("--wrench-conf", type=float, default=0.30)
    parser.add_argument("--device", default="6")
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def _pick_tool_class(model, result) -> list[dict]:
    names = result.names
    tool_ids = {
        int(class_id)
        for class_id, name in names.items()
        if str(name).strip().lower() == "tool"
    }
    if not tool_ids:
        return []
    rows: list[dict] = []
    if result.boxes is None:
        return rows
    for box, score, class_id in zip(
        result.boxes.xyxy.cpu().tolist(),
        result.boxes.conf.cpu().tolist(),
        result.boxes.cls.cpu().tolist(),
    ):
        class_id = int(class_id)
        if class_id in tool_ids:
            rows.append({"box_xyxy": [float(value) for value in box], "confidence": float(score)})
    return rows


def main() -> None:
    args = parse_args()
    devices = {item.strip() for item in str(args.device).split(",") if item.strip()}
    if not devices <= {"6", "8", "cpu"}:
        raise SystemExit("Only physical GPU ids 6 and 8 (or cpu for a smoke test) are allowed.")
    args.output.mkdir(parents=True, exist_ok=True)
    overlay_dir = args.output / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    workpiece_model = YOLO(str(args.workpiece_model))
    wrench_model = YOLO(str(args.wrench_model))
    records: list[dict] = []
    images = sorted(path for path in args.images.rglob("*") if path.suffix.lower() in IMAGE_EXTS)
    for image in images:
        workpiece_result = workpiece_model.predict(
            source=str(image), conf=args.workpiece_conf, imgsz=args.imgsz, device=args.device, verbose=False
        )[0]
        wrench_result = wrench_model.predict(
            source=str(image), conf=args.wrench_conf, imgsz=args.imgsz, device=args.device, verbose=False
        )[0]
        record = {
            "image": str(image.resolve()),
            "candidates": {
                "workpiece": _pick_tool_class(workpiece_model, workpiece_result),
                "wrench": _pick_tool_class(wrench_model, wrench_result),
            },
            "polygon_required": True,
            "status": "candidate_boxes_only",
        }
        records.append(record)
        workpiece_result.save(filename=str(overlay_dir / f"{image.stem}_workpiece.jpg"))
        wrench_result.save(filename=str(overlay_dir / f"{image.stem}_wrench.jpg"))

    (args.output / "candidates.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "README.txt").write_text(
        "候選框僅供人工修成 workpiece/wrench 多邊形；不能直接拿來訓練 YOLO-seg。\n"
        "主工件模型的 tool → workpiece；額外 yolo11n_tool 的 tool → wrench。\n",
        encoding="utf-8",
    )
    print(f"Processed {len(images)} images. Candidate JSON: {args.output / 'candidates.json'}")


if __name__ == "__main__":
    main()
