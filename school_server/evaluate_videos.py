from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO


def evaluate(model_path: Path, video_path: Path, output_dir: Path, device: str, conf: float) -> dict:
    model_name = model_path.parent.parent.name if model_path.parent.name == "weights" else model_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    video_output = output_dir / f"{model_name}__{video_path.stem}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = cv2.VideoWriter(
        str(video_output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    cap.release()

    class_names = {0: "hole", 1: "screw", 2: "tool"}
    class_counts: Counter[str] = Counter()
    confidence_sum = 0.0
    detections = 0
    frames = 0
    frames_with_detection = 0
    max_detections = 0
    model = YOLO(str(model_path))
    results = model.predict(
        source=str(video_path),
        stream=True,
        imgsz=640,
        conf=conf,
        iou=0.7,
        device=device,
        verbose=False,
    )
    writer = cv2.VideoWriter(
        str(video_output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for result in results:
        frames += 1
        boxes = result.boxes
        count = len(boxes) if boxes is not None else 0
        max_detections = max(max_detections, count)
        if count:
            frames_with_detection += 1
            for cls, score in zip(boxes.cls.tolist(), boxes.conf.tolist()):
                class_counts[class_names.get(int(cls), str(int(cls)))] += 1
                confidence_sum += float(score)
            detections += count
        writer.write(result.plot())
    writer.release()
    return {
        "model": model_name,
        "model_path": str(model_path),
        "video": str(video_path),
        "frames": frames,
        "frames_with_detection": frames_with_detection,
        "detection_frame_rate": frames_with_detection / frames if frames else 0.0,
        "total_detections": detections,
        "mean_detections_per_frame": detections / frames if frames else 0.0,
        "max_detections_in_frame": max_detections,
        "mean_confidence": confidence_sum / detections if detections else 0.0,
        "class_detection_counts": dict(class_counts),
        "confidence_threshold": conf,
        "output_video": str(video_output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run identical video-only evaluation for YOLO11n C/D/E.")
    parser.add_argument("--model", action="append", required=True, type=Path)
    parser.add_argument("--video", action="append", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("school_training/video_evaluation"))
    parser.add_argument("--device", default="6")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    summaries = []
    for model_path in args.model:
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        for video_path in args.video:
            summaries.append(evaluate(model_path, video_path, args.output, args.device, args.conf))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
