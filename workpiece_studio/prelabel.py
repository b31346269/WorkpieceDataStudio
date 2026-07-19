from __future__ import annotations

import os
import uuid
from pathlib import Path
from threading import RLock
from typing import Any

from PIL import Image


LOCAL_ULTRALYTICS_CONFIG = (
    Path(__file__).resolve().parent.parent / "workspace" / ".ultralytics"
)
LOCAL_ULTRALYTICS_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(LOCAL_ULTRALYTICS_CONFIG))
LOCAL_MATPLOTLIB_CONFIG = (
    Path(__file__).resolve().parent.parent / "workspace" / ".matplotlib"
)
LOCAL_MATPLOTLIB_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_MATPLOTLIB_CONFIG))


ALIASES = {
    "hole": "hole",
    "holes": "hole",
    "screw": "screw",
    "screws": "screw",
    "bolt": "screw",
    "bolts": "screw",
    "tool": "tool",
    "tools": "tool",
    "wrench": "tool",
    "spanner": "tool",
    "ratchet": "tool",
}


class YoloPrelabeler:
    def __init__(self) -> None:
        self._models: dict[str, Any] = {}
        self._lock = RLock()

    def predict(
        self,
        model_path: Path,
        image: Image.Image,
        project_classes: list[str],
        confidence: float,
    ) -> list[dict[str, Any]]:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed. Run setup.ps1 -WithML."
            ) from exc

        key = str(model_path.resolve())
        with self._lock:
            model = self._models.get(key)
            if model is None:
                model = YOLO(key)
                self._models[key] = model
            result = model.predict(
                source=image,
                imgsz=640,
                conf=confidence,
                rect=False,
                verbose=False,
            )[0]

        boxes: list[dict[str, Any]] = []
        if result.boxes is None:
            return boxes
        xyxy = result.boxes.xyxy.detach().cpu().tolist()
        classes = result.boxes.cls.detach().cpu().tolist()
        confidences = result.boxes.conf.detach().cpu().tolist()
        model_names = result.names
        normalized_project = {
            name.strip().lower(): index for index, name in enumerate(project_classes)
        }
        for coordinates, model_class, score in zip(xyxy, classes, confidences):
            raw_name = str(model_names[int(model_class)]).strip().lower()
            mapped_name = ALIASES.get(raw_name, raw_name)
            if mapped_name not in normalized_project:
                continue
            x1, y1, x2, y2 = coordinates
            boxes.append(
                {
                    "id": uuid.uuid4().hex,
                    "class_id": normalized_project[mapped_name],
                    "x1": max(0.0, min(float(image.width), float(x1))),
                    "y1": max(0.0, min(float(image.height), float(y1))),
                    "x2": max(0.0, min(float(image.width), float(x2))),
                    "y2": max(0.0, min(float(image.height), float(y2))),
                    "confidence": round(float(score), 6),
                    "source": "model",
                }
            )
        return boxes


prelabeler = YoloPrelabeler()
