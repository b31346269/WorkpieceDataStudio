from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .storage import REQUIRED_QUALITY_CHECKS, ProjectStore, utc_now


def _quote_yaml(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def export_yolov8(
    store: ProjectStore,
    project_id: str,
    split_name: str = "train",
) -> tuple[Path, dict[str, Any]]:
    project = store.get_project(project_id)
    candidates = [
        candidate
        for candidate in store.list_candidates(project_id)
        if candidate["status"] == "approved"
        and candidate.get("generation", {}).get("training_eligible", True)
        and all(
            candidate.get("quality_checks", {}).get(name, False)
            for name in REQUIRED_QUALITY_CHECKS
        )
    ]
    if not candidates:
        raise ValueError(
            "No approved real-generation candidates are available. "
            "Mock previews are intentionally excluded."
        )

    exports = store.project_root(project_id) / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output_path = exports / f"{project_id}-{stamp}.yolov8.zip"

    with tempfile.TemporaryDirectory(prefix="export-", dir=exports) as temporary:
        root = Path(temporary)
        images_dir = root / split_name / "images"
        labels_dir = root / split_name / "labels"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        manifest_items: list[dict[str, Any]] = []

        for index, candidate in enumerate(candidates, 1):
            stem = f"generated_{index:06d}_{candidate['id'][:8]}"
            source_image = store.candidate_image(project_id, candidate["id"])
            image_name = f"{stem}{source_image.suffix.lower()}"
            shutil.copy2(source_image, images_dir / image_name)
            label_lines: list[str] = []
            width = float(candidate["width"])
            height = float(candidate["height"])
            for box in candidate["boxes"]:
                x1 = max(0.0, min(width, float(box["x1"])))
                y1 = max(0.0, min(height, float(box["y1"])))
                x2 = max(0.0, min(width, float(box["x2"])))
                y2 = max(0.0, min(height, float(box["y2"])))
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue
                center_x = ((x1 + x2) / 2) / width
                center_y = ((y1 + y2) / 2) / height
                box_width = (x2 - x1) / width
                box_height = (y2 - y1) / height
                label_lines.append(
                    f"{int(box['class_id'])} {center_x:.8f} {center_y:.8f} "
                    f"{box_width:.8f} {box_height:.8f}"
                )
            (labels_dir / f"{stem}.txt").write_text(
                "\n".join(label_lines) + ("\n" if label_lines else ""),
                encoding="utf-8",
            )
            manifest_items.append(
                {
                    "candidate_id": candidate["id"],
                    "image": f"{split_name}/images/{image_name}",
                    "label": f"{split_name}/labels/{stem}.txt",
                    "box_count": len(label_lines),
                    "generation": candidate["generation"],
                }
            )

        yaml_lines = [
            f"path: .",
            f"train: {split_name}/images",
            f"val: {split_name}/images",
            f"test: {split_name}/images",
            "names:",
        ]
        yaml_lines.extend(
            f"  {index}: {_quote_yaml(name)}"
            for index, name in enumerate(project["classes"])
        )
        (root / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
        manifest = {
            "project_id": project_id,
            "created_at": utc_now(),
            "classes": project["classes"],
            "candidate_count": len(manifest_items),
            "warning": (
                "Synthetic images belong in train only. Keep a separate real validation "
                "and test set when combining this export with the main dataset."
            ),
            "items": manifest_items,
        }
        (root / "generation_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        with zipfile.ZipFile(
            output_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            for path in root.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(root).as_posix())

    return output_path, {
        "filename": output_path.name,
        "candidate_count": len(candidates),
        "created_at": utc_now(),
    }
