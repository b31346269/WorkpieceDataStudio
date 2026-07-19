from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from threading import RLock
from typing import Any, BinaryIO

from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT / "workspace"
PROJECTS = WORKSPACE / "projects"
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
EXPECTED_CLASSES = ["hole", "screw", "tool"]
REQUIRED_QUALITY_CHECKS = (
    "workpiece_geometry",
    "holes_realistic",
    "screws_realistic",
    "tool_separate",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-").lower()
    return value[:48] or "project"


class ProjectStore:
    def __init__(self) -> None:
        PROJECTS.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def list_projects(self) -> list[dict[str, Any]]:
        projects: list[dict[str, Any]] = []
        for path in sorted(PROJECTS.glob("*/project.json")):
            try:
                projects.append(self._read_json(path))
            except (OSError, json.JSONDecodeError):
                continue
        return projects

    def create_project(self, name: str, classes: list[str]) -> dict[str, Any]:
        project_id = f"{slugify(name)}-{uuid.uuid4().hex[:8]}"
        root = self.project_root(project_id)
        for child in (
            "references",
            "candidates",
            "metadata",
            "imports",
            "exports",
            "models",
        ):
            (root / child).mkdir(parents=True, exist_ok=True)
        project = {
            "id": project_id,
            "name": name.strip(),
            "classes": classes,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "source_dataset": None,
        }
        self._write_json(root / "project.json", project)
        return project

    def get_project(self, project_id: str) -> dict[str, Any]:
        return self._read_json(self.project_root(project_id) / "project.json")

    def save_reference(
        self,
        project_id: str,
        filename: str,
        source: BinaryIO,
    ) -> dict[str, Any]:
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image format: {suffix or '(none)'}")
        reference_id = uuid.uuid4().hex
        safe_name = f"{reference_id}{suffix}"
        path = self.project_root(project_id) / "references" / safe_name
        with path.open("wb") as target:
            shutil.copyfileobj(source, target)
        try:
            with Image.open(path) as image:
                width, height = image.size
                image.verify()
        except Exception:
            path.unlink(missing_ok=True)
            raise ValueError("The uploaded file is not a valid image.")
        metadata = {
            "id": reference_id,
            "filename": safe_name,
            "original_name": Path(filename).name,
            "width": width,
            "height": height,
            "created_at": utc_now(),
        }
        self._write_json(
            self.project_root(project_id) / "metadata" / f"reference-{reference_id}.json",
            metadata,
        )
        return metadata

    def save_yolo_model(
        self,
        project_id: str,
        filename: str,
        source: BinaryIO,
    ) -> dict[str, Any]:
        if Path(filename).suffix.lower() != ".pt":
            raise ValueError("The YOLO model must be an Ultralytics .pt file.")
        model_id = uuid.uuid4().hex
        safe_name = f"{model_id}.pt"
        path = self.project_root(project_id) / "models" / safe_name
        with path.open("wb") as target:
            shutil.copyfileobj(source, target)
        if path.stat().st_size < 1024:
            path.unlink(missing_ok=True)
            raise ValueError("The uploaded .pt file is too small to be a model.")
        digest = self._sha256_file(path)
        for existing in self.list_yolo_models(project_id):
            existing_path = self.project_root(project_id) / "models" / existing["filename"]
            if not existing_path.is_file():
                continue
            existing_digest = existing.get("sha256") or self._sha256_file(existing_path)
            if existing_digest == digest:
                path.unlink(missing_ok=True)
                return {**existing, "duplicate": True}
        metadata = {
            "id": model_id,
            "filename": safe_name,
            "original_name": Path(filename).name,
            "size": path.stat().st_size,
            "sha256": digest,
            "created_at": utc_now(),
        }
        self._write_json(
            self.project_root(project_id) / "metadata" / f"model-{model_id}.json",
            metadata,
        )
        return metadata

    def list_yolo_models(self, project_id: str) -> list[dict[str, Any]]:
        root = self.project_root(project_id) / "metadata"
        return [self._read_json(path) for path in sorted(root.glob("model-*.json"))]

    def get_yolo_model(self, project_id: str, model_id: str) -> Path:
        metadata = self._read_json(
            self.project_root(project_id) / "metadata" / f"model-{model_id}.json"
        )
        return self.project_root(project_id) / "models" / metadata["filename"]

    def delete_yolo_model(self, project_id: str, model_id: str) -> dict[str, Any]:
        if not re.fullmatch(r"[a-f0-9]{32}", model_id):
            raise ValueError("Invalid model id.")
        root = self.project_root(project_id)
        metadata_path = root / "metadata" / f"model-{model_id}.json"
        metadata = self._read_json(metadata_path)
        model_path = root / "models" / metadata["filename"]
        with self._lock:
            model_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
        return metadata

    def list_references(self, project_id: str) -> list[dict[str, Any]]:
        root = self.project_root(project_id) / "metadata"
        return [
            self._read_json(path)
            for path in sorted(root.glob("reference-*.json"))
        ]

    def get_reference(self, project_id: str, reference_id: str) -> tuple[dict[str, Any], Path]:
        metadata = self._read_json(
            self.project_root(project_id) / "metadata" / f"reference-{reference_id}.json"
        )
        return metadata, self.project_root(project_id) / "references" / metadata["filename"]

    def import_yolo_zip(
        self,
        project_id: str,
        filename: str,
        source: BinaryIO,
    ) -> dict[str, Any]:
        root = self.project_root(project_id)
        destination = root / "imports" / "source-yolov8.zip"
        with destination.open("wb") as target:
            shutil.copyfileobj(source, target)
        try:
            summary = self.inspect_yolo_zip(destination)
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        project = self.get_project(project_id)
        project["source_dataset"] = {
            "filename": Path(filename).name,
            "stored_as": destination.name,
            **summary,
        }
        project["updated_at"] = utc_now()
        self._write_json(root / "project.json", project)
        return project["source_dataset"]

    def inspect_yolo_zip(self, path: Path) -> dict[str, Any]:
        image_count = 0
        label_count = 0
        box_count = 0
        class_ids: set[int] = set()
        invalid_rows: list[str] = []
        with zipfile.ZipFile(path, "r") as archive:
            for info in archive.infolist():
                pure = PurePosixPath(info.filename.replace("\\", "/"))
                if pure.is_absolute() or ".." in pure.parts:
                    raise ValueError(f"Unsafe ZIP entry: {info.filename}")
                suffix = pure.suffix.lower()
                parts = {part.lower() for part in pure.parts}
                if suffix in ALLOWED_IMAGE_SUFFIXES and "images" in parts:
                    image_count += 1
                if suffix == ".txt" and "labels" in parts:
                    label_count += 1
                    text = archive.read(info).decode("utf-8-sig", errors="replace")
                    for line_number, raw in enumerate(text.splitlines(), 1):
                        row = raw.strip().split()
                        if not row:
                            continue
                        try:
                            class_id = int(float(row[0]))
                            values = [float(value) for value in row[1:]]
                        except ValueError:
                            invalid_rows.append(f"{info.filename}:{line_number}")
                            continue
                        if len(values) not in (4,) and len(values) < 6:
                            invalid_rows.append(f"{info.filename}:{line_number}")
                            continue
                        class_ids.add(class_id)
                        box_count += 1
        if image_count == 0:
            raise ValueError("No images were found under an images/ folder.")
        if label_count == 0:
            raise ValueError("No YOLO label files were found under a labels/ folder.")
        return {
            "image_count": image_count,
            "label_count": label_count,
            "annotation_count": box_count,
            "class_ids": sorted(class_ids),
            "invalid_rows": invalid_rows[:50],
            "inspected_at": utc_now(),
        }

    def create_candidate(
        self,
        project_id: str,
        image: Image.Image,
        generation: dict[str, Any],
        boxes: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        candidate_id = uuid.uuid4().hex
        filename = f"{candidate_id}.jpg"
        root = self.project_root(project_id)
        image_path = root / "candidates" / filename
        image.convert("RGB").save(image_path, "JPEG", quality=96, subsampling=0)
        metadata = {
            "id": candidate_id,
            "filename": filename,
            "width": image.width,
            "height": image.height,
            "status": "pending",
            "boxes": boxes or [],
            "quality_checks": {
                name: False for name in REQUIRED_QUALITY_CHECKS
            },
            "generation": generation,
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
        self._write_json(root / "metadata" / f"candidate-{candidate_id}.json", metadata)
        return metadata

    def list_candidates(self, project_id: str) -> list[dict[str, Any]]:
        root = self.project_root(project_id) / "metadata"
        candidates = [
            self._read_json(path)
            for path in sorted(root.glob("candidate-*.json"))
        ]
        candidates.sort(key=lambda item: item["created_at"])
        return candidates

    def get_candidate(self, project_id: str, candidate_id: str) -> dict[str, Any]:
        return self._read_json(
            self.project_root(project_id) / "metadata" / f"candidate-{candidate_id}.json"
        )

    def update_candidate(
        self,
        project_id: str,
        candidate_id: str,
        status: str,
        boxes: list[dict[str, Any]],
        quality_checks: dict[str, bool] | None = None,
    ) -> dict[str, Any]:
        metadata = self.get_candidate(project_id, candidate_id)
        metadata["status"] = status
        metadata["boxes"] = boxes
        metadata["quality_checks"] = {
            name: bool((quality_checks or {}).get(name, False))
            for name in REQUIRED_QUALITY_CHECKS
        }
        metadata["updated_at"] = utc_now()
        self._write_json(
            self.project_root(project_id) / "metadata" / f"candidate-{candidate_id}.json",
            metadata,
        )
        return metadata

    def candidate_image(self, project_id: str, candidate_id: str) -> Path:
        metadata = self.get_candidate(project_id, candidate_id)
        return self.project_root(project_id) / "candidates" / metadata["filename"]

    def delete_project(self, project_id: str) -> dict[str, Any]:
        root = self.project_root(project_id)
        project = self._read_json(root / "project.json")
        with self._lock:
            shutil.rmtree(root)
        return project

    def project_root(self, project_id: str) -> Path:
        if not re.fullmatch(r"[a-z0-9_-]+", project_id):
            raise ValueError("Invalid project id.")
        root = (PROJECTS / project_id).resolve()
        projects_root = PROJECTS.resolve()
        if projects_root not in root.parents:
            raise ValueError("Invalid project path.")
        return root

    def _read_json(self, path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write_json(self, path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=path.parent,
                delete=False,
                suffix=".tmp",
            ) as handle:
                json.dump(value, handle, ensure_ascii=False, indent=2)
                temp_path = Path(handle.name)
            temp_path.replace(path)

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
