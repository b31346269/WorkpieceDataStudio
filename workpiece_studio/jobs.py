from __future__ import annotations

import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import RLock
from typing import Any

from .generation import GenerationSettings, generate_image
from .prelabel import prelabeler
from .schemas import GenerationRequest
from .storage import ProjectStore, utc_now


AUTO_SCREEN_MAX_HOLES = 8
AUTO_SCREEN_MAX_SCREWS = 3
AUTO_SCREEN_MAX_TOTAL = 10
AUTO_SCREEN_ROI_MARGIN = 0.22


def screen_generated_boxes(
    boxes: list[dict[str, Any]],
    classes: list[str],
    width: int,
    height: int,
) -> dict[str, Any]:
    """Reject obviously dense center-workpiece annotations before manual review."""
    class_ids = {name: index for index, name in enumerate(classes)}
    hole_id = class_ids.get("hole")
    screw_id = class_ids.get("screw")
    if hole_id is None or screw_id is None or not boxes:
        return {
            "evaluated": False,
            "passed": True,
            "reason": "Pre-label boxes or expected classes were unavailable.",
        }

    min_x = width * AUTO_SCREEN_ROI_MARGIN
    max_x = width * (1.0 - AUTO_SCREEN_ROI_MARGIN)
    min_y = height * AUTO_SCREEN_ROI_MARGIN
    max_y = height * (1.0 - AUTO_SCREEN_ROI_MARGIN)
    center_boxes = []
    for box in boxes:
        center_x = (float(box["x1"]) + float(box["x2"])) / 2.0
        center_y = (float(box["y1"]) + float(box["y2"])) / 2.0
        if min_x <= center_x <= max_x and min_y <= center_y <= max_y:
            center_boxes.append(box)

    hole_count = sum(box["class_id"] == hole_id for box in center_boxes)
    screw_count = sum(box["class_id"] == screw_id for box in center_boxes)
    reasons = []
    if hole_count > AUTO_SCREEN_MAX_HOLES:
        reasons.append(
            f"central hole detections {hole_count} exceed {AUTO_SCREEN_MAX_HOLES}"
        )
    if screw_count > AUTO_SCREEN_MAX_SCREWS:
        reasons.append(
            f"central screw detections {screw_count} exceed {AUTO_SCREEN_MAX_SCREWS}"
        )
    total_count = hole_count + screw_count
    if total_count > AUTO_SCREEN_MAX_TOTAL:
        reasons.append(
            f"central hole and screw detections {total_count} exceed {AUTO_SCREEN_MAX_TOTAL}"
        )
    return {
        "evaluated": True,
        "passed": not reasons,
        "hole_count": hole_count,
        "screw_count": screw_count,
        "total_count": total_count,
        "max_holes": AUTO_SCREEN_MAX_HOLES,
        "max_screws": AUTO_SCREEN_MAX_SCREWS,
        "max_total": AUTO_SCREEN_MAX_TOTAL,
        "roi_margin": AUTO_SCREEN_ROI_MARGIN,
        "reason": "; ".join(reasons) if reasons else "Within count limits.",
    }


class GenerationJobs:
    def __init__(self, store: ProjectStore) -> None:
        self.store = store
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="generation")
        self.jobs: dict[str, dict[str, Any]] = {}
        self.lock = RLock()

    def submit(
        self,
        project_id: str,
        request: GenerationRequest,
    ) -> dict[str, Any]:
        self.store.get_project(project_id)
        self.store.get_reference(project_id, request.reference_id)
        if request.prelabel and request.yolo_model_id:
            self.store.get_yolo_model(project_id, request.yolo_model_id)
        job_id = uuid.uuid4().hex
        job = {
            "id": job_id,
            "project_id": project_id,
            "status": "queued",
            "completed": 0,
            "total": request.count,
            "candidate_ids": [],
            "warnings": [],
            "error": None,
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
        with self.lock:
            self.jobs[job_id] = job
        self.executor.submit(self._run, job_id, project_id, request)
        return dict(job)

    def get(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            return dict(self.jobs[job_id])

    def _patch(self, job_id: str, **updates: Any) -> None:
        with self.lock:
            self.jobs[job_id].update(updates)
            self.jobs[job_id]["updated_at"] = utc_now()

    def _run(
        self,
        job_id: str,
        project_id: str,
        request: GenerationRequest,
    ) -> None:
        self._patch(job_id, status="running")
        try:
            project = self.store.get_project(project_id)
            _, reference_path = self.store.get_reference(
                project_id,
                request.reference_id,
            )
            model_path = None
            if request.prelabel and request.yolo_model_id:
                model_path = self.store.get_yolo_model(
                    project_id,
                    request.yolo_model_id,
                )
            base_seed = request.seed if request.seed is not None else uuid.uuid4().int % 2_147_483_647
            candidate_ids: list[str] = []
            warnings: list[str] = []

            for index in range(request.count):
                seed = base_seed + index
                settings = GenerationSettings(
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt,
                    strength=request.strength,
                    ip_adapter_scale=request.ip_adapter_scale,
                    guidance_scale=request.guidance_scale,
                    steps=request.steps,
                    seed=seed,
                    framing=request.framing,
                    quality_mode=request.quality_mode,
                    scene_preset=request.scene_preset,
                )
                image, runtime = generate_image(
                    request.provider,
                    reference_path,
                    settings,
                )
                boxes: list[dict[str, Any]] = []
                if model_path is not None:
                    try:
                        boxes = prelabeler.predict(
                            model_path,
                            image,
                            project["classes"],
                            request.confidence,
                        )
                    except Exception as exc:
                        warnings.append(f"Pre-label failed for seed {seed}: {exc}")

                generation = {
                    "reference_id": request.reference_id,
                    "prompt": request.prompt,
                    "negative_prompt": request.negative_prompt,
                    "seed": seed,
                    "strength": request.strength,
                    "ip_adapter_scale": request.ip_adapter_scale,
                    "guidance_scale": request.guidance_scale,
                    "steps": request.steps,
                    "quality_mode": request.quality_mode,
                    "scene_preset": request.scene_preset,
                    "framing": request.framing,
                    "prelabel_model_id": request.yolo_model_id or None,
                    **runtime,
                }
                auto_screen = screen_generated_boxes(
                    boxes,
                    project["classes"],
                    image.width,
                    image.height,
                )
                generation["auto_screen"] = auto_screen
                candidate = self.store.create_candidate(
                    project_id,
                    image,
                    generation,
                    boxes,
                )
                if not auto_screen["passed"]:
                    candidate = self.store.update_candidate(
                        project_id,
                        candidate["id"],
                        "rejected",
                        boxes,
                        {},
                    )
                candidate_ids.append(candidate["id"])
                self._patch(
                    job_id,
                    completed=index + 1,
                    candidate_ids=list(candidate_ids),
                    warnings=list(warnings),
                )
            self._patch(job_id, status="completed")
        except Exception as exc:
            self._patch(
                job_id,
                status="failed",
                error=str(exc),
                traceback=traceback.format_exc(limit=12),
            )
