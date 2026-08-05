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
                candidate = self.store.create_candidate(
                    project_id,
                    image,
                    generation,
                    boxes,
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
