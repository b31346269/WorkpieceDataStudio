from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


ReviewStatus = Literal["pending", "approved", "rejected"]


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    classes: list[str] = Field(default_factory=lambda: ["hole", "screw", "tool"])

    @field_validator("classes")
    @classmethod
    def validate_classes(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        if not cleaned or len(set(cleaned)) != len(cleaned):
            raise ValueError("Class names must be non-empty and unique.")
        return cleaned


class BoxAnnotation(BaseModel):
    id: str
    class_id: int = Field(ge=0)
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float | None = Field(default=None, ge=0, le=1)
    source: Literal["model", "manual"] = "manual"


class CandidateUpdate(BaseModel):
    status: ReviewStatus
    boxes: list[BoxAnnotation]
    quality_checks: dict[str, bool] = Field(default_factory=dict)


class GenerationRequest(BaseModel):
    reference_id: str
    prompt: str = Field(min_length=3, max_length=6000)
    negative_prompt: str = Field(default="", max_length=6000)
    count: int = Field(default=4, ge=1, le=100)
    seed: int | None = None
    strength: float = Field(default=0.34, ge=0.05, le=0.95)
    ip_adapter_scale: float = Field(default=0.82, ge=0, le=1.5)
    guidance_scale: float = Field(default=6.5, ge=0, le=20)
    steps: int = Field(default=32, ge=4, le=100)
    quality_mode: Literal[
        "strict",
        "shape_variation",
        "balanced",
        "creative",
    ] = "strict"
    scene_preset: Literal[
        "factory_mixed",
        "assembly_line",
        "machine_enclosure",
        "maintenance_bench",
        "conveyor_fixture",
        "warehouse_inspection",
        "custom",
    ] = "factory_mixed"
    framing: Literal["focus_crop", "letterbox"] = "focus_crop"
    provider: Literal[
        "diffusers",
        "sdxl_controlnet",
        "flux2_klein",
        "mock",
    ] = "diffusers"
    prelabel: bool = True
    confidence: float = Field(default=0.18, ge=0.01, le=0.99)
    yolo_model_id: str = ""


class ExportRequest(BaseModel):
    split_name: str = Field(default="train", pattern=r"^[A-Za-z0-9_-]+$")
