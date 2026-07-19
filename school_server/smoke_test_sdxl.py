from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from workpiece_studio.generation import GenerationSettings, generate_image
from workpiece_studio.prelabel import prelabeler
from workpiece_studio.storage import ProjectStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("strict", "shape_variation"),
        default="strict",
    )
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = ProjectStore()
    projects = store.list_projects()
    if not projects:
        raise RuntimeError("No project is available for the SDXL smoke test.")
    project = projects[0]
    references = store.list_references(project["id"])
    models = store.list_yolo_models(project["id"])
    if not references:
        raise RuntimeError("No reference image is available.")
    if not models:
        raise RuntimeError("best.pt is not available.")

    _, reference_path = store.get_reference(project["id"], references[0]["id"])
    shape_mode = args.mode == "shape_variation"
    settings = GenerationSettings(
        prompt=(
            "Top-down realistic smartphone photograph of a "
            + ("different redesigned industrial motor housing" if shape_mode else "industrial motor housing")
            + ", cast aluminum, mechanically plausible circular threaded holes "
            "and correctly seated screws, natural laboratory lighting"
        ),
        negative_prompt="",
        strength=0.64 if shape_mode else 0.30,
        ip_adapter_scale=0.60 if shape_mode else 0.82,
        guidance_scale=6.0,
        steps=40 if shape_mode else 36,
        seed=20260719,
        framing="focus_crop",
        quality_mode=args.mode,
    )
    image, runtime = generate_image("sdxl_controlnet", reference_path, settings)
    model_path = store.get_yolo_model(project["id"], models[0]["id"])
    boxes = prelabeler.predict(model_path, image, project["classes"], 0.18)
    output_name = args.output or (
        "sdxl-shape-smoke.jpg" if shape_mode else "sdxl-smoke.jpg"
    )
    output = ROOT / "workspace" / Path(output_name).name
    image.save(output, "JPEG", quality=96, subsampling=0)
    print(
        json.dumps(
            {
                "output": str(output),
                "width": image.width,
                "height": image.height,
                "reference": references[0]["original_name"],
                "prelabel_model": models[0]["original_name"],
                "box_count": len(boxes),
                "boxes": boxes,
                "runtime": runtime,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
