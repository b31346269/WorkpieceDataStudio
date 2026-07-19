from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from workpiece_studio.generation import GenerationSettings, generate_image
from workpiece_studio.prelabel import prelabeler
from workpiece_studio.storage import ProjectStore


def main() -> None:
    store = ProjectStore()
    projects = store.list_projects()
    if not projects:
        raise RuntimeError("No project is available for the FLUX.2 smoke test.")
    project = projects[0]
    references = store.list_references(project["id"])
    models = store.list_yolo_models(project["id"])
    if not references:
        raise RuntimeError("No reference image is available.")
    if not models:
        raise RuntimeError("best.pt is not available.")

    _, reference_path = store.get_reference(project["id"], references[0]["id"])
    settings = GenerationSettings(
        prompt=(
            "A realistic top-down smartphone photograph of a substantially "
            "redesigned industrial motor housing. Use a wider asymmetric cast "
            "aluminum body, a different outer silhouette, repositioned mounting "
            "ears, new cooling fins and a new cavity arrangement. Preserve natural "
            "machining marks, physically drilled circular holes and rigid fasteners."
        ),
        negative_prompt="",
        strength=0.64,
        ip_adapter_scale=0.60,
        guidance_scale=1.0,
        steps=4,
        seed=20260719,
        framing="focus_crop",
        quality_mode="shape_variation",
    )
    image, runtime = generate_image("flux2_klein", reference_path, settings)
    model_path = store.get_yolo_model(project["id"], models[0]["id"])
    boxes = prelabeler.predict(model_path, image, project["classes"], 0.18)
    output = ROOT / "workspace" / "flux2-shape-smoke.jpg"
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
