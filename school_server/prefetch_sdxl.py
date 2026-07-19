from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from workpiece_studio.generation import _sdxl_controlnet_generator


def main() -> None:
    _sdxl_controlnet_generator._load()
    print(
        json.dumps(
            _sdxl_controlnet_generator.runtime,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
