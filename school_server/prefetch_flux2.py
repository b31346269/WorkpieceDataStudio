from __future__ import annotations

import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from workpiece_studio.generation import _flux2_klein_generator


def main() -> None:
    snapshot_download(
        repo_id=_flux2_klein_generator.MODEL_ID,
        allow_patterns=[
            "model_index.json",
            "scheduler/**",
            "tokenizer/**",
            "text_encoder/**",
            "transformer/**",
            "vae/**",
        ],
    )
    _flux2_klein_generator._load()
    print(
        json.dumps(
            _flux2_klein_generator.runtime,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
