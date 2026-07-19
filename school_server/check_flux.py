from __future__ import annotations

import importlib.util

import diffusers
import huggingface_hub
import torch
import transformers
from huggingface_hub import HfApi


MODEL_IDS = (
    "black-forest-labs/FLUX.2-klein-9b-fp8",
    "black-forest-labs/FLUX.2-klein-9B",
)


def main() -> None:
    for model_id in MODEL_IDS:
        model = HfApi().model_info(model_id, files_metadata=True)
        files = [
            (item.rfilename, item.size or 0)
            for item in model.siblings
            if item.rfilename.endswith(
                (".json", ".safetensors", ".model", ".txt")
            )
        ]
        total_bytes = sum(size for _, size in files)
        print(f"model={model.id}")
        print(f"gated={model.gated}")
        print(f"selected_files={len(files)}")
        print(f"selected_size_gb={total_bytes / 1024**3:.2f}")
        for name, size in sorted(files, key=lambda item: item[1], reverse=True)[:12]:
            print(f"{size / 1024**3:7.2f} GB  {name}")
    print(f"diffusers={diffusers.__version__}")
    print(f"transformers={transformers.__version__}")
    print(f"huggingface_hub={huggingface_hub.__version__}")
    print(f"torch={torch.__version__}")
    print(f"cuda={torch.cuda.is_available()}")
    print(
        "flux2_klein_pipeline="
        f"{importlib.util.find_spec('diffusers.pipelines.flux2') is not None}"
    )


if __name__ == "__main__":
    main()
