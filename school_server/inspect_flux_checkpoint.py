from __future__ import annotations

from huggingface_hub import hf_hub_download
from safetensors import safe_open


MODEL_ID = "black-forest-labs/FLUX.2-klein-9b-fp8"
WEIGHT_NAME = "flux-2-klein-9b-fp8.safetensors"


def main() -> None:
    checkpoint = hf_hub_download(repo_id=MODEL_ID, filename=WEIGHT_NAME)
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        print(f"keys={len(keys)}")
        print(f"metadata={handle.metadata()}")
        for key in keys[:80]:
            tensor = handle.get_slice(key)
            print(f"{key}\tshape={tensor.get_shape()}\tdtype={tensor.get_dtype()}")


if __name__ == "__main__":
    main()
