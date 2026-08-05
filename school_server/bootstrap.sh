#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# This account may use physical GPU 2, 3, or 6 only. GPU 6 is the currently
# allocated card; keep it as the safe default for setup checks and training.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"

ENV_DIR="$ROOT/.school-env"
CONDA_BIN="${CONDA_BIN:-/opt/miniconda3/bin/conda}"
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  if [[ ! -x "$CONDA_BIN" ]]; then
    echo "Conda was not found at $CONDA_BIN." >&2
    exit 1
  fi
  "$CONDA_BIN" create --prefix "$ENV_DIR" python=3.11 pip -y
fi

PYTHON="$ENV_DIR/bin/python"
"$PYTHON" -m pip install --upgrade pip
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
if ! "$PYTHON" -c \
  "import torch; raise SystemExit(0 if torch.__version__.startswith('2.5.1+cu121') else 1)"; then
  "$PYTHON" -m pip install --upgrade --force-reinstall \
    torch==2.5.1 torchvision==0.20.1 --index-url "$TORCH_INDEX_URL"
fi
"$PYTHON" -m pip install -r requirements-core.txt -r requirements-ml.txt

"$PYTHON" - <<'PY'
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA ready: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Visible GPU: {torch.cuda.get_device_name(0)}")
PY

echo "School server environment is ready."
