#!/bin/bash
set -e
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo " Environment Setup (uv + PyTorch 2.10 + CUDA 12.8)"
echo "============================================"

PIP_MIRROR="https://pypi.org/simple/"
PYTORCH_INDEX="https://download.pytorch.org/whl/cu128"

# --- Detect available Python (conda, system, or uv-managed) ---
find_python() {
    for candidate in python3.12 python3.10 python3; do
        if command -v "$candidate" &>/dev/null; then
            echo "$candidate"
            return 0
        fi
    done
    if [ -n "${CONDA_PREFIX:-}" ]; then
        local conda_py="${CONDA_PREFIX}/bin/python"
        if [ -x "$conda_py" ]; then echo "$conda_py"; return 0; fi
    fi
    echo ""
    return 1
}

# --- Install uv if missing ---
UV_AVAILABLE=1
if ! command -v uv &>/dev/null; then
    echo "[1/5] Installing uv ..."
    if curl -LsSf https://astral.sh/uv/install.sh | sh 2>/dev/null; then
        export PATH="$HOME/.local/bin:$PATH"
    else
        echo "  uv install failed — falling back to pip"
        UV_AVAILABLE=0
    fi
else
    echo "[1/5] uv already installed: $(uv --version)"
fi

# --- Create venv ---
VENV_DIR="$PROJ_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[2/5] Creating Python venv ..."
    if [ "$UV_AVAILABLE" -eq 1 ]; then
        uv venv "$VENV_DIR" --python 3.12 2>/dev/null \
            || uv venv "$VENV_DIR" --python 3.10 2>/dev/null \
            || uv venv "$VENV_DIR" 2>/dev/null
    fi
    if [ ! -d "$VENV_DIR" ]; then
        PYTHON_BIN=$(find_python)
        if [ -z "$PYTHON_BIN" ]; then
            echo "[ERROR] No Python found. Install Python 3.10+ or conda."
            exit 1
        fi
        echo "  Using $PYTHON_BIN to create venv"
        "$PYTHON_BIN" -m venv "$VENV_DIR" || {
            echo "[ERROR] Failed to create venv with $PYTHON_BIN"
            exit 1
        }
    fi
else
    echo "[2/5] Venv exists: $VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# --- Install PyTorch CUDA + project dependencies ---
echo "[3/5] Installing PyTorch 2.10.0 + CUDA 12.8 + project deps ..."
if [ "$UV_AVAILABLE" -eq 1 ]; then
    uv pip install "torch==2.10.0" "torchvision" "torchaudio" \
        -r "$PROJ_DIR/requirements.txt" \
        --index-url "$PYTORCH_INDEX" \
        --extra-index-url "$PIP_MIRROR" \
        --index-strategy unsafe-best-match
else
    pip install --upgrade pip
    pip install "torch==2.10.0" "torchvision" "torchaudio" \
        --index-url "$PYTORCH_INDEX"
    pip install -r "$PROJ_DIR/requirements.txt"
fi

# --- Optional: flash-attention (skip if already attempted) ---
_FA_MARKER="$VENV_DIR/.flash_attn_attempted"
if [ ! -f "$_FA_MARKER" ]; then
    echo "[5/5] Installing flash-attn + flash-linear-attention (optional, first time only) ..."
    if [ "$UV_AVAILABLE" -eq 1 ]; then
        uv pip install flash-attn --no-build-isolation 2>/dev/null || echo "  flash-attn skipped"
        uv pip install flash-linear-attention causal-conv1d --no-build-isolation 2>/dev/null \
            || echo "  flash-linear-attention skipped"
    else
        pip install flash-attn --no-build-isolation 2>/dev/null || echo "  flash-attn skipped"
        pip install flash-linear-attention causal-conv1d --no-build-isolation 2>/dev/null \
            || echo "  flash-linear-attention skipped"
    fi
    touch "$_FA_MARKER"
else
    echo "[5/5] Flash-attn already attempted (skip rebuild)"
fi

# --- Verify ---
echo ""
echo "============================================"
python -c "
import torch
print(f'  PyTorch  : {torch.__version__}')
print(f'  CUDA     : {torch.version.cuda}')
print(f'  GPUs     : {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'    GPU {i}: {torch.cuda.get_device_name(i)}')
"
echo "============================================"
echo ""
echo "Setup complete!"
echo "  Activate:  source $VENV_DIR/bin/activate"
echo "  Run:       bash scripts/run_all_experiments.sh"
