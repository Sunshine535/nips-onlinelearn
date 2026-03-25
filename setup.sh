#!/bin/bash
set -e
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo " Environment Setup"
echo "============================================"

PIP_MIRROR="https://mirrors.aliyun.com/pypi/simple/"
PYTORCH_INDEX="https://download.pytorch.org/whl/cu128"

# --- Detect existing PyTorch + CUDA ---
SYSTEM_TORCH_OK=0
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    SYSTEM_TORCH_OK=1
    echo "[INFO] System Python already has PyTorch + CUDA"
    python3 -c "import torch; print(f'  PyTorch={torch.__version__}  CUDA={torch.version.cuda}  GPUs={torch.cuda.device_count()}')"
fi

_pip_install() {
    if command -v uv &>/dev/null && [ -n "$VIRTUAL_ENV" ]; then
        uv pip install "$@"
    else
        pip3 install "$@"
    fi
}

if [ "$SYSTEM_TORCH_OK" -eq 1 ] && [ -z "$FORCE_VENV" ]; then
    # --- Server mode: system Python already has torch+CUDA, just install missing deps ---
    echo "[1/3] Using system Python (torch+CUDA detected)"
    echo "[2/3] Installing project dependencies ..."
    _pip_install -r "$PROJ_DIR/requirements.txt" -i "$PIP_MIRROR" 2>&1 | tail -5
    echo "[3/3] Installing flash-attn (optional) ..."
    _pip_install flash-attn --no-build-isolation 2>/dev/null || echo "  flash-attn skipped (optional)"
else
    # --- Local/fresh mode: create venv and install everything ---
    if ! command -v uv &>/dev/null; then
        echo "[1/5] Installing uv ..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    else
        echo "[1/5] uv already installed: $(uv --version)"
    fi

    VENV_DIR="$PROJ_DIR/.venv"
    if [ ! -d "$VENV_DIR" ]; then
        echo "[2/5] Creating Python venv ..."
        uv venv "$VENV_DIR" --python 3.12 2>/dev/null \
            || uv venv "$VENV_DIR" --python 3.10 2>/dev/null \
            || uv venv "$VENV_DIR"
    else
        echo "[2/5] Venv exists: $VENV_DIR"
    fi
    source "$VENV_DIR/bin/activate"

    echo "[3/5] Installing PyTorch + CUDA ..."
    uv pip install torch torchvision torchaudio --index-url "$PYTORCH_INDEX"

    echo "[4/5] Installing project dependencies ..."
    uv pip install -r "$PROJ_DIR/requirements.txt" \
        --index-url "$PIP_MIRROR" \
        --extra-index-url "$PYTORCH_INDEX" \
        --index-strategy unsafe-best-match

    echo "[5/5] Installing flash-attn (optional) ..."
    uv pip install flash-attn --no-build-isolation 2>/dev/null || echo "  flash-attn skipped (optional)"
fi

# --- Verify ---
echo ""
echo "============================================"
python3 -c "
import torch, transformers, peft
print(f'  PyTorch      : {torch.__version__}')
print(f'  CUDA         : {torch.version.cuda}')
print(f'  GPUs         : {torch.cuda.device_count()}')
print(f'  transformers : {transformers.__version__}')
print(f'  peft         : {peft.__version__}')
for i in range(torch.cuda.device_count()):
    print(f'    GPU {i}: {torch.cuda.get_device_name(i)}')
"
echo "============================================"
echo ""
echo "Setup complete!"
echo "  Run: bash run.sh"
