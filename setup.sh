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
    # Also verify transformers is new enough for Qwen3.5
    TRANSFORMERS_OK=0
    python3 -c "
import transformers
v = tuple(int(x) for x in transformers.__version__.split('.')[:2])
assert v >= (4, 48)
" 2>/dev/null && TRANSFORMERS_OK=1

    if [ "$TRANSFORMERS_OK" -eq 1 ]; then
        SYSTEM_TORCH_OK=1
        echo "[INFO] System Python has PyTorch + CUDA + compatible transformers"
        python3 -c "import torch, transformers; print(f'  PyTorch={torch.__version__}  CUDA={torch.version.cuda}  GPUs={torch.cuda.device_count()}  transformers={transformers.__version__}')"
    else
        echo "[INFO] System has torch+CUDA but transformers is too old for Qwen3.5"
        echo "[INFO] Will create isolated venv with up-to-date packages"
        export FORCE_VENV=1
    fi
fi

_pip_install() {
    if command -v uv &>/dev/null && [ -n "$VIRTUAL_ENV" ]; then
        uv pip install "$@"
    else
        pip3 install --break-system-packages "$@"
    fi
}

if [ "$SYSTEM_TORCH_OK" -eq 1 ] && [ -z "$FORCE_VENV" ]; then
    # --- Server mode: system Python already has torch+CUDA, just install missing deps ---
    echo "[1/3] Using system Python (torch+CUDA detected)"
    echo "[2/3] Installing project dependencies ..."
    _pip_install -r "$PROJ_DIR/requirements.txt" -i "$PIP_MIRROR" 2>&1 | tail -5
    echo "[3/3] Installing optional accelerators ..."
    _pip_install flash-attn --no-build-isolation 2>/dev/null || echo "  flash-attn skipped"
    _pip_install flash-linear-attention causal-conv1d --no-build-isolation 2>/dev/null || echo "  flash-linear-attention skipped"
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

    echo "[3/5] Installing PyTorch CUDA + project deps ..."
    uv pip install torch torchvision torchaudio \
        -r "$PROJ_DIR/requirements.txt" \
        --index-url "$PYTORCH_INDEX" \
        --extra-index-url "$PIP_MIRROR" \
        --index-strategy unsafe-best-match

    echo "[5/5] Installing optional accelerators ..."
    uv pip install flash-attn --no-build-isolation 2>/dev/null || echo "  flash-attn skipped"
    uv pip install flash-linear-attention causal-conv1d --no-build-isolation 2>/dev/null || echo "  flash-linear-attention skipped"
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
