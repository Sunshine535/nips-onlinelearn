#!/usr/bin/env bash
set -euo pipefail

_PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$_PROJ_ROOT/.venv/bin/activate" ]; then source "$_PROJ_ROOT/.venv/bin/activate"; fi
export PATH="$HOME/.local/bin:$PATH"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="${PROJECT_DIR}/configs/spm_config.yaml"

cd "$PROJECT_DIR"
mkdir -p outputs/spm_training outputs/eval

echo "========================================"
echo "  SPM: Streaming Parameter Memory"
echo "  Config: $CONFIG"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "========================================"

echo "=== Step 1: Train SPM ==="
python3 scripts/train_spm.py \
    --config "$CONFIG" \
    --output_dir outputs \
    --num_sessions 100 \
    2>&1 | tee outputs/spm_training/train.log

echo ""
echo "=== Step 2: Evaluate SPM ==="
python3 scripts/eval_spm.py \
    --config "$CONFIG" \
    --model_dir outputs/spm_training/final \
    --output_dir outputs/eval \
    2>&1 | tee outputs/eval/eval.log

echo ""
echo "=== Pipeline complete ==="
