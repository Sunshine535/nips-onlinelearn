#!/usr/bin/env bash
set -euo pipefail

_PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$_PROJ_ROOT/.venv/bin/activate" ]; then source "$_PROJ_ROOT/.venv/bin/activate"; fi
export PATH="$HOME/.local/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="${SPM_CONFIG:-${PROJECT_DIR}/configs/spm_config.yaml}"

source "$SCRIPT_DIR/gpu_utils.sh"
auto_setup

cd "$PROJECT_DIR"
mkdir -p outputs/spm_training outputs/eval logs

echo "========================================"
echo "  SPM: Streaming Parameter Memory"
echo "  Config: $CONFIG"
echo "  GPUs: $NUM_GPUS × $GPU_CLASS"
echo "========================================"

echo "=== Step 1: Train SPM ==="
GPU0=$(gpu_at_index 0)
CUDA_VISIBLE_DEVICES="$GPU0" python3 scripts/train_spm.py \
    --config "$CONFIG" \
    --output_dir outputs \
    --num_sessions 100 \
    --resume_from_checkpoint auto \
    2>&1 | tee logs/spm_train.log

echo ""
echo "=== Step 2: Evaluate SPM ==="
CUDA_VISIBLE_DEVICES="$GPU0" python3 scripts/eval_spm.py \
    --config "$CONFIG" \
    --model_dir outputs/spm_training/final \
    --output_dir outputs/eval \
    2>&1 | tee logs/spm_eval.log

echo ""
echo "=== Pipeline complete ==="
