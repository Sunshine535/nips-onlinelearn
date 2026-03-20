#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="${PROJECT_DIR}/configs/spm_config.yaml"

cd "$PROJECT_DIR"
mkdir -p outputs/spm_training outputs/eval

echo "========================================"
echo "  SPM: Streaming Parameter Memory"
echo "  Model: Qwen/Qwen3.5-9B"
echo "  GPUs: 8x A100-80GB"
echo "========================================"

echo "=== Step 1: Train SPM ==="
torchrun \
    --nproc_per_node=8 \
    --master_port=29700 \
    scripts/train_spm.py \
        --config "$CONFIG" \
        --output_dir outputs \
        --num_sessions 100 \
    2>&1 | tee outputs/spm_training/train.log

echo ""
echo "=== Step 2: Evaluate SPM ==="
python scripts/eval_spm.py \
    --config "$CONFIG" \
    --model_dir outputs/spm_training/final \
    --output_dir outputs/eval \
    2>&1 | tee outputs/eval/eval.log

echo ""
echo "=== Pipeline complete ==="
