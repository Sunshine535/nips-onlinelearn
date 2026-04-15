#!/bin/bash
# =============================================================================
# Re-run all 13 methods with adapter-state fix (8-GPU parallel)
# Usage: bash run_fixed.sh
# =============================================================================
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ_DIR"

export HF_HOME="${HF_HOME:-$(dirname "$PROJ_DIR")/.cache/hf}"
export TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=1

MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
MAX_SESSIONS=50
OUTPUT_DIR="$PROJ_DIR/outputs/streaming_eval/Qwen_Qwen3.5-9B_fixed"
LOG_DIR="$PROJ_DIR/logs"
EVAL_SCRIPT="$PROJ_DIR/scripts/run_streaming_eval.py"

if [ -f "$PROJ_DIR/.venv/bin/activate" ]; then
    source "$PROJ_DIR/.venv/bin/activate"
fi

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "$NUM_GPUS" -eq 0 ] && NUM_GPUS=1

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

ALL_METHODS=(
    mirror_lora
    spm
    sdft
    single_lora_ewc_replay
    dual_lora_ema
    param_matched_lora
    single_lora_replay
    frozen
    single_lora_sgd
    single_lora_ewc
    dual_lora_heuristic
    dual_lora_periodic_avg
    retrieval_augmented
)

echo "============================================================"
echo " Re-run with adapter-state fix"
echo " Model   : $MODEL"
echo " GPUs    : $NUM_GPUS"
echo " Output  : $OUTPUT_DIR"
echo " Time    : $(date)"
echo "============================================================"

PIDS=()
gpu_idx=0
for method in "${ALL_METHODS[@]}"; do
    gpu=$((gpu_idx % NUM_GPUS))
    log="$LOG_DIR/fixed_${method}.log"
    echo "  GPU $gpu: $method -> $log"
    CUDA_VISIBLE_DEVICES=$gpu nohup python3 -u "$EVAL_SCRIPT" \
        --task dialogue --methods "$method" --model "$MODEL" \
        --output_dir "$OUTPUT_DIR" --max_sessions "$MAX_SESSIONS" \
        --no_nli --device cuda:0 \
        > "$log" 2>&1 &
    PIDS+=($!)
    gpu_idx=$((gpu_idx + 1))
done

echo ""
echo "Launched ${#PIDS[@]} methods on $NUM_GPUS GPUs"
echo "Waiting..."

FAILED=0
for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || FAILED=$((FAILED + 1))
done

if [ $FAILED -gt 0 ]; then
    echo "[WARN] $FAILED/${#PIDS[@]} jobs failed. Check logs/"
else
    echo "[OK] All jobs done."
fi

echo ""
echo "Collecting results..."
python3 "$PROJ_DIR/collect_results.py"
echo ""
echo "Done: $(date)"
