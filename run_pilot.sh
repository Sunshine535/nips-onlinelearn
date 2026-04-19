#!/bin/bash
# =============================================================================
# Pilot: Tool-Use Forgetting phenomenon validation
# 3 conditions in parallel on 3 GPUs, ~8-12 hours total on Qwen3.5-0.5B
# =============================================================================
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ_DIR"

export HF_HOME="${HF_HOME:-$(dirname "$PROJ_DIR")/.cache/hf}"
export TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=1

MODEL="${MODEL:-Qwen/Qwen3.5-0.5B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJ_DIR/outputs/pilot}"
LOG_DIR="$PROJ_DIR/logs"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

if [ -f "$PROJ_DIR/.venv/bin/activate" ]; then
    source "$PROJ_DIR/.venv/bin/activate"
fi

echo "============================================================"
echo " Pilot: Tool-Use Forgetting validation"
echo " Model : $MODEL"
echo " Output: $OUTPUT_DIR"
echo " Time  : $(date)"
echo "============================================================"

# GPU 0: Baseline (no fine-tune, just baseline eval)
CUDA_VISIBLE_DEVICES=0 nohup python3 scripts/pilot_tool_forgetting.py \
    --condition baseline --model "$MODEL" --gpu 0 \
    --output_dir "$OUTPUT_DIR" \
    > "$LOG_DIR/pilot_baseline.log" 2>&1 &
PID_BASE=$!

# GPU 1: Drift (Alpaca fine-tune)
CUDA_VISIBLE_DEVICES=1 nohup python3 scripts/pilot_tool_forgetting.py \
    --condition alpaca --model "$MODEL" --gpu 0 \
    --output_dir "$OUTPUT_DIR" --max_steps 500 --eval_every 50 \
    > "$LOG_DIR/pilot_alpaca.log" 2>&1 &
PID_DRIFT=$!

# GPU 2: NLU control (commonsense_qa fine-tune, matched size)
CUDA_VISIBLE_DEVICES=2 nohup python3 scripts/pilot_tool_forgetting.py \
    --condition nlu_control --model "$MODEL" --gpu 0 \
    --output_dir "$OUTPUT_DIR" --max_steps 500 --eval_every 50 \
    > "$LOG_DIR/pilot_nlu_control.log" 2>&1 &
PID_CTRL=$!

echo "Launched 3 pilot conditions on 3 GPUs"
echo "  PID $PID_BASE  : baseline (GPU 0)"
echo "  PID $PID_DRIFT : alpaca drift (GPU 1)"
echo "  PID $PID_CTRL  : nlu_control (GPU 2)"
echo ""
echo "Check progress with:"
echo "  tail -f logs/pilot_alpaca.log"
echo ""
echo "Waiting for all jobs..."

FAILED=0
for pid in $PID_BASE $PID_DRIFT $PID_CTRL; do
    wait "$pid" 2>/dev/null || FAILED=$((FAILED + 1))
done

if [ $FAILED -gt 0 ]; then
    echo "[WARN] $FAILED/3 jobs failed. Check logs/"
    exit 1
fi

echo "[OK] All 3 conditions done. Running analysis..."
echo ""
python3 scripts/pilot_analyze.py --input_dir "$OUTPUT_DIR"

echo ""
echo "Pilot done: $(date)"
