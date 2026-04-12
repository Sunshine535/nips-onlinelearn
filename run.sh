#!/bin/bash
# =============================================================================
# Streaming Mirror-LoRA: Full 8-GPU Parallel Experiment Pipeline
#
# Usage:
#   bash run.sh                    # auto-detect GPUs, run everything
#   bash run.sh --model Qwen/Qwen3.5-9B --sessions 50
#   nohup bash run.sh > run.log 2>&1 &   # background
#
# Environment (set by caller or defaults):
#   HF_HOME       — HuggingFace cache dir (auto-set by caller)
#   NUM_GPUS      — override GPU count (default: auto-detect)
# =============================================================================
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ_DIR"

# --- HF cache: shared across projects in parent dir ---
export HF_HOME="${HF_HOME:-$(dirname "$PROJ_DIR")/.cache/hf}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME"

# ── Defaults ──
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
MODEL_SMALL="${MODEL_SMALL:-Qwen/Qwen3.5-0.8B}"
MAX_SESSIONS="${MAX_SESSIONS:-50}"
SESSIONS_PER_PERSONA="${SESSIONS_PER_PERSONA:-5}"
TURNS_PER_SESSION="${TURNS_PER_SESSION:-15}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJ_DIR/outputs/streaming_eval}"
LOG_DIR="${PROJ_DIR}/logs"

# ── Parse args ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)       MODEL="$2";            shift 2;;
        --model-small) MODEL_SMALL="$2";      shift 2;;
        --sessions)    MAX_SESSIONS="$2";     shift 2;;
        --turns)       TURNS_PER_SESSION="$2"; shift 2;;
        --output)      OUTPUT_DIR="$2";       shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

# ── Environment ──
source "$PROJ_DIR/scripts/gpu_utils.sh"
auto_setup

export TOKENIZERS_PARALLELISM=false

# Activate venv if present
if [ -f "$PROJ_DIR/.venv/bin/activate" ]; then
    source "$PROJ_DIR/.venv/bin/activate"
    echo "[env] venv: $PROJ_DIR/.venv"
fi

# Dependency check
python3 -c "import torch, transformers, peft" 2>/dev/null || {
    echo "[ERROR] Missing deps. Run: bash setup.sh"; exit 1
}

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo ""
echo "============================================================"
echo " Streaming Mirror-LoRA — Full Experiment Pipeline"
echo " Model (large) : $MODEL"
echo " Model (small) : $MODEL_SMALL"
echo " Sessions      : $MAX_SESSIONS"
echo " Turns/session : $TURNS_PER_SESSION"
echo " GPUs          : $NUM_GPUS"
echo " Output        : $OUTPUT_DIR"
echo " Time          : $(date)"
echo "============================================================"
echo ""

# ── Pre-download models (single process, uses HF cache) ──
echo "[1/4] Pre-downloading models..."
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
for m in ['$MODEL', '$MODEL_SMALL']:
    print(f'  Downloading {m}...')
    AutoTokenizer.from_pretrained(m)
    AutoModelForCausalLM.from_pretrained(m, device_map='cpu', torch_dtype='auto')
    print(f'  {m} cached.')
" 2>&1 | grep -v "^$\|Warning\|HTTP\|token"
echo "[1/4] Models cached."
echo ""

# ── All 13 methods ──
ALL_METHODS=(
    mirror_lora
    spm
    frozen
    single_lora_sgd
    single_lora_ewc
    single_lora_replay
    single_lora_ewc_replay
    param_matched_lora
    dual_lora_ema
    dual_lora_periodic_avg
    dual_lora_heuristic
    retrieval_augmented
    sdft
)

EVAL_SCRIPT="$PROJ_DIR/scripts/run_streaming_eval.py"
COMMON_ARGS="--task dialogue --max_sessions $MAX_SESSIONS --sessions_per_persona $SESSIONS_PER_PERSONA --turns_per_session $TURNS_PER_SESSION --device cuda:0 --no_nli"

# ── Helper: launch one method on one GPU ──
launch_method() {
    local gpu_id=$1 model=$2 method=$3 out_dir=$4 tag=$5
    local log_file="$LOG_DIR/${tag}_gpu${gpu_id}_${method}.log"
    echo "  GPU $gpu_id: $method ($model) → $log_file"
    CUDA_VISIBLE_DEVICES=$gpu_id nohup python3 -u "$EVAL_SCRIPT" \
        --methods "$method" --model "$model" --output_dir "$out_dir" \
        $COMMON_ARGS \
        > "$log_file" 2>&1 &
}

# ── Helper: wait for all background jobs, fail if any fail ──
wait_all() {
    local label=$1
    local pids=("${@:2}")
    local failed=0
    for pid in "${pids[@]}"; do
        wait "$pid" || { echo "[WARN] PID $pid failed"; failed=$((failed+1)); }
    done
    if [ $failed -gt 0 ]; then
        echo "[WARN] $label: $failed/$((${#pids[@]})) jobs failed"
    else
        echo "[$label] All jobs completed successfully."
    fi
}

# ======================================================================
# Phase 2: Large model evaluation (Qwen3.5-9B on N GPUs)
# ======================================================================
echo "[2/4] Large model evaluation: $MODEL"

OUT_LARGE="$OUTPUT_DIR/$(echo "$MODEL" | tr '/' '_')"
mkdir -p "$OUT_LARGE"

# Distribute methods across GPUs round-robin
PIDS_LARGE=()
gpu_idx=0
for method in "${ALL_METHODS[@]}"; do
    phys_gpu=$(gpu_at_index $gpu_idx)
    launch_method "$phys_gpu" "$MODEL" "$method" "$OUT_LARGE" "large"
    PIDS_LARGE+=($!)
    gpu_idx=$(( (gpu_idx + 1) % NUM_GPUS ))
done

echo "  Launched ${#PIDS_LARGE[@]} methods on $NUM_GPUS GPUs (round-robin)"
echo "  Waiting for large model experiments..."
wait_all "Phase 2" "${PIDS_LARGE[@]}"
echo "[2/4] Done."
echo ""

# ======================================================================
# Phase 3: Small model evaluation (Qwen3.5-0.8B — fast, full sweep)
# ======================================================================
echo "[3/4] Small model evaluation: $MODEL_SMALL"

OUT_SMALL="$OUTPUT_DIR/$(echo "$MODEL_SMALL" | tr '/' '_')"
mkdir -p "$OUT_SMALL"

PIDS_SMALL=()
gpu_idx=0
for method in "${ALL_METHODS[@]}"; do
    phys_gpu=$(gpu_at_index $gpu_idx)
    launch_method "$phys_gpu" "$MODEL_SMALL" "$method" "$OUT_SMALL" "small"
    PIDS_SMALL+=($!)
    gpu_idx=$(( (gpu_idx + 1) % NUM_GPUS ))
done

echo "  Launched ${#PIDS_SMALL[@]} methods on $NUM_GPUS GPUs"
echo "  Waiting for small model experiments..."
wait_all "Phase 3" "${PIDS_SMALL[@]}"
echo "[3/4] Done."
echo ""

# ======================================================================
# Phase 4: Synthetic benchmark (CPU, fast)
# ======================================================================
echo "[4/4] Synthetic benchmark..."
python3 "$PROJ_DIR/scripts/run_synthetic_v2.py" \
    --output_dir "$OUTPUT_DIR/synthetic_v2" \
    > "$LOG_DIR/synthetic_v2.log" 2>&1 || echo "[WARN] Synthetic v2 had errors"
echo "[4/4] Done."
echo ""

# ======================================================================
# Collect results
# ======================================================================
echo "============================================================"
echo " Collecting results..."
echo "============================================================"
python3 -c "
import json, os, glob

for d in sorted(glob.glob('$OUTPUT_DIR/*/dialogue_comparison.json')):
    tag = os.path.basename(os.path.dirname(d))
    with open(d) as f:
        data = json.load(f)
    print(f'\\n=== {tag} ===')
    print(f'{\"Method\":<28} {\"AvgRet\":>8} {\"PPL\":>8} {\"Loss\":>8} {\"Forget\":>8}')
    print('-' * 65)
    for name, m in sorted(data.items()):
        rets = m.get('session_retentions', [])
        avg_ret = sum(rets)/len(rets) if rets else 0
        ppl = m.get('perplexity', 0)
        loss = m.get('avg_loss', 0)
        forget = m.get('forgetting_rate', 0)
        print(f'{name:<28} {avg_ret:>8.4f} {ppl:>8.2f} {loss:>8.4f} {forget:>8.4f}')
" 2>/dev/null || echo "[INFO] Result collection skipped (some experiments may still be finishing)"

echo ""
echo "============================================================"
echo " Pipeline completed at $(date)"
echo " Results in: $OUTPUT_DIR"
echo "============================================================"
