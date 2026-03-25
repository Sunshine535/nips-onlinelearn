#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/gpu_utils.sh"
auto_setup

PROJ_DIR_ROOT="$(dirname "$SCRIPT_DIR")"
if [ -f "$PROJ_DIR_ROOT/.venv/bin/activate" ]; then
    source "$PROJ_DIR_ROOT/.venv/bin/activate"
fi
export PATH="$HOME/.local/bin:$PATH"

# Verify key dependencies before running
if ! python3 -c "import torch, transformers, peft" 2>/dev/null; then
    echo "[ERROR] Missing dependencies. Run: bash setup.sh"
    exit 1
fi

# --- Phase resume ---
PHASE_MARKER_DIR="$PROJ_DIR_ROOT/results/.phase_markers"
mkdir -p "$PHASE_MARKER_DIR"
FORCE_RERUN="${FORCE_RERUN:-0}"
phase_done() { touch "$PHASE_MARKER_DIR/phase_${1}.done"; echo "[PHASE $1] Completed at $(date)"; }
is_phase_done() {
    [[ "$FORCE_RERUN" == "1" ]] && return 1
    [[ -f "$PHASE_MARKER_DIR/phase_${1}.done" ]] && echo "[PHASE $1] Already completed. Skipping." && return 0
    return 1
}

export WANDB_PROJECT="onlinelearn"
PROJECT_DIR="$PROJ_DIR_ROOT"
CONFIG="${PROJECT_DIR}/configs/spm_config.yaml"
SPM_DIR="${PROJECT_DIR}/outputs/spm_training"
PPO_DIR="${PROJECT_DIR}/outputs/ppo_policy"
EVAL_DIR="${PROJECT_DIR}/outputs/streaming_eval"
LOG_DIR="${PROJECT_DIR}/logs"

mkdir -p "$SPM_DIR" "$PPO_DIR" "$EVAL_DIR" "$LOG_DIR"

echo "========================================="
echo " SPM Full Experiment Pipeline"
echo " GPUs: $NUM_GPUS × $GPU_CLASS"
echo "========================================="

# Phase 1: SPM Training
if ! is_phase_done 1; then
    echo ">>> Phase 1: Streaming Parameter Memory training"
    python3 "${SCRIPT_DIR}/train_spm.py" \
        --config "${CONFIG}" --output_dir "${PROJECT_DIR}/outputs" \
        --num_sessions 100 --probe_interval 10 \
        2>&1 | tee "${LOG_DIR}/phase1_spm.log"
    phase_done 1
fi

# Phase 2: PPO Integration
if ! is_phase_done 2; then
    echo ">>> Phase 2: PPO consolidation policy training"
    python3 "${SCRIPT_DIR}/train_ppo_integration.py" \
        --config "${CONFIG}" --output_dir "${PPO_DIR}" \
        --num_episodes 50 --turns_per_episode 40 --ppo_epochs 4 \
        2>&1 | tee "${LOG_DIR}/phase2_ppo.log"
    phase_done 2
fi

# Phase 3: Streaming Evaluation
if ! is_phase_done 3; then
    echo ">>> Phase 3: Streaming evaluation (PersonaChat + LIGHT)"
    python3 "${SCRIPT_DIR}/eval_streaming.py" \
        --config "${CONFIG}" --output_dir "${EVAL_DIR}" \
        --num_sessions 50 --methods no_adapt full_ft ewc spm \
        --datasets personachat light \
        2>&1 | tee "${LOG_DIR}/phase3_eval.log"
    phase_done 3
fi

# Phase 4: Ablations (parallel across GPUs; gpu_id captured before each subshell to avoid GPU_IDX races)
if ! is_phase_done 4; then
    echo ">>> Phase 4: Ablations"

    echo "  4a: Consolidation frequency"
    GPU_IDX=0
    PIDS=()
    LABELS=()
    for freq in 5 10 20 50; do
        ABL_DIR="${PROJECT_DIR}/outputs/ablation_freq_${freq}"
        gpu_id=$((GPU_IDX % NUM_GPUS))
        (
            mkdir -p "$ABL_DIR"
            python3 -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
cfg['working_memory']['max_turns_before_consolidation'] = ${freq}
cfg['streaming']['consolidation_frequency'] = ${freq}
with open('${ABL_DIR}/config.yaml', 'w') as f:
    yaml.dump(cfg, f)
"
            CUDA_VISIBLE_DEVICES="${gpu_id}" python3 "${SCRIPT_DIR}/train_spm.py" \
                --config "${ABL_DIR}/config.yaml" --output_dir "$ABL_DIR" --num_sessions 30
            CUDA_VISIBLE_DEVICES="${gpu_id}" python3 "${SCRIPT_DIR}/eval_streaming.py" \
                --config "${ABL_DIR}/config.yaml" --output_dir "${ABL_DIR}/eval" \
                --num_sessions 20 --methods spm --datasets personachat
        ) > "${LOG_DIR}/ablation_freq_${freq}.log" 2>&1 &
        PIDS+=($!)
        LABELS+=("freq_${freq}")
        GPU_IDX=$((GPU_IDX + 1))
    done
    FAIL=0
    for j in "${!PIDS[@]}"; do
        wait "${PIDS[$j]}" || { echo "ERROR: ${LABELS[$j]} failed"; FAIL=1; }
    done
    if [ "$FAIL" -ne 0 ]; then exit 1; fi

    echo "  4b: EWC lambda sweep"
    GPU_IDX=0
    PIDS=()
    LABELS=()
    for ewc_lambda in 100 1000 5000 10000 50000; do
        ABL_DIR="${PROJECT_DIR}/outputs/ablation_ewc_${ewc_lambda}"
        gpu_id=$((GPU_IDX % NUM_GPUS))
        (
            mkdir -p "$ABL_DIR"
            python3 -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
cfg['long_term_memory']['ewc_lambda'] = ${ewc_lambda}
with open('${ABL_DIR}/config.yaml', 'w') as f:
    yaml.dump(cfg, f)
"
            CUDA_VISIBLE_DEVICES="${gpu_id}" python3 "${SCRIPT_DIR}/train_spm.py" \
                --config "${ABL_DIR}/config.yaml" --output_dir "$ABL_DIR" --num_sessions 30
        ) > "${LOG_DIR}/ablation_ewc_${ewc_lambda}.log" 2>&1 &
        PIDS+=($!)
        LABELS+=("ewc_${ewc_lambda}")
        GPU_IDX=$((GPU_IDX + 1))
    done
    FAIL=0
    for j in "${!PIDS[@]}"; do
        wait "${PIDS[$j]}" || { echo "ERROR: ${LABELS[$j]} failed"; FAIL=1; }
    done
    if [ "$FAIL" -ne 0 ]; then exit 1; fi

    echo "  4c: LoRA rank"
    GPU_IDX=0
    PIDS=()
    LABELS=()
    for rank in 4 8 16 32 64; do
        ABL_DIR="${PROJECT_DIR}/outputs/ablation_rank_${rank}"
        gpu_id=$((GPU_IDX % NUM_GPUS))
        (
            mkdir -p "$ABL_DIR"
            python3 -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
cfg['working_memory']['lora_r'] = ${rank}
cfg['working_memory']['lora_alpha'] = ${rank} * 2
with open('${ABL_DIR}/config.yaml', 'w') as f:
    yaml.dump(cfg, f)
"
            CUDA_VISIBLE_DEVICES="${gpu_id}" python3 "${SCRIPT_DIR}/train_spm.py" \
                --config "${ABL_DIR}/config.yaml" --output_dir "$ABL_DIR" --num_sessions 30
        ) > "${LOG_DIR}/ablation_rank_${rank}.log" 2>&1 &
        PIDS+=($!)
        LABELS+=("rank_${rank}")
        GPU_IDX=$((GPU_IDX + 1))
    done
    FAIL=0
    for j in "${!PIDS[@]}"; do
        wait "${PIDS[$j]}" || { echo "ERROR: ${LABELS[$j]} failed"; FAIL=1; }
    done
    if [ "$FAIL" -ne 0 ]; then exit 1; fi

    phase_done 4
fi

# Phase 5: Aggregate results
if ! is_phase_done 5; then
    echo ">>> Phase 5: Aggregate results"
    python3 -c "
import json, os, glob
results = {}
eval_file = '${EVAL_DIR}/streaming_eval_results.json'
if os.path.exists(eval_file):
    with open(eval_file) as f:
        results['main'] = json.load(f)
for ablation in glob.glob('${PROJECT_DIR}/outputs/ablation_*/eval/streaming_eval_results.json'):
    name = ablation.split('ablation_')[1].split('/')[0]
    with open(ablation) as f:
        results[f'ablation_{name}'] = json.load(f)
with open('${EVAL_DIR}/all_results_aggregated.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'Aggregated {len(results)} experiment results')
"
    phase_done 5
fi

echo "========================================="
echo " All experiments complete!"
echo "========================================="

DONE_FILE="$PROJECT_DIR/results/.pipeline_done"
mkdir -p "$(dirname "$DONE_FILE")"
cat > "$DONE_FILE" << DONEEOF
{
  "project": "nips-onlinelearn",
  "completed_at": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "hostname": "$(hostname)",
  "gpus": "${NUM_GPUS:-unknown}",
  "status": "PIPELINE_COMPLETE"
}
DONEEOF
echo "[PIPELINE_COMPLETE] Run 'bash collect_results.sh' to package results."
