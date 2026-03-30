#!/usr/bin/env bash
set -euo pipefail

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

TORCHRUN=$(get_torchrun_cmd)

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
EVAL_DIR="${PROJECT_DIR}/outputs/streaming_eval"
LOG_DIR="${PROJECT_DIR}/logs"

mkdir -p "$SPM_DIR" "$EVAL_DIR" "$LOG_DIR"

echo "======================================================="
echo " SPM: Two-Timescale Behavioral Distillation Pipeline"
echo " GPUs: $NUM_GPUS × $GPU_CLASS"
echo " Config: $CONFIG"
echo "======================================================="

# ========================
# Phase 1: SPM Main Training
# ========================
if ! is_phase_done 1; then
    echo ">>> Phase 1: SPM Training (Behavioral Distillation Consolidation)"
    CUDA_VISIBLE_DEVICES=0 python3 "${SCRIPT_DIR}/train_spm.py" \
        --config "${CONFIG}" --output_dir "${PROJECT_DIR}/outputs" \
        --num_sessions 100 --probe_interval 10 \
        --resume_from_checkpoint auto \
        2>&1 | tee "${LOG_DIR}/phase1_spm.log"
    phase_done 1
fi

# ========================
# Phase 2: Streaming Evaluation (7 methods × 2 datasets)
# ========================
if ! is_phase_done 2; then
    echo ">>> Phase 2: Streaming evaluation (all methods × PersonaChat + LIGHT)"
    CUDA_VISIBLE_DEVICES=0 python3 "${SCRIPT_DIR}/eval_streaming.py" \
        --config "${CONFIG}" --output_dir "${EVAL_DIR}" \
        --num_sessions 50 \
        --methods frozen single_lora single_lora_ewc_replay param_matched retrieval_augmented dual_lora_ewc spm \
        --datasets personachat light \
        2>&1 | tee "${LOG_DIR}/phase2_eval.log"
    phase_done 2
fi

# ========================
# Phase 3: Core Ablations (parallel across GPUs)
# ========================
if ! is_phase_done 3; then
    echo ">>> Phase 3: Ablations"

    # --- 3a: Beta (KL distillation weight) sweep ---
    echo "  3a: Beta sweep (beta=0, 0.1, 0.5, 1.0, 5.0)"
    GPU_IDX=0; PIDS=(); LABELS=()
    for beta in 0 0.1 0.5 1.0 5.0; do
        ABL_DIR="${PROJECT_DIR}/outputs/ablation_beta_${beta}"
        gpu_id=$((GPU_IDX % NUM_GPUS))
        phys_gpu=$(gpu_at_index $gpu_id)
        (
            mkdir -p "$ABL_DIR"
            CUDA_VISIBLE_DEVICES="${phys_gpu}" python3 "${SCRIPT_DIR}/train_spm.py" \
                --config "${CONFIG}" --output_dir "$ABL_DIR" --num_sessions 30 \
                --beta "$beta" --resume_from_checkpoint auto
        ) > "${LOG_DIR}/ablation_beta_${beta}.log" 2>&1 &
        PIDS+=($!); LABELS+=("beta_${beta}"); GPU_IDX=$((GPU_IDX + 1))
    done
    FAIL=0
    for j in "${!PIDS[@]}"; do
        wait "${PIDS[$j]}" || { echo "ERROR: ${LABELS[$j]} failed"; FAIL=1; }
    done
    if [ "$FAIL" -ne 0 ]; then exit 1; fi

    # --- 3b: Gamma (Fisher trust-region) sweep ---
    echo "  3b: Gamma sweep (gamma=0, 0.01, 0.1, 1.0)"
    GPU_IDX=0; PIDS=(); LABELS=()
    for gamma in 0 0.01 0.1 1.0; do
        ABL_DIR="${PROJECT_DIR}/outputs/ablation_gamma_${gamma}"
        gpu_id=$((GPU_IDX % NUM_GPUS))
        phys_gpu=$(gpu_at_index $gpu_id)
        (
            mkdir -p "$ABL_DIR"
            CUDA_VISIBLE_DEVICES="${phys_gpu}" python3 "${SCRIPT_DIR}/train_spm.py" \
                --config "${CONFIG}" --output_dir "$ABL_DIR" --num_sessions 30 \
                --gamma "$gamma" --resume_from_checkpoint auto
        ) > "${LOG_DIR}/ablation_gamma_${gamma}.log" 2>&1 &
        PIDS+=($!); LABELS+=("gamma_${gamma}"); GPU_IDX=$((GPU_IDX + 1))
    done
    FAIL=0
    for j in "${!PIDS[@]}"; do
        wait "${PIDS[$j]}" || { echo "ERROR: ${LABELS[$j]} failed"; FAIL=1; }
    done
    if [ "$FAIL" -ne 0 ]; then exit 1; fi

    # --- 3c: WM LoRA rank sweep ---
    echo "  3c: WM rank sweep (rank=4, 8, 16, 32)"
    GPU_IDX=0; PIDS=(); LABELS=()
    for rank in 4 8 16 32; do
        ABL_DIR="${PROJECT_DIR}/outputs/ablation_rank_${rank}"
        gpu_id=$((GPU_IDX % NUM_GPUS))
        phys_gpu=$(gpu_at_index $gpu_id)
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
            CUDA_VISIBLE_DEVICES="${phys_gpu}" python3 "${SCRIPT_DIR}/train_spm.py" \
                --config "${ABL_DIR}/config.yaml" --output_dir "$ABL_DIR" --num_sessions 30 \
                --resume_from_checkpoint auto
        ) > "${LOG_DIR}/ablation_rank_${rank}.log" 2>&1 &
        PIDS+=($!); LABELS+=("rank_${rank}"); GPU_IDX=$((GPU_IDX + 1))
    done
    FAIL=0
    for j in "${!PIDS[@]}"; do
        wait "${PIDS[$j]}" || { echo "ERROR: ${LABELS[$j]} failed"; FAIL=1; }
    done
    if [ "$FAIL" -ne 0 ]; then exit 1; fi

    # --- 3d: Reservoir size sweep ---
    echo "  3d: Reservoir size sweep (500, 2000, 5000, 10000)"
    GPU_IDX=0; PIDS=(); LABELS=()
    for rsz in 500 2000 5000 10000; do
        ABL_DIR="${PROJECT_DIR}/outputs/ablation_reservoir_${rsz}"
        gpu_id=$((GPU_IDX % NUM_GPUS))
        phys_gpu=$(gpu_at_index $gpu_id)
        (
            mkdir -p "$ABL_DIR"
            python3 -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
cfg['long_term_memory']['max_memory_buffer'] = ${rsz}
with open('${ABL_DIR}/config.yaml', 'w') as f:
    yaml.dump(cfg, f)
"
            CUDA_VISIBLE_DEVICES="${phys_gpu}" python3 "${SCRIPT_DIR}/train_spm.py" \
                --config "${ABL_DIR}/config.yaml" --output_dir "$ABL_DIR" --num_sessions 30 \
                --resume_from_checkpoint auto
        ) > "${LOG_DIR}/ablation_reservoir_${rsz}.log" 2>&1 &
        PIDS+=($!); LABELS+=("reservoir_${rsz}"); GPU_IDX=$((GPU_IDX + 1))
    done
    FAIL=0
    for j in "${!PIDS[@]}"; do
        wait "${PIDS[$j]}" || { echo "ERROR: ${LABELS[$j]} failed"; FAIL=1; }
    done
    if [ "$FAIL" -ne 0 ]; then exit 1; fi

    phase_done 3
fi

# ========================
# Phase 4: Aggregate results
# ========================
if ! is_phase_done 4; then
    echo ">>> Phase 4: Aggregate results"
    python3 -c "
import json, os, glob
results = {}
eval_file = '${EVAL_DIR}/streaming_eval_results.json'
if os.path.exists(eval_file):
    with open(eval_file) as f:
        results['main'] = json.load(f)

for ablation in sorted(glob.glob('${PROJECT_DIR}/outputs/ablation_*/spm_training/training_log.json')):
    name = ablation.split('ablation_')[1].split('/')[0]
    with open(ablation) as f:
        data = json.load(f)
    results[f'ablation_{name}'] = {
        'num_sessions': len(data),
        'final_loss': data[-1]['avg_loss'] if data else None,
        'final_consol_loss': data[-1]['consolidation_loss'] if data else None,
    }

out_path = '${EVAL_DIR}/all_results_aggregated.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'Aggregated {len(results)} experiment results -> {out_path}')
"
    phase_done 4
fi

echo "======================================================="
echo " All experiments complete!"
echo " Results: ${EVAL_DIR}/all_results_aggregated.json"
echo "======================================================="

DONE_FILE="$PROJECT_DIR/results/.pipeline_done"
mkdir -p "$(dirname "$DONE_FILE")"
cat > "$DONE_FILE" << DONEEOF
{
  "project": "nips-onlinelearn",
  "method": "Behavioral Distillation Consolidation (Two-Timescale Residual LoRA)",
  "completed_at": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "hostname": "$(hostname)",
  "gpus": "${NUM_GPUS:-unknown}",
  "status": "PIPELINE_COMPLETE"
}
DONEEOF
echo "[PIPELINE_COMPLETE] Run 'bash collect_results.sh' to package results."
