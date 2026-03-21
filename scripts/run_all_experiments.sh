#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/gpu_utils.sh"
auto_setup

# --- Activate project venv (created by setup.sh) ---
PROJ_DIR_ROOT="$(dirname "$SCRIPT_DIR")"
if [ -f "$PROJ_DIR_ROOT/.venv/bin/activate" ]; then
    source "$PROJ_DIR_ROOT/.venv/bin/activate"
fi
export PATH="$HOME/.local/bin:$PATH"

# ─── SPM: Full Experiment Pipeline ──────────────────────────────────────────
# spm_training → ppo_integration → eval (PersonaChat + LIGHT) → ablations
export WANDB_PROJECT="onlinelearn"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="${PROJECT_DIR}/configs/spm_config.yaml"
SPM_DIR="${PROJECT_DIR}/outputs/spm_training"
PPO_DIR="${PROJECT_DIR}/outputs/ppo_policy"
EVAL_DIR="${PROJECT_DIR}/outputs/streaming_eval"

mkdir -p "$SPM_DIR" "$PPO_DIR" "$EVAL_DIR"

echo "========================================="
echo " SPM Full Experiment Pipeline"
echo " Config: ${CONFIG}"
echo "========================================="

# ─── Phase 1: SPM Training ─────────────────────────────────────────────────
echo ""
echo ">>> Phase 1: Streaming Parameter Memory training"
python "${SCRIPT_DIR}/train_spm.py" \
    --config "${CONFIG}" \
    --output_dir "${PROJECT_DIR}/outputs" \
    --num_sessions 100 \
    --probe_interval 10

echo "  SPM training complete."

# ─── Phase 2: PPO Integration ──────────────────────────────────────────────
echo ""
echo ">>> Phase 2: PPO consolidation policy training"
python "${SCRIPT_DIR}/train_ppo_integration.py" \
    --config "${CONFIG}" \
    --output_dir "${PPO_DIR}" \
    --num_episodes 50 \
    --turns_per_episode 40 \
    --ppo_epochs 4

echo "  PPO training complete."

# ─── Phase 3: Streaming Evaluation (4 methods × 2 datasets) ────────────────
echo ""
echo ">>> Phase 3: Streaming evaluation (PersonaChat + LIGHT)"
python "${SCRIPT_DIR}/eval_streaming.py" \
    --config "${CONFIG}" \
    --output_dir "${EVAL_DIR}" \
    --num_sessions 50 \
    --methods no_adapt full_ft ewc spm \
    --datasets personachat light

echo "  Evaluation complete."

# ─── Phase 4: Ablations ────────────────────────────────────────────────────
echo ""
echo ">>> Phase 4: Ablation - consolidation frequency"
for freq in 5 10 20 50; do
    ABLATION_DIR="${PROJECT_DIR}/outputs/ablation_freq_${freq}"
    mkdir -p "$ABLATION_DIR"

    # Create temp config with modified frequency
    python3 -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
cfg['working_memory']['max_turns_before_consolidation'] = ${freq}
cfg['streaming']['consolidation_frequency'] = ${freq}
with open('${ABLATION_DIR}/config.yaml', 'w') as f:
    yaml.dump(cfg, f)
"
    python "${SCRIPT_DIR}/train_spm.py" \
        --config "${ABLATION_DIR}/config.yaml" \
        --output_dir "$ABLATION_DIR" \
        --num_sessions 30

    python "${SCRIPT_DIR}/eval_streaming.py" \
        --config "${ABLATION_DIR}/config.yaml" \
        --output_dir "${ABLATION_DIR}/eval" \
        --num_sessions 20 \
        --methods spm \
        --datasets personachat
done

echo ""
echo ">>> Phase 4b: Ablation - EWC lambda sweep"
for ewc_lambda in 100 1000 5000 10000 50000; do
    ABLATION_DIR="${PROJECT_DIR}/outputs/ablation_ewc_${ewc_lambda}"
    mkdir -p "$ABLATION_DIR"

    python3 -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
cfg['long_term_memory']['ewc_lambda'] = ${ewc_lambda}
with open('${ABLATION_DIR}/config.yaml', 'w') as f:
    yaml.dump(cfg, f)
"
    python "${SCRIPT_DIR}/train_spm.py" \
        --config "${ABLATION_DIR}/config.yaml" \
        --output_dir "$ABLATION_DIR" \
        --num_sessions 30
done

echo ""
echo ">>> Phase 4c: Ablation - LoRA rank"
for rank in 4 8 16 32 64; do
    ABLATION_DIR="${PROJECT_DIR}/outputs/ablation_rank_${rank}"
    mkdir -p "$ABLATION_DIR"

    python3 -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
cfg['working_memory']['lora_r'] = ${rank}
cfg['working_memory']['lora_alpha'] = ${rank} * 2
with open('${ABLATION_DIR}/config.yaml', 'w') as f:
    yaml.dump(cfg, f)
"
    python "${SCRIPT_DIR}/train_spm.py" \
        --config "${ABLATION_DIR}/config.yaml" \
        --output_dir "$ABLATION_DIR" \
        --num_sessions 30
done

# ─── Phase 5: Analysis ─────────────────────────────────────────────────────
echo ""
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

echo ""
echo "========================================="
echo " All experiments complete!"
echo "========================================="
echo " Results directories:"
echo "   SPM Training: ${SPM_DIR}"
echo "   PPO Policy:   ${PPO_DIR}"
echo "   Evaluation:   ${EVAL_DIR}"
echo "   Ablations:    ${PROJECT_DIR}/outputs/ablation_*"
echo "========================================="
