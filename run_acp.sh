#!/usr/bin/env bash
# =============================================================================
# run_acp.sh — Deploy and run SPM pipeline on ACP cluster
#
# Usage:
#   bash run_acp.sh          # full pipeline
#   bash run_acp.sh setup    # install deps only
#   bash run_acp.sh run      # run pipeline only (deps assumed ready)
# =============================================================================
set -euo pipefail

PROJECT_DIR=/data/szs/250010072/nwh/nips-onlinelearn
DATA_DIR=/data/szs/share/onlinelearn
MODEL_PATH=/data/szs/share/Qwen3.5-9B

export HF_HOME="${DATA_DIR}/.cache/huggingface"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

cd "$PROJECT_DIR"

# ---- Symlink persistent output directories ----
setup_symlinks() {
    mkdir -p "${DATA_DIR}/results" "${DATA_DIR}/logs"
    ln -sfn "${DATA_DIR}/results" "$PROJECT_DIR/results"
    ln -sfn "${DATA_DIR}/logs"    "$PROJECT_DIR/logs"
    mkdir -p "$HF_HOME"
    echo "[symlinks] results -> ${DATA_DIR}/results"
    echo "[symlinks] logs    -> ${DATA_DIR}/logs"
}

# ---- Install dependencies ----
setup_deps() {
    echo "============================================"
    echo " Installing dependencies (PyTorch 2.10 + CUDA 12.8)"
    echo "============================================"

    PYTORCH_INDEX="https://download.pytorch.org/whl/cu128"
    PIP_MIRROR="https://pypi.org/simple/"

    pip install "torch==2.10.0" "torchvision" "torchaudio" \
        --index-url "$PYTORCH_INDEX" \
        --extra-index-url "$PIP_MIRROR"

    pip install -r "$PROJECT_DIR/requirements.txt" \
        -i "$PIP_MIRROR"

    pip install deepspeed accelerate trl peft \
        -i "$PIP_MIRROR"

    pip install flash-attn --no-build-isolation 2>/dev/null \
        || echo "[WARN] flash-attn install failed, continuing without it"
}

# ---- Generate ACP config with local model path ----
generate_config() {
    python3 -c "
import yaml, os
with open('configs/spm_config.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['model']['base_model'] = '${MODEL_PATH}'
os.makedirs('configs', exist_ok=True)
with open('configs/spm_config_acp.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
print('[config] Generated configs/spm_config_acp.yaml')
print(f'  base_model = ${MODEL_PATH}')
"
    export SPM_CONFIG="${PROJECT_DIR}/configs/spm_config_acp.yaml"
}

# ---- GPU health check (use total_memory, not total_mem) ----
gpu_check() {
    echo "============================================"
    echo " GPU Check"
    echo "============================================"
    python3 -c "
import torch
n = torch.cuda.device_count()
print(f'  PyTorch    : {torch.__version__}')
print(f'  CUDA       : {torch.version.cuda}')
print(f'  GPU count  : {n}')
for i in range(n):
    props = torch.cuda.get_device_properties(i)
    gb = props.total_memory / (1024 ** 3)
    print(f'  GPU {i}: {props.name}  {gb:.1f} GB')
assert n > 0, 'No GPUs detected!'
"
    echo "============================================"
}

# ---- Run pipeline ----
run_pipeline() {
    generate_config
    gpu_check

    echo ""
    echo "============================================"
    echo " Starting SPM pipeline"
    echo " Project : $PROJECT_DIR"
    echo " Data    : $DATA_DIR"
    echo " Model   : $MODEL_PATH"
    echo " Config  : $SPM_CONFIG"
    echo " Time    : $(date)"
    echo "============================================"

    bash scripts/run_all_experiments.sh 2>&1 | tee "${DATA_DIR}/logs/run_acp_$(date +%Y%m%d_%H%M%S).log"
}

# ---- Main ----
MODE="${1:-all}"
case "$MODE" in
    setup)
        setup_symlinks
        setup_deps
        gpu_check
        ;;
    run)
        setup_symlinks
        run_pipeline
        ;;
    all|*)
        setup_symlinks
        setup_deps
        run_pipeline
        ;;
esac
