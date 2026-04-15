#!/bin/bash
# =============================================================================
# Full NeurIPS Experiment Pipeline (E6-E9 + ablations + multi-seed)
#
# Usage:
#   bash run_full.sh                          # run everything
#   bash run_full.sh --phase 2                # only Phase 2 (classification)
#   bash run_full.sh --phase 3                # only Phase 3 (ablations)
#   bash run_full.sh --seeds "42 123 456"     # multi-seed
#   nohup bash run_full.sh > run_full.log 2>&1 &
#
# Prerequisites:
#   - transformers>=4.51.0 (Qwen3.5 support)
#   - run.sh Phase 2/3 already completed (dialogue results exist)
# =============================================================================
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ_DIR"

export HF_HOME="${HF_HOME:-$(dirname "$PROJ_DIR")/.cache/hf}"
export TOKENIZERS_PARALLELISM=false

# ── Defaults ──
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
MODEL_SMALL="${MODEL_SMALL:-Qwen/Qwen3.5-0.8B}"
MAX_SESSIONS="${MAX_SESSIONS:-50}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJ_DIR/outputs/streaming_eval}"
LOG_DIR="${PROJ_DIR}/logs"
SEEDS="${SEEDS:-42}"
PHASE="${PHASE:-all}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --phase)  PHASE="$2";       shift 2;;
        --model)  MODEL="$2";       shift 2;;
        --seeds)  SEEDS="$2";       shift 2;;
        --output) OUTPUT_DIR="$2";  shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

source "$PROJ_DIR/scripts/gpu_utils.sh" 2>/dev/null || true
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
[ "$NUM_GPUS" -eq 0 ] && NUM_GPUS=1

if [ -f "$PROJ_DIR/.venv/bin/activate" ]; then
    source "$PROJ_DIR/.venv/bin/activate"
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

EVAL_SCRIPT="$PROJ_DIR/scripts/run_streaming_eval.py"

echo "============================================================"
echo " Full NeurIPS Experiment Pipeline"
echo " Model       : $MODEL"
echo " GPUs        : $NUM_GPUS"
echo " Seeds       : $SEEDS"
echo " Phase       : $PHASE"
echo " Time        : $(date)"
echo "============================================================"

# ── Helper ──
launch() {
    local gpu_id=$1 task=$2 method=$3 model=$4 tag=$5
    shift 5
    local log_file="$LOG_DIR/${tag}_gpu${gpu_id}_${task}_${method}.log"
    echo "  GPU $gpu_id: $task/$method → $log_file"
    local out="$OUTPUT_DIR/$(echo "$model" | tr '/' '_')"
    mkdir -p "$out"
    CUDA_VISIBLE_DEVICES=$gpu_id nohup python3 -u "$EVAL_SCRIPT" \
        --task "$task" --methods "$method" --model "$model" \
        --output_dir "$out" --max_sessions "$MAX_SESSIONS" \
        --no_nli "$@" \
        > "$log_file" 2>&1 &
}

wait_jobs() {
    local label=$1; shift
    local failed=0
    for pid in "$@"; do
        wait "$pid" 2>/dev/null || failed=$((failed+1))
    done
    [ $failed -gt 0 ] && echo "[$label] $failed jobs failed" || echo "[$label] All done."
}

# ======================================================================
# Phase 1: Classification task (E7) — AG News concept drift
# ======================================================================
if [[ "$PHASE" == "all" || "$PHASE" == "1" ]]; then
    echo ""
    echo "[Phase 1] E7: Classification ($MODEL)"

    ALL_METHODS=(mirror_lora spm frozen single_lora_sgd single_lora_ewc
                 single_lora_replay single_lora_ewc_replay param_matched_lora
                 dual_lora_ema dual_lora_periodic_avg dual_lora_heuristic
                 retrieval_augmented sdft)

    PIDS=()
    gpu_idx=0
    for method in "${ALL_METHODS[@]}"; do
        gpu=$((gpu_idx % NUM_GPUS))
        launch "$gpu" "classification" "$method" "$MODEL" "e7_large"
        PIDS+=($!)
        gpu_idx=$((gpu_idx + 1))
    done
    wait_jobs "E7-large" "${PIDS[@]}"

    # Small model too
    PIDS=()
    gpu_idx=0
    for method in "${ALL_METHODS[@]}"; do
        gpu=$((gpu_idx % NUM_GPUS))
        launch "$gpu" "classification" "$method" "$MODEL_SMALL" "e7_small"
        PIDS+=($!)
        gpu_idx=$((gpu_idx + 1))
    done
    wait_jobs "E7-small" "${PIDS[@]}"
    echo "[Phase 1] Done."
fi

# ======================================================================
# Phase 2: Multi-seed runs for error bars (E6 + E7, mirror_lora + top baselines)
# ======================================================================
if [[ "$PHASE" == "all" || "$PHASE" == "2" ]]; then
    echo ""
    echo "[Phase 2] Multi-seed runs"

    KEY_METHODS=(mirror_lora spm single_lora_ewc_replay dual_lora_ema)

    for seed in $SEEDS; do
        [ "$seed" == "42" ] && continue  # already run
        echo "  Seed: $seed"
        PIDS=()
        gpu_idx=0
        for method in "${KEY_METHODS[@]}"; do
            for task in dialogue classification; do
                gpu=$((gpu_idx % NUM_GPUS))
                log="$LOG_DIR/seed${seed}_gpu${gpu}_${task}_${method}.log"
                out="$OUTPUT_DIR/$(echo "$MODEL" | tr '/' '_')/seed_${seed}"
                mkdir -p "$out"
                echo "  GPU $gpu: seed=$seed $task/$method"
                CUDA_VISIBLE_DEVICES=$gpu nohup python3 -u "$EVAL_SCRIPT" \
                    --task "$task" --methods "$method" --model "$MODEL" \
                    --output_dir "$out" --max_sessions "$MAX_SESSIONS" \
                    --seed "$seed" --no_nli \
                    > "$log" 2>&1 &
                PIDS+=($!)
                gpu_idx=$((gpu_idx + 1))
            done
        done
        wait_jobs "Seed-$seed" "${PIDS[@]}"
    done
    echo "[Phase 2] Done."
fi

# ======================================================================
# Phase 3: Ablation study (E8) — component necessity
# ======================================================================
if [[ "$PHASE" == "all" || "$PHASE" == "3" ]]; then
    echo ""
    echo "[Phase 3] E8: Ablation study"

    # Ablations are Mirror-LoRA variants with specific components disabled
    # We run these by modifying config and using mirror_lora method

    ABLATION_OUT="$OUTPUT_DIR/$(echo "$MODEL_SMALL" | tr '/' '_')/ablations"
    ABLATION_CFGS="$PROJ_DIR/configs/ablations"
    mkdir -p "$ABLATION_OUT"

    # Use pre-created configs from configs/ablations/*.yaml
    PIDS=()
    gpu_idx=0
    for cfg_file in "$ABLATION_CFGS"/*.yaml; do
        name=$(basename "$cfg_file" .yaml)
        gpu=$((gpu_idx % NUM_GPUS))
        out="$ABLATION_OUT/${name}"
        log="$LOG_DIR/ablation_${name}.log"
        mkdir -p "$out"
        echo "  GPU $gpu: ablation=$name (config: $cfg_file)"
        CUDA_VISIBLE_DEVICES=$gpu nohup python3 -u "$EVAL_SCRIPT" \
            --task dialogue --methods mirror_lora --model "$MODEL_SMALL" \
            --config "$cfg_file" --output_dir "$out" \
            --max_sessions 20 --no_nli \
            > "$log" 2>&1 &
        PIDS+=($!)
        gpu_idx=$((gpu_idx + 1))
    done
    wait_jobs "E8-ablation" "${PIDS[@]}"
    echo "[Phase 3] Done."
fi

: << 'REMOVED_INLINE_GENERATOR'
    python3 -u << 'PYEOF'
import json, os, sys, subprocess, yaml, copy

proj = os.environ.get("PROJ_DIR", ".")
model = os.environ.get("MODEL_SMALL", "Qwen/Qwen3.5-0.8B")
ablation_dir = os.environ.get("ABLATION_DIR", "outputs/ablations")
eval_script = os.path.join(proj, "scripts/run_streaming_eval.py")

# Base config
base_config = {
    "working_memory": {
        "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "learning_rate": 5e-4, "online_update_steps": 3,
    },
    "long_term_memory": {
        "lora_r": 64, "lora_alpha": 128, "lora_dropout": 0.02,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
        "consolidation_lr": 1e-4, "consolidation_epochs": 3,
        "max_memory_buffer": 5000,
    },
    "consolidation": {"beta": 1.0, "gamma": 0.0},
}

# Ablation variants
ablations = {
    "full": {},  # baseline (full Mirror-LoRA)
    "no_kl_distill": {"consolidation": {"beta": 0.0}},
    "no_trust_region": {"consolidation": {"gamma": 0.0}},
    "high_trust_region": {"consolidation": {"gamma": 10.0}},
    "wm_rank4": {"working_memory": {"lora_r": 4, "lora_alpha": 8}},
    "wm_rank32": {"working_memory": {"lora_r": 32, "lora_alpha": 64}},
    "lt_rank16": {"long_term_memory": {"lora_r": 16, "lora_alpha": 32}},
    "lt_rank128": {"long_term_memory": {"lora_r": 128, "lora_alpha": 256}},
    "no_replay": {"long_term_memory": {"max_memory_buffer": 0}},
    "small_replay": {"long_term_memory": {"max_memory_buffer": 500}},
    "fast_consol": {"long_term_memory": {"consolidation_epochs": 1}},
    "slow_consol": {"long_term_memory": {"consolidation_epochs": 10}},
}

os.makedirs(ablation_dir, exist_ok=True)

for name, overrides in ablations.items():
    cfg = copy.deepcopy(base_config)
    for section, params in overrides.items():
        cfg[section].update(params)

    cfg_path = os.path.join(ablation_dir, f"config_{name}.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)

    print(f"Ablation: {name} -> {cfg_path}")

print(f"\nGenerated {len(ablations)} ablation configs in {ablation_dir}")
print("Run each with:")
print(f"  python3 {eval_script} --task dialogue --methods mirror_lora "
      f"--model $MODEL_SMALL --config <config> --output_dir <out>")
PYEOF
REMOVED_INLINE_GENERATOR

# ======================================================================
# Phase 4: Efficiency benchmark (E9)
# ======================================================================
if [[ "$PHASE" == "all" || "$PHASE" == "4" ]]; then
    echo ""
    echo "[Phase 4] E9: Efficiency benchmark"

    python3 -u << 'PYEOF'
import json, os, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

model_name = os.environ.get("MODEL_SMALL", "Qwen/Qwen3.5-0.8B")
output_dir = os.environ.get("OUTPUT_DIR", "outputs/streaming_eval")

print(f"Efficiency benchmark: {model_name}")

# Measure memory and latency for different configs
configs = {
    "frozen": {"rank": 0, "dual": False},
    "single_lora_r16": {"rank": 16, "dual": False},
    "single_lora_r80": {"rank": 80, "dual": False},
    "dual_lora_16_64": {"rank": 16, "dual": True, "lt_rank": 64},
    "mirror_lora": {"rank": 16, "dual": True, "lt_rank": 64},
}

results = {}
device = "cuda:0" if torch.cuda.is_available() else "cpu"

for name, cfg in configs.items():
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    base_params = sum(p.numel() for p in model.parameters())
    trainable = 0

    if cfg["rank"] > 0:
        lora_cfg = LoraConfig(
            r=cfg["rank"], lora_alpha=cfg["rank"]*2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Measure forward latency
    input_ids = tokenizer("Hello world", return_tensors="pt").input_ids.to(device)
    times = []
    with torch.no_grad():
        for _ in range(10):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(input_ids)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    peak_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0

    results[name] = {
        "base_params_M": base_params / 1e6,
        "trainable_params_M": trainable / 1e6,
        "peak_gpu_memory_GB": round(peak_mem, 2),
        "avg_forward_latency_ms": round(sum(times[2:])/len(times[2:]), 2),
        "overhead_pct": round(trainable / base_params * 100, 3) if base_params > 0 else 0,
    }
    print(f"  {name}: {results[name]}")
    del model

eff_path = os.path.join(output_dir, "efficiency_benchmark.json")
with open(eff_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {eff_path}")
PYEOF
    echo "[Phase 4] Done."
fi

# ======================================================================
# Collect all results
# ======================================================================
echo ""
echo "============================================================"
echo " Collecting all results..."
echo "============================================================"

python3 -u << 'PYEOF'
import json, os, glob

output_dir = os.environ.get("OUTPUT_DIR", "outputs/streaming_eval")

# Collect dialogue results
for d in sorted(glob.glob(f'{output_dir}/*/dialogue_comparison.json')):
    tag = os.path.basename(os.path.dirname(d))
    with open(d) as f:
        data = json.load(f)
    print(f'\n=== {tag} (dialogue) ===')
    print(f'{"Method":<28} {"AvgRet":>8} {"PPL":>8} {"Forget":>8}')
    print('-' * 60)
    for name, m in sorted(data.items()):
        rets = m.get('session_retentions', [])
        avg_ret = sum(rets)/len(rets) if rets else 0
        ppl = m.get('perplexity', 0)
        forget = m.get('forgetting_rate', 0)
        print(f'{name:<28} {avg_ret:>8.4f} {ppl:>8.2f} {forget:>8.4f}')

# Collect classification results
for d in sorted(glob.glob(f'{output_dir}/*/classification_comparison.json')):
    tag = os.path.basename(os.path.dirname(d))
    with open(d) as f:
        data = json.load(f)
    print(f'\n=== {tag} (classification) ===')
    print(f'{"Method":<28} {"Accuracy":>8} {"Forget":>8} {"FwdTfr":>8} {"Regret":>10}')
    print('-' * 70)
    for name, m in sorted(data.items()):
        print(f'{name:<28} {m.get("overall_accuracy",0):>8.4f} '
              f'{m.get("forgetting",0):>8.4f} '
              f'{m.get("forward_transfer",0):>8.4f} '
              f'{m.get("dynamic_regret",0):>10.2f}')

# Collect ablation results
abl_dir = glob.glob(f'{output_dir}/*/ablations/results_*')
if abl_dir:
    print(f'\n=== Ablation Study ===')
    print(f'{"Variant":<28} {"AvgRet":>8} {"PPL":>8}')
    print('-' * 50)
    for d in sorted(abl_dir):
        name = os.path.basename(d).replace('results_', '')
        comp = os.path.join(d, 'dialogue_comparison.json')
        if os.path.exists(comp):
            with open(comp) as f:
                data = json.load(f)
            for _, m in data.items():
                rets = m.get('session_retentions', [])
                avg_ret = sum(rets)/len(rets) if rets else 0
                ppl = m.get('perplexity', 0)
                print(f'{name:<28} {avg_ret:>8.4f} {ppl:>8.2f}')

# Efficiency
eff = os.path.join(output_dir, 'efficiency_benchmark.json')
if os.path.exists(eff):
    with open(eff) as f:
        data = json.load(f)
    print(f'\n=== Efficiency ===')
    print(f'{"Config":<24} {"Params(M)":>10} {"Train(M)":>10} {"Mem(GB)":>8} {"Lat(ms)":>8}')
    print('-' * 65)
    for name, m in data.items():
        print(f'{name:<24} {m["base_params_M"]:>10.1f} {m["trainable_params_M"]:>10.2f} '
              f'{m["peak_gpu_memory_GB"]:>8.2f} {m["avg_forward_latency_ms"]:>8.2f}')

PYEOF

echo ""
echo "============================================================"
echo " Pipeline complete: $(date)"
echo "============================================================"
