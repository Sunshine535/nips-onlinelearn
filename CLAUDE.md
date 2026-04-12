# Project: Streaming Mirror-LoRA (NeurIPS 2026 Best Paper Target)

## Paper Title
Dual-Timescale Adaptation Is Necessary and Sufficient for Streaming Low-Rank Learning under Multi-Rate Drift

## Central Thesis
Multi-rate drift creates an impossible tradeoff for single-timescale adaptation; separating fast tracking from slow consolidation resolves it.

## Current Pipeline Stage
**Stage 2 (Implementation) + Auto-Review-Loop Round 2**

## Models
- **Primary**: `Qwen/Qwen3.5-9B` (latest architecture, 9B params)
- **Small**: `Qwen/Qwen3.5-0.8B` (fast iteration)
- HF_HOME is auto-set by the run environment; models cached once, shared across projects

## Running Experiments
```bash
bash run.sh                                    # 8-GPU full pipeline
bash run.sh --model Qwen/Qwen3.5-9B --sessions 50  # customize
```

## Source Code
- `src/streaming_mirror_lora.py` — **OUR METHOD** (StreamingMirrorLoRA + Fisher trust-region consolidation)
- `src/streaming_memory.py` — SPM dual-LoRA baseline (with `_set_adapters_layerwise` fix for PEFT 0.18)
- `src/synthetic_benchmark.py` — synthetic two-rate drift benchmark
- `src/baselines.py` — 11 baseline methods
- `scripts/run_streaming_eval.py` — unified LLM evaluation (13 methods, 8-GPU parallel)
- `scripts/run_synthetic_v2.py` — V2 synthetic experiments

## Key Bug Fixes Applied (2026-04-09)
1. **PEFT dual-adapter**: `set_adapter()` in PEFT 0.18 can't accept lists → use layer-level `BaseTunerLayer.set_adapter()`
2. **fp16 LoRA gradients**: zero-init B causes gradient underflow → cast LoRA params to fp32
3. **zero_init**: was zeroing both LoRA A and B → only zero B (keep A random for gradient flow)
4. **Fisher merge**: WM (r=16) and LT (r=64) rank mismatch → Fisher trust-region approach
5. **KL distillation NaN**: fp16 log_softmax → compute KL in fp32

## Key Results
- **Synthetic**: 30-1000x dual vs single timescale regret separation
- **Qwen2.5-7B**: Mirror-LoRA AvgRet=0.169 vs SPM 0.138 (+22%), PPL 23.6 vs 298.1
- **Fisher trust-region**: prevents catastrophic PPL degradation during consolidation
