# Behavioral Distillation Consolidation for Two-Timescale Streaming Persona Retention

Two-timescale residual LoRA with session-end KL-distillation consolidation for online persona adaptation in frozen LLMs.

**Core idea**: A zero-initialized Working-Memory LoRA (WM) captures session-local adaptations atop a persistent Long-Term LoRA (LT). At session boundaries, behavioral distillation transfers knowledge from WM into LT via a split-objective:
- **Behavior transfer**: KL divergence on current-session data (teacher = LT+WM, student = LT')
- **Retention**: NTP replay on reservoir-sampled past data

## Quick Start

```bash
# 1. Install dependencies
bash setup.sh

# 2. Run full pipeline (training → evaluation → ablations)
bash run.sh

# 3. (Optional) background
nohup bash run.sh > run.log 2>&1 &
tail -f run.log
```

### Resume After Interruption

Re-run `bash run.sh` — completed phases are automatically skipped; training resumes from the latest checkpoint. To force re-run: `FORCE_RERUN=1 bash run.sh`.

### Check Completion

```bash
cat results/.pipeline_done       # Shows PIPELINE_COMPLETE when all phases finish
ls results/.phase_markers/       # Per-phase completion markers
```

## Project Structure

```
nips-onlinelearn/
├── configs/
│   └── spm_config.yaml               # Hyperparameters (beta, gamma, LoRA rank, etc.)
├── src/
│   └── streaming_memory.py           # Core: dual-LoRA + behavioral distillation consolidation
├── scripts/
│   ├── gpu_utils.sh                  # Auto GPU detection
│   ├── run_all_experiments.sh        # Master pipeline
│   ├── train_spm.py                  # SPM streaming training
│   ├── eval_streaming.py             # 7-method × 2-dataset evaluation
│   └── eval_spm.py                   # Per-model evaluation
├── refine-logs/
│   └── FINAL_PROPOSAL.md            # NeurIPS-ready proposal (research-refine output)
└── outputs/                          # All experiment outputs
```

## Method

| Component | Description | Params |
|-----------|-------------|--------|
| Base model | Qwen/Qwen3.5-9B (frozen) | 9B |
| Working-Memory LoRA | r=16, zero-init per session, 3-step NTP updates | ~13M |
| Long-Term LoRA | r=64, persistent, behavioral distillation consolidation | ~52M |

### Core Equation

```
L = L_ntp(θ_LT'; B_reservoir)  +  β · E_{x∈B_session}[D_KL(p_{LT+WM}(·|x) || p_{LT'}(·|x))]  +  γ · Fisher_trust_region
```

### Algorithm (per session k)

1. **Zero-init WM**: Reset WM-LoRA to zero (residual over LT)
2. **Online adaptation**: For each turn, 1-3 NTP gradient steps on WM-LoRA
3. **Buffer data**: Add each turn to the session buffer
4. **Consolidation** (session end):
   - Cache teacher logits from combined LT+WM on session data
   - Switch to LT-only, optimize split objective:
     - NTP on reservoir samples (retention)
     - KL distillation on session data (behavior transfer)
5. **Merge session data into reservoir** (reservoir sampling)

## Experiments

### Pipeline Phases

| # | Phase | Description | Output |
|---|-------|-------------|--------|
| 1 | SPM Training | 100-session streaming training | `outputs/spm_training/` |
| 2 | PPO Policy (auxiliary) | PPO consolidation policy training (auxiliary) | `outputs/ppo_policy/` |

> **Note**: Phase 2 (PPO Policy) is experimental and not required for the main contribution. The core method (SPM with behavioral distillation consolidation) is fully evaluated in Phases 1 and 3. Phase 2 explores an optional RL-based consolidation policy that is orthogonal to the primary claims.

| 3 | Evaluation | 7 methods × 2 datasets (PersonaChat, LIGHT) | `outputs/streaming_eval/` |
| 4a | Ablation: beta | KL weight sweep (0, 0.1, 0.5, 1.0, 5.0) | `outputs/ablation_beta_*/` |
| 4b | Ablation: gamma | Fisher trust-region sweep (0, 0.01, 0.1, 1.0) | `outputs/ablation_gamma_*/` |
| 4c | Ablation: WM rank | LoRA rank sweep (4, 8, 16, 32) | `outputs/ablation_rank_*/` |
| 4d | Ablation: reservoir | Buffer size sweep (500, 2k, 5k, 10k) | `outputs/ablation_reservoir_*/` |
| 5 | Aggregation | Collect all results | `outputs/streaming_eval/all_results_aggregated.json` |

### Baselines (7 methods)

| Method | Description |
|--------|-------------|
| Frozen | No adaptation (lower bound) |
| Single-LoRA | Online NTP, no protection |
| Single-LoRA + EWC + Replay | r=16, continual learning baseline |
| Param-matched Single-LoRA | r=80 (~65M, matched to WM+LT) |
| Retrieval-Augmented | Top-k context prepending |
| Dual-LoRA + EWC | Parameter-space consolidation |
| **SPM (Ours)** | Behavioral distillation consolidation |

### Target Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Semantic Retention F1 | NLI-based persona fact retention | ≥ 0.65 |
| Adaptation Speed | Turns to reach RetF1 ≥ 0.5 | ≤ 5 |
| Forgetting Rate | Per-session retention decline | ≤ 0.10 |
| Generation Quality | PPL degradation vs. frozen | ≤ 5% |

## GPU Budget

| Phase | Est. GPU-hours |
|-------|---------------|
| SPM training | ~150 |
| Evaluation (7×2) | ~60 |
| Ablation studies | ~200 |
| Buffer + contingency | ~90 |
| **Total** | **~500** |

## Citation

```bibtex
@inproceedings{spm2026,
  title={Behavioral Distillation Consolidation for Two-Timescale Streaming Persona Retention in Large Language Models},
  author={Anonymous},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026}
}
```

## License

[MIT License](LICENSE)
