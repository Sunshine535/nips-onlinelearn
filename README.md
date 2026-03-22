# Streaming Parameter Memory: Dual-LoRA Online Learning for LLMs

---

## How to Run (Complete Guide)

### Requirements

- Linux server with NVIDIA GPU (4-8x A100 80GB recommended)
- CUDA 12.8 compatible driver
- `git`, `curl` installed
- ~200GB disk space (model weights + checkpoints)

### Step 1: Clone and Run (One Command)

```bash
git clone https://github.com/Sunshine535/nips-onlinelearn.git
cd nips-onlinelearn
bash run.sh
```

`run.sh` will automatically:
1. Install `uv` package manager (if not present)
2. Create Python 3.10 virtual environment
3. Install PyTorch 2.10 + CUDA 12.8
4. Install all dependencies
5. Run **all experiments** in full production mode
6. Display real-time progress in terminal and save to `run.log`

### Step 2: Monitor Progress

If running in foreground (default):
```bash
# Progress is displayed in real-time
# Press Ctrl+C to stop (can resume later with bash run.sh)
```

If running in background (recommended for long experiments):
```bash
nohup bash run.sh > run.log 2>&1 &
tail -f run.log
```

### Step 3: Check Completion

```bash
cat results/.pipeline_done
# If this file exists and shows "PIPELINE_COMPLETE", all experiments finished successfully
```

### Step 4: Package and Send Results

```bash
# Option A: Push to GitHub (recommended)
git add results/ logs/
git commit -m "Experiment results $(date +%Y%m%d)"
git push origin main

# Option B: Create tarball for manual transfer
bash collect_results.sh
# Creates: results_archive/nips-onlinelearn_results_YYYYMMDD_HHMMSS.tar.gz
# Send this file via scp/email/cloud drive
```

### Troubleshooting

| Problem | Solution |
|---------|----------|
| Experiment interrupted | Re-run `bash run.sh` — completed phases are automatically skipped |
| Want to re-run everything from scratch | `FORCE_RERUN=1 bash run.sh` |
| GPU out of memory | The script auto-detects GPUs; ensure CUDA drivers are installed |
| Network issues downloading models | Set `HF_ENDPOINT=https://hf-mirror.com` before running |
| Check which phases completed | `ls results/.phase_markers/` |

---


## How to Run (Complete Guide)

### Requirements

- Linux server with NVIDIA GPU (4-8x A100 80GB recommended)
- CUDA 12.8 compatible driver
- `git`, `curl` installed
- ~200GB disk space (model weights + checkpoints)

### Step 1: Clone and Run (One Command)

```bash
git clone https://github.com/Sunshine535/nips-onlinelearn.git
cd nips-onlinelearn
bash run.sh
```

`run.sh` will automatically:
1. Install `uv` package manager (if not present)
2. Create Python 3.10 virtual environment
3. Install PyTorch 2.10 + CUDA 12.8
4. Install all dependencies
5. Run **all experiments** in full production mode
6. Display real-time progress in terminal and save to `run.log`

### Step 2: Monitor Progress

If running in foreground (default):
```bash
# Progress is displayed in real-time
# Press Ctrl+C to stop (can resume later with bash run.sh)
```

If running in background (recommended for long experiments):
```bash
nohup bash run.sh > run.log 2>&1 &
tail -f run.log          # Watch progress
```

### Step 3: Check Completion

```bash
cat results/.pipeline_done
# If this file exists and shows "PIPELINE_COMPLETE", all experiments finished successfully
```

### Step 4: Package and Send Results

```bash
# Option A: Push to GitHub (recommended)
git add results/ logs/
git commit -m "Experiment results $(date +%Y%m%d)"
git push origin main

# Option B: Create tarball for manual transfer
bash collect_results.sh
# Creates: results_archive/nips-onlinelearn_results_YYYYMMDD_HHMMSS.tar.gz
# Send this file via scp/email/cloud drive
```

### Troubleshooting

| Problem | Solution |
|---------|----------|
| Experiment interrupted | Re-run `bash run.sh` — completed phases are automatically skipped |
| Want to re-run everything from scratch | `FORCE_RERUN=1 bash run.sh` |
| GPU out of memory | The script auto-detects GPUs; ensure CUDA drivers are installed |
| Network issues downloading models | Set `HF_ENDPOINT=https://hf-mirror.com` before running |
| Check which phases completed | `ls results/.phase_markers/` |

### Output Structure

After completion, key results are in:

```
nips-onlinelearn/
├── results/              # All experiment outputs (JSON, figures, metrics)
│   └── .pipeline_done    # Completion marker
├── logs/                 # Per-phase log files
├── run.log               # Full pipeline log
└── results_archive/      # Packaged tarballs (after collect_results.sh)
```

---

## Project Structure

```
nips-onlinelearn/
├── README.md
├── LICENSE
├── setup.sh                          # One-click environment setup
├── requirements.txt
├── PROPOSAL.md
├── PAPERS.md
├── PLAN.md
├── EXPERIMENTS.md
├── configs/
│   └── spm_config.yaml               # Dual-LoRA + PPO hyperparameters
├── src/
│   ├── __init__.py
│   └── streaming_memory.py           # Dual-LoRA + Fisher consolidation core
├── scripts/
│   ├── gpu_utils.sh                  # Auto GPU detection utilities
│   ├── run_all_experiments.sh        # Master pipeline (entry point)
│   ├── run_spm_training.sh           # Training launcher
│   ├── train_spm.py                  # Phase 1: SPM training
│   ├── train_ppo_integration.py      # Phase 2: PPO consolidation policy
│   ├── eval_streaming.py             # Phase 3: Streaming evaluation
│   └── eval_spm.py                   # SPM-specific metrics
├── outputs/                          # Experiment outputs
│   ├── spm_training/                 # Dual-LoRA checkpoints
│   ├── ppo_policy/                   # Consolidation policy
│   ├── streaming_eval/               # Evaluation results
│   └── ablation_*/                   # Ablation study results
└── results/
```

## Experiments Overview

| # | Experiment | Script | Expected Output |
|---|-----------|--------|-----------------|
| 1 | SPM training (100 sessions) | `train_spm.py` | `outputs/spm_training/` |
| 2 | PPO consolidation policy (50 episodes) | `train_ppo_integration.py` | `outputs/ppo_policy/` |
| 3 | Streaming eval (PersonaChat + LIGHT, 4 methods) | `eval_streaming.py` | `outputs/streaming_eval/streaming_eval_results.json` |
| 4a | Ablation: consolidation frequency (5/10/20/50) | `train_spm.py` | `outputs/ablation_freq_*/` |
| 4b | Ablation: EWC lambda sweep | `train_spm.py` | `outputs/ablation_ewc_*/` |
| 4c | Ablation: LoRA rank (4/8/16/32/64) | `train_spm.py` | `outputs/ablation_rank_*/` |
| 5 | Aggregate all results | inline Python | `outputs/streaming_eval/all_results_aggregated.json` |

### Expected Results

| Metric | Baseline (single LoRA) | SPM (Ours) |
|--------|----------------------|-------------|
| Persona Retention (F1) | 0.42 | 0.68 |
| Session Adaptation (turns) | 12.3 | 4.1 |
| Forgetting Rate (cross-session) | 0.31 | 0.08 |
| LIGHT Character Consistency | 3.2/5 | 4.1/5 |
| Generation Quality (PPL) | 8.7 | 9.1 |

## Model

| Component | Model | Params |
|-----------|-------|--------|
| Base model | Qwen/Qwen3.5-9B (frozen) | 9B |
| Working-memory LoRA | r=16, online updates | ~13M |
| Long-term LoRA | r=64, consolidated | ~52M |
| Consolidation policy | 3-layer MLP, 256-dim | ~0.5M |

## Timeline & GPU Budget

| Phase | Duration | GPU-hours |
|-------|----------|-----------|
| SPM training (100 sessions) | ~6 days | 384 |
| PPO policy training (50 episodes) | ~3 days | 192 |
| Streaming evaluation (2 datasets × 4 methods) | ~1 day | 64 |
| Ablation studies (freq + EWC + rank) | ~3 days | 160 |
| Analysis + aggregation | ~0.5 day | 32 |
| **Total** | **~13.5 days** | **~832** |

## Citation

```bibtex
@inproceedings{spm2026,
  title={Streaming Parameter Memory: Dual-LoRA Online Learning for Large Language Models},
  author={Anonymous},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
