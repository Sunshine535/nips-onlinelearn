# SPM: Streaming Parameter Memory — Short-term to Long-term LoRA Consolidation

[![NeurIPS 2026 Submission](https://img.shields.io/badge/NeurIPS-2026-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()

## Overview

**Streaming Parameter Memory (SPM)** maintains dual LoRA modules during inference: a **working-memory LoRA** that updates in real-time during conversation, and a **long-term LoRA** that stores consolidated persistent knowledge. A Fisher Information-based consolidation policy decides *when* and *what* to transfer from working memory to long-term storage, enabling genuine online personalization without catastrophic forgetting.

### Key Insight

Human memory consolidates short-term experiences into long-term storage during sleep/rest via importance-weighted replay. SPM mirrors this in parameter space: the working-memory LoRA captures immediate conversational context (user preferences, facts mentioned in-session), while a learned consolidation policy periodically merges important parameter updates into the long-term LoRA using Fisher Information as an importance score.

## Why SPM?

| Approach | Mechanism | Limitation |
|----------|-----------|------------|
| CharacterFlywheel (Meta, 2026) | Offline iterative LoRA from interaction logs | Requires batch retraining; 15+ generation cycles for convergence |
| ROSA2 (2026) | Test-time adaptation with words+weights | No explicit short/long-term separation; no consolidation policy |
| T2PAM (NeurIPS 2025) | Task-specific parameter allocation | Static allocation; no online streaming updates |
| DEAL (2025) | Dynamic expert allocation | Expert-level granularity only; no per-conversation memory |
| **SPM (Ours)** | **Dual LoRA + Fisher consolidation** | **Online streaming, explicit memory separation, learned policy** |

## Architecture

```
User Input ──→ Base Model (Qwen3.5-9B, frozen)
                    │
                    ├── Long-term LoRA (persistent knowledge)
                    │        ↑ consolidation (Fisher-weighted merge)
                    └── Working-memory LoRA (session context)
                             ↑ online gradient updates
                             │
              Consolidation Policy Network
              (decides when/what to transfer)
```

### Components

1. **Working-Memory LoRA (WM-LoRA)**: Rank-16 adapter updated via online gradient descent during conversation. Captures in-session user preferences, mentioned facts, and interaction patterns. Reset or decayed between sessions.

2. **Long-Term LoRA (LT-LoRA)**: Rank-64 adapter storing consolidated knowledge. Updated only through the consolidation policy. Persists across sessions.

3. **Consolidation Policy**: Small MLP (3-layer, 256-dim) trained via RL that takes as input:
   - Fisher Information diagonal of WM-LoRA parameters
   - Cosine similarity between WM-LoRA and LT-LoRA weight directions
   - Session interaction statistics (turn count, topic diversity)
   - Outputs: per-parameter merge coefficients in [0, 1]

4. **Fisher Information Estimator**: Maintains running diagonal Fisher estimate from conversation log-likelihoods, used as importance weighting for consolidation.

## Quick Start

```bash
conda create -n spm python=3.11 && conda activate spm
pip install -r requirements.txt

# Phase 1: Train base dual-LoRA system
bash scripts/train_dual_lora.sh

# Phase 2: Train consolidation policy via RL
bash scripts/train_consolidation_policy.sh

# Phase 3: Evaluate on PersonaChat + LIGHT
bash scripts/eval_personachat.sh
bash scripts/eval_light.sh
```

## Hardware Requirements

| Component | Requirement |
|-----------|-------------|
| GPU | 8× A100-80GB (training), 1× A100-80GB (inference) |
| VRAM per GPU | ~45GB (Qwen3.5-9B + dual LoRA + optimizer states) |
| Storage | ~200GB (model + datasets + checkpoints) |
| Training time | ~48h Phase 1, ~24h Phase 2, ~8h evaluation |

## Repository Structure

```
nips-onlinelearn/
├── src/
│   ├── model/
│   │   ├── dual_lora.py           # Dual LoRA architecture
│   │   ├── working_memory.py      # WM-LoRA online update logic
│   │   ├── long_term_memory.py    # LT-LoRA consolidation target
│   │   └── fisher_estimator.py    # Diagonal Fisher computation
│   ├── policy/
│   │   ├── consolidation_net.py   # Consolidation policy network
│   │   ├── rl_trainer.py          # PPO training for policy
│   │   └── reward.py              # Reward: consistency + persona retention
│   ├── data/
│   │   ├── persona_chat.py        # PersonaChat dataloader
│   │   ├── light_dialogue.py      # LIGHT environment interface
│   │   └── streaming_buffer.py    # Online conversation buffer
│   └── eval/
│       ├── persona_retention.py   # Long-term persona consistency
│       ├── session_adaptation.py  # In-session adaptation speed
│       └── forgetting_metrics.py  # Catastrophic forgetting measurement
├── configs/
│   ├── dual_lora_qwen9b.yaml
│   ├── consolidation_ppo.yaml
│   └── eval_config.yaml
├── scripts/
│   ├── train_dual_lora.sh
│   ├── train_consolidation_policy.sh
│   ├── eval_personachat.sh
│   └── eval_light.sh
├── PROPOSAL.md
├── PAPERS.md
├── PLAN.md
└── requirements.txt
```

## Expected Results

| Metric | Baseline (single LoRA) | SPM (Ours) | Target |
|--------|----------------------|-------------|--------|
| Persona Retention (F1) | 0.42 | 0.68+ | 0.70 |
| Session Adaptation (turns to align) | 12.3 | 4.1 | <5 |
| Forgetting Rate (cross-session) | 0.31 | 0.08 | <0.10 |
| LIGHT Character Consistency | 3.2/5 | 4.1/5 | 4.0/5 |
| Generation Quality (Perplexity) | 8.7 | 9.1 | <10.0 |

## Citation

```bibtex
@inproceedings{spm2026,
  title={Streaming Parameter Memory: Short-term to Long-term LoRA Consolidation for Online Personalization},
  author={Anonymous},
  booktitle={NeurIPS},
  year={2026}
}
```

## License

MIT License
