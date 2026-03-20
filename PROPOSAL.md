# Research Proposal: Streaming Parameter Memory (SPM)

## Title
Streaming Parameter Memory: Short-term to Long-term LoRA Consolidation for Online Personalization

## Problem Statement

Current LLM personalization approaches operate in offline batch mode: collect interaction data → retrain LoRA → deploy updated model. This creates a fundamental latency gap — the model cannot adapt *during* a conversation, only *between* conversations. CharacterFlywheel (Meta, 2026) requires 15 iterative generation cycles for convergence; ROSA2 performs test-time adaptation but lacks explicit memory separation, leading to interference between session-specific and persistent knowledge.

**Core question**: Can we build a streaming parameter memory system that adapts *within* a conversation while maintaining *across*-conversation persistent knowledge, using principled importance-based consolidation?

## Hypothesis

A dual-LoRA architecture with learned Fisher Information-based consolidation achieves:
1. **Faster in-session adaptation** than single-LoRA online learning (by isolating working memory)
2. **Better cross-session retention** than naive online updates (by protecting long-term parameters)
3. **Comparable generation quality** to offline fine-tuning (by importance-weighted consolidation)

## Theoretical Foundation

### Complementary Learning Systems (CLS) Theory
McClelland et al. (1995) proposed that biological memory uses complementary fast (hippocampus) and slow (neocortex) learning systems. SPM instantiates this in parameter space:
- **Fast system (WM-LoRA)**: High learning rate (1e-4), rapid adaptation, session-scoped
- **Slow system (LT-LoRA)**: Updated only via consolidation, persistent across sessions

### Fisher Information as Importance
The diagonal Fisher Information F_ii = E[(∂log p(y|x,θ)/∂θ_i)²] measures how much parameter θ_i contributes to the model's predictions. We use this as a principled importance score for consolidation:

**Consolidation rule**: θ_LT ← θ_LT + α · σ(Policy(F, sim, stats)) ⊙ (θ_WM - θ_LT)

where σ(Policy(·)) outputs per-parameter merge coefficients based on Fisher scores, parameter similarity, and session statistics.

### Why Not Simple EWC/EMA?
- **EWC** protects old parameters but doesn't distinguish short/long-term
- **EMA** merges uniformly without importance weighting
- **SPM** learns *what* to consolidate and *when*, via an RL-trained policy

## Method

### Phase 1: Dual LoRA Training (Weeks 1-3)
- Initialize Qwen3.5-9B with WM-LoRA (rank 16) and LT-LoRA (rank 64)
- Train WM-LoRA online update mechanism on PersonaChat conversations
- Training objective: next-token prediction with persona-conditioned prefix
- WM-LoRA update rule: single-step SGD on each conversation turn
- Fisher estimation: exponential moving average over turn-level gradients

### Phase 2: Consolidation Policy Training (Weeks 3-5)
- Policy network: MLP(Fisher_diag ⊕ cos_sim ⊕ session_stats) → merge_coefficients
- RL setup: PPO with reward = λ₁·persona_retention + λ₂·adaptation_speed - λ₃·forgetting_penalty
- Episode: simulated multi-session interaction (5-10 sessions, 10-20 turns each)
- Consolidation triggered every K turns (K learned as part of policy)

### Phase 3: Evaluation (Weeks 5-7)
- **PersonaChat**: 1,155 test personas, 5 sessions per persona, measure retention across sessions
- **LIGHT**: 663 characters, situated dialogue in fantasy setting, measure character consistency
- **Ablations**: WM-only, LT-only, EMA consolidation, random consolidation, Fisher-only (no RL)

## Key Experiments

### Experiment 1: Online Adaptation Speed
Measure turns-to-alignment on PersonaChat: how many turns until the model's responses reflect the user's stated persona traits.
- Metric: Persona-trait F1 between generated responses and ground-truth persona
- Baselines: frozen model, single LoRA online, ROSA2-style adaptation

### Experiment 2: Cross-Session Retention
After 5 sessions with the same persona, measure how much persona information is retained without explicit re-statement.
- Metric: Unprompted persona recall rate, persona consistency score (NLI-based)
- Baselines: session-independent, naive LoRA accumulation, EWC-regularized

### Experiment 3: Consolidation Policy Analysis
Visualize what the policy learns to consolidate: which layers, which parameter directions, at what conversation points.
- Fisher magnitude distribution across layers at consolidation time
- Correlation between merge coefficients and downstream retention

### Experiment 4: Scaling with Session Count
Stress test: 50 sessions per persona. Does LT-LoRA saturate? Does forgetting accumulate?
- Track retention curve over session count
- Compare with CharacterFlywheel's iterative retraining trajectory

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| WM-LoRA online updates destabilize generation | High | Constrain WM-LoRA norm; use gradient clipping |
| Fisher estimates too noisy from few turns | Medium | EMA smoothing; minimum sample threshold |
| Consolidation policy doesn't converge | Medium | Pre-train with heuristic rule; warm-start RL |
| Dual LoRA inference overhead too large | Low | Merge LT-LoRA into base; only WM-LoRA adds overhead |
| PersonaChat too simple to show gains | Low | LIGHT provides more complex character scenarios |

## Timeline

| Week | Task | Deliverable |
|------|------|-------------|
| 1-2 | Dual LoRA infrastructure + WM-LoRA online updates | Working online update loop |
| 3 | Fisher estimator + consolidation policy network | Policy network training code |
| 4-5 | RL training of consolidation policy | Trained policy checkpoint |
| 5-6 | PersonaChat + LIGHT evaluation | Full metrics table |
| 6-7 | Ablations + analysis + paper writing | Draft paper |

## Compute Budget

- Phase 1 training: 8× A100-80GB × 48h = 384 GPU-hours
- Phase 2 RL training: 8× A100-80GB × 24h = 192 GPU-hours
- Evaluation + ablations: 8× A100-80GB × 16h = 128 GPU-hours
- **Total: ~704 GPU-hours**

## Success Criteria

1. Persona retention F1 ≥ 0.65 (vs. ~0.42 single-LoRA baseline)
2. Session adaptation ≤ 5 turns (vs. ~12 turns baseline)
3. Forgetting rate ≤ 0.10 per session (vs. ~0.31 baseline)
4. No more than 5% perplexity degradation vs. frozen base model
