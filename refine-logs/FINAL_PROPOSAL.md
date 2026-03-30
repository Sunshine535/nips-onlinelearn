# Research Proposal: Behavioral Distillation Consolidation for Two-Timescale Streaming Persona Retention in LLMs

## Problem Anchor

- **Bottom-line problem**: Large language models cannot adapt *within* a conversation session to user-specific knowledge while simultaneously retaining that knowledge *across* sessions. Current personalization approaches operate in batch mode, creating a fundamental latency gap.

- **Must-solve bottleneck**: Existing online adaptation methods face a fundamental plasticity-stability trade-off: fast in-session adaptation causes catastrophic forgetting, while strong forgetting prevention makes adaptation too slow. No mechanism provides both rapid session-local plasticity and durable cross-session stability simultaneously.

- **Non-goals**: NOT (1) general CL for LLMs, (2) RAG replacement, (3) multi-user scaling, (4) full model fine-tuning.

- **Constraints**: ~500 GPU-hours, Qwen3.5-9B frozen, PersonaChat + LIGHT, 6 weeks, NeurIPS 2026.

- **Success condition**: Semantic Retention F1 ≥ 0.65, adaptation ≤ 5 turns, forgetting ≤ 0.10/session, PPL degradation ≤ 5%.

## Technical Gap

Online parametric personalization of LLMs requires within-session adaptation and cross-session retention. Current methods conflate two timescales in one parameter space, forcing a plasticity-stability trade-off with no structural solution:

- **Single-LoRA online**: High plasticity → catastrophic forgetting
- **Single-LoRA + EWC/replay**: Moderate protection → slow adaptation
- **Retrieval-augmented**: No forgetting → no parametric behavioral adaptation
- **Offline re-training**: No real-time adaptation → always one session behind

**The gap**: No method provides structural separation of session-local and persistent adaptation with a principled knowledge transfer mechanism between them.

## Method Thesis

A zero-initialized residual working-memory LoRA captures session-local adaptations atop a persistent long-term LoRA. At session boundaries, behavioral distillation transfers session knowledge into LT via KL minimization on current-session data, while NTP replay on reservoir data preserves past knowledge. This achieves the first behavior-space consolidation for streaming persona retention in frozen LLMs.

## Contribution Focus

- **Dominant contribution**: Behavioral distillation consolidation — a single split-objective equation that transfers session-local parametric knowledge into persistent storage via KL divergence minimization (behavior transfer on current session) combined with NTP replay (retention on reservoir).
- **Supporting mechanism**: Two-timescale residual LoRA lifecycle (zero-init WM per session, persistent LT across sessions).
- **Not individually novel**: LoRA, KL distillation, reservoir sampling, EWC.

## Proposed Method

### Complexity Budget
- **Frozen backbone**: Qwen3.5-9B (9B params, fully frozen)
- **New trainable**: Working-Memory LoRA (~13M, r=16, zero-init/session), Long-Term LoRA (~52M, r=64, persistent)
- **Not used**: PPO, attention over memory, per-persona adapters, external models

### Core Equation (Main Novelty)

$$\Delta_{LT}' = \arg\min_{\Delta} \left[ \mathcal{L}_{ntp}(\Delta; B_{all}) + \beta \cdot \mathbb{E}_{x \in B_k}\left[D_{KL}\left(p_{\theta+\Delta_{LT}+\Delta_{WM_k}}(\cdot|x) \| p_{\theta+\Delta}(\cdot|x)\right)\right] \right]$$

Where:
- $B_{all}$: Reservoir buffer (all past sessions) — for **retention**
- $B_k$: Current session data — for **behavior transfer**
- $D_{KL}$: Teacher = combined (LT+WM), Student = updated LT
- $\beta$: Distillation weight (default 1.0)

### Algorithm

```
Initialize: LT-LoRA Δ_LT, Reservoir B_all = ∅

For each session k = 1, 2, ...:
  Δ_WM ← 0  (zero-init residual over LT)
  B_k ← ∅

  For each turn t in session k:
    c_t = conversation context (system + user persona + dialogue history)
    Generate response using (θ_base + Δ_LT + Δ_WM)
    Update Δ_WM ← Δ_WM - η·∇_{Δ_WM} L_ntp(c_t)  [1-3 steps, lr=5e-4]
    B_k ← B_k ∪ {c_t}

  # Session-end consolidation
  For step = 1 to N_consolidation:
    L_retention = L_ntp(Δ_LT'; sample(B_all))     # retention on old data
    L_transfer = D_KL(p_{LT+WM}(·|x) ∥ p_{LT'}(·|x))  # behavior transfer on B_k
    L = L_retention + β · L_transfer
    Update Δ_LT' via gradient descent

  Δ_LT ← Δ_LT'
  B_all ← reservoir_update(B_all, B_k)
  Discard Δ_WM
```

### Working Memory (Residual Design)
- **Initialization**: Zero-init (LoRA_B = 0) at each session start
- **Inference**: h = W_base·x + Δ_LT·x + Δ_WM·x (both adapters active)
- **Update**: Per-turn NTP gradient descent on WM only, 1-3 steps, lr=5e-4
- **Discarded at session end**: WM encodes session-local adaptations only

### Online Learning Signal
- Input: Full conversation context including user messages with persona facts
- Labels: Next tokens in conversation (user + assistant turns)
- The user's persona information (e.g., "I'm a teacher", "I have two cats") is in the input context
- WM learns to generate responses conditioned on this context → persona consistency improves
- Deployment-realistic: no oracle labels needed

### Replay Buffer
- **Reservoir sampling**: Uniform probability replacement
- **Size**: 5000 examples
- **Split usage**: B_k for KL distillation, B_all for NTP replay

### Failure Modes and Diagnostics
1. **KL distillation collapse/divergence**: Monitor KL loss; apply temperature scaling if needed
2. **WM destabilization**: Monitor per-turn PPL; gradient clipping, reduce steps
3. **LT saturation**: Monitor retention plateau; increase rank if needed

### Novelty and Elegance Argument

**Conceptual ancestors** (acknowledged):
- Fast weights (Schmidhuber 1992, Hinton & Plaut 1987): Dual timescale concept
- CLS theory (McClelland et al. 1995): Complementary learning systems inspiration
- LwF (Li & Hoiem 2017): Output distillation for continual learning
- Progressive networks (Rusu et al. 2016): Per-task architecture

**Our novelty**: The specific combination of (1) two-timescale residual LoRA lifecycle + (2) split-objective behavioral distillation consolidation (KL on current session + NTP replay on reservoir), applied to (3) streaming persona retention in frozen LLMs. None of the above works address this setting with this mechanism.

## Evaluation

### Methods (7)
1. Frozen (no adaptation)
2. Single-LoRA online (r=16, no protection)
3. Single-LoRA + EWC + replay (r=16)
4. Parameter-matched single-LoRA + EWC + replay (r=80, same total params)
5. Retrieval-augmented personalization (embedding similarity, top-5 context)
6. Dual-LoRA + EWC consolidation (same architecture, parameter-space consolidation)
7. **SPM (ours)**: Dual-LoRA + KL distillation consolidation

### Datasets
- PersonaChat validation (1,155 personas, multi-session)
- LIGHT test (663 characters, fantasy dialogue)

### Primary Metrics (3)
1. **Semantic Persona Retention F1** (NLI-based entailment check — paraphrase-robust)
2. **Adaptation Speed** (turns to consistent persona alignment)
3. **Cross-Session Forgetting Rate** (retention drop per additional session)

### Secondary Metrics
- Perplexity, BLEU-4, per-turn latency (ms), amortized consolidation latency (ms/session)

### Ablations
- β sweep: {0.1, 0.5, 1.0, 2.0, 5.0}
- Fisher trust-region: γ ∈ {0, 100, 1000, 5000} (tested as optional stabilizer)
- WM initialization: zero-init vs. random-init vs. copy-from-LT
- Consolidation trigger: session-end vs. surprise-triggered vs. every-N-turns
- Buffer strategy: reservoir vs. gradient-norm importance vs. loss-based
- LoRA ranks: WM r∈{4,8,16}, LT r∈{16,32,64}

### Claims
1. SPM > all single-adapter methods on retention + adaptation speed
2. Timescale separation is key (parameter-matched ablation proves capacity alone doesn't explain gains)
3. Behavioral distillation > parameter-space consolidation (same architecture, different consolidation)

## Compute & Timeline
- **Estimated GPU-hours**: ~500
  - Stage 1 SPM training: 300
  - Stage 2 Evaluation: 120
  - Stage 3 Ablations: 80
- **Timeline**: 6 weeks
  - Weeks 1-1.5: Code implementation and sanity checks
  - Weeks 2-3.5: Training (100 sessions)
  - Weeks 4-4.5: Evaluation (7 methods × 2 datasets)
  - Weeks 5-6: Ablations + paper writing
