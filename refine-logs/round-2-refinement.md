# Round 2 Refinement

## Problem Anchor
[VERBATIM — UNCHANGED]
- **Bottom-line problem**: LLMs cannot adapt within a session while retaining knowledge across sessions. Batch-mode personalization creates a latency gap.
- **Must-solve bottleneck**: Plasticity-stability trade-off in online adaptation. No mechanism provides both simultaneously.
- **Non-goals**: NOT general CL, NOT RAG replacement, NOT multi-user scaling, NOT full model FT.
- **Constraints**: ~500 GPU-hours, Qwen3.5-9B frozen, PersonaChat + LIGHT, 6 weeks, NeurIPS 2026.
- **Success condition**: Retention F1 ≥ 0.65, adaptation ≤ 5 turns, forgetting ≤ 0.10/session, PPL ≤ +5%.

## Anchor Check
- **Original bottleneck**: Single parameter space conflates session-local and persistent knowledge.
- **Why revised method still addresses it**: The two-timescale residual LoRA + behavioral distillation consolidation directly separates timescales. The revised online signal clarification ensures user-specific knowledge actually enters the model.
- **Reviewer suggestions rejected as drift**: None — all suggestions accepted and integrated.

## Simplicity Check
- **Dominant contribution after revision**: Two-timescale residual LoRA + behavioral distillation consolidation — now expressed as ONE equation with NO optional components in the main method.
- **Components removed or merged**:
  - Surprise trigger → moved to ablation only (not part of main method)
  - Fisher trust-region → moved to ablation only (main method uses KL distillation alone)
  - Gradient-norm importance → replaced with reservoir sampling in main method (importance weighting tested in ablation)
- **Why the remaining mechanism is still the smallest adequate route**: The main method is now literally two things: (1) a second LoRA slot (zero-init residual) and (2) a KL-distillation step at session end. Nothing else.

## Changes Made

### 1. Clarified Online Learning Signal (CRITICAL fix)
- **Reviewer said**: Explain how user-specific knowledge enters the gradient. Self-supervised on own outputs could be degenerate.
- **Action**: Clarified that the online learning signal is next-token prediction on the FULL conversation context, where the USER's messages contain the new knowledge:
  - User says "I'm a teacher and I have two cats" → this IS the new information
  - Model learns to predict responses CONDITIONED on this user context
  - The gradient updates WM to better predict responses consistent with the stated persona
  - This is NOT self-training on model outputs — it's learning from the dialogue structure itself
  - Label = next token in the full conversation (including user turns and assistant turns)
  - The user's messages are the "data" and the assistant's responses are the "supervision"
  - In deployment: use next-token prediction on the actual conversation as it unfolds
- **Reasoning**: This is exactly how a deployed dialogue system would learn from interaction. The model sees the user's persona facts in context and updates its parameters to be consistent with them.
- **Impact**: Removes the main blocking issue. The learning pathway is now clearly specified and deployment-realistic.

### 2. Further Simplification of Main Method (Contribution Quality fix)
- **Reviewer said**: Remove surprise trigger, Fisher, and importance weighting from main method.
- **Action**: Main method is now ONLY:
  - Zero-init residual WM-LoRA per session
  - Per-turn NTP updates on WM (1-3 steps)
  - Reservoir-sampled replay buffer
  - Session-end KL distillation consolidation into LT
  ALL other components (surprise trigger, Fisher trust-region, importance weighting) are tested only in ablation.
- **Reasoning**: Maximum parsimony. One paper, one mechanism.
- **Impact**: The paper can now be summarized in one paragraph without mentioning any optional components.

### 3. Added Parameter-Matched and Nonparametric Baselines (Validation fix)
- **Reviewer said**: Need parameter-matched baseline and nonparametric personalization baseline.
- **Action**: Added two new baselines:
  - **Parameter-matched single-LoRA**: Single LoRA with r=80 (same total trainable params as WM r=16 + LT r=64) + EWC + replay. This controls for "extra capacity" explanation.
  - **Retrieval-augmented personalization**: Store all user turns in a retrieval buffer, prepend top-k relevant turns as context at inference. This is the nonparametric personalization frontier.
  - Also report amortized consolidation latency (time for session-end KL distillation, averaged over sessions).
- **Impact**: The comparison now isolates timescale separation from capacity and addresses the strongest alternative explanation.

### 4. Refined Novelty Positioning (Venue Readiness fix)
- **Reviewer said**: Tone down "first" claim and frame around behavioral distillation, not just dual adapters.
- **Action**: Reframed novelty as:
  - "We propose behavioral distillation consolidation for streaming LLM personalization" — the core novelty is the consolidation MECHANISM (behavior-space KL distillation at session boundaries), not just having two adapters.
  - Related work acknowledgment: Schmidhuber/Hinton fast-weights, complementary learning systems, progressive neural networks, PackNet — these are acknowledged as conceptual ancestors, but none operate in the streaming online LLM personalization setting with behavior-space consolidation.
  - Claim reframed from "first dual-adapter" to "first behavior-space consolidation for streaming parametric personalization of frozen LLMs."
- **Impact**: More defensible novelty claim that survives related work scrutiny.

## Revised Proposal

# Research Proposal: Behavioral Distillation Consolidation for Two-Timescale Online LLM Personalization

## Problem Anchor
[Same — see above]

## Technical Gap

Online parametric personalization of LLMs requires adapting to user-specific knowledge during conversation while retaining that knowledge across sessions. This demands simultaneously high plasticity (rapid in-session adaptation) and high stability (cross-session retention).

Existing approaches fail because they conflate two fundamentally different timescales in a single parameter space:
- **Single-LoRA online**: High plasticity → catastrophic forgetting
- **Single-LoRA + EWC/replay**: Moderate protection → slow adaptation
- **Retrieval-augmented**: No forgetting → no parametric adaptation (behavior doesn't evolve)
- **Offline re-training**: No real-time adaptation → one session behind

**The gap**: No method structurally separates session-local and persistent adaptation with a principled knowledge transfer mechanism between them.

## Method Thesis

A zero-initialized residual working-memory LoRA captures session-local adaptations atop a persistent long-term LoRA. At session boundaries, behavioral distillation (KL divergence minimization) transfers the combined model's knowledge into the long-term adapter alone. This achieves the first behavior-space consolidation for streaming parametric personalization of frozen LLMs.

## Contribution Focus

- **Dominant contribution**: Behavioral distillation consolidation — a single equation that transfers session-local parametric knowledge into persistent storage via KL divergence minimization between the session-adapted model (teacher) and the updated long-term adapter (student).
- **Supporting mechanism**: Two-timescale residual LoRA lifecycle (zero-init WM per session, persistent LT across sessions) — an architectural pattern that enables the consolidation mechanism.
- **Not claimed as novel**: LoRA, EWC, Fisher, reservoir sampling, KL distillation individually.

## Proposed Method

### Complexity Budget
- **Frozen**: Qwen3.5-9B (9B params)
- **Trainable**: WM-LoRA (~13M, r=16, zero-init/session), LT-LoRA (~52M, r=64, persistent)
- **Not used**: PPO, attention over memory, per-persona adapters, external models

### System Overview

```
Session k starts:
  Δ_WM ← 0  (zero-init residual over LT)

For each turn t in session k:
  Input: conversation context c_t = [system prompt, user msg, ...]
  Forward: h = f(c_t; θ_base + Δ_LT + Δ_WM)
  Online update: Δ_WM ← Δ_WM - η·∇_{Δ_WM} L_ntp(c_t)
    (1-3 gradient steps, lr=5e-4, only WM params)
  Store c_t in reservoir buffer B (uniform probability replacement)

Session k ends:
  Consolidation via behavioral distillation:
    Δ_LT' = argmin_{Δ} [ L_ntp(Δ; B) + β·E_{x∈B}[D_KL(p_{LT+WM}(·|x) ∥ p_{Δ}(·|x))] ]
  Δ_LT ← Δ_LT'
  Discard Δ_WM (zero-init next session)
```

### Core Mechanism: Behavioral Distillation Consolidation

**The central equation (main paper claim):**

Δ_LT' = argmin_{Δ} [ L_ntp(Δ; B) + β · E_{x∈B}[D_KL(p_{θ_base+Δ_LT+Δ_WM}(·|x) ∥ p_{θ_base+Δ}(·|x))] ]

- **L_ntp**: Standard next-token prediction loss on buffer B — ensures LT continues to fit the data
- **D_KL**: Behavioral distillation from (LT+WM) teacher to updated-LT student — transfers WM's session-local knowledge into LT WITHOUT losing LT's existing knowledge (because the teacher includes both)
- **β**: Balances task fit vs. behavioral preservation (tuned; default 1.0)

**Why this works**: The KL term forces the updated LT to mimic the behavior of the combined (LT+WM) model. Since (LT+WM) = (old persistent knowledge) + (new session knowledge), the distillation naturally transfers the session knowledge while maintaining the persistent knowledge. This is strictly more principled than parameter-space EWC because it operates on the model's actual output distribution, not on individual parameter values.

**What enters the gradient (online learning signal)**:
- Input: Full conversation up to turn t, including user's messages containing persona facts
- Labels: Next tokens in the conversation (both user and assistant turns)
- The user's persona information ("I'm a teacher", "I love hiking") is in the INPUT context
- WM learns to generate responses conditioned on this context → persona consistency improves
- This is deployment-realistic: no oracle labels needed

### Working Memory (Residual Design)
- **Zero-init**: LoRA_B = 0 at session start → model starts from LT-adapted state
- **Combined inference**: h = W_base·x + Δ_LT·x + Δ_WM·x
- **Online update**: 1-3 NTP gradient steps on Δ_WM only per turn
- **Discarded at session end**: WM encodes session-local adaptations only

### Replay Buffer
- **Reservoir sampling**: Uniform probability replacement (no importance weighting in main method)
- **Size**: 5000 examples
- **Used for**: Consolidation distillation at session end

### Training Plan

**Stage 1: SPM Training (100 sessions, ~300 GPU-hours)**
- PersonaChat train, grouped by persona, 20 turns/session
- Per-turn: 1-3 WM gradient steps
- Per-session-end: KL distillation consolidation into LT
- Log: per-session retention, forgetting curve, loss trajectory

**Stage 2: Streaming Evaluation (~120 GPU-hours)**
- 7 methods × 2 datasets × 50 sessions
- Methods:
  1. Frozen (no adaptation)
  2. Single-LoRA online (r=16, no protection)
  3. Single-LoRA + EWC + replay (r=16, best single-adapter CL)
  4. Parameter-matched single-LoRA + EWC + replay (r=80, same total params)
  5. Retrieval-augmented personalization (nonparametric: store+retrieve user turns)
  6. Dual-LoRA + task-loss-only consolidation (no KL distillation)
  7. **SPM (ours)**: Dual-LoRA + KL distillation consolidation
- Datasets: PersonaChat validation, LIGHT test
- Metrics:
  - Persona Retention F1 (NLI-based consistency score)
  - Adaptation Speed (turns to alignment)
  - Forgetting Rate (retention drop per session)
  - Perplexity (PPL)
  - BLEU-4
  - Per-turn latency (ms)
  - Amortized consolidation latency (ms/session)

**Stage 3: Ablation Studies (~80 GPU-hours)**
- KL distillation: β sweep (0.1, 0.5, 1.0, 2.0, 5.0)
- Optional Fisher trust-region: γ ∈ {0, 100, 1000, 5000}
- WM initialization: zero-init vs. random-init vs. copy-from-LT
- Consolidation trigger: session-end-only vs. surprise-triggered vs. every-N-turns
- Replay buffer: reservoir vs. gradient-norm importance vs. loss-based importance
- LoRA ranks: WM r∈{4,8,16}, LT r∈{16,32,64}

### Failure Modes

1. **KL distillation collapses**: Detection via KL→0 or exploding. Fix: temperature scaling, buffer diversity check.
2. **WM destabilizes**: Detection via PPL spike. Fix: gradient clipping, fewer steps.
3. **LT saturates**: Detection via retention plateau. Fix: larger rank, periodic Fisher reset.

### Novelty and Elegance Argument

**Conceptual ancestors** (acknowledged, not claimed as identical):
- Fast weights (Schmidhuber 1992, Hinton & Plaut 1987): Dual timescale concept exists, but not for LoRA, not for LLMs, not for streaming personalization
- CLS theory (McClelland et al. 1995): Inspiration, but no concrete mechanism for LoRA consolidation
- Progressive neural networks (Rusu et al. 2016): Per-task columns, not consolidated. We use one persistent adapter.
- PackNet (Mallya & Lazebnik 2018): Pruning-based CL, not behavior-space consolidation.
- Knowledge distillation for CL (Li & Hoiem 2017, LwF): Output distillation to prevent forgetting in multi-task CL. We extend this to a two-timescale adapter system for streaming personalization.

**Our novelty**: The specific combination of (1) two-timescale residual LoRA lifecycle + (2) behavioral distillation as the consolidation mechanism, applied to (3) streaming online personalization of frozen LLMs. None of the above works address this setting.

## Claim-Driven Validation

### Claim 1: SPM achieves superior plasticity-stability trade-off
- 50 PersonaChat sessions, 7 methods
- Key comparison: SPM vs. single-LoRA+EWC+replay (best single-adapter CL)
- Expected: Retention F1 ≥ 0.65 vs. ≤ 0.50 for best single-adapter

### Claim 2: Timescale separation is the key mechanism
- Ablation: parameter-matched single-LoRA (r=80, same params) vs. dual-LoRA (r=16+r=64)
- Same consolidation, different architecture
- Expected: Dual-adapter retains ≥ 15% more retention even with fewer per-adapter params

### Claim 3: Behavioral distillation is better than parameter-space consolidation
- Ablation: dual-LoRA + task-loss-only vs. dual-LoRA + KL distillation
- Expected: KL distillation variant retains more behavioral consistency

## Compute & Timeline
- ~500 GPU-hours total
- 6 weeks: 1.5 weeks code → 2 weeks training → 1 week eval → 1.5 weeks writing
