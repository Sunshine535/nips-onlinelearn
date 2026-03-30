# Round 3 Refinement

## Problem Anchor
[VERBATIM — UNCHANGED through all rounds]

## Anchor Check
- **Original bottleneck**: Plasticity-stability trade-off in single-parameter-space online adaptation.
- **Why revised method still addresses it**: Two-timescale residual LoRA + behavioral distillation consolidation directly separates timescales. Refined consolidation objective now uses current-session data for behavior transfer and reservoir for retention.
- **Reviewer suggestions rejected as drift**: Narrowed claim language per reviewer advice — this is "streaming persona/behavioral retention," not "general online LLM personalization."

## Simplicity Check
- **Dominant contribution**: Behavioral distillation consolidation with split objective — still ONE equation.
- **Components adjusted**: Consolidation objective now has two explicit data sources (clarity, not complexity).
- **Why still smallest adequate route**: Same two components (residual WM + KL distillation), just with clearer data flow.

## Changes Made

### 1. Split Consolidation Data Sources (Method Specificity fix)
- **Reviewer said**: Clarify whether KL teacher acts on full reservoir or current-session data. Risk: projecting current WM behavior onto older contexts.
- **Action**: Split the consolidation into two complementary terms:
  - **Behavior transfer** (KL): Computed on CURRENT SESSION data B_k only. Teacher = (LT+WM_k), student = LT'. This transfers session-k's behavioral knowledge.
  - **Retention** (NTP replay): Computed on RESERVOIR data B_all. Standard NTP loss on old data to prevent LT' from forgetting previously consolidated knowledge.
  - Combined: Δ_LT' = argmin_{Δ} [ L_ntp(Δ; B_all) + β · E_{x∈B_k}[D_KL(p_{LT+WM}(·|x) ∥ p_{Δ}(·|x))] ]
- **Reasoning**: KL distillation should only project WM behavior onto contexts where WM was active (current session). Replay on old data prevents catastrophic forgetting of past sessions. This split is cleaner and more principled.
- **Impact**: Makes the data flow unambiguous. Implementable in one function call.

### 2. Narrowed Claim Language (Venue Readiness fix)
- **Reviewer said**: Don't overstate "general online LLM personalization" when benchmarks are persona retention.
- **Action**: Paper title and claims now specifically say "streaming persona retention and behavioral adaptation" rather than "general online LLM personalization." The method IS general, but the evaluation validates the persona/behavioral retention setting specifically.
- **Impact**: Claims proportional to evidence. Avoids reviewer pushback on generalizability.

### 3. Precise Parameter-Space Consolidation Baseline (Contribution Quality fix)
- **Reviewer said**: Define "parameter-space consolidation" baseline very precisely.
- **Action**: The parameter-space consolidation baseline is defined as:
  - Same dual-LoRA architecture (WM r=16, LT r=64)
  - Same session lifecycle (zero-init WM, consolidation at session end)
  - Consolidation via: L_ntp(Δ_LT; B_all) + λ · Σ_i F_i · (Δ_LT,i - Δ_LT,i^prev)² (EWC)
  - NO KL distillation
  - This isolates the value of behavior-space distillation vs. parameter-space regularization
- **Impact**: Claim 3 is now airtight — exactly the same setup, only the consolidation mechanism differs.

### 4. Paraphrase-Based Persona Probes (Validation fix)
- **Reviewer said**: Add paraphrase probes so success is not just lexical recall.
- **Action**: Persona Retention metric now has two sub-metrics:
  - **Lexical Retention**: Direct keyword matching (does the response contain persona facts?)
  - **Semantic Retention (NLI-based)**: Given persona statement P and generated response R, use a pre-trained NLI model to check if R entails P. This captures paraphrased retention, not just exact match.
  - **Primary metric**: Semantic Retention F1 (NLI-based). Lexical is secondary.
- **Impact**: Much stronger evaluation. Avoids "gaming" through keyword memorization.

### 5. Strengthened Retrieval Baseline (Validation fix)
- **Reviewer said**: Make retrieval baseline strong, not tokenistic.
- **Action**: Retrieval-augmented baseline now includes:
  - Store all user turns with persona-relevant content
  - At inference: retrieve top-5 most relevant past turns via embedding similarity
  - Prepend retrieved turns as additional context in the system prompt
  - Uses the same frozen Qwen3.5-9B base model
  - This is a genuine strong nonparametric alternative, not a straw man
- **Impact**: If SPM beats this, it demonstrates that parametric adaptation provides value beyond retrieval.

### 6. Locked Primary Metrics (Validation fix)
- **Reviewer said**: Lock around 3 primary metrics. Don't sprawl.
- **Action**: Three primary metrics:
  1. **Semantic Persona Retention F1** (NLI-based, primary)
  2. **Adaptation Speed** (turns to consistent persona alignment)
  3. **Cross-Session Forgetting Rate** (retention drop per additional session)
  Secondary (reported but not central): PPL, BLEU-4, per-turn latency, consolidation latency
- **Impact**: Clear hierarchy of metrics. Reviewers know what to focus on.

## Revised Proposal

# Behavioral Distillation Consolidation for Two-Timescale Streaming Persona Retention in LLMs

## Problem Anchor
[Unchanged]

## Technical Gap
Online parametric personalization of LLMs requires within-session adaptation and cross-session retention. Current methods conflate two timescales in one parameter space, forcing a plasticity-stability trade-off with no structural solution.

## Method Thesis
A zero-initialized residual WM-LoRA captures session-local adaptations atop a persistent LT-LoRA. At session boundaries, behavioral distillation transfers session knowledge into LT via KL minimization on current-session data, while NTP replay on reservoir data preserves past knowledge.

## Contribution
- **Dominant**: Behavioral distillation consolidation — one split-objective equation.
- **Supporting**: Two-timescale residual LoRA lifecycle.
- **Not individually novel**: LoRA, KL distillation, reservoir sampling.

## Method

### Core Equation

Δ_LT' = argmin_{Δ} [ L_ntp(Δ; B_all) + β · E_{x∈B_k}[D_KL(p_{θ+Δ_LT+Δ_WM_k}(·|x) ∥ p_{θ+Δ}(·|x))] ]

Where:
- B_all: Reservoir buffer (all past sessions) — for retention
- B_k: Current session data — for behavior transfer
- D_KL: Teacher = (LT+WM), Student = updated LT
- β: Distillation weight (default 1.0)

### Algorithm

```
Initialize: LT-LoRA Δ_LT (random or pretrained), Buffer B_all = ∅

For each session k = 1, 2, ...:
  Δ_WM ← 0  (zero-init residual)
  B_k ← ∅

  For each turn t in session k:
    c_t = conversation context up to turn t
    Generate response using (θ_base + Δ_LT + Δ_WM)
    Update Δ_WM ← Δ_WM - η·∇_{Δ_WM} L_ntp(c_t)  [1-3 steps]
    B_k ← B_k ∪ {c_t}

  # Session-end consolidation
  For step = 1 to N_consolidation:
    Sample batch from B_all → compute L_ntp(Δ_LT'; B_all)
    Sample batch from B_k → compute D_KL(p_{LT+WM}(·|x) ∥ p_{LT'}(·|x))
    L = L_ntp + β · D_KL
    Update Δ_LT' via gradient descent

  Δ_LT ← Δ_LT'
  B_all ← reservoir_update(B_all, B_k)
  Discard Δ_WM
```

### Working Memory
- Zero-init LoRA_B = 0 (residual over LT)
- Per-turn: 1-3 NTP gradient steps on WM only
- Discarded at session end

### Replay Buffer
- Reservoir sampling, 5000 capacity
- Current-session buffer B_k for KL, full reservoir B_all for NTP replay

## Evaluation

### Methods (7 total)
1. Frozen (no adaptation)
2. Single-LoRA online (r=16, no protection)
3. Single-LoRA + EWC + replay (r=16)
4. Parameter-matched single-LoRA + EWC + replay (r=80, same total params)
5. Retrieval-augmented personalization (embedding-based retrieval, top-5 context)
6. Dual-LoRA + EWC consolidation (parameter-space, no KL — precise ablation of Claim 3)
7. **SPM (ours)**: Dual-LoRA + KL distillation consolidation

### Datasets
- PersonaChat validation (1,155 personas, multi-session)
- LIGHT test (663 characters, fantasy dialogue)

### Primary Metrics (3)
1. **Semantic Persona Retention F1** (NLI-based entailment check)
2. **Adaptation Speed** (turns to consistent alignment)
3. **Cross-Session Forgetting Rate** (retention drop per session)

### Secondary Metrics
- PPL, BLEU-4, per-turn latency (ms), amortized consolidation latency (ms/session)

### Ablations
- β sweep: {0.1, 0.5, 1.0, 2.0, 5.0}
- Fisher trust-region: γ ∈ {0, 100, 1000, 5000}
- WM init: zero vs. random vs. copy-from-LT
- Consolidation trigger: session-end vs. surprise vs. every-N-turns
- Buffer: reservoir vs. gradient-norm vs. loss-based
- LoRA ranks: WM r∈{4,8,16}, LT r∈{16,32,64}

## Claims
1. SPM > all single-adapter methods on retention + adaptation speed
2. Timescale separation is key (param-matched ablation proves capacity doesn't explain the gain)
3. Behavioral distillation > parameter-space consolidation (same architecture, different consolidation)

## Compute: ~500 GPU-hours (300 training + 120 eval + 80 ablation), 6 weeks
