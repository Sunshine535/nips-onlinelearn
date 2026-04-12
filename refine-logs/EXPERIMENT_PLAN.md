# Experiment Plan: Streaming Mirror-LoRA

**Paper**: Dual-Timescale Adaptation Is Necessary and Sufficient for Streaming Low-Rank Learning under Multi-Rate Drift
**Target**: NeurIPS 2026 (best paper)
**Budget**: ~500 GPU-hours
**Date**: 2026-04-07

---

## Claims and Evidence Map

| # | Claim | Experiment | Primary Metric |
|---|-------|-----------|---------------|
| C1 | Single-timescale fails under multi-rate drift | E1 | Dynamic regret vs T |
| C2 | Dual-timescale is sufficient for O(sqrt(T)) | E1 | Dynamic regret vs T |
| C3 | Phase transition matches theory | E2 | Regret heatmap |
| C4 | Fisher-precision averaging is optimal consolidation | E5 | Post-merge loss |
| C5 | Consolidation decomposes into distillation + replay | E5 + E8 | Ablation delta |
| C6 | B* proportional to sqrt(C_m / rho_s) | E3 | Excess loss vs B |
| C7 | Invariant-selective > indiscriminate consolidation | E4 | Regret + param recovery |
| C8 | Thesis holds on dialogue | E6 | Online RetF1, forgetting |
| C9 | Thesis holds beyond dialogue | E7 | Online accuracy, forgetting |
| C10 | Each component is necessary | E8 | Ablation matrix |
| C11 | Gains not from extra compute/rank | E9 | Perf-normalized-by-budget |

---

## Experiment Blocks

### Block 1: Synthetic Theory Validation (Priority: MUST-RUN)

**E1: Controlled Two-Rate Latent Drift**
- Data: y_t = f(x_t; U_t, a_t) where U_t is slow subspace, a_t fast coefficients
- U_t: rank-r subspace rotating at rate rho_s per step
- a_t: coefficients with K_T jump changepoints of size Delta
- Ground truth known -> direct measurement of subspace recovery + coefficient tracking
- Compare: Single-LoRA-SGD vs Dual-Timescale vs EMA vs Oracle
- Output: Regret curves (Fig 3), subspace recovery error
- Compute: ~20 GPU-hours

**E2: Phase Transition Diagram**
- Grid over (rho_s, rho_f) with 10x10 resolution
- For each cell: run single-timescale and dual-timescale, record regret ratio
- Plot failure boundary where single-timescale becomes linear
- Overlay theoretical prediction
- Output: Phase diagram heatmap (Fig 4)
- Compute: ~40 GPU-hours

**E3: Consolidation Frequency Sweep**
- Fix synthetic stream, sweep B in {1, 2, 5, 10, 20, 50, 100, 200, 500}
- Measure excess loss vs B
- Overlay B* = sqrt(C_m / rho_s) prediction
- Output: U-shaped curve (Fig 5)
- Compute: ~15 GPU-hours

**E4: Invariant-Selective Consolidation Test**
- Synthetic with known invariant + transient coordinates
- Compare: consolidate-all vs consolidate-invariant-only vs consolidate-transient-only
- Measure regret + parameter recovery on invariant coordinates
- Output: Selective consolidation bar chart (Fig 6)
- Compute: ~15 GPU-hours

**E5: Merge Rule Comparison**
- After each consolidation block, compare:
  - Fisher-precision averaging
  - Uniform averaging
  - EMA (alpha=0.1)
  - Confidence-weighted heuristic
  - Random merge coefficients
- Measure post-merge loss and downstream regret
- Compute: ~10 GPU-hours

### Block 2: Real-World Dialogue Stream (Priority: MUST-RUN)

**E6: Streaming Persona Retention**
- Dataset: PersonaChat (1155 personas x 5 sessions) + LIGHT (663 characters)
- Protocol: Online adaptation without task boundaries
  - For each persona/character:
    - 5 sessions, 15 turns each
    - WM-LoRA adapts online per turn
    - Consolidation at learned frequency B*
    - Measure retention across sessions
- Methods (11 baselines + ours):
  1. Frozen (lower bound)
  2. Single-LoRA SGD
  3. Single-LoRA + EWC
  4. Single-LoRA + Replay
  5. Single-LoRA + EWC + Replay
  6. Param-matched Single-LoRA (r=80, ~65M)
  7. Dual-LoRA + EMA
  8. Dual-LoRA + Periodic Averaging
  9. Dual-LoRA + Heuristic Consolidation
  10. Retrieval-Augmented
  11. SDFT (self-distillation baseline)
  12. **Streaming Mirror-LoRA (Ours)**
- Metrics:
  - Semantic Retention F1 (NLI-based)
  - Adaptation Speed (turns to RetF1 >= 0.5)
  - Forgetting Rate (per-session decline)
  - Perplexity degradation vs frozen
  - Dynamic regret proxy (cumulative online loss)
- Long-horizon stress test: 100 sessions per persona (subset)
- Compute: ~150 GPU-hours

### Block 3: Real-World Non-Dialogue Stream (Priority: MUST-RUN)

**E7: Streaming Text Classification under Domain Drift**
- Dataset: News topic classification with temporal drift
  - Option A: 20Newsgroups with synthetic temporal ordering
  - Option B: ArXiv papers with real temporal ordering (cs.* categories, 2020-2026)
  - Option C: AG News with concept drift injection
- Protocol: Online LoRA adaptation to evolving topic distribution
  - Slow drift: gradual topic proportion shift
  - Fast drift: sudden topic appearance/disappearance
- Same 12-method comparison as E6
- Metrics:
  - Online accuracy (rolling window)
  - Forgetting (accuracy on old topics after drift)
  - Forward transfer (speed of adaptation to new topics)
  - Dynamic regret proxy
- Compute: ~80 GPU-hours

### Block 4: Ablation Matrix (Priority: MUST-RUN)

**E8: Component Ablation**
- Full method vs remove-one-component:

| Variant | Dual-TS | Manifold | Fisher Merge | Selective | Replay | Distill |
|---------|---------|----------|-------------|-----------|--------|---------|
| Full    | Y | Y | Y | Y | Y | Y |
| No dual | N | Y | Y | Y | Y | Y |
| No manifold | Y | N | Y | Y | Y | Y |
| No Fisher | Y | Y | N (uniform) | Y | Y | Y |
| No selective | Y | Y | Y | N (all) | Y | Y |
| No replay | Y | Y | Y | Y | N | Y |
| No distill | Y | Y | Y | Y | Y | N |
| B/4 | Y | Y | Y | Y | Y | Y |
| 4B | Y | Y | Y | Y | Y | Y |

- Run on: synthetic + PersonaChat + non-dialogue
- Compute: ~80 GPU-hours

### Block 5: Efficiency Analysis (Priority: SHOULD-RUN)

**E9: Rank and Compute Tradeoff**
- Sweep WM rank: {4, 8, 16, 32, 64}
- Sweep LT rank: {16, 32, 64, 128}
- Measure performance-per-parameter and performance-per-FLOP
- Compare: rank-matched single-LoRA vs dual-LoRA
- Wall-clock latency per consolidation event
- Compute: ~40 GPU-hours

---

## Compute Budget

| Block | Experiments | GPU-hours | Priority |
|-------|-----------|-----------|----------|
| 1. Synthetic | E1-E5 | 100 | MUST |
| 2. Dialogue | E6 | 150 | MUST |
| 3. Non-dialogue | E7 | 80 | MUST |
| 4. Ablation | E8 | 80 | MUST |
| 5. Efficiency | E9 | 40 | SHOULD |
| Buffer | Debug, reruns | 50 | — |
| **Total** | | **500** | |

---

## Run Order

1. **E1-E5 (synthetic)** — Validate theory first. If separation theorem doesn't show in practice, abort and rethink.
2. **E6 (dialogue)** — Validate on primary domain.
3. **E7 (non-dialogue)** — Validate generality.
4. **E8 (ablations)** — Confirm each component matters.
5. **E9 (efficiency)** — Final polish for reviewer satisfaction.

**Kill criterion**: If E1 does not show clear separation between single and dual-timescale in synthetic, the theory is wrong — stop and diagnose before real experiments.

---

## Figure Plan

| Fig | Content | Section | Purpose |
|-----|---------|---------|---------|
| 1 | Central setup schematic: stream + two-rate drift + dual-timescale learner | Intro | Communicate thesis in 20 seconds |
| 2 | Separation theorem illustration: single-timescale tradeoff cartoon | Theory | Intuition for necessity |
| 3 | Synthetic regret curves: single vs dual vs baselines | Synthetic | Main theorem validation |
| 4 | Phase transition diagram: (rho_s, rho_f) grid | Synthetic | Theory-to-practice correspondence |
| 5 | Consolidation frequency: U-shaped loss vs B with B* marker | Synthetic | Corollary 2 validation |
| 6 | Selective consolidation: invariant vs transient vs all | Synthetic | Corollary 3 validation |
| 7 | Real-world results: dialogue + non-dialogue panels | Real-world | Thesis generality |
| 8 | Ablation summary: component removal impact | Ablation | Every piece matters |

---

## Implementation Delta from Existing SPM Codebase

| Component | Status | Work Needed |
|-----------|--------|------------|
| Dual-LoRA architecture | EXISTS | Refactor to separate subspace U + coefficients a |
| Fisher estimation | EXISTS | Upgrade to precision-weighted merge |
| Behavioral distillation | EXISTS | Already implements consolidation KL |
| Reservoir replay | EXISTS | Keep as-is |
| Grassmann manifold updates | NEW | Implement online mirror descent on Grassmann |
| Invariant-selective mask | NEW | Implement cross-window gradient stability detector |
| Synthetic benchmark | NEW | Build controlled two-rate drift data generator |
| Non-dialogue task | NEW | Add streaming classification pipeline |
| NLI-based retention metric | PARTIAL | Upgrade from ROUGE-L proxy to real NLI |
| Strong baselines | PARTIAL | Add EMA, periodic avg, SDFT, subspace tracking |
| Adaptation speed metric | MISSING | Implement turns-to-threshold metric |

---

## Tracker

| Experiment | Status | Score | Notes |
|-----------|--------|-------|-------|
| E1 | PENDING | — | |
| E2 | PENDING | — | |
| E3 | PENDING | — | |
| E4 | PENDING | — | |
| E5 | PENDING | — | |
| E6 | PENDING | — | |
| E7 | PENDING | — | |
| E8 | PENDING | — | |
| E9 | PENDING | — | |
