# Idea Discovery Report

**Direction**: Online learning and streaming adaptation for LLMs (NeurIPS 2026 Best Paper target)
**Date**: 2026-04-07
**Pipeline**: research-lit -> idea-creator -> novelty-check -> research-review

---

## Executive Summary

**Recommended idea**: Prove that dual-timescale adaptation is necessary and sufficient for streaming low-rank learning under multi-rate drift. A separation theorem shows single-timescale suffers Omega(T) regret while dual-timescale achieves O(sqrt(T)). Three practical corollaries follow: Fisher-precision consolidation, optimal merge frequency, and invariant-selective merging. One algorithm (Streaming Mirror-LoRA) instantiates all corollaries. Validated on synthetic benchmark with known slow/fast factors + real streaming tasks in dialogue and non-dialogue domains.

**Why best paper**: First necessity proof for dual-timescale in CL. Algorithm falls out of theory, not designed ad hoc. Broad impact beyond persona — applies to any streaming adaptation.

---

## Literature Landscape

See `LITERATURE_LANDSCAPE.md` for full survey (40+ papers, 2025-2026).

**Key competitors**:
| Paper | Venue | Threat | Gap vs Our Work |
|-------|-------|--------|----------------|
| SDFT | arXiv Jan 2026 | Highest | No necessity proof, no dual-timescale, no streaming |
| KeepLoRA | ICLR 2026 | High | Batch setting, no streaming regret bounds |
| CONEC-LoRA | arXiv Oct 2025 | High | Requires task boundaries |
| StelLA | NeurIPS 2025 | Medium | Offline Stiefel manifold, no streaming |
| SPRInG | arXiv Jan 2026 | Medium | Retrieval-based, no regret theory |

**Critical gap filled**: No prior work proves necessity of dual-timescale for online low-rank adaptation. No prior work gives regret bounds for boundary-free streaming LoRA.

---

## Ranked Ideas

### 1. RECOMMENDED: Dual-Timescale Separation Theorem + Streaming Mirror-LoRA

**Title**: *Dual-Timescale Adaptation Is Necessary and Sufficient for Streaming Low-Rank Learning under Multi-Rate Drift*

**Central Thesis**: Multi-rate drift creates an impossible tradeoff for single-timescale adaptation; separating fast tracking from slow consolidation resolves it.

**Theory (3 theorems + 3 corollaries)**:
1. **Theorem 1 (Necessity)**: Under structured two-scale drift, any single-timescale learner achieving O(sqrt(T)) on stationary streams suffers Omega(T) on multi-rate drift
2. **Theorem 2 (Sufficiency)**: Dual-timescale achieves O(sqrt(T)) regret under bounded slow subspace drift + bounded fast coefficient drift
3. **Corollary 1**: Optimal consolidation = Fisher-precision averaging = distillation + replay
4. **Corollary 2**: B* proportional to sqrt(merge_cost / slow_drift_rate)
5. **Corollary 3**: Invariant-selective consolidation weakly improves regret, strictly when transient drift is nonzero

**Algorithm**: Streaming Mirror-LoRA — online mirror updates on low-rank manifold + periodic Fisher-weighted invariant-selective consolidation

**Novelty assessment**:
- Separation theorem for streaming LoRA: **NOVEL** (closest: two-timescale SA theory, but never applied to online LoRA / structured low-rank drift)
- Streaming Mirror-LoRA: **NOVEL** as regret-bounded online method (StelLA/RiemannLoRA exist but offline, no streaming regret)
- Invariant-selective consolidation: **PARTIALLY NOVEL** (IRM+CL exists conceptually but not as formal corollary of streaming regret theory)
- Combined package A+B+C: **NOVEL** (no existing work combines all three)

**Reviewer score**: 6/10 (current state) -> 8-9/10 (after restructuring per reviewer demands)

**Pilot feasibility**: Existing SPM codebase provides ~70% of implementation. Need to add: manifold updates, synthetic benchmark, non-dialogue experiments, stronger baselines.

**Compute**: ~500 GPU-hours

---

### 2. BACKUP: Forward Transfer by Predictive Subspace Routing
- Learn adaptation directions anticipating future drift
- Beautiful but harder to validate empirically
- ~1000 GPU-hours
- **Status**: Deprioritized — Idea 1 has stronger theory + lower risk

### 3. ELIMINATED: Pure Benchmark Paper (1000 Sessions)
- High impact but benchmarks alone don't win best paper
- **Status**: Absorbed into Idea 1 as synthetic benchmark component

---

## Eliminated Ideas (from Phase 2)

| Idea | Reason for Elimination |
|------|----------------------|
| Geodesic LoRA only | Subsumed by Streaming Mirror-LoRA |
| Change-point control only | "Smart scheduling" may underwhelm reviewers |
| Reusable episodic modes | Depends on stream recurrence structure |
| TTT + CL unification | Theory too messy |
| Minimal sufficient memory | Hard to get decisive empirical win |

---

## Refined Proposal Summary

**Problem anchor**: Streaming low-rank adaptation under multi-rate drift (no task boundaries)

**One-sentence method**: Dual-timescale Mirror-LoRA with Fisher-precision invariant-selective consolidation, provably necessary and sufficient for sublinear dynamic regret.

**Dominant contribution**: First necessity proof that dual-timescale is required for streaming PEFT, with matching algorithm and optimal consolidation theory.

**Tagline**: "Multi-rate drift creates an impossible tradeoff for single-timescale adaptation; separating fast tracking from slow consolidation resolves it."

---

## Key Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Theorem assumptions too idealized | HIGH | Validate with synthetic phase transition experiments showing exact correspondence |
| Grassmann manifold overhead | MEDIUM | Ablate: show it helps but isn't the main story; main story is dual-timescale |
| Invariant identification brittle | MEDIUM | Use multiple proxy criteria (gradient stability, cross-window consistency) |
| Non-dialogue experiments weak | MEDIUM | Choose streaming text classification with clear multi-rate drift |
| Strong baselines close the gap | LOW-MEDIUM | If EMA or dual-adapter heuristic is competitive, the thesis still holds — they ARE dual-timescale |

---

## Next Steps

1. `/research-refine-pipeline` — Refine method details + experiment plan
2. Implement: upgrade SPM codebase to Streaming Mirror-LoRA
3. `/run-experiment` — Deploy synthetic + real experiments
4. `/auto-review-loop` — Iterate until submission-ready

---

## Files

- Literature landscape: `LITERATURE_LANDSCAPE.md`
- Raw idea candidates: `IDEA_CANDIDATES_RAW.md`
- Refined proposal: `refine-logs/FINAL_PROPOSAL.md` (to be updated)
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md` (to be created)
