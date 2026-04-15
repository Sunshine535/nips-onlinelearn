# Narrative Report: Streaming Mirror-LoRA

**Paper**: Dual-Timescale Adaptation Is Necessary and Sufficient for Streaming Low-Rank Learning under Multi-Rate Drift
**Target**: NeurIPS 2026
**Date**: 2026-04-14
**Pipeline Stage**: Stage 3 (partial) — synthetic experiments complete, LLM experiments pending

---

## 1. Problem Statement & Core Claim

**Problem**: In streaming learning with LoRA-adapted LLMs, the data distribution drifts at multiple rates — fast coefficient changes (topic shifts, user preferences) and slow subspace evolution (language patterns, domain drift). Single-timescale adaptation fails because decaying step sizes needed for convergence prevent fast recovery from abrupt changes.

**Core Claim**: Separating adaptation into fast (coefficient tracking) and slow (subspace consolidation) timescales is both *necessary* and *sufficient* for sublinear regret under multi-rate drift, provided the timescales are genuinely separated.

**Contribution Summary**:
1. **Necessity theorem** (PROVEN): Any monotone-step learner incurs Ω(T) regret under adversarial two-rate drift
2. **Sufficiency theorem** (PARTIALLY PROVEN): Dual-timescale achieves O(T^{3/4}) — honest rate, not overclaimed
3. **Mirror-LoRA algorithm**: Practical instantiation with Fisher trust-region consolidation
4. **Timescale separation condition**: Theory predicts *when* dual-timescale fails (validated empirically)

---

## 2. Method Summary

**Mirror-LoRA** maintains two LoRA adapters:
- **Working Memory (WM, rank 16)**: Fast-adapted via standard AdamW on current data
- **Long-Term Memory (LT, rank 64)**: Periodically consolidated via behavioral merge:
  - KL distillation from WM on session data
  - NTP replay on reservoir buffer
  - Fisher trust-region to prevent catastrophic forgetting

**Key design choices**:
- Function-space behavioral merge (rank-agnostic, avoids parameter-space mismatch)
- Consolidation period B* ∝ (C_merge / ρ_s²)^{1/3} (cube-root, corrected from original square-root)
- Selective consolidation using gradient-sign stability (conditional dominance)

---

## 3. Key Quantitative Results

### Synthetic Experiments (COMPLETE)

| Experiment | Result | Status |
|------------|--------|--------|
| E1: Coefficient-only separation | 524x dual/single at T=20000 | ✅ STRONG |
| E1: Full-drift regime | 0.61-0.71x (dual WORSE) | ⚠️ GENUINE LIMITATION |
| V2: K_T sweep (decreasing jumps) | 30-117x separation | ✅ STRONG |
| V2: Timescale separation sweep | 96x at ρ_s=0.0001; 0.68x at ρ_s=0.005 | ✅ VALIDATES THEORY |
| E3: Consolidation frequency sweep | Near-monotone, best at B=2 | ⚠️ DOESN'T MATCH B* FORMULA |
| E4: Selectivity | dual_selective > dual_all (worse) | ⚠️ CONDITIONAL CLAIM |
| E5: Merge rules | Fisher-weighted < uniform < EMA | ⚠️ HEURISTIC |

**Key insight**: The timescale separation sweep (V2) transforms the full-drift negative result from a weakness into a *prediction*. The theory correctly identifies when dual-timescale will fail.

### LLM Experiments (PENDING — BLOCKED)

| Experiment | Status | Blocker |
|------------|--------|---------|
| E6: Dialogue (PersonaChat/LIGHT) | ❌ NOT RUN | Transformers version |
| E7: Non-dialogue (AG News/ArXiv) | ❌ NOT RUN | Transformers version |
| E8: Ablations (WM rank, LT rank, B) | ❌ NOT RUN | Transformers version |
| E9: Efficiency (wall-clock, memory) | ❌ NOT RUN | Transformers version |

**Fix required**: `pip install transformers>=4.51.0` (Qwen3.5 architecture support)

**CLAUDE.md claims** (Qwen2.5-7B: AvgRet=0.169, PPL=23.6):
**STATUS: UNVERIFIED** — no result files found. These appear to be aspirational claims drafted before experiments ran. Must be treated as hypotheses, not results.

---

## 4. Theory Status

| Claim | Status | Evidence |
|-------|--------|----------|
| Theorem 1 (Necessity): Ω(T) for monotone-step | ✅ PROVEN | 3-round adversarial review (GPT-5.4 xhigh) |
| Theorem 2 (Sufficiency): O(T^{3/4}) | ⚠️ PARTIALLY PROVEN | Fast track OK; slow track cites Riemannian OGD |
| Timescale separation condition | ✅ STATED + VALIDATED | Proposition + V2 sweep data |
| Corollary 1 (Behavioral merge) | ✅ HONEST | Function-space, rank-agnostic |
| Corollary 2 (Optimal B*) | ⚠️ HEURISTIC | Cube-root corrected; code uses square-root |
| Corollary 3 (Selective consolidation) | ⚠️ CONDITIONAL | State-dependent, not unconditional |

**5 remaining proof gaps** (2-3 days work):
1. Riemannian OGD: verify geodesic smoothness for linear-Gaussian
2. OLS concentration: add sample-covariance bound
3. Theorem 1 burn-in: explicit T_0
4. Within-window drift: derive additive bias
5. Corollary 3 retraction: bound eigenvalue-thresholding error

---

## 5. Honest Assessment

### What the paper CAN claim (with current evidence):
1. **Theoretical**: Monotone-step learners provably fail under multi-rate drift; dual-timescale achieves honest O(T^{3/4}) rate
2. **Algorithmic**: Mirror-LoRA is a well-motivated dual-LoRA system with Fisher trust-region
3. **Empirical (synthetic)**: Clear separation under controlled conditions; theory predicts when it fails
4. **Timescale separation**: Novel theoretical + empirical contribution

### What the paper CANNOT claim (without LLM experiments):
1. ❌ Any specific performance numbers on Qwen or any LLM
2. ❌ Superiority over baselines on real tasks
3. ❌ Practical usefulness beyond synthetic setting

### Risks:
1. **LLM experiments may not show improvement** — the theory gap (strongly convex → non-convex LLM) is acknowledged but the heuristic may still not work
2. **13 baselines are implemented** — some may outperform Mirror-LoRA
3. **E3 frequency sweep** doesn't match B* formula — may need hyperparameter tuning

---

## 6. Next Steps (Priority Order)

### Immediate (today):
1. ✅ Fix transformers version requirement
2. ✅ Add timescale separation condition to theory
3. ✅ Write narrative report

### Next GPU session (1-2 days):
4. `pip install transformers>=4.51.0` on GPU server
5. Run LLM smoke test: `python scripts/run_streaming_eval.py --model Qwen/Qwen3.5-9B --methods mirror_lora,spm,single_lora --sessions 5 --gpus 0`
6. If smoke test passes: full E6-E9 (estimated 300 GPU-hours on 8×A100)

### After experiments (1-2 days):
7. Auto-review Round 2 with LLM results
8. Fill remaining proof gaps
9. Paper writing pipeline

---

## 7. Figure/Table Inventory

| Figure | Exists | Source |
|--------|--------|--------|
| Timescale separation sweep (ρ_s vs ratio) | ✅ Data exists | outputs/synthetic_v2/ |
| Coefficient-only separation (T vs regret) | ✅ Data exists | outputs/synthetic/e1 |
| Method comparison table (LLM) | ❌ MISSING | Awaiting E6-E7 |
| Ablation tables | ❌ MISSING | Awaiting E8 |
| Architecture diagram | ❌ MISSING | Need to create |
| Theory illustration (dual vs single trajectories) | ❌ MISSING | Need to create |

---

## 8. Research Methodology Notes

Per the user's research methodology principles:
- The full-drift negative result is treated as a **design constraint** and **mechanism evidence**, not a project endpoint
- The Qwen performance claims are treated as **unverified hypotheses**, not facts
- The project is in **research pushing mode** — next step is running LLM experiments, not writing up current results as final
