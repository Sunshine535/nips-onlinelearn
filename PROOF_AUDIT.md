# Proof Audit: Streaming Mirror-LoRA

**Paper**: Dual-Timescale Adaptation Is Necessary and Sufficient for Streaming Low-Rank Learning under Multi-Rate Drift
**Date**: 2026-04-14
**Reviewer**: Codex GPT-5.4 (xhigh) + Claude Opus 4 cross-verification
**Round**: 1
**Verdict**: **FAIL — Acceptance gate NOT met**

---

## Executive Summary

The project claims 3 theorems + 3 corollaries but **no formal proofs exist anywhere in the repository**. The theoretical claims are stated as one-line assertions in proposal documents. Cross-model adversarial review (GPT-5.4 xhigh) identified **30 issues**, including **8 FATAL** and **10 CRITICAL** problems. Key findings:

1. **No proofs exist** — all claims are unproven assertions
2. **Strong convexity assumption is false** — fails for LLM NTP loss AND the synthetic benchmark's per-sample loss
3. **"Any single-timescale algorithm" is indefensible** — proof sketch only applies to decaying-LR SGD, not Adam/AdaGrad/restarts
4. **Corollaries 1-3 are empirically contradicted** — saved synthetic results show Fisher merge, selective consolidation, and optimal B* claims fail
5. **Theory-code gap** — the synthetic benchmark algorithm differs fundamentally from the LLM algorithm
6. **Dead code** — Fisher-precision merge, invariant detector, and optimal interval are defined but never called

---

## Issue Ledger (30 Issues)

### FATAL Issues (8)

| ID | Category | Location | Statement | Why Invalid | Fix Strategy |
|----|----------|----------|-----------|-------------|-------------|
| 1 | UNJUSTIFIED_ASSERTION | All claims | 3 theorems + 3 corollaries presented as formal results | No proofs exist anywhere | WRITE_PROOFS or DOWNGRADE_TO_CONJECTURES |
| 2 | HIDDEN_ASSUMPTION | Thm1-2 | l_t(theta) is mu-strongly convex | Fails for LLM NTP loss; fails for per-sample squared loss (rank-1 Hessian) | WEAKEN_CLAIM: use expected/batch loss |
| 3 | MODEL_MISMATCH | Setting | Strong convexity in theta supports (U,a) analysis | theta=Ua is bilinear, non-identifiable (rotation gauge) | ADD_DERIVATION: fix gauge, use manifold analysis |
| 6 | QUANTIFIER_ERROR | Thm1 | "Any single-timescale algorithm" | Proof sketch only covers decaying-LR SGD; Adam/restarts are counterexamples | WEAKEN_CLAIM: restrict to specific class |
| 16 | CLAIM_EVIDENCE_MISMATCH | Thm2 | Synthetic supports sufficiency | Saved E1 shows dual WORSE than single in full drift regime | WEAKEN_CLAIM: narrow to coefficient-only |
| 17 | MODEL_MISMATCH | Thm-to-algorithm | Synthetic learner validates LLM algorithm | Window OLS != WM AdamW + KL/replay | SEPARATE: toy theorem vs practical heuristic |
| 18 | IMPLEMENTATION_MISMATCH | Cor1 | Fisher-precision merge theta_merge = ... | WM (r=16) and LT (r=64) have different shapes; merge undefined | WEAKEN_CLAIM or require equal shapes |
| 19 | UNJUSTIFIED_ASSERTION | Cor1/MC-5 | Fisher merge "decomposes into" distillation + replay | No derivation exists; code implements iterative KL+replay, not closed-form | REPHRASE as heuristic analogy |

### CRITICAL Issues (10)

| ID | Category | Location | Statement | Why Invalid |
|----|----------|----------|-----------|-------------|
| 4 | NONUNIQUENESS | Setting | V_s, V_f well-defined | Basis-dependent; U_t Frobenius distance is gauge-dependent |
| 5 | HIDDEN_ASSUMPTION | Setting | theta*_t = U_t a_t + r_t for LLMs | Only enforced in synthetic toy; no LLM evidence |
| 7 | CLAIM_EVIDENCE_MISMATCH | Thm1 sketch | Recovery = Theta(sqrt(tau)) | Only for decaying-LR SGD under strong convexity |
| 8 | HIDDEN_ASSUMPTION | Thm1 sketch | Recovery costs don't overlap | Same order as jump spacing; arithmetic error in PROOF_SKELETON |
| 10 | MODEL_MISMATCH | Synthetic E1 | E1 tests K_T=Theta(sqrt(T)) | Fixed jump_interval gives K_T=Theta(T) |
| 13 | PROBABILISTIC_MODE | All theorems | Reg_T bounds hold | Pathwise? Expectation? w.h.p.? Unspecified |
| 15 | BOUND_MISMATCH | Thm2 summary | Dual achieves O(sqrt(T)) | With V_f=Theta(sqrt(T)), bound is O(T^{3/4}) |
| 20 | DEAD_CODE | Cor1-3 impl | Mirror-LoRA implements corollaries | Fisher merge, invariant mask, optimal B* never called |
| 23 | DEFINITIONAL_GAP | Cor3 | "Invariant coordinates" well-defined | No formal definition; detector is ad-hoc heuristic |
| 27 | CLAIM_EVIDENCE_MISMATCH | Thm-baselines | Necessity explains LLM baseline weakness | Baselines use constant-lr AdamW, not decaying SGD |

### MAJOR Issues (7)

| ID | Category | Location | Statement | Why Invalid |
|----|----------|----------|-----------|-------------|
| 9 | EDGE_CASE_FAILURE | Thm1 | Omega(T) with K_T=sqrt(T) | No lower bound on Delta; if Delta->0, regret->0 |
| 11 | IMPLEMENTATION_MISMATCH | Phase diagram | rho_f = fast drift rate | rho_f unused when jumps active; just sets jump_interval |
| 12 | DEFINITIONAL_GAP | Synthetic | Code measures dynamic regret | Compares to generator theta*, not per-round minimizer |
| 14 | PARAMETER_DEPENDENCE | Thm2 | C1,C2,C3 constants | Hide essential d,r,mu,L,sigma,B dependence |
| 21 | CIRCULAR_DEPENDENCE | Cor2/Thm2 | B* from Thm2 | Thm2 constants depend on B; can't optimize over B |
| 24 | CLAIM_EVIDENCE_MISMATCH | Cor3/E4 | Selective >= indiscriminate | E4 shows opposite: dual_selective > dual_all (worse) |
| 25 | CLAIM_EVIDENCE_MISMATCH | Cor1/E5 | Fisher merge optimal | E5 shows fisher_weighted worse than uniform and EMA |

### MINOR Issues (5)

| ID | Category | Location | Statement | Why Invalid |
|----|----------|----------|-----------|-------------|
| 22 | CLAIM_EVIDENCE_MISMATCH | Cor2/E3 | Frequency sweep shows U-shape | E3 is nearly monotone, best at B=2 |
| 26 | IMPLEMENTATION_MISMATCH | E4/E5 | Tests validate corollaries | Synthetic mechanisms differ from LLM code |
| 28 | EDGE_CASE_FAILURE | Setting | Applies for r<=d | r=d makes Grassmann a singleton |
| 29 | HIDDEN_ASSUMPTION | Synthetic | Jump size fixed | Coefficient norm unbounded, grows with K_T |
| 30 | CLAIM_EVIDENCE_MISMATCH | Paper-wide | Synthetic validates theorem | At best validates narrow toy intuition |

---

## Counterexample Log

### CE-1: Strong convexity failure (Issue #2)
**Target**: Assumption A1 (mu-strong convexity)
**Construction**: Take d=2, single sample x_t. Loss l_t(theta) = 0.5*(theta^T x_t - y_t)^2. Hessian = x_t x_t^T, which is rank-1, NOT positive definite for d>1.
**Status**: VERIFIED COUNTEREXAMPLE. Even the synthetic benchmark violates this assumption per-sample.

### CE-2: Rotation gauge (Issue #3)
**Target**: V_s = sum ||U_{t+1} - U_t||_F well-defined
**Construction**: Take U_t and U_t' = U_t @ R where R is orthogonal. Both represent the SAME subspace on Gr(d,r) but ||U_t - U_t'||_F can be arbitrarily large.
**Status**: VERIFIED COUNTEREXAMPLE.

### CE-3: Adam as single-timescale (Issue #6)
**Target**: "Any single-timescale algorithm" suffers Omega(T)
**Construction**: Adam with constant learning rate and adaptive preconditioner. After a jump, the effective step size in the direction of the jump is large (because the second-moment estimate is initially small for the new direction). Recovery may be fast.
**Status**: CANDIDATE COUNTEREXAMPLE — needs empirical verification.

### CE-4: Delta -> 0 (Issue #9)
**Target**: Omega(T) lower bound
**Construction**: Set Delta_T = T^{-2}. Then total drift V_f = K_T * Delta_T = sqrt(T) * T^{-2} -> 0. Regret also -> 0, not Omega(T).
**Status**: VERIFIED COUNTEREXAMPLE.

### CE-5: Saved E1 full-drift results (Issue #16)
**Target**: Theorem 2 sufficiency
**Construction**: From saved outputs, at T=5000 with full drift: single=953652, dual=1363734. Dual is WORSE.
**Status**: VERIFIED COUNTEREXAMPLE from project's own experiments.

### CE-6: Saved E4 selectivity results (Issue #24)
**Target**: Corollary 3 (selective >= indiscriminate)
**Construction**: From saved outputs: dual_selective=1320842 vs dual_all=1224875. Selective is WORSE.
**Status**: VERIFIED COUNTEREXAMPLE from project's own experiments.

### CE-7: Saved E5 merge results (Issue #25)
**Target**: Corollary 1 (Fisher merge optimal)
**Construction**: From saved outputs: Fisher-weighted worse than uniform and EMA.
**Status**: VERIFIED COUNTEREXAMPLE from project's own experiments.

---

## Acceptance Gate Assessment

| Criterion | Status |
|-----------|--------|
| Zero open FATAL issues | **FAIL** — 8 FATAL issues |
| Every theorem has explicit hypotheses + proof | **FAIL** — no proofs exist |
| Every application discharges hypotheses | **FAIL** — no hypothesis discharge |
| All O/Theta/o statements have declared dependence | **FAIL** — constants unspecified |
| Counterexample pass on all key lemmas | **FAIL** — 7 counterexamples found |

**OVERALL: FAIL**

---

## Salvage Analysis

### What IS salvageable

1. **Narrow toy lower bound**: For a restricted class of decaying-step-size, single-recursion algorithms on expected quadratic loss with known fixed subspace and discrete jumps, a lower bound of Omega(T) on expected regret is plausible and provable.

2. **Narrow toy upper bound**: For window-OLS on coefficients + slow decaying subspace gradient, an upper bound on expected regret is provable in the same linear-Gaussian setting.

3. **Empirical separation**: The coefficient-only isolation test (synthetic_benchmark.py lines 603-617) cleanly shows the mechanism: window OLS recovers in constant time while decaying SGD slows down. This is solid empirical evidence for a specific algorithm comparison.

4. **LLM heuristic**: The dual-LoRA + KL distillation + replay approach is a well-motivated heuristic inspired by CLS theory, even without formal regret bounds. The Mirror-LoRA Fisher trust-region genuinely prevents PPL degradation (CLAUDE.md key results).

### What MUST change

| Current Claim | Salvageable Version |
|---------------|-------------------|
| "Theorem 1 (Necessity): ANY single-timescale..." | "Proposition 1: For decaying-step OGD on strongly-convex batch loss..." |
| "Theorem 2 (Sufficiency): O(sqrt(T))" | "Proposition 2: Window-OLS + slow gradient achieves O(T^{3/4}) in expectation" |
| "Cor1: Fisher merge = distill + replay" | "Remark: Under Laplace approximation, precision-weighted merge is analogous to distillation" |
| "Cor2: B* = sqrt(C/rho)" | "Heuristic: B* = sqrt(C/rho) from stylized cost model" |
| "Cor3: Selective dominates" | DROP or "Conjecture: Under formal invariant definition..." |
| "Theory-derived algorithm" | "Theory-inspired algorithm" |

### Recommended restructuring

**Option A (recommended): Narrow theorem + strong empirical story**
- Prove a clean, narrow separation result for the linear-Gaussian setting
- Clearly label it as a "motivating theorem" for the real algorithm
- Focus the paper on the empirical story: Mirror-LoRA works well on real LLMs
- Relegate the theorem to Section 3 "Motivating Theory" (1-2 pages)
- Main contribution becomes: a well-designed dual-timescale heuristic + comprehensive evaluation

**Option B: Full theory paper**
- Requires 3-6 months of additional theoretical work
- Must handle non-convexity, gauge invariance, probabilistic bounds
- High risk: may not succeed before NeurIPS deadline

**Option C: Reject/withdraw theorem claims entirely**
- Pure empirical paper with strong baselines and ablations
- Weaker framing but avoids theoretical landmines

---

## Round 1 Complete

**Next steps**:
1. Author decides between Option A/B/C
2. If Option A: write the narrow theorem proofs (estimated 2-3 pages of LaTeX)
3. Re-submit for Round 2 review
4. Wire dead code (Fisher merge, invariant detector, optimal B*) or remove claims

---

## Appendix: Codex Review Thread

**ThreadId**: `019d879d-1c44-7621-aefe-14786da021ba`
**Model**: GPT-5.4 (xhigh reasoning)
**Round**: 1 of 3 max
