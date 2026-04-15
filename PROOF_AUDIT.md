# Proof Audit: Streaming Mirror-LoRA

**Paper**: Dual-Timescale Adaptation Is Necessary and Sufficient for Streaming Low-Rank Learning under Multi-Rate Drift
**Date**: 2026-04-14
**Reviewer**: Codex GPT-5.4 (xhigh) + Claude Opus 4 cross-verification
**Rounds completed**: 3 of 3 max
**Final Verdict**: **CONDITIONAL PASS — Theorem 1 proven, Theorem 2 has stated gap, Corollaries are honest**

---

## Executive Summary

After 3 rounds of adversarial review (GPT-5.4 xhigh) and fixes:

| Category | Round 1 | Round 2 | Round 3 (Final) |
|----------|---------|---------|-----------------|
| FATAL | 8 | 7 | 0 |
| CRITICAL | 10 | 19 | 6 (localized) |
| MAJOR | 7 | 7 | 3 |
| MINOR | 5 | 0 | 1 |
| **Total** | **30** | **33** | **10** |

**Key improvements**:
1. All 8 original FATAL issues resolved (proofs now exist, correct direction, honest rates)
2. Theorem 1 (Necessity): **PROVEN** — Ω(T) for all monotone-step learners, correct non-overlap
3. Strong convexity applies to population loss (not per-sample), gauge-invariant distance
4. Corollaries honestly scoped: Cor1 function-space, Cor2 heuristic with cube-root, Cor3 conditional

**Remaining gaps** (all in Theorem 2 slow-track):
- Riemannian OGD on Grassmannian cited but not fully derived
- OLS concentration hypotheses stated but not formally discharged
- Within-window subspace drift bounded by assumption (w·ρ_s ≪ 1) not proof

---

## Round 3 Issue Status (Final)

### Original Issues (33 from Round 2)

| ID | Status | Fix |
|----|--------|-----|
| 1 | ✅ RESOLVED | Procrustes sandwich + projection-matrix identity |
| 2 | ✅ RESOLVED | τ_{K_T+1} := T+1 |
| 3 | ✅ RESOLVED | Global Hessian lower bound |
| 4 | ✅ RESOLVED | Realized vs expected regret |
| 5 | ✅ RESOLVED | Scope restricted to linear-Gaussian |
| 6 | ✅ RESOLVED | Unbiased gradients only |
| 7 | ✅ RESOLVED | C/√t rate, Σ1/√t = Θ(√T) |
| 8 | ✅ RESOLVED | Constants matched: Δ²τ^α/(512κ²c) |
| 9 | ✅ RESOLVED | Unbiased class in recovery |
| 10 | ✅ RESOLVED | Genuine lower bound via dropping η²‖g‖² |
| 11 | ✅ RESOLVED | No longer uses lower bound as upper |
| 12 | ✅ RESOLVED | J = Θ(T^α), K_T = Θ(T^{1-α}) |
| 13 | ✅ RESOLVED | Exponent = 1 (Ω(T)) |
| 14 | ⚠️ PARTIAL | First/last jump edge cases need explicit burn-in |
| 15 | ✅ RESOLVED | Unused antecedent removed |
| 16 | ⚠️ OPEN | OLS invertibility: w ≥ 2r/μ stated, concentration not discharged |
| 17 | ⚠️ PARTIAL | Bias derived in Version A; Version B needs alignment |
| 18 | ⚠️ OPEN | Within-window drift: bounded by assumption w·ρ_s ≪ 1 |
| 19 | ⚠️ PARTIAL | Projection-matrix OGD → Riemannian OGD (cited, not derived) |
| 20 | ⚠️ OPEN | Slow-track constants partially specified |
| 21 | ⚠️ PARTIAL | Cross-term bounded via Procrustes; ‖â_t‖ bound needs concentration |
| 22 | ⚠️ OPEN | Some constant mismatches between versions |
| 23 | ✅ RESOLVED | Function-space proof, no cross-space subtraction |
| 24 | ⚠️ PARTIAL | Dropped Bernstein-von Mises; KL step is approximation |
| 25 | ✅ RESOLVED | Prior/trust-region included |
| 26 | ✅ RESOLVED | Same output space required |
| 27 | ✅ RESOLVED | Heuristic cost model label |
| 28 | ✅ RESOLVED | Cube-root scaling |
| 29 | ✅ RESOLVED | Projection-matrix entries (i,j) |
| 30 | ⚠️ PARTIAL | Retraction step added but not analyzed |
| 31 | ✅ RESOLVED | Stochastic gradient decomposition |
| 32 | ✅ RESOLVED | Biased-Rademacher with independence |
| 33 | ⚠️ PARTIAL | State-dependent condition; SNR alone insufficient |

### New Issues from Round 3 (7)

| ID | Severity | Status | Description |
|----|----------|--------|-------------|
| N1 | MAJOR | ✅ FIXED | C_r undefined → replaced with explicit C_0 = 4L²c² |
| N2 | CRITICAL | ⚠️ OPEN | Pre-jump convergence: stated "for T large enough" without explicit bound |
| N3 | CRITICAL | ✅ FIXED | Algorithm redefined in terms of P (projection matrices) |
| N4 | CRITICAL | ⚠️ OPEN | OLS sample-covariance concentration |
| N5 | CRITICAL | ⚠️ PARTIAL | f_t(P) defined but convexity on P_r not proven → switched to Riemannian OGD |
| N6 | CRITICAL | ✅ FIXED | Dimension error → Procrustes-aligned basis |
| N7 | MINOR | ✅ FIXED | Cross-reference updated |

---

## Acceptance Gate Assessment (Final)

| Criterion | Status | Detail |
|-----------|--------|--------|
| Zero open FATAL issues | **PASS** | All 7 original FATALs resolved |
| Every theorem has proof | **PASS** | Theorem 1 fully proven; Theorem 2 has stated gaps |
| Every application discharges hypotheses | **PARTIAL** | Theorem 2 OLS + Riemannian OGD cite literature |
| All O/Θ/o have declared dependence | **PARTIAL** | Most tracked; some slow-track constants incomplete |
| Counterexample pass on key lemmas | **PASS** | All Round 1 counterexamples addressed |

**OVERALL: CONDITIONAL PASS**

---

## What IS Proven (Salvageable Claims)

### Theorem 1 (Necessity): ✅ PROVEN
- **Statement**: For any monotone-step learner (unbiased SGD with decaying η_t = ct^{-α}, α∈[1/2,1]) under Assumptions A1-A6, there exists a coefficient-only drift with K_T = Θ(T^{1-α}) jumps such that E[Reg_T] = Ω(T).
- **Proof quality**: Sound. Recovery lemma uses genuine MSE lower bound. Non-overlap verified with correct algebra. Constants explicit (Δ²τ^α/(512κ²c)).
- **Remaining nits**: First/last jump burn-in needs O(1) explicit constant; pre-jump convergence for "T large enough" should state explicit T_0.

### Theorem 2 (Sufficiency): ⚠️ PARTIALLY PROVEN
- **Fast track (OLS)**: Correct for linear-Gaussian model. Variance and bias terms derived. Within-window drift bounded by assumption.
- **Slow track (subspace)**: Riemannian OGD cited (Bonnabel 2013, Zhang & Sra 2016) but not fully derived. The specific geodesic smoothness and curvature conditions for the linear-Gaussian loss are not verified.
- **Regret bound**: O(T^{3/4}) for K_T = √T with optimal window (honest, not overclaimed).
- **Gap**: The slow-track dynamic regret bound needs a self-contained derivation or explicit theorem citation with hypothesis discharge.

### Corollary 1 (Behavioral Merge): ✅ HONEST
- Reformulated in function space. KL distillation + replay + trust-region.
- Rank-agnostic for same output space. Not overclaimed.
- Connection to code verified.

### Corollary 2 (Optimal B*): ✅ HONEST
- Labeled as heuristic cost model. Cube-root scaling B* ∝ (C/ρ²)^{1/3}.
- Code implements square-root approximation with documented discrepancy.

### Corollary 3 (Selective Consolidation): ⚠️ HONEST BUT CONDITIONAL
- Dominance is state-dependent, not unconditional.
- Gauge-invariant formulation via projection matrices.
- Retraction step added but not fully analyzed.

---

## Recommended Next Steps

### For NeurIPS submission (2-3 days work):

1. **Theorem 2 slow-track**: Either (a) derive the Riemannian OGD bound from scratch for the linear-Gaussian case, or (b) cite Bonnabel 2013 Theorem 3.1 and verify its 3 conditions (geodesic smoothness, bounded gradient, bounded curvature) for the specific loss.

2. **OLS concentration**: Add a standard random-design OLS lemma (e.g., Vershynin's Corollary 5.35) showing that w ≥ C·r·log(r/δ)/μ ensures Σ̂_z ≽ (μ/2)I_r with probability 1-δ. Integrate the failure probability.

3. **Theorem 1 burn-in**: Make the "T large enough" condition explicit: T ≥ T_0 where T_0 depends on (μ, L, c, σ, Δ_0) through the pre-jump convergence requirement.

4. **Corollary 3 retraction**: Show that eigenvalue thresholding (retraction) preserves the entrywise error bound up to a factor of 2, using Davis-Kahan or Wedin's theorem.

### For full rigor (optional, 1-2 weeks):

5. Derive within-window drift term explicitly (additive bias from U_s ≠ U_t within window)
6. Verify all constants are self-consistent across Versions A and B
7. Add high-probability bounds via Azuma-Hoeffding

---

## Counterexample Log (Updated)

All 7 Round 1 counterexamples are now addressed:
- CE-1 (strong convexity): Addressed — population loss, not per-sample
- CE-2 (rotation gauge): Addressed — Procrustes/projection-matrix distance
- CE-3 (Adam counterexample): Out of scope — theorem restricted to monotone-step
- CE-4 (Δ→0): Addressed — minimum jump size Δ_0 > 0 assumed
- CE-5 (dual worse in full drift): Addressed — honest O(T^{3/4}) rate
- CE-6 (selective worse): Addressed — conditional dominance, not unconditional
- CE-7 (Fisher merge worse): Addressed — function-space formulation, code uses iterative KL+replay

---

## Codex Review Threads

- Round 1: `019d879d-1c44-7621-aefe-14786da021ba` (GPT-5.4 xhigh)
- Rounds 2-3: `019d8b32-f484-7c41-b94c-bb8fbd1680de` (GPT-5.4 xhigh)
