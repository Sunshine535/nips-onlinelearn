# Proof Skeleton: Streaming Mirror-LoRA

**Paper**: Dual-Timescale Adaptation Is Necessary and Sufficient for Streaming Low-Rank Learning under Multi-Rate Drift
**Date**: 2026-04-14
**Status**: Phase 0.5 — Proof Obligation Ledger

---

## 1. Dependency DAG

```
A1 (mu-strong convexity) ──┐
A2 (L-smoothness)          ├──> Thm1 (Necessity) ──> Main Claim
A3 (two-rate decomposition)┤                         ┌─> Cor1 (Optimal Merge)
A4 (bounded slow drift V_s)├──> Thm2 (Sufficiency) ──┼─> Cor2 (Optimal B*)
A5 (bounded fast drift V_f)┤                         └─> Cor3 (Selectivity)
A6 (bounded noise sigma)   ┘
D1 (theta*_t = U_t a_t + r_t) ──> A3
D2 (U_t on Gr(d,r))           ──> A4
D3 (a_t piecewise constant)   ──> A5
```

**Cycle check**: NO CYCLES DETECTED. All edges point forward.

---

## 2. Complete Claim Inventory

### Theorem 1 (Necessity)
**Source**: FINAL_PROPOSAL.md:35, IDEA_REPORT.md:43
**Statement**: Any single-timescale algorithm achieving O(sqrt(T)) on stationary streams suffers Omega(T) on valid two-scale drift sequences with K_T = Theta(sqrt(T)) fast jumps.

### Theorem 2 (Sufficiency)
**Source**: FINAL_PROPOSAL.md:37, IDEA_REPORT.md:44
**Statement**: Dual-timescale with fast online updates + periodic slow consolidation achieves Reg_T <= C1 sqrt(T(V_f+1)) + C2 sqrt(T(V_s+1)) + C3.

### Corollary 1 (Optimal Merge)
**Source**: FINAL_PROPOSAL.md:39
**Statement**: Fisher-precision averaging theta_merge = (sum F_i)^{-1}(sum F_i theta_i). Decomposes into distillation on recent data + replay on past data.

### Corollary 2 (Optimal Frequency)
**Source**: FINAL_PROPOSAL.md:41, code: StreamingMirrorLoRA.compute_optimal_interval
**Statement**: B* proportional to sqrt(C_merge / rho_drift).

### Corollary 3 (Selectivity)
**Source**: FINAL_PROPOSAL.md:43
**Statement**: Consolidating only invariant coordinates weakly dominates indiscriminate consolidation.

---

## 3. Assumption Ledger

| ID | Assumption | Stated In | Used By | Verified Where |
|----|-----------|-----------|---------|----------------|
| A1 | l_t(theta) is mu-strongly convex in theta | FINAL_PROPOSAL.md:33 | Thm1, Thm2 | **UNVERIFIED** — NTP loss for LLMs is NOT strongly convex |
| A2 | l_t(theta) is L-smooth in theta | FINAL_PROPOSAL.md:33 | Thm1, Thm2 | **UNVERIFIED** — smoothness in LoRA subspace not established |
| A3 | theta*_t = U_t a_t + r_t decomposition | FINAL_PROPOSAL.md:33 | Thm1, Thm2 | **PARTIALLY** — synthetic benchmark enforces this by construction |
| A4 | V_s := sum ||U_{t+1} - U_t||_F bounded | Implied | Thm2 | **UNVERIFIED** — no bound on subspace drift in real data |
| A5 | V_f := sum ||a_{t+1} - a_t|| bounded, K_T jumps | Implied | Thm1, Thm2 | **UNVERIFIED** — real data has no discrete jumps |
| A6 | Noise sigma bounded | Implied | Thm1, Thm2 | Trivially true for bounded loss |
| A7 | rank r known and fixed | Implied by architecture | All | Hardcoded (r=16, r=64) |
| A8 | Single-timescale = one learning rate schedule | Thm1 premise | Thm1 | **CRITICAL**: definition scope unclear |

---

## 4. Typed Symbol Table

| Symbol | Type | Domain | Depends On | Notes |
|--------|------|--------|-----------|-------|
| T | scalar | N+ | — | Time horizon |
| d | scalar | N+ | — | Input dimension |
| r | scalar | N+ | — | True rank of optimal subspace |
| theta*_t | vector | R^d | t, U_t, a_t | Time-varying optimum |
| U_t | matrix | Gr(d,r) | t, rho_s | Slowly drifting subspace |
| a_t | vector | R^r | t, rho_f, K_T | Fast-varying coefficients |
| K_T | scalar | N+ | T | Number of fast jumps in [0,T] |
| V_s | scalar | R+ | T, rho_s | Total slow variation |
| V_f | scalar | R+ | T, rho_f, K_T | Total fast variation |
| mu | scalar | R+ | — | Strong convexity constant |
| L | scalar | R+ | — | Smoothness constant |
| sigma | scalar | R+ | — | Noise level |
| eta | scalar | R+ | t (may decay) | Learning rate |
| B | scalar | N+ | — | Consolidation period |
| B* | scalar | R+ | C_merge, rho_s | Optimal consolidation period |
| C_merge | scalar | R+ | — | Cost of one consolidation |
| rho_s | scalar | R+ | — | Slow drift rate per step |
| rho_f | scalar | R+ | — | Fast drift rate per step |
| F_i | scalar/matrix | R+/PSD | data, model | Diagonal Fisher information |
| Reg_T | scalar | R | T | Dynamic regret = sum l_t(theta_t) - l_t(theta*_t) |
| Delta | scalar | R+ | — | Jump magnitude (coefficient) |

---

## 5. Canonical Quantified Statements

### Theorem 1 (Necessity) — ATTEMPTED FORMALIZATION

```
For all d >= r >= 1, for all mu > 0, L >= mu:
  For any single-timescale algorithm A with learning rate schedule {eta_t}
  satisfying eta_t = Theta(t^{-alpha}) for some alpha in (0,1]:
    IF A achieves Reg_T^{stationary} = O(sqrt(T)) on streams with V_s = V_f = 0,
    THEN there exists a two-rate drift sequence with:
      K_T = Theta(sqrt(T)) jumps of size Delta > 0,
      V_s = 0 (no subspace drift),
    such that:
      Reg_T^{A} = Omega(T)
```

**ISSUES WITH FORMALIZATION**:
- "Single-timescale" is defined only as "one learning rate schedule" — what about adaptive methods (Adam)?
- Is alpha restricted to (0,1]? What about constant eta?
- The Omega(T) lower bound likely requires K_T * recovery_cost_per_jump growing as T.
- Does this hold for all single-timescale algorithms, or only decaying-LR SGD?

### Theorem 2 (Sufficiency) — ATTEMPTED FORMALIZATION

```
For all d >= r >= 1, mu > 0, L >= mu, sigma > 0:
  Given a dual-timescale algorithm with:
    - Fast: per-step update with effective rate eta_f (non-decaying)
    - Slow: consolidation every B steps
  Under assumptions A1-A6:
    Reg_T <= C1 * sqrt(T * (V_f + 1)) + C2 * sqrt(T * (V_s + 1)) + C3
  where C1, C2, C3 depend on (d, r, mu, L, sigma, eta_f, B).
```

**ISSUES WITH FORMALIZATION**:
- C1, C2, C3 dependence on (d, r, mu, L, sigma, eta_f, B) is UNSPECIFIED.
- The bound must be compared against OPTIMAL Reg_T to be meaningful — what is the information-theoretic lower bound?
- "Non-decaying effective rate" — what specific mechanism? Sliding-window OLS? Fixed-eta SGD?

### Corollary 2 — ATTEMPTED FORMALIZATION

```
Given the dual-timescale regret bound from Thm2, optimizing over B:
  B* = argmin_B [ (cost_per_consolidation / B) + B * rho_s_effective ]
  => B* = sqrt(C_merge / rho_s)
```

**ISSUES**: Assumes linear drift cost and amortized consolidation cost — needs proof that these approximations hold.

---

## 6. Micro-Claim Inventory

### MC-1: Single-timescale recovery cost from jump at time tau
```
Context: [Thm1 proof sketch, decaying LR eta_t = c/sqrt(t)]
Goal: After coefficient jump of size Delta at time tau, single-timescale learner
      needs O(Delta^2 * sqrt(tau)) steps to recover
Rule: eta_tau = c/sqrt(tau), so recovery time ~ Delta / eta_tau = Delta * sqrt(tau) / c
Side-conditions: Assumes gradient provides direction toward optimum (strong convexity) ✓
Status: CLAIMED in code comments (synthetic_benchmark.py:18-20), NO FORMAL PROOF
```

### MC-2: Cumulative single-timescale recovery cost = Omega(T)
```
Context: [Thm1 proof, K_T = Theta(sqrt(T)) jumps at times tau_k ~ k^2]
Goal: sum_{k=1}^{K_T} recovery_cost_k = Omega(T)
Rule: sum_{k=1}^{sqrt(T)} sqrt(k * J) where J = jump_interval
      ~ sqrt(J) * sum_{k=1}^{sqrt(T)} sqrt(k)
      ~ sqrt(J) * T^{3/4}
Side-conditions: This gives Omega(T^{3/4}), NOT Omega(T)!
Status: **POTENTIAL GAP** — the claimed Omega(T) may actually be Omega(T^{3/4})
```

### MC-3: Dual-timescale constant recovery cost per jump
```
Context: [Thm1 proof / Thm2 proof, window OLS for coefficients]
Goal: After any jump at any time tau, dual-timescale recovers in O(window_size) steps
Rule: Sliding-window OLS ages out old data after exactly window_size steps
Side-conditions: Requires U_hat ≈ U_true (subspace well-tracked) ✓ (if slow drift)
Status: Correct for the synthetic benchmark; needs formal analysis for general case
```

### MC-4: Fisher-precision averaging = optimal Bayes merge
```
Context: [Corollary 1]
Goal: theta_merge = (sum F_i)^{-1}(sum F_i theta_i) is the MAP estimate
Rule: Under Gaussian posterior approximation, each theta_i has posterior
      N(theta_i, F_i^{-1}), product of Gaussians gives precision-weighted mean
Side-conditions: Requires diagonal Fisher approximation; requires posterior Gaussianity
Status: STANDARD RESULT (Laplace approximation), but diagonal Fisher drops cross-terms
```

### MC-5: Fisher-precision merge = distillation + replay
```
Context: [Corollary 1, second part]
Goal: theta_merge can be realized as: distillation on recent data + replay on past data
Rule: CLAIMED but NO DERIVATION in any project file
Status: **UNJUSTIFIED** — this decomposition is the core algorithmic bridge from
        theory to practice, but no formal argument connects the Fisher merge formula
        to the KL distillation + NTP replay implementation
```

### MC-6: B* = sqrt(C_merge / rho_drift) is optimal consolidation period
```
Context: [Corollary 2, code: compute_optimal_interval]
Goal: Minimize total regret = (amortized merge cost) + (drift cost between merges)
Rule: Classical optimization: f(B) = C/B + rho*B, f'(B)=0 => B* = sqrt(C/rho)
Side-conditions: Assumes linear drift cost between merges; assumes constant merge cost
Status: The optimization is trivial. The HARD part is proving that drift cost is
        indeed linear in B, which requires strong convexity + bounded gradient.
```

### MC-7: Selective consolidation weakly dominates
```
Context: [Corollary 3]
Goal: Consolidating only invariant coordinates has regret <= consolidating all
Rule: If transient coordinates have zero long-term signal (mean gradient = 0),
      consolidating them adds noise without benefit
Side-conditions: Requires a formal definition of "invariant" vs "transient"
Status: Intuitive but NO FORMAL PROOF. The InvariantDetector uses cosine similarity
        heuristic, not a theoretically grounded criterion.
```

---

## 7. Limit-Order Map

| Statement | Limit | Uniform In | Source |
|-----------|-------|-----------|--------|
| Reg_T^{single} = Omega(T) | T -> infinity | fixed d, r, mu, L, Delta, jump_interval | Thm1 |
| Reg_T^{dual} = O(sqrt(T)) | T -> infinity | fixed d, r, B, window_size | Thm2 |
| recovery_single ~ Delta^2 * sqrt(tau) | tau -> infinity | fixed Delta, c | MC-1 |
| B* = sqrt(C_merge / rho_s) | — | static formula | Cor2 |
| Regret ratio single/dual = 30-1000x | empirical | specific hyperparams | CLAUDE.md key results |

**AMBIGUITY FLAGS**:
- Thm2 O(sqrt(T)): is the constant inside independent of d? of r? Almost certainly not.
- Thm1 Omega(T): are the constants constructive? The adversarial drift sequence construction needs to be explicit.
- All regret bounds: dynamic regret (sum l_t(theta_t) - l_t(theta*_t)) vs static regret (vs best fixed theta)?

---

## 8. Critical Observations (Pre-Review)

### OBSERVATION 1: NO FORMAL PROOFS EXIST
The project claims 3 theorems + 3 corollaries but contains NO formal proofs anywhere — not in .tex files, not in .md files, not in code comments. The theoretical claims exist only as one-line statements in `FINAL_PROPOSAL.md` and `IDEA_REPORT.md`. The synthetic benchmark (`src/synthetic_benchmark.py`) provides empirical evidence for the separation, but this is NOT a proof.

### OBSERVATION 2: MC-2 GAP — Omega(T) vs Omega(T^{3/4})
The informal argument for Theorem 1's Omega(T) lower bound appears to give only Omega(T^{3/4}). With K_T = Theta(sqrt(T)) uniformly spaced jumps and recovery cost O(sqrt(tau)) per jump:
```
sum_{k=1}^{sqrt(T)} sqrt(k * T/sqrt(T)) = sqrt(sqrt(T)) * sum_{k=1}^{sqrt(T)} sqrt(k)
                                          ~ T^{1/4} * T^{3/4} = T
```
Wait — this might actually work if the jumps are uniformly spaced at interval J = T/K_T = sqrt(T):
```
sum_{k=1}^{K_T} sqrt(k * J) = sum_{k=1}^{sqrt(T)} sqrt(k * sqrt(T))
                              = T^{1/4} * sum_{k=1}^{sqrt(T)} sqrt(k)
                              ~ T^{1/4} * (sqrt(T))^{3/2} / (3/2)
                              = T^{1/4} * T^{3/4}
                              = T
```
So the Omega(T) bound MAY hold with K_T = sqrt(T) and uniform spacing. But this requires:
1. Each jump at time tau_k = k * sqrt(T) produces recovery cost exactly Theta(sqrt(tau_k))
2. The recovery costs don't overlap (each jump is fully recovered before the next)
3. Condition 2 requires recovery time << jump interval = sqrt(T), but recovery time at tau_k = k*sqrt(T) is Theta(k^{1/2} * T^{1/4}), which is at most T^{1/4} * T^{1/4} = T^{1/2} < sqrt(T) ✓

### OBSERVATION 3: Strong convexity assumption is problematic
The NTP loss for language models is HIGHLY non-convex. Claiming mu-strong convexity even in the LoRA subspace is dubious. The synthetic benchmark bypasses this by using a linear prediction model (y_t = theta^T x_t) which IS strongly convex in theta. But this doesn't validate the theory for LLMs.

### OBSERVATION 4: Code-theory gap in consolidation
Corollary 1 claims Fisher-precision averaging = distillation + replay. The CODE implements KL distillation + NTP replay (streaming_memory.py:247-341). But NO derivation connects the theoretical Fisher merge formula to the KL divergence objective. This is MC-5 — a critical unjustified assertion.

### OBSERVATION 5: Synthetic benchmark supports intuition, not theorem
The synthetic benchmark (synthetic_benchmark.py) demonstrates that sliding-window OLS beats decaying-LR SGD on coefficient recovery. This is a specific instance of the claimed separation. But:
- It only tests one specific single-timescale method (decaying SGD)
- It doesn't test against adaptive methods (Adam, AdaGrad)
- The Omega(T) claim is about ALL single-timescale methods

### OBSERVATION 6: Reviewer Memory warning
REVIEWER_MEMORY.md:6 explicitly flags: "Theory/experiment mismatch — synthetic benchmark supports coefficient-jump intuition, not the claimed necessity/sufficiency theorem"
