# Research Proposal: Dual-Timescale Adaptation Is Necessary and Sufficient for Streaming Low-Rank Learning under Multi-Rate Drift

## Problem Anchor

- **Bottom-line problem**: Streaming low-rank adaptation (online LoRA) under non-stationary data with no task boundaries. The data distribution drifts at multiple rates simultaneously — slow structural drift + fast contextual shifts.

- **Must-solve bottleneck**: Single-timescale online adaptation faces an impossible tradeoff: the learning rate needed to average gradient noise kills responsiveness to late abrupt changes, and vice versa. No existing work proves this is fundamental or provides a provably optimal solution.

- **Non-goals**: NOT (1) batch continual learning with task boundaries, (2) architecture-level changes to transformers, (3) retrieval-only solutions, (4) full model fine-tuning.

- **Constraints**: ~500 GPU-hours, frozen base LLM, NeurIPS 2026 best paper target.

- **Success condition**: (1) Separation theorem with airtight proof, (2) phase transition experimentally validated, (3) clear wins over 11+ baselines on synthetic + 2 real domains.

## Technical Gap

Online low-rank adaptation of LLMs requires tracking non-stationary distributions in parameter-efficient subspaces. The fundamental tension:

- **Single-LoRA + small LR**: Averages noise well -> misses abrupt shifts (high regret on fast component)
- **Single-LoRA + large LR**: Tracks fast shifts -> amplifies noise (high regret on slow component)
- **All existing LoRA-CL methods**: Assume discrete task boundaries. No boundary-free streaming theory exists.

**The gap**: No proof that this single-timescale impossibility is fundamental. No regret-bounded algorithm for streaming LoRA. No formal connection between optimal consolidation and distillation+replay.

## Central Thesis

Multi-rate drift creates an impossible tradeoff for single-timescale adaptation; separating fast tracking from slow consolidation resolves it.

## Method

### Theoretical Framework

**Setting**: Streaming prediction with loss l_t(theta) that is mu-strongly convex and L-smooth in adapted low-rank parameters. Optimum decomposes as theta*_t = U_t a_t + r_t where U_t is slowly drifting rank-r subspace, a_t are fast-varying coefficients.

**Theorem 1 (Necessity)**: Any single-timescale algorithm achieving O(sqrt(T)) stationary regret suffers Omega(T) on valid two-scale drift sequences with K_T = Theta(sqrt(T)) fast jumps.

**Theorem 2 (Sufficiency)**: Dual-timescale with fast online updates + periodic slow consolidation achieves Reg_T <= C1 sqrt(T(V_f+1)) + C2 sqrt(T(V_s+1)) + C3.

**Corollary 1 (Optimal Merge)**: Fisher-precision averaging theta_merge = (sum F_i)^{-1}(sum F_i theta_i). Decomposes into distillation on recent data + replay on past data.

**Corollary 2 (Optimal Frequency)**: B* proportional to sqrt(merge_cost / Fisher_drift_rate).

**Corollary 3 (Selectivity)**: Consolidating only invariant coordinates weakly dominates indiscriminate consolidation.

### Algorithm: Streaming Mirror-LoRA

One algorithm box implementing all corollaries:
1. **Fast-timescale**: Online mirror updates on low-rank manifold for WM-LoRA (coefficients a_t)
2. **Slow-timescale**: Periodic Fisher-precision consolidation into LT-LoRA (subspace U_t)
3. **Selectivity mask**: Identify invariant coordinates via gradient stability + cross-window consistency
4. **Consolidation realization**: KL distillation on recent data + NTP replay on reservoir data

### Architecture

| Component | Params | Role |
|-----------|--------|------|
| Base LLM | Frozen | Foundation |
| WM-LoRA (fast) | r=16, ~13M | Track fast drift, reset-able |
| LT-LoRA (slow) | r=64, ~52M | Accumulate invariant knowledge |
| Fisher estimator | Diagonal | Importance + merge weights |
| Reservoir buffer | Bounded | Replay data for consolidation |
| Selectivity detector | Lightweight | Invariant/transient classification |

## Contribution Focus

- **Dominant**: Separation theorem — first necessity proof for dual-timescale in streaming PEFT
- **Supporting**: Optimal consolidation theory (Fisher-precision = distill + replay)
- **Algorithmic**: Streaming Mirror-LoRA (regret-bounded, theory-derived)
- **Empirical**: Synthetic benchmark + two real-domain validation
- **NOT independently novel**: LoRA, Fisher information, manifold optimization, invariant risk minimization — these are building blocks, not claims

## Experiment Plan

See `refine-logs/EXPERIMENT_PLAN.md` for detailed plan (9 experiments, 500 GPU-hours).

**Key experiments**:
1. Synthetic two-rate drift -> separation theorem validation
2. Phase transition diagram -> theory-to-practice correspondence
3. Consolidation frequency sweep -> B* validation
4. Invariant-selective test -> Corollary 3 validation
5. PersonaChat + LIGHT dialogue stream -> real-world validation
6. Streaming text classification -> non-dialogue generality
7. Full ablation matrix -> every component justified by theory
8. 11+ baselines including EMA, periodic avg, dual-adapter heuristic, SDFT, subspace tracking

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Theorem assumptions too strong | HIGH | Validate with synthetic phase transitions; state caveats honestly |
| Manifold overhead for marginal gains | MEDIUM | Ablate manifold updates; main story is dual-timescale, not geometry |
| Strong baselines close the gap | MEDIUM | If EMA/dual-heuristic works, it validates thesis — they ARE dual-timescale |
| Non-dialogue experiments weak | MEDIUM | Choose clean classification with controllable drift |
| Invariant detection unreliable | LOW-MEDIUM | Multiple proxy criteria; ablate to show marginal vs essential |

## Paper Framing

### What this paper IS:
- A theorem that dual-timescale is necessary for streaming PEFT
- An algorithm that follows from the theorem
- Experiments that validate the theorem's predictions

### What this paper is NOT:
- Three separate contributions stapled together
- A LoRA variant paper
- A persona/dialogue paper
- A manifold optimization paper

### Key sentence (repeat in abstract, intro, conclusion):
"Multi-rate drift creates an impossible tradeoff for single-timescale adaptation; separating fast tracking from slow consolidation resolves it."
