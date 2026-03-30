# Round 2 Review (GPT-5.4)

**Overall Score: 7.5/10**
**Verdict: REVISE**

## Scores

| Dimension | Score | Change |
|---|---:|---:|
| Problem Fidelity | 8.4 | +1.4 |
| Method Specificity | 7.3 | +2.3 |
| Contribution Quality | 7.2 | +1.2 |
| Frontier Leverage | 7.4 | +1.4 |
| Feasibility | 8.0 | +2.0 |
| Validation Focus | 7.1 | +1.1 |
| Venue Readiness | 7.0 | +1.0 |

## Key Remaining Issues

### 1. Online Learning Signal (CRITICAL)
- Self-supervised on own outputs could be degenerate self-training
- Need to explain how user-specific facts enter the gradient
- This is the main blocking issue for acceptance

### 2. Contribution Still Slightly Diluted (IMPORTANT)
- Replay, surprise trigger, optional Fisher still too many moving parts
- "First" claim exposed to prior fast/slow-weights + distillation

### 3. Missing Baselines (IMPORTANT)
- Need parameter-matched baseline (same total params, single adapter)
- Need nonparametric personalization baseline (retrieval/profile-based)
- Need amortized consolidation cost in latency reporting

### 4. Novelty Positioning (IMPORTANT)
- Frame around behavioral distillation for streaming personalization, not just dual adapters
- Exhaustive related work needed for "first" claims

## Simplification Opportunities
1. Session-end consolidation as only core trigger (surprise → ablation)
2. Remove Fisher from main method (→ fallback if KL-only fails)
3. Replace gradient-norm importance with simpler reservoir buffer

## Modernization Opportunities
1. Reframe as behavior-space online personalization, not adapter CL
2. Add strong memory/profile baseline
3. Add parameter-count-matched baseline
