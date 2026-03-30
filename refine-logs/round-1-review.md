# Round 1 Review (GPT-5.4)

**Overall Score: 5.9/10**
**Verdict: REVISE**

## Scores

| Dimension | Score | Brief reason |
|---|---:|---|
| Problem Fidelity | 7 | Mostly on-anchor. Caveat: online supervision signal unspecified. |
| Method Specificity | 5 | Core transfer mechanism undefined. WM init, WM/LT composition, consolidation equation, buffer policy unclear. |
| Contribution Quality | 6 | Promising core idea diluted by PPO and over-claiming "selective consolidation." |
| Frontier Leverage | 6 | LoRA right, but consolidation in old CL parameter space, not LLM behavior space. |
| Feasibility | 6 | Plausible if simplified. 5 updates/turn + Fisher + PPO heavier than it looks. |
| Validation Focus | 6 | Misses strongest simple baseline and wall-clock latency measurement. |
| Venue Readiness | 6 | Borderline. Reads as incremental CL composition. |

## Critical Issues

### Method Specificity (5/10) — CRITICAL
- WM randomly initialized each session → wastes first turns
- Both adapters active at inference but LT consolidation never consumes WM delta
- Replay importance 1/loss contradicts surprise story
- **Fix**: Make WM a residual over LT, zero-init each session. Consolidate by distilling (LT+WM) into LT' with KL/task loss + Fisher trust-region. Replace 1/loss with novelty-based weighting.

### Contribution Quality (6/10) — CRITICAL
- Too many "moving novelties": dual adapters, Fisher/EWC, surprise gating, PPO
- PPO weakens parsimony
- **Fix**: Cut PPO. One mechanism: two-timescale low-rank memory + one consolidation rule. Use deterministic trigger.

## Important Issues

### Frontier Leverage (6/10) — IMPORTANT
- Consolidation in parameter space (EWC) vs behavior space
- **Fix**: Distill token distributions from LT+WM into LT with KL on replay, optionally keep Fisher as secondary stabilizer.

### Feasibility (6/10) — IMPORTANT
- Operationally busy for 7-week project
- **Fix**: Zero-init WM, consolidate at session end or surprise threshold, estimate Fisher once per session, reduce LT rank.

### Validation Focus (6/10) — IMPORTANT
- Missing key baseline and latency
- **Fix**: Minimal set = single LoRA, single LoRA + replay/EWC, dual LoRA + fixed consolidation, dual LoRA + final rule. Report per-turn latency.

### Venue Readiness (6/10) — IMPORTANT
- Reviewers may see "dual-adapter CL with standard tools"
- **Fix**: Sharpen title around one equation-level mechanism for WM→LT transfer.

## Simplification Opportunities
1. Delete PPO; use session-end or surprise-threshold consolidation
2. Make WM a zero-init residual over LT, not separate random adapter
3. Merge replay/consolidation/retention into one objective: task/KL distillation + stabilizer

## Modernization Opportunities
1. Replace parameter-space EWC with behavior-space distillation (LT+WM → LT)
2. Define online signal with deployment-realistic supervision (not oracle labels)
3. If adaptive timing needed, use learned scalar gate from loss/KL/gradient, not PPO

## Drift Warning
Potential drift if per-turn loss uses oracle assistant responses. Clarify online supervision channel.

<details>
<summary>Raw GPT-5.4 Response</summary>

The core instinct is good: a fast disposable adapter plus a slow persistent adapter is the right shape for this anchor. The proposal is not yet sharp enough because the actual consolidation path is underspecified, and the PPO policy turns a potentially elegant paper into a module stack.

[Full verbatim response saved above]
</details>
