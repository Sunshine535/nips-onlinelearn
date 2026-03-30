# Refinement Report

**Problem**: Online LLM personalization (streaming persona retention)
**Initial Approach**: Dual-LoRA + Fisher/EWC + PPO consolidation policy
**Date**: 2026-03-30
**Rounds**: 4 / 5
**Final Score**: 9.0 / 10
**Final Verdict**: READY

## Problem Anchor
LLMs cannot adapt within a session while retaining knowledge across sessions. The plasticity-stability trade-off in single-parameter-space online adaptation is the core bottleneck.

## Output Files
- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Final proposal: `refine-logs/FINAL_PROPOSAL.md`

## Score Evolution

| Round | Problem Fidelity | Method Specificity | Contribution Quality | Frontier Leverage | Feasibility | Validation Focus | Venue Readiness | Overall | Verdict |
|-------|------------------|--------------------|----------------------|-------------------|-------------|------------------|-----------------|---------|---------|
| 1     | 7                | 5                  | 6                    | 6                 | 6           | 6                | 6               | 5.9     | REVISE  |
| 2     | 8.4              | 7.3                | 7.2                  | 7.4               | 8.0         | 7.1              | 7.0             | 7.5     | REVISE  |
| 3     | 9.0              | 8.5                | 8.0                  | 8.0               | 8.4         | 8.6              | 7.8             | 8.3     | REVISE  |
| 4     | 9.4              | 9.0                | 8.6                  | -                 | 9.2         | 9.1              | -               | 9.0     | READY   |

## Round-by-Round Review Record

| Round | Main Reviewer Concerns | What Was Changed | Result |
|-------|-------------------------|------------------|--------|
| 1     | Transfer mechanism undefined; PPO sprawl; parameter-space EWC; too busy | WM→zero-init residual; PPO removed; KL distillation; simplified loop | Resolved |
| 2     | Online signal unclear; overclaimed; missing baselines | Clarified NTP signal; narrowed claims; added param-matched + retrieval baselines | Resolved |
| 3     | KL teacher data ambiguity; metrics need NLI; retrieval baseline weak | Split B_k/B_all; NLI-based retention; strengthened retrieval | Resolved |
| 4     | No blocking issues | - | READY |

## Final Proposal Snapshot
1. Two-timescale residual LoRA: zero-init WM per session, persistent LT across sessions
2. Behavioral distillation consolidation: KL on current-session data (behavior transfer) + NTP replay on reservoir (retention)
3. One equation captures the entire mechanism
4. 7 methods × 2 datasets × 3 primary metrics evaluation
5. ~500 GPU-hours, 6 weeks

## Method Evolution Highlights
1. **Most important simplification**: Removing PPO — turned a 4-component system into a 1-equation method
2. **Most important mechanism upgrade**: Parameter-space EWC → behavior-space KL distillation
3. **Most important framing improvement**: Split consolidation (B_k for transfer, B_all for retention)

## Pushback / Drift Log
| Round | Reviewer Said | Author Response | Outcome |
|-------|---------------|-----------------|---------|
| 1 | "PPO makes it a module stack" | Agreed, removed PPO entirely | Accepted |
| 1 | "Random WM init wastes turns" | Agreed, switched to zero-init residual | Accepted |
| 2 | "Self-training on own outputs is degenerate" | Clarified: NTP on conversation where user messages contain persona facts | Accepted |
| 3 | "KL on full reservoir risks projecting WM onto old contexts" | Split into B_k for KL + B_all for replay | Accepted |
| 3 | "Don't overclaim general online personalization" | Narrowed to "streaming persona retention" | Accepted |

## Remaining Weaknesses
1. Novelty is incremental (strong synthesis of known ingredients, not a conceptual breakthrough)
2. Evaluation limited to persona/behavioral retention benchmarks (PersonaChat + LIGHT)
3. Single base model (Qwen3.5-9B) — generalization to other LLMs untested

## Next Steps
- **READY**: Proceed to implementation → experiment deployment → auto-review-loop
- Implement the refined method in code (update `src/streaming_memory.py`)
- Test on SSH server for code validation
- Run full evaluation pipeline
