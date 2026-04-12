# Reviewer Memory

## Round 1 — Score: 2/10
- **Suspicion**: "Special sauce" claims collapse under ablation — selective hurts, Fisher hurts, full dual hurts
- **Unresolved**: Full-drift dual is WORSE than single (ratio 0.6-0.7x). E4 selective consolidation hurts. E5 Fisher merge is worst.
- **Patterns**: Theory/experiment mismatch — synthetic benchmark supports coefficient-jump intuition, not the claimed necessity/sufficiency theorem
- **Bug found**: Mirror-LoRA invariant path nonfunctional — gradient zeroed before capture
- **E2 landmine**: Phase diagram likely anti-claim; rho_f is mis-specified
- **Decision needed**: Pivot to narrower paper, or repair benchmark + algorithm before theorem claims
