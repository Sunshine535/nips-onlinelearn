# Auto Review Log: Streaming Mirror-LoRA

## Round 1 — Nightmare Review (2026-04-09)

### Assessment (Summary)
- Score: 2/10
- Verdict: NOT READY
- Difficulty: nightmare
- Key criticisms:
  1. Central thesis contradicted by full-drift results (dual WORSE than single)
  2. Dual regret scales linearly in constant-jump setting, not O(sqrt(T)) as claimed
  3. rho_f unused in synthetic benchmark — not real multi-rate drift
  4. E3/E4/E5 synthetic ablations don't match actual Mirror-LoRA mechanisms
  5. Invariant detector bug: gradients zeroed before capture
  6. B* formula not evidenced — no U-shape in frequency sweep
  7. No proofs, no LLM results, no paper draft

### Debate Transcript

**Claude's Rebuttal:**

1. W1 (full-drift): Partially accepted — timescales not properly separated in experiment
2. W2 (linear not sqrt): Rejected — dual per-step regret is constant (0.5), O(sqrt(T)) requires K_T=O(sqrt(T))
3. W5 (bug): Accepted

**GPT's Ruling:**
1. W1: PARTIALLY SUSTAINED — narrow to claim/benchmark mismatch
2. W2: PARTIALLY SUSTAINED — theory/experiment gap acknowledged
3. W5: OVERRULED — bug confirmed

Score unchanged: 2/10

### Actions Taken (Round 1 Fixes)

1. **Fixed invariant detector gradient capture bug** — gradients now captured before zero_grad in `_update_fisher_and_snapshot()`
2. **Created V2 synthetic experiments** (`scripts/run_synthetic_v2.py`):
   - Exp A: K_T = O(sqrt(T)) decreasing-jump stream — validates actual O(sqrt(T)) claim
   - Exp B: Properly separated timescales (rho_s=0.001, jump_interval=20)
   - Exp C: Timescale separation ratio sweep
3. **GPU environment ready**: torch 2.10.0+cu128, 4x H100 80GB, all deps installed
4. **Smoke test passed**: StreamingMirrorLoRA runs on GPU with Qwen2.5-0.5B

### Results
- V2 synthetic experiments: RUNNING (CPU)
- LLM smoke test: PASSED
- Full LLM evaluation: PENDING (awaiting V2 results to validate theory first)

### Status
- Continuing to Round 2
- Waiting for V2 synthetic results before submitting for re-review
