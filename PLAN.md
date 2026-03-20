# Execution Plan — Streaming Parameter Memory (SPM)

## Project Timeline: 7 Weeks

---

## Week 1: Infrastructure + Dual LoRA Architecture
**Goal**: Working dual-LoRA module on Qwen3.5-9B with correct forward pass.

### Tasks
- [ ] Set up conda environment with PyTorch 2.3+, transformers, peft, trl
- [ ] Implement `DualLoRA` wrapper: manages WM-LoRA (rank 16) and LT-LoRA (rank 64) on same base model
- [ ] Forward pass: output = base(x) + LT_LoRA(x) + WM_LoRA(x) with separate scaling
- [ ] Verify correct gradient isolation: WM-LoRA receives gradients, LT-LoRA frozen during online updates
- [ ] Load Qwen3.5-9B and confirm dual-LoRA inference produces coherent outputs
- [ ] Memory profiling: confirm 8× A100 can hold model + dual LoRA + optimizer states

### Implementation Details
```python
class DualLoRA(nn.Module):
    def __init__(self, base_model, wm_rank=16, lt_rank=64):
        self.base = base_model  # frozen
        self.wm_lora = LoRAAdapter(base_model, rank=wm_rank)  # trainable online
        self.lt_lora = LoRAAdapter(base_model, rank=lt_rank)  # updated only via consolidation
        self.wm_lora.requires_grad_(True)
        self.lt_lora.requires_grad_(False)
```

### Validation Criteria
- [ ] Dual-LoRA forward pass matches single-LoRA perplexity on 100 test samples (±0.5)
- [ ] WM-LoRA gradient updates don't leak into LT-LoRA or base model
- [ ] Peak VRAM < 72GB per GPU (leaves headroom for optimizer states)

---

## Week 2: Working Memory Online Update + Fisher Estimator
**Goal**: WM-LoRA updates in real-time during conversation. Fisher estimator running.

### Tasks
- [ ] Implement `OnlineUpdater`: single-step SGD on WM-LoRA per conversation turn
- [ ] Learning rate schedule: start 1e-4, decay within session (cosine over turns)
- [ ] Gradient clipping: max_norm=1.0 on WM-LoRA gradients to prevent destabilization
- [ ] WM-LoRA norm constraint: project back to sphere if ||WM-LoRA||_F > threshold
- [ ] Implement `FisherEstimator`: diagonal Fisher from turn-level log-likelihood gradients
- [ ] Fisher EMA: F_t = β·F_{t-1} + (1-β)·g_t² where β=0.99
- [ ] Unit tests: Fisher estimates are non-negative, decay correctly, respond to input distribution

### Fisher Estimation Details
```python
class FisherEstimator:
    def __init__(self, beta=0.99):
        self.beta = beta
        self.fisher_diag = None

    def update(self, wm_lora, loss):
        grads = torch.autograd.grad(loss, wm_lora.parameters())
        fisher_step = {n: g**2 for n, g in zip(names, grads)}
        if self.fisher_diag is None:
            self.fisher_diag = fisher_step
        else:
            for n in self.fisher_diag:
                self.fisher_diag[n] = self.beta * self.fisher_diag[n] + (1-self.beta) * fisher_step[n]
```

### Validation Criteria
- [ ] WM-LoRA perplexity on persona-specific text decreases within 5 turns
- [ ] Fisher estimates converge (variance < 10% of mean after 20 turns)
- [ ] Online update adds < 50ms latency per turn on single A100

---

## Week 3: Consolidation Policy Network
**Goal**: Policy network architecture implemented, ready for RL training.

### Tasks
- [ ] Implement `ConsolidationPolicy`: MLP(512→256→256→1) per parameter group
- [ ] Input features per parameter group:
  - Fisher diagonal (normalized): importance signal
  - Cosine similarity between WM and LT parameter directions: redundancy signal
  - WM-LoRA parameter magnitude: activation signal
  - Session statistics: turn count, unique entities, topic transitions
- [ ] Output: sigmoid-activated merge coefficient per parameter group (0=keep, 1=fully merge)
- [ ] Consolidation operation: θ_LT ← θ_LT + α_group · (θ_WM - θ_LT)
- [ ] Implement multi-session simulator: generates synthetic conversation sequences
- [ ] Define reward function: R = 0.5·retention + 0.3·adaptation - 0.2·forgetting

### Policy Architecture
```
Input (per group):
  [Fisher_norm(d), cos_sim(1), wm_mag(1), session_stats(4)] → dim=d+6

MLP:
  Linear(d+6, 256) → ReLU → Linear(256, 256) → ReLU → Linear(256, 1) → Sigmoid

Output: merge_coefficient ∈ [0, 1]
```

### Parameter Grouping Strategy
- Group by: (layer_index, module_type) → ~120 groups for Qwen3.5-9B
- Fisher features: mean, max, std of Fisher diagonal within group → 3 features
- Total input dim per group: 3 + 1 + 1 + 4 = 9

### Validation Criteria
- [ ] Policy network outputs valid merge coefficients (all in [0,1])
- [ ] Single consolidation step completes in < 1 second
- [ ] LT-LoRA changes are consistent with policy output magnitudes

---

## Week 4: RL Training of Consolidation Policy
**Goal**: Trained consolidation policy that outperforms heuristic baselines.

### Tasks
- [ ] Set up PPO training loop with trl library
- [ ] Episode structure: 5 sessions × 15 turns, consolidation every 5 turns
- [ ] Reward computation at session boundaries:
  - Persona retention: NLI-based consistency between response and persona profile
  - Adaptation speed: number of turns until persona-trait F1 > 0.5
  - Forgetting penalty: drop in retention from session N to session N+1
- [ ] Baseline comparisons during training:
  - Heuristic: merge top-K Fisher parameters by magnitude
  - EMA: uniform exponential moving average (α=0.1)
  - Random: random merge coefficients
- [ ] Hyperparameter search: lr ∈ {1e-5, 3e-5, 1e-4}, clip_range ∈ {0.1, 0.2}
- [ ] Early stopping: policy reward plateaus for 500 episodes

### RL Training Setup
- PPO with GAE (λ=0.95, γ=0.99)
- Policy lr: 3e-5, value lr: 1e-4
- Batch size: 32 episodes, mini-batch: 8
- Max episodes: 10,000 (expect convergence by ~5,000)
- Estimated time: 24h on 8× A100

### Validation Criteria
- [ ] Policy reward monotonically increases (smoothed) during training
- [ ] Trained policy retention > heuristic baseline by ≥ 5%
- [ ] Policy learns non-trivial behavior (not all-zero or all-one merge coefficients)
- [ ] Merge coefficients correlate with Fisher importance (Spearman ρ > 0.3)

---

## Week 5: PersonaChat Evaluation
**Goal**: Complete PersonaChat benchmark results with ablations.

### Tasks
- [ ] Prepare PersonaChat test split: 1,155 personas, 5 synthetic sessions each
- [ ] Session generation: use Qwen3.5-9B to generate user turns matching persona
- [ ] Metrics implementation:
  - Persona Retention F1: NLI(response, persona_trait) averaged over traits
  - Session Adaptation: turns until F1 > 0.5 on session-specific traits
  - Cross-Session Forgetting: retention(session_N) - retention(session_{N+1})
  - Perplexity: on held-out next-turn prediction
- [ ] Run full evaluation: SPM, single-LoRA-online, frozen, EMA-consolidation, EWC
- [ ] Ablation: WM-only (no consolidation), LT-only (no online updates), Fisher-only (no RL)
- [ ] Statistical significance: bootstrap 95% CI over persona-level results

### Evaluation Protocol
```
For each persona P:
  For session s in 1..5:
    Reset WM-LoRA (or apply decay)
    For turn t in 1..15:
      Generate response with dual LoRA
      Update WM-LoRA with online gradient
      Update Fisher estimates
    Apply consolidation policy (WM→LT)
  Measure: retention(s=5), avg_adaptation_speed, avg_forgetting
```

---

## Week 6: LIGHT Evaluation + Analysis
**Goal**: LIGHT benchmark results + consolidation policy analysis.

### Tasks
- [ ] Set up LIGHT environment: character descriptions + situated dialogue
- [ ] Metrics: Character Consistency Score (human-eval proxy via NLI), Action Appropriateness
- [ ] Run evaluation: SPM vs baselines on 100 randomly sampled LIGHT characters
- [ ] Analysis experiments:
  - Layer-wise consolidation patterns: which layers does policy prefer to consolidate?
  - Temporal patterns: does consolidation increase/decrease over sessions?
  - Fisher vs. merge coefficient correlation: is the policy using Fisher meaningfully?
  - t-SNE of LT-LoRA evolution over sessions: does it converge to persona-specific regions?
- [ ] Scaling analysis: vary number of sessions (1, 5, 10, 20, 50) and plot retention curves

### Expected Findings
- Early layers: higher consolidation (general linguistic patterns)
- Late layers: selective consolidation (persona-specific features)
- Consolidation frequency increases as WM-LoRA diverges more from LT-LoRA
- Fisher importance peaks at persona-relevant tokens (names, preferences, facts)

---

## Week 7: Paper Writing + Final Experiments
**Goal**: Complete draft paper.

### Paper Outline
1. **Introduction** (1.5 pages): Motivation from CLS theory, production need (CharacterFlywheel), our contribution
2. **Related Work** (1 page): Online adaptation, LoRA methods, continual learning, CLS in ML
3. **Method** (2.5 pages): Dual LoRA architecture, Fisher estimator, consolidation policy, RL training
4. **Experiments** (3 pages): PersonaChat, LIGHT, ablations, scaling, analysis
5. **Discussion** (0.5 pages): Limitations, broader impact
6. **Conclusion** (0.5 pages)

### Tasks
- [ ] Write complete paper draft
- [ ] Generate all figures: architecture diagram, retention curves, layer-wise analysis, t-SNE
- [ ] Run any additional experiments identified during writing
- [ ] Internal review and revision
- [ ] Prepare supplementary material: full results tables, hyperparameter sensitivity

---

## Compute Budget Summary

| Phase | GPUs | Hours | GPU-Hours |
|-------|------|-------|-----------|
| Week 1-2: Infrastructure + WM-LoRA | 8× A100 | 48 | 384 |
| Week 3-4: Policy training (PPO) | 8× A100 | 24 | 192 |
| Week 5: PersonaChat eval + ablations | 8× A100 | 16 | 128 |
| Week 6: LIGHT eval + analysis | 8× A100 | 8 | 64 |
| Week 7: Additional experiments | 8× A100 | 8 | 64 |
| **Total** | | **104** | **832** |

---

## Critical Path Dependencies

```
Week 1 (Dual LoRA) → Week 2 (Online Update + Fisher)
                          ↓
                   Week 3 (Policy Network)
                          ↓
                   Week 4 (RL Training)
                          ↓
                   Week 5 (PersonaChat) ─→ Week 7 (Paper)
                          ↓
                   Week 6 (LIGHT + Analysis) ─→ Week 7 (Paper)
```

**Bottleneck**: Week 4 (RL training). If consolidation policy doesn't converge, fallback to heuristic Fisher-threshold policy + ablation showing RL improvement is marginal but direction is promising.

## Risk Mitigations

1. **WM-LoRA instability**: Pre-trained WM-LoRA on PersonaChat before online mode; use as initialization
2. **Fisher noise**: Increase EMA β to 0.999; require minimum 10 turns before first consolidation
3. **RL non-convergence**: Start with behavior cloning from heuristic policy; fine-tune with PPO
4. **Evaluation variance**: 1,155 personas provides sufficient statistical power; report bootstrap CIs
