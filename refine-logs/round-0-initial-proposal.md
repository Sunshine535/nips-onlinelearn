# Research Proposal: Streaming Parameter Memory — Fisher-Informed Dual-LoRA Consolidation for Online LLM Personalization

## Problem Anchor

- **Bottom-line problem**: Large language models cannot adapt *within* a conversation session to user-specific knowledge while simultaneously retaining that knowledge *across* sessions. Current personalization approaches (offline LoRA fine-tuning, retrieval-augmented prompting) operate in batch mode, creating a fundamental latency gap where the model is always one session behind.

- **Must-solve bottleneck**: Existing online adaptation methods (e.g., single-LoRA continuous updates, EWC regularization) face a fundamental trade-off: fast in-session adaptation causes catastrophic forgetting of previously learned user preferences, while strong forgetting prevention makes in-session adaptation too slow. There is no mechanism that provides both rapid session-local plasticity and durable cross-session stability simultaneously.

- **Non-goals**: This project does NOT aim to (1) build a general-purpose continual learning system for LLMs that handles arbitrary distribution shifts, (2) replace retrieval-augmented generation for factual knowledge injection, (3) scale to millions of users simultaneously (we target single-user streaming), or (4) fine-tune the entire base model online.

- **Constraints**:
  - Compute: ~700 GPU-hours total (8× A100-80GB × 4 days training + evaluation)
  - Base model: Qwen3.5-9B (frozen), parameter-efficient adaptation only
  - Data: PersonaChat (multi-turn personas), LIGHT (character-playing in fantasy setting)
  - Timeline: 7 weeks
  - Venue: NeurIPS 2026

- **Success condition**: SPM achieves (1) Persona Retention F1 ≥ 0.65 after 5+ sessions (vs ≤ 0.42 for single-LoRA), (2) within-session adaptation in ≤ 5 turns (vs ≥ 12 turns baseline), (3) cross-session forgetting rate ≤ 0.10 per session (vs ≥ 0.31 baseline), and (4) no more than 5% perplexity degradation vs. frozen model.

## Technical Gap

### Why Current Methods Fail

1. **Single-LoRA online updates** (naive baseline): Each conversation turn updates the same adapter. New information overwrites old knowledge because gradient updates have no structural separation between session-local and persistent knowledge. After 5 sessions, early-session persona traits are nearly completely forgotten.

2. **EWC (Elastic Weight Consolidation)**: Applies a quadratic penalty around previously important parameters, but treats all parameters uniformly. EWC does not distinguish between "this parameter encodes session-specific style" and "this parameter encodes persistent user facts." Result: either over-regularization (slow adaptation) or under-regularization (forgetting).

3. **Retrieval-augmented approaches**: Store conversation history and retrieve at inference. These avoid the forgetting problem entirely but cannot perform *parametric* adaptation — the model's internal representations don't evolve. For tasks requiring style adaptation, behavioral alignment, or implicit preference learning, retrieval is insufficient.

4. **CharacterFlywheel (Meta, 2026)**: Requires 15 iterative generation-evaluation cycles for convergence, making it impractical for real-time streaming. It operates in batch mode per character, not in a streaming online setting.

### Why Naive Fixes Are Insufficient

- **Larger LoRA rank**: Does not address the structural conflation of session-local and persistent knowledge.
- **More replay**: Reduces forgetting but slows adaptation because replay data dilutes new session signals.
- **Separate adapters per session**: Does not transfer knowledge across sessions at all.
- **Simple parameter averaging/EMA**: Averages without importance weighting, destroying task-relevant structure.

### Smallest Adequate Intervention

The fundamental issue is that a single parameter space conflates two timescales of adaptation. The minimal fix is to separate parameters into two pools with different update dynamics and connect them via a principled consolidation mechanism:
- **Working Memory LoRA (WM)**: Fast-adapting, session-scoped, re-initialized each session
- **Long-Term Memory LoRA (LT)**: Slowly consolidating, persistent across sessions, protected by importance weighting
- **Fisher-Informed Consolidation**: Transfer important knowledge from WM to LT using diagonal Fisher Information as the importance measure, combined with experience replay and EWC protection

### Core Technical Claim

A dual-LoRA architecture with Fisher-informed selective consolidation achieves superior online personalization by structurally separating session-local plasticity from cross-session persistence, where the consolidation policy is learned end-to-end via reinforcement learning to adaptively determine *when* and *how much* to consolidate based on information-theoretic surprise signals.

## Method Thesis

- **One-sentence thesis**: Structurally separating fast session-local adaptation (working-memory LoRA) from slow cross-session consolidation (long-term LoRA), connected by a Fisher-informed, surprise-gated consolidation mechanism, achieves the best known trade-off between in-session plasticity and cross-session retention for online LLM personalization.

- **Why this is the smallest adequate intervention**: The only new components beyond standard LoRA are (1) the dual-adapter lifecycle management (which adapter is active when) and (2) the consolidation mechanism (Fisher estimation + selective merge). Both are lightweight and attach to the existing PEFT infrastructure without modifying the base model.

- **Why this route is timely in the foundation-model era**: With 9B+ parameter frozen LLMs as the backbone, parameter-efficient methods are the only viable route for online adaptation. The LoRA ecosystem (PEFT library) provides the substrate; SPM contributes the *temporal management policy* that existing LoRA methods lack entirely.

## Contribution Focus

- **Dominant contribution**: A principled dual-LoRA consolidation mechanism (Fisher-informed selective merge + surprise-gated timing) that achieves simultaneous rapid in-session adaptation and durable cross-session retention, grounded in complementary learning systems (CLS) theory.

- **Optional supporting contribution**: A learned consolidation policy (lightweight MLP trained via PPO) that adaptively determines consolidation timing and magnitude based on Fisher statistics, replacing fixed-interval heuristics.

- **Explicit non-contributions**: We do NOT claim novelty in (1) the LoRA method itself, (2) EWC or Fisher estimation (standard tools), (3) the PPO algorithm, or (4) the base model architecture. Our novelty is in the *architectural pattern* of dual timescale LoRA management and the *consolidation policy* that connects them.

## Proposed Method

### Complexity Budget
- **Frozen / reused backbone**: Qwen3.5-9B (9B params, fully frozen)
- **New trainable components**: (1) Working-Memory LoRA (~13M params, r=16), (2) Long-Term LoRA (~52M params, r=64), (3) Consolidation policy MLP (~0.5M params)
- **Tempting additions intentionally not used**: (1) Multi-head attention over memory buffer — replaced by simpler importance-weighted sampling, (2) Separate adapter per persona — doesn't scale and prevents transfer, (3) Large retrieval system — we focus on parametric adaptation

### System Overview

```
Session Start → Initialize WM-LoRA (random init)
                     ↓
For each turn:
  1. Encode turn with WM-LoRA active → compute loss
  2. Online gradient update on WM-LoRA (5 steps, lr=5e-4)
  3. Store (input, label, importance=1/loss) in Memory Buffer
  4. Compute state vector: [Fisher_mag, buffer_fill, turn_idx, recent_loss, loss_trend, turns_since_consolidation, session_idx, total_turns]
  5. Policy decides: CONSOLIDATE or WAIT
     ↓ (if CONSOLIDATE)
  6. Switch to LT-LoRA active
  7. Estimate diagonal Fisher on LT-LoRA using replay buffer
  8. Consolidation training: replay buffer + EWC penalty (λ=5000)
  9. Store LT-LoRA checkpoint for next EWC anchor
 10. Switch back to WM-LoRA, reset update counter
                     ↓
Session End → WM-LoRA is discarded (re-initialized next session)
              LT-LoRA persists

Inference: Activate both [WM, LT] adapters simultaneously
```

### Core Mechanism: Fisher-Informed Selective Consolidation

**Input**: Current WM-LoRA parameters θ_WM, current LT-LoRA parameters θ_LT, memory buffer B

**Output**: Updated θ_LT that incorporates important knowledge from θ_WM

**Process**:
1. **Fisher Estimation**: Compute diagonal Fisher F_i = E[(∂ℓ/∂θ_i)²] over replay buffer samples on LT-LoRA. This tells us which LT parameters are currently important for past knowledge.

2. **EWC-Protected Training**: Train LT-LoRA on replay buffer with loss:
   L = L_task(θ_LT; B) + λ · Σ_i F_i · (θ_LT,i - θ_LT,i^prev)²
   
   This allows LT to learn new patterns from the replay buffer while protecting parameters critical for previously consolidated knowledge.

3. **Importance-Weighted Replay**: Samples are drawn from the memory buffer with probability proportional to their importance score (inverse loss at encoding time). This biases consolidation toward high-quality, informative interactions.

**Why this is the main novelty**: Unlike standard EWC which applies a single regularization uniformly, our dual-LoRA setup allows EWC to operate *only on the consolidation path* (LT ← WM), while leaving the working memory completely free for rapid adaptation. This structural separation is the key insight.

### Supporting Component: Learned Consolidation Policy

**Input**: 8-dimensional state vector [Fisher_magnitude, buffer_fill_ratio, turn_position, avg_recent_loss, loss_trend, turns_since_last_consolidation, session_progress, total_progress]

**Output**: Binary action (CONSOLIDATE / WAIT) with learned value baseline

**Architecture**: 3-layer MLP (8 → 64 → 64 → 2) with separate value head (8 → 64 → 1)

**Training**: PPO with reward = quality_improvement - forgetting_penalty - consolidation_cost
- quality_improvement = max(0, loss_before - loss_after) × 2.0
- forgetting_penalty = max(0, retention_before - retention_after) × 5.0
- consolidation_cost = 0.1 (if consolidated)

**Why it does not create contribution sprawl**: The policy is a tiny MLP (<0.5M params) that replaces a fixed-interval heuristic. It is an implementation detail of the consolidation mechanism, not a separate contribution. The paper's main contribution remains the dual-LoRA architecture; the policy simply makes the timing adaptive.

### Modern Primitive Usage

- **Which primitive**: Parameter-efficient fine-tuning (LoRA) as the adaptation substrate; PPO as the consolidation policy trainer
- **Exact role**: LoRA provides the parameter-efficient adaptation interface; PPO trains the tiny consolidation decision network. Neither is used as a generative model, planner, or teacher.
- **Why more natural than old-school alternatives**: LoRA is the standard for parameter-efficient adaptation of frozen LLMs. A hand-tuned consolidation schedule would require extensive hyperparameter search per domain; PPO automates this with a principled reward signal.

### Integration into Base Generator

- **Base model**: Qwen3.5-9B, fully frozen (not a single base parameter is trained)
- **Attachment point**: Two LoRA adapters attached via PEFT library's multi-adapter API
- **What is trainable**: WM-LoRA (online, per-turn), LT-LoRA (consolidation only), Policy MLP (PPO)
- **Inference order**: Both adapters active simultaneously; PEFT handles the forward pass with combined LoRA projections

### Training Plan

**Stage 1: SPM Training (100 sessions)**
- Load PersonaChat train conversations, group by persona
- For each session: 20 turns of online WM-LoRA updates + periodic consolidation
- Consolidation uses fixed interval (every 10 turns) — policy not yet active
- Output: Trained LT-LoRA, populated memory buffer, forgetting curves

**Stage 2: PPO Consolidation Policy Training (50 episodes)**
- Load pre-trained SPM from Stage 1
- For each episode: 40 turns with policy-decided consolidation
- Reward: quality improvement - forgetting penalty - consolidation cost
- Output: Trained policy checkpoint

**Stage 3: Streaming Evaluation**
- 4 methods × 2 datasets × 50 sessions
- Methods: no_adapt, full_ft (single LoRA), EWC, SPM (ours)
- Datasets: PersonaChat validation, LIGHT test
- Metrics: Perplexity, BLEU-4, Persona Retention F1, Knowledge Retention@10, Retention@30

**Stage 4: Ablation Studies**
- Consolidation frequency: 5, 10, 20, 50 turns
- EWC lambda: 100, 1000, 5000, 10000
- LoRA rank: 4, 8, 16, 32, 64

### Failure Modes and Diagnostics

1. **WM-LoRA online updates destabilize generation**
   - Detection: Loss spikes > 3× moving average, or perplexity > 2× baseline
   - Mitigation: Gradient clipping (norm ≤ 1.0), constrain WM-LoRA norm

2. **Fisher estimates too noisy from few turns**
   - Detection: Fisher variance across consecutive estimates > mean
   - Mitigation: EMA smoothing of Fisher (α=0.9), minimum 10 samples before first consolidation

3. **Consolidation policy collapses to always/never consolidate**
   - Detection: Consolidation rate < 0.05 or > 0.8 for 10+ consecutive episodes
   - Mitigation: Entropy bonus in PPO reward, warm-start with heuristic rule

4. **LT-LoRA saturates after many sessions**
   - Detection: Retention plateau despite continued training
   - Mitigation: Progressive rank increase, periodic Fisher re-estimation

### Novelty and Elegance Argument

**Closest work**:
- **EWC (Kirkpatrick et al., 2017)**: Uses Fisher to protect parameters but operates on a single parameter set. SPM adds structural separation (dual adapters) that allows free working memory.
- **Progressive Memory Banks (PMB)**: Stores separate snapshots per task. SPM consolidates into a single persistent adapter, avoiding linear storage growth.
- **ROSA2 (2025)**: Test-time adaptation via LoRA but lacks explicit memory separation. SPM's dual-adapter architecture provides principled working/long-term distinction.
- **CharacterFlywheel (Meta, 2026)**: Iterative offline refinement, not streaming. SPM operates in true streaming mode.

**Exact difference**: SPM is the first method to combine (1) structural dual-adapter separation with (2) Fisher-informed consolidation in (3) a true streaming online setting for LLM personalization. The novelty is in the *architectural pattern* and *consolidation mechanism*, not in any individual component.

## Claim-Driven Validation Sketch

### Claim 1: SPM achieves better plasticity-stability trade-off than single-adapter methods

- **Minimal experiment**: Compare SPM vs. single-LoRA, EWC, and frozen model on 50 PersonaChat sessions
- **Baselines / ablations**: no_adapt, full_ft, EWC, SPM_no_policy (fixed interval)
- **Metric**: Persona Retention F1 after N sessions, Session Adaptation Speed (turns to alignment)
- **Expected evidence**: SPM Retention F1 ≥ 0.65 vs. ≤ 0.42 single-LoRA; Adaptation speed ≤ 5 turns vs. ≥ 12

### Claim 2: The dual-adapter separation is the key mechanism, not just EWC or replay

- **Minimal experiment**: Ablation removing structural separation (single LoRA + EWC + replay vs. dual LoRA + EWC + replay)
- **Baselines / ablations**: SPM_full vs. SPM_single_adapter (same EWC + replay but one adapter)
- **Metric**: Retention gap after 50 sessions
- **Expected evidence**: Dual-adapter variant retains ≥ 15% more persona F1 than single-adapter with same consolidation

## Experiment Handoff Inputs
- **Must-prove claims**: (1) SPM > single-LoRA and EWC on retention+adaptation, (2) Dual-adapter separation is necessary
- **Must-run ablations**: Consolidation frequency sweep, EWC lambda sweep, LoRA rank sweep, policy vs. fixed interval
- **Critical datasets / metrics**: PersonaChat (Retention F1, Adaptation Speed), LIGHT (Character Consistency)
- **Highest-risk assumptions**: (1) WM-LoRA online updates are stable enough for useful adaptation in 5 steps, (2) Fisher estimation with small replay buffer is accurate enough for meaningful importance weighting

## Compute & Timeline Estimate
- **Estimated GPU-hours**: ~700 (8× A100-80GB)
  - Stage 1 SPM training: 384 GPU-hours
  - Stage 2 PPO policy: 192 GPU-hours
  - Stage 3 Evaluation: 64 GPU-hours
  - Stage 4 Ablations: 60 GPU-hours
- **Data / annotation cost**: Zero (PersonaChat and LIGHT are publicly available)
- **Timeline**: 7 weeks (2 weeks coding, 2 weeks training, 1 week PPO, 1 week eval, 1 week writing)
