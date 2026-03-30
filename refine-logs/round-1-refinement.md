# Round 1 Refinement

## Problem Anchor
[VERBATIM FROM ROUND 0]
- **Bottom-line problem**: Large language models cannot adapt *within* a conversation session to user-specific knowledge while simultaneously retaining that knowledge *across* sessions. Current personalization approaches (offline LoRA fine-tuning, retrieval-augmented prompting) operate in batch mode, creating a fundamental latency gap where the model is always one session behind.
- **Must-solve bottleneck**: Existing online adaptation methods face a fundamental trade-off: fast in-session adaptation causes catastrophic forgetting, while strong forgetting prevention makes adaptation too slow. No mechanism provides both rapid session-local plasticity and durable cross-session stability simultaneously.
- **Non-goals**: NOT (1) general CL for LLMs, (2) RAG replacement, (3) multi-user scaling, (4) full model fine-tuning.
- **Constraints**: ~700 GPU-hours, Qwen3.5-9B frozen, PersonaChat + LIGHT, 7 weeks, NeurIPS 2026.
- **Success condition**: Retention F1 ≥ 0.65, adaptation ≤ 5 turns, forgetting ≤ 0.10/session, PPL degradation ≤ 5%.

## Anchor Check
- **Original bottleneck**: Single parameter space conflates session-local and persistent knowledge, forcing a plasticity-stability trade-off.
- **Why revised method still addresses it**: The dual-adapter residual design with behavior-space distillation consolidation directly separates the two timescales while the KL-distillation transfer is more principled than parameter-space EWC alone.
- **Reviewer suggestions rejected as drift**: The drift warning about oracle supervision is valid — we clarify that online supervision uses the conversation turns themselves (user message + model's own response), not oracle assistant labels. This is deployment-realistic.

## Simplicity Check
- **Dominant contribution after revision**: Two-timescale residual LoRA with behavior-space distillation consolidation — one equation-level mechanism.
- **Components removed or merged**:
  - PPO policy → REMOVED entirely. Consolidation triggers are now deterministic (session-end + optional surprise threshold).
  - Parameter-space EWC → DEMOTED to optional secondary stabilizer. Primary consolidation is now KL distillation.
  - Random WM init → REPLACED with zero-init residual over LT.
  - Complex 1/loss importance weighting → REPLACED with novelty-based gradient-norm weighting.
- **Reviewer suggestions rejected as unnecessary complexity**: None — all simplification suggestions accepted.
- **Why the remaining mechanism is still the smallest adequate route**: The only new components are (1) the dual-adapter lifecycle (WM as zero-init residual over LT) and (2) the consolidation distillation step. Both are minimal additions to standard PEFT.

## Changes Made

### 1. WM as Zero-Init Residual over LT (Method Specificity fix)
- **Reviewer said**: Make WM a residual adapter over LT, initialized to zero. Define explicit inference as Δ_total = Δ_LT + αΔ_WM.
- **Action**: Accepted. WM-LoRA is now zero-initialized at each session start. During inference, both adapters are active with the combined effect being LT's persistent knowledge plus WM's session-local adaptation delta. This means turn 1 of every session starts from the LT-adapted model, not random noise.
- **Reasoning**: Zero-init means the first turns of each session already benefit from accumulated long-term knowledge. Random init wasted those turns.
- **Impact on core method**: Significantly simplifies the architecture and makes the "residual" framing mathematically clean.

### 2. Behavior-Space Distillation Consolidation (Frontier Leverage fix)
- **Reviewer said**: Replace parameter-space EWC with behavior-space distillation from LT+WM into LT.
- **Action**: Accepted. The consolidation step now trains LT' by minimizing:
  L_consolidate = L_task(θ_LT'; B) + β · D_KL(p_{LT+WM}(·|x) || p_{LT'}(·|x))
  where B is the replay buffer and p_{LT+WM} is the teacher (combined adapters), p_{LT'} is the student (updated LT).
  Fisher-based trust region is optionally retained as a secondary stabilizer:
  + γ · Σ_i F_i · (θ_LT',i - θ_LT,i)² (γ set small, e.g., 100, or 0 to disable)
- **Reasoning**: KL distillation in token-distribution space is more natural for LLMs than parameter-space penalties. It directly preserves the model's behavioral knowledge.
- **Impact on core method**: This is now the central equation of the paper — the single mechanism-level novelty.

### 3. PPO Removed (Contribution Quality fix)
- **Reviewer said**: Cut PPO from the main method. Use deterministic trigger.
- **Action**: Accepted. Consolidation now triggers at (1) session end (always), and (2) optionally mid-session when surprise exceeds a threshold (loss spike > 2× moving average). The PPO policy is entirely removed from the main method.
- **Reasoning**: PPO added a separate contribution that diluted the main story. Deterministic triggers are simpler, more interpretable, and sufficient for the core claim.
- **Impact on core method**: The paper is now about ONE mechanism: two-timescale residual LoRA + KL-distillation consolidation.

### 4. Simplified Operational Loop (Feasibility fix)
- **Reviewer said**: Reduce operational complexity. Consolidate once per session, estimate Fisher once.
- **Action**: Accepted. The per-turn loop is now:
  - 1-3 gradient steps on WM (reduced from 5)
  - Store turn in replay buffer with gradient-norm importance weighting
  - At session end: one consolidation pass (KL distillation + optional Fisher trust-region)
  No mid-session Fisher estimation. Fisher computed once per consolidation on the replay buffer.
- **Reasoning**: Fewer moving parts per turn, lower latency, and no repeated Fisher estimation.
- **Impact on core method**: Makes the method deployable and testable within the 7-week timeline.

### 5. Clarified Online Supervision (Drift fix)
- **Reviewer said**: Clarify the online supervision channel.
- **Action**: The online learning signal is the model's own generated response given the user turn. Specifically:
  - User sends message → model generates response → (input=user+system, label=generated response)
  - Optionally, if user provides explicit correction or preference signal, use that instead
  - This is deployment-realistic: no oracle assistant labels needed
- **Reasoning**: Avoids the drift into supervised dialogue fine-tuning with hidden labels.
- **Impact on core method**: The method works in true deployment mode, not just benchmark mode.

### 6. Strengthened Baselines (Validation fix)
- **Reviewer said**: Add single LoRA + replay/EWC as a baseline and report per-turn latency.
- **Action**: Accepted. Updated baseline set:
  1. Frozen (no adaptation)
  2. Single LoRA online (naive)
  3. Single LoRA + EWC + replay (best single-adapter CL)
  4. Dual LoRA + fixed consolidation (our method, fixed-interval ablation)
  5. Dual LoRA + KL distillation consolidation (our full method)
  Add per-turn latency (ms) as a reported metric.
- **Impact on validation**: The ablation now directly isolates the value of (a) dual-adapter structure and (b) KL distillation consolidation.

## Revised Proposal

# Research Proposal: Two-Timescale Residual LoRA with Behavioral Distillation for Online LLM Personalization

## Problem Anchor
[Same as above — verbatim]

## Technical Gap

Current online adaptation methods for LLMs face a fundamental plasticity-stability dilemma. Single-LoRA online updates conflate session-local and persistent knowledge in one parameter set, causing catastrophic forgetting. EWC-style regularization treats all parameters uniformly without timescale separation. Retrieval-augmented approaches avoid forgetting but cannot perform parametric behavioral adaptation.

The gap: no existing method provides **structural** separation of two adaptation timescales (session-local vs. persistent) with a **principled** mechanism for transferring session-acquired knowledge into permanent storage without destroying prior knowledge.

## Method Thesis

- **One-sentence thesis**: A zero-initialized residual working-memory LoRA captures session-local adaptations on top of a persistent long-term LoRA, with end-of-session behavioral distillation transferring valuable knowledge from the combined model into the long-term adapter — achieving the first structural separation of adaptation timescales for online LLM personalization.

- **Why smallest adequate intervention**: The only additions to standard single-LoRA are (1) a second adapter slot (already supported by PEFT) and (2) a KL-distillation consolidation step at session boundaries. No new architectures, no RL, no external models.

- **Why timely**: Frozen 9B+ LLMs with LoRA are the standard deployment pattern. SPM adds temporal management to this existing substrate — a capability gap the LoRA ecosystem currently has no answer for.

## Contribution Focus

- **Dominant contribution**: Two-timescale residual LoRA architecture with behavioral distillation consolidation — a single, equation-level mechanism for managing online adaptation across sessions.
- **Explicit non-contributions**: LoRA itself, EWC, Fisher estimation, the base model architecture. All are reused tools; the novelty is in their composition into a principled two-timescale system.

## Proposed Method

### Complexity Budget
- **Frozen backbone**: Qwen3.5-9B (9B params, fully frozen)
- **New trainable**: Working-Memory LoRA (~13M, r=16, zero-init per session), Long-Term LoRA (~52M, r=64, persistent)
- **Tempting additions NOT used**: PPO policy, attention-based memory, per-persona adapters, retrieval systems

### System Overview

```
Session Start → WM-LoRA ← zero init (residual over LT)
                     ↓
For each turn t in session:
  1. User sends message x_t
  2. Model generates response y_t using (LT + WM)
  3. Online update: 1-3 gradient steps on WM-LoRA only
     Loss: L_ntp(y_t | x_t; θ_base + Δ_LT + Δ_WM)
  4. Store (x_t, y_t, ||∇||) in Memory Buffer B
                     ↓
Session End (or surprise trigger):
  5. Consolidation: Distill behavior of (LT + WM) into LT'
     L = L_task(θ_LT'; B) + β · D_KL(p_{LT+WM} || p_{LT'})
     + γ · Σ_i F_i · (θ_LT' - θ_LT)²  [optional trust-region]
  6. θ_LT ← θ_LT'
  7. Discard WM (will be zero-init next session)
```

### Core Mechanism: Behavioral Distillation Consolidation

**The central equation:**

θ_LT' = argmin_{θ} [ L_task(θ; B) + β · E_{x~B}[D_KL(p_{θ_base+Δ_LT+Δ_WM}(·|x) || p_{θ_base+Δ_θ}(·|x))] + γ · Σ_i F_i · (θ_i - θ_LT,i)² ]

Where:
- **L_task**: Next-token prediction loss on replay buffer B
- **D_KL**: KL divergence between the combined-adapter teacher and the updated LT student
- **Fisher trust-region**: Optional stabilizer protecting previously important LT parameters
- **β**: Distillation weight (controls how much behavioral knowledge to transfer)
- **γ**: Trust-region weight (controls how much to protect old LT knowledge; can be 0)

**Why this is the main novelty**: This single equation captures the entire consolidation mechanism. It (1) trains LT on new data, (2) preserves the behavioral knowledge of the combined WM+LT model via KL distillation, and (3) optionally protects previously important parameters via Fisher trust-region. No other component is needed.

**Input/Output**:
- Input: Replay buffer B, current Δ_WM (session-learned delta), current Δ_LT
- Output: Updated Δ_LT' that incorporates session knowledge
- WM is then discarded (zero-init next session)

### Working Memory Adapter (Residual Design)

- **Initialization**: Zero-init (LoRA_B = 0) at each session start
- **Inference**: h = W_base·x + Δ_LT·x + Δ_WM·x (both adapters active simultaneously)
- **Update**: Per-turn gradient descent on WM only, 1-3 steps, lr=5e-4
- **Why zero-init**: Session starts from the full LT-adapted model immediately. Early turns are productive from turn 1. The WM accumulates session-local deltas as a residual correction.

### Consolidation Trigger
- **Primary**: End of every session (always fires)
- **Optional**: Mid-session surprise trigger when per-turn loss exceeds 2× moving average (indicates genuinely new information worth consolidating early)
- **Why deterministic**: Simpler, more reproducible, and sufficient for the core claim. Adaptive timing is an orthogonal research question.

### Replay Buffer
- **Size**: 5000 examples max
- **Importance weighting**: Gradient norm ||∇_WM L|| at encoding time (high gradient norm = model was learning a lot from this example → more important for consolidation)
- **Eviction**: Replace lowest-importance examples when buffer is full

### Training Plan

**Stage 1: SPM Training (100 sessions, ~300 GPU-hours)**
- Load PersonaChat train, group by persona into sessions of 20 turns
- For each session: online WM updates (1-3 steps/turn) + consolidation at session end
- Track: retention curve, forgetting rate, per-turn loss, consolidation quality
- Checkpoint every 20 sessions

**Stage 2: Streaming Evaluation (~100 GPU-hours)**
- 5 methods × 2 datasets × 50 sessions
- Methods: frozen, single-LoRA, single-LoRA+EWC+replay, dual-LoRA+fixed-consolidation, dual-LoRA+KL-distillation (ours)
- Datasets: PersonaChat validation, LIGHT test
- Metrics: Persona Retention F1, Adaptation Speed (turns), Forgetting Rate, PPL, BLEU-4, Per-turn latency (ms)

**Stage 3: Ablation Studies (~100 GPU-hours)**
- Consolidation method: task-loss only vs. +KL distillation vs. +Fisher trust-region vs. +both
- WM initialization: zero-init vs. random-init vs. copy-from-LT
- Consolidation frequency: every session vs. every 5 sessions vs. surprise-triggered
- LoRA ranks: WM r∈{4,8,16}, LT r∈{16,32,64}

### Failure Modes and Diagnostics

1. **WM online updates destabilize generation**
   - Detection: Per-turn PPL spike > 2× moving average
   - Mitigation: Gradient clipping (norm ≤ 1.0), reduce to 1 update step, lower lr

2. **KL distillation collapses or diverges**
   - Detection: KL loss goes to 0 (collapse) or explodes
   - Mitigation: Clamp KL, use temperature scaling, verify buffer diversity

3. **LT saturates after many sessions**
   - Detection: Retention plateau despite continued consolidation
   - Mitigation: Increase LT rank, reset Fisher estimates periodically

### Novelty and Elegance Argument

**Closest work**:
- **EWC**: Parameter-space protection on single adapter. We add structural separation + behavior-space distillation.
- **ROSA2**: Test-time LoRA adaptation without memory separation. We add persistent LT + consolidation.
- **CharacterFlywheel**: Offline iterative, not streaming. We operate in true online mode.
- **Progressive Memory Banks**: Store snapshots per task. We consolidate into one persistent adapter.

**Exact difference**: SPM is the first to provide (1) structural two-timescale LoRA separation with (2) behavioral distillation consolidation for (3) true streaming online LLM personalization. The novelty is captured in a single equation (the consolidation objective) that can be described without mentioning PPO, attention mechanisms, or external models.

## Claim-Driven Validation Sketch

### Claim 1: Dual-LoRA + KL distillation consolidation achieves better plasticity-stability than single-adapter methods
- **Experiment**: 50 PersonaChat sessions, 5 methods
- **Baselines**: frozen, single-LoRA, single-LoRA+EWC+replay
- **Metric**: Persona Retention F1, Adaptation Speed, Forgetting Rate
- **Expected**: Retention F1 ≥ 0.65 vs. ≤ 0.42 (single-LoRA), ≤ 0.50 (single-LoRA+EWC)

### Claim 2: Structural dual-adapter separation is necessary (not just better consolidation)
- **Experiment**: Ablation comparing single-LoRA+KL-distillation vs. dual-LoRA+KL-distillation
- **Same consolidation mechanism, different architecture**
- **Expected**: Dual-adapter retains ≥ 15% more Retention F1

## Experiment Handoff Inputs
- **Must-prove claims**: (1) SPM > single-adapter CL methods, (2) dual-adapter separation is necessary
- **Must-run ablations**: consolidation method components, WM init strategy, consolidation frequency, LoRA rank
- **Critical metrics**: Persona Retention F1, Forgetting Rate, Per-turn Latency
- **Highest-risk assumption**: KL distillation from combined (LT+WM) to LT' is stable and effective

## Compute & Timeline Estimate
- **Estimated GPU-hours**: ~500 (reduced from 700 by removing PPO)
  - Stage 1 SPM training: 300
  - Stage 2 Evaluation: 100
  - Stage 3 Ablations: 100
- **Timeline**: 6 weeks (1.5 weeks code, 2 weeks training, 1 week eval, 1.5 weeks writing)
