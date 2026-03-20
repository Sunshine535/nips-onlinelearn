# Related Papers — Streaming Parameter Memory (SPM)

## Core Related Work

### 1. CharacterFlywheel — Meta, 2026 (Production System)
- **Paper**: "CharacterFlywheel: Iterative Persona Refinement through Multi-Generation LoRA Training"
- **Venue**: Meta Technical Report, 2026
- **Key idea**: Iterative offline pipeline: collect user interactions → filter by engagement → fine-tune character LoRA → deploy → repeat. Converges after ~15 generation cycles in production.
- **Limitation**: Strictly offline; requires batch retraining. No in-session adaptation. 15 cycles = weeks of latency for new characters.
- **SPM differentiation**: SPM adapts *within* a single session via streaming WM-LoRA updates. No batch retraining required. Consolidation is continuous, not iterative.

### 2. ROSA2 — 2026
- **Paper**: "ROSA2: Test-Time Adaptation with Words and Weights"
- **Venue**: Preprint, 2026
- **Key idea**: Extends ROSA to jointly adapt both prompt (words) and LoRA parameters (weights) at test time. Single unified adaptation mechanism.
- **Limitation**: No explicit short/long-term separation. All adaptations are ephemeral or cumulative without principled distinction.
- **SPM differentiation**: SPM explicitly separates working memory from long-term storage. Fisher-based consolidation provides principled importance scoring vs. ROSA2's uniform adaptation.

### 3. T2PAM — NeurIPS 2025
- **Paper**: "T2PAM: Task-to-Parameter Allocation Module for Efficient Multi-Task LoRA"
- **Venue**: NeurIPS 2025
- **Key idea**: Learns to allocate LoRA parameters to tasks via a routing mechanism. Static allocation at deployment time.
- **Limitation**: No online adaptation; allocation is fixed after training. Cannot handle new tasks without retraining.
- **Relevance**: SPM's consolidation policy is analogous to T2PAM's allocation — but operates continuously and online.

### 4. ROSA — NeurIPS 2025
- **Paper**: "ROSA: Random Orthogonal Subspace Adaptation"
- **Venue**: NeurIPS 2025
- **Key idea**: Adapts LLMs in random orthogonal subspaces to avoid interference between tasks. Efficient multi-task adaptation.
- **Limitation**: Subspace selection is random, not learned. No memory consolidation mechanism.
- **Relevance**: SPM could incorporate orthogonal constraints between WM-LoRA and LT-LoRA subspaces.

### 5. DEAL — 2025
- **Paper**: "DEAL: Dynamic Expert Allocation for Lifelong Learning in LLMs"
- **Key idea**: Dynamically allocates MoE experts to new tasks, freezing old experts. Expert-level granularity.
- **Limitation**: Expert-level is too coarse for per-conversation personalization. Requires MoE architecture.
- **SPM differentiation**: SPM operates at parameter level, not expert level. Works with any transformer + LoRA.

### 6. GainLoRA — 2025
- **Paper**: "GainLoRA: Gradient-Aware Importance-Normalized LoRA"
- **Key idea**: Uses gradient magnitude to determine per-parameter LoRA rank allocation.
- **Relevance**: Gradient importance is related to Fisher Information. GainLoRA applies this at training time; SPM applies it for online consolidation.

### 7. SELF-PARAM — 2025
- **Paper**: "SELF-PARAM: Self-Supervised Parameter Memory for Continual Learning"
- **Key idea**: Self-supervised replay of parameter snapshots to prevent forgetting in continual learning.
- **Limitation**: Requires storing parameter snapshots (memory overhead). Replay-based, not streaming.
- **SPM differentiation**: SPM consolidates in real-time without storing snapshots. Fisher importance replaces replay.

## Foundational Work

### 8. LoRA — Hu et al., ICLR 2022
- **Paper**: "LoRA: Low-Rank Adaptation of Large Language Models"
- **Key idea**: Low-rank decomposition W = W₀ + BA for efficient fine-tuning.
- **Relevance**: SPM's dual LoRA architecture builds directly on LoRA. Both WM and LT modules use standard LoRA parameterization.

### 9. Elastic Weight Consolidation (EWC) — Kirkpatrick et al., 2017
- **Paper**: "Overcoming catastrophic forgetting in neural networks"
- **Key idea**: Use Fisher Information diagonal to penalize changes to important parameters.
- **Relevance**: SPM uses Fisher for consolidation (transfer), not for regularization (protection). Opposite application direction.

### 10. Complementary Learning Systems — McClelland et al., 1995
- **Paper**: "Why there are complementary learning systems in the hippocampus and neocortex"
- **Key idea**: Fast hippocampal learning + slow neocortical consolidation.
- **Relevance**: Direct theoretical inspiration for SPM's dual-memory architecture.

## Evaluation Benchmarks

### 11. PersonaChat — Zhang et al., 2018
- **Paper**: "Personalizing Dialogue Agents: I have a dog, do you have pets too?"
- **Venue**: ACL 2018
- **Stats**: 10,907 dialogues, 1,155 personas, 5 profile sentences each
- **Use in SPM**: Primary evaluation benchmark for persona retention and adaptation.

### 12. LIGHT — Urbanek et al., 2019
- **Paper**: "Learning to Speak and Act in a Fantasy Text Adventure Game"
- **Venue**: EMNLP 2019
- **Stats**: 663 characters, 1,755 locations, situated dialogue
- **Use in SPM**: Complex character consistency evaluation in open-ended dialogue.

## Positioning Matrix

| Method | Online? | Dual Memory? | Learned Consolidation? | Parameter-level? |
|--------|---------|-------------|----------------------|-----------------|
| CharacterFlywheel | ✗ (offline batch) | ✗ | ✗ (heuristic filter) | ✓ (LoRA) |
| ROSA2 | ✓ (test-time) | ✗ | ✗ (gradient-based) | ✓ (words+weights) |
| T2PAM | ✗ (static) | ✗ | ✓ (learned routing) | ✓ (LoRA) |
| DEAL | ✗ (dynamic alloc) | ✗ | ✗ (rule-based) | ✗ (expert-level) |
| EWC | ✗ (regularization) | ✗ | ✗ (Fisher penalty) | ✓ |
| **SPM (Ours)** | **✓** | **✓** | **✓** | **✓** |

## Key Papers to Cite in Introduction

1. CharacterFlywheel — motivates the problem (production need for personalization)
2. ROSA2 — closest competitor (test-time adaptation, but no memory separation)
3. CLS Theory (McClelland 1995) — theoretical foundation
4. EWC (Kirkpatrick 2017) — Fisher Information as importance measure
5. LoRA (Hu 2022) — underlying parameterization

## Papers to Watch (May Appear Before Submission)

- Any Meta follow-up on CharacterFlywheel with online components
- ROSA3 or follow-up with memory mechanisms
- Any CLS-inspired LLM adaptation work from DeepMind/Google Brain
- NeurIPS 2025 workshop papers on personalized LLMs
