# Literature Landscape: Online Learning & Streaming Adaptation for LLMs (2025-2026)

**Survey date**: 2026-04-07
**Sources**: arXiv, Google Scholar, Semantic Scholar, NeurIPS/ICML/ICLR proceedings

---

## Direct Competitors to SPM

| # | Paper | Venue | Threat | Key Difference from SPM |
|---|-------|-------|--------|------------------------|
| 1 | **SDFT** (Self-Distillation for CL) | arXiv Jan 2026 | **HIGHEST** | On-policy self-distillation, no dual adapter — simpler |
| 2 | **SPRInG** (Selective Parametric + Retrieval) | arXiv Jan 2026 | HIGH | Drift-driven selective adaptation + retrieval, no dual-timescale |
| 3 | **CONEC-LoRA** (Dual LoRA consolidation) | arXiv Oct 2025 | HIGH | Dual LoRA (shared+specific), but requires task boundaries |
| 4 | **KeepLoRA** (Residual subspace) | ICLR 2026 | HIGH | Single LoRA, principal/residual split — cleaner, but batch |
| 5 | **GainLoRA** (Gated LoRA) | NeurIPS 2025 | HIGH | Gated integration, but linear parameter growth |
| 6 | **D-MoLE** (MoE-LoRA) | ICML 2025 | HIGH | MoE routing alternative to dual-timescale |
| 7 | **SEAL** (Meta-learned TTT) | NeurIPS 2025 | MEDIUM | RL-learned TTT policy, model decides how to update |

## LoRA-Based Continual Learning Landscape

- **BiLoRA** (CVPR 2025) — almost-orthogonal via bilinear frequency decomposition
- **OLieRA** (arXiv Sep 2025) — orthogonality via Lie groups, SOTA on LLM CL
- **Merge before Forget** (arXiv Dec 2025) — sequential merge into single LoRA
- **PLAN** (ICCV 2025) — proactive subspace allocation anticipating interference
- **TreeLoRA** (ICML 2025) — hierarchical adapter tree with bandit routing
- **SMoLoRA** (ICCV 2025) — sparse MoE-LoRA for dual catastrophic forgetting
- **LoRAC-IPC** (Pattern Recog. 2025) — critical parameter freezing within LoRA
- **Text-to-LoRA** (ICML 2025) — hypernetwork generates LoRA from text description
- **DragLoRA** (ICML 2025) — online optimization of adapters
- **Meta-UCF** — hypernetwork-generated task-conditioned LoRA, constant memory

## Test-Time Training / Adaptation

- **In-Place TTT** (ICLR 2026) — fast-weight MLP projection update at inference
- **TTT-E2E** (arXiv Dec 2025) — context-to-weight compression via meta-learning
- **Meta-TTL** (arXiv Apr 2026) — bi-level optimization for TTT policy discovery
- **TLM** (arXiv May 2025) — domain adaptation via unlabeled test data

## Dual Memory / CLS-Inspired

- **Titans** (Google, Jan 2025) — short-term attention + neural long-term memory + persistent params
- **HiCL** (AAAI 2025) — hippocampal-inspired sparse coding + EWC consolidation
- **CH-HNN** (Nature Comms 2025) — corticohippocampal hybrid ANN+SNN
- **ITDMS** (2025) — information-theoretic fast/slow buffer

## Persona / Dialogue

- **PALACE** (ACL 2025) — persona knowledge graph + VAE-LoRA, multi-session
- **OpenCharacter** (arXiv Jan 2025) — 20K synthetic personas, SFT-based
- **PLUM** (Apple, L2M2 2025) — LoRA personalization from conversations, offline
- **PCL** (ACL 2025) — persona-aware contrastive learning for role-playing
- **MDRP** (arXiv Mar 2026) — memory-driven role-playing evaluation taxonomy

## Distillation for Continual Learning

- **SDFT** — on-policy self-distillation, demo-conditioned teacher
- **EvolveR** (arXiv Oct 2025) — self-distillation of experience into strategic principles
- **Continual Policy Distillation** (arXiv Jan 2026) — teacher-student CRL framework
- **SuRe** (arXiv Nov 2025) — surprise-driven prioritized replay

## Fisher Information / Importance

- **On Fisher Computation in CL** (arXiv Feb 2025) — standardization paper, critical for any Fisher-based method
- **LoRAC-IPC** — importance-based critical parameter freezing

---

## Key Trends (2025-2026)

1. **LoRA-based CL is dominant** — orthogonal, gated, MoE, merged variants all appearing at top venues
2. **TTT/TTA surging** — shifting compute to inference time
3. **Theoretical grounding required** — KeepLoRA, Spurious Forgetting provide formal proofs
4. **Strong baselines expected** — must beat O-LoRA, EWC, replay, vanilla LoRA, full FT
5. **Scalability evidence** — two model families, 100+ task sequences valued
6. **Streaming is the gap** — almost all CL work assumes discrete task boundaries

## Open Problems / Opportunities

1. **Online LoRA under non-stationary streams** — no task boundaries, continuous drift → SPM's target
2. **Regret theory for online low-rank adaptation** — online convex optimization + LoRA unconnected
3. **Forward transfer in LoRA-CL** — past knowledge accelerating new adaptation
4. **Continual alignment** — RLHF/DPO under distribution shift with forgetting guarantees
5. **Long-horizon benchmarks (100+ adaptations)** — existing benchmarks use 5-20 tasks
6. **Compute-optimal allocation** — when to TTT vs. retrain vs. consolidate

## SPM Positioning

**Unique combination**: streaming (no task boundaries) + dual-timescale LoRA + behavioral distillation + persona-specific

**Must differentiate from SDFT**: SDFT uses in-place self-distillation without dual adapters. SPM's advantage is explicit fast/slow separation enabling immediate rollback and principled consolidation scheduling. Must show empirically that dual-timescale > in-place distillation for streaming persona retention.

**Must address KeepLoRA**: KeepLoRA achieves stability+plasticity in single LoRA via subspace analysis. SPM must show that dual-timescale is necessary for streaming (not just helpful for batch CL).
