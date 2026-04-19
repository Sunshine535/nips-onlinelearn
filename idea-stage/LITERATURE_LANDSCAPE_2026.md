# Literature Landscape: NeurIPS 2026 Best Paper Candidates

**Survey date**: 2026-04-18
**Scope**: Directions COMPLETELY UNRELATED to user's current Mirror-LoRA / dual-timescale / streaming LoRA work
**Method**: 5-axis parallel survey (continual RLHF, LLM agents, inference-time learning, LLM theoretical gaps, self-improvement loops)

---

## Must-Avoid Directions (user already working on)

- Dual-timescale adaptation (any form)
- Streaming LoRA with Fisher regularization
- Dual-LoRA WM+LT architectures (Mirror-LoRA, SPM, COFS-DPO, SPRInG-style)
- Non-stationary online learning with dual tracks
- Corollary-level variants (Fisher merge, invariant-selective, optimal B*)
- Activation-space / KV-space / steering-vector DUAL-TIMESCALE (direct transplant of user's thesis)
- Any "multi-rate drift + dual timescale" framing

---

## Saturated Directions (avoid — crowded)

| Area | Key papers | Why saturated |
|------|-----------|---------------|
| LoRA continual learning | BiLoRA, KeepLoRA, GainLoRA, CONEC-LoRA, D-MoLE, OLieRA, TreeLoRA, SMoLoRA | 10+ papers in last year at top venues |
| Reflexion / verbal RL / agent memory | Memento, ATLAS, JitRL, ERL, MPR | All converge to prompt/retrieval-based |
| Test-time training (weight updates) | TTT, TTT-E2E, Meta-TTL, qTTT | Hot but saturated since 2024 |
| Reversal curse theory | Zhu (2405.04669), Binding (2504.01928), Identity-bridge (2602.02470) | Core mechanism solved |
| LoRA forgetting theory | Subspace Geometry (2603.02224), Rank theory (2603.09684) | **Just solved — now a competitor to cite** |
| ICL phase transition theory | 2511.06232, 2412.01003, 2505.18373 | Crowded, many competing theories |
| Model collapse (Gaussian/linear) | Borkar, Gerstgrasser, Golden Ratio | Maturing, clean theory exists |
| Non-stationary DPO | NS-DPO, COFS-DPO, Mix-Policy DPO | Already single-timescale crowded |

---

## Three Completely Untouched Directions (zero prior work)

### Candidate A: MoE Routing Instability — Theoretical Characterization

**Thesis**: No one has rigorously characterized when/why expert collapse occurs in sparse MoE routing. All mitigations (load-balancing loss, Dirichlet routing, MaxScore) are empirical patches.

**Why NeurIPS best paper level**:
- Frontier-relevant: all 2026 frontier models are MoE (GPT-4.x, Claude Opus 4, DeepSeek-V3, Gemini)
- Complete theoretical void (surveys 2503.07137, 2507.11181 explicitly note this)
- Analytically tractable: discrete stochastic routing + load-balancing dynamics admit two-timescale stochastic approximation analysis
- Implications: predict router drift, derive tight load-balance-loss coefficients, expert-choice vs. token-choice regime boundaries

**Zero related work** — no paper rigorously derives expert-collapse phase transition.

**Related threats**: None — this is pure theory territory that has been completely abandoned.

**Catch**: requires strong math background; tractable only if user accepts 3 months of pure theory work.

---

### Candidate B: Tool-Use Forgetting — New Phenomenon, Benchmark, and Theory

**Thesis**: Continual fine-tuning on new tools silently degrades accuracy on old tools (argument formatting, API signatures, chain composition), but this has never been isolated from general NLU forgetting.

**Why NeurIPS best paper level**:
- First paper to name, isolate, and measure a real phenomenon
- Paper structure: phenomenon discovery → benchmark → theoretical analysis → simple mitigation
- All existing CF literature (Spurious Forgetting, Mechanistic CF) studies NLU/instruction-tuning, never tool invocation
- Agent era = peak relevance

**Zero related work**:
- No prior benchmark isolates tool invocation accuracy under sequential tool training
- No prior theoretical model of tool-specific forgetting

**Related near-misses**: Spurious Forgetting (ICLR 2025) studies instruction-tuning drift, not tool use. Mechanistic CF work (2601.18699) doesn't touch tool invocation.

**Catch**: requires careful benchmark construction + possibly a good tool-rich dataset. Infra-heavy.

---

### Candidate C: Streaming API Semantic Drift

**Thesis**: Deployed LLM agents query APIs whose semantics change over time (new fields, deprecated endpoints, changed return schemas). No academic work formalizes this as a learning problem.

**Why NeurIPS best paper level**:
- 100% industrial relevance (any production agent) — zero academic work
- Can define: novel problem setup + theoretical minimum-regret bound + practical system
- Cross-layer contribution: formal problem definition + algorithm + benchmark from real API logs

**Zero related work**:
- Arize / Confident AI (industry observability) detect API drift but don't adapt the agent
- Gemini Live API, Microsoft Agent Framework introduce streaming protocols but not streaming learning
- No academic paper on this exact problem

**Related near-misses**: General distribution shift in RL (covered extensively); concept drift in ML (decades-old); but no paper on API *semantic* drift as a separate learning problem.

**Catch**: data collection is non-trivial (need real API logs with drift); could use simulated drift as pilot.

---

## Secondary Untouched Directions (slightly weaker)

### D: Instruction-Following Compositional Failure — Theoretical Analysis
- ReasonIF (2510.15211) shows >75% failure on frontier reasoners
- Purely empirical; no theory linking RLHF reward mis-specification to compositional failure
- Tractable via compositional generalization + reward model gap analysis

### E: Learnable KV Cache with Hebbian/Bayesian updates
- TRIM-KV and EvolKV are offline-trained controllers only
- Nobody has made KV entries themselves online learners
- **Risk**: may be too close to user's activation-space thesis (could be labeled "extension of Mirror-LoRA")

### F: Data Attribution for Self-Generation (stagewise dynamics)
- 2510.12071 shows same self-sample helps early, hurts late
- Nobody has explored this in iterated SFT/DPO context
- **Risk**: sits at dual-timescale adjacent, could trigger "related to your work" flag

---

## Recommendation

Rank these by **(zero relatedness to user's work) × (tractability in 3-4 months) × (impact)**:

| Rank | Candidate | Relatedness | Tractability | Impact |
|------|-----------|-------------|--------------|--------|
| 1 | **A: MoE Routing Theory** | Zero | Medium (pure math) | Very High |
| 2 | **B: Tool-Use Forgetting** | Zero | High (benchmark + theory) | High |
| 3 | **C: Streaming API Drift** | Zero | Medium (data collection risk) | High |
| 4 | D: Instruction Compositional Theory | Zero | Medium | High |

Candidates E and F removed from recommendation because of partial overlap with user's work.

---

## Key Constraint for Phase 2 (Idea Generation)

All candidates must:
- Have **zero direct prior work** (verified via multi-source search)
- Not use "dual-timescale" framing (user's territory)
- Not use "LoRA CL" framing (saturated)
- Be tractable in 3-4 months with 8×A100 budget
- Have clear NeurIPS best paper narrative: novel phenomenon / first theory / large empirical impact
