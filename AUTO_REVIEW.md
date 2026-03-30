# Auto Review Log: nips-onlinelearn

## Round 1 — Code / Artifact Review (2026-03-31)

### Assessment (Summary)
- Score: 3.0/10
- Verdict: NOT READY TO PRODUCE EXPERIMENTAL RESULTS
- Review scope:
  1. Code quality and completeness
  2. Multi-GPU support
  3. Checkpoint resume support
  4. Whether the code is ready to produce results

### Verification Performed
- `python3 -m compileall src scripts` passes.
- `bash -n run.sh setup.sh collect_results.sh scripts/*.sh` passes.
- No test suite is present in the repository.
- The current local environment for this review does not have `torch` installed, so runtime verification here was limited to static inspection plus the recorded `run.log` and `logs/phase1_spm.log`.

### Main Findings

#### 1. Critical: the main evaluation pipeline does not evaluate the phase-1 trained SPM artifact
- `scripts/run_all_experiments.sh:54-61` runs phase 1 training and writes `outputs/spm_training`.
- `scripts/run_all_experiments.sh:93-97` launches `eval_streaming.py` without any checkpoint/model argument.
- `scripts/eval_streaming.py:447-463` exposes no `--model_dir` or equivalent argument for loading trained SPM weights.
- `scripts/eval_streaming.py:275-299` constructs a fresh `SPMMethod`, and `scripts/eval_streaming.py:494-520` reloads a fresh base model per method/dataset.
- Net effect: the headline 7-method comparison in phase 3 is disconnected from phase 1 and does not measure the trained SPM model saved by the pipeline.

#### 2. Critical: there is direct evidence that the repository is not currently runnable end-to-end
- The recorded pipeline run in `run.log:13-27` fails in phase 1 with `RuntimeError: Cannot load model Qwen/Qwen3.5-9B`.
- The same failure is visible in `logs/phase1_spm.log:3-14`.
- The base model is hard-coded at `configs/spm_config.yaml:1-2`.
- `run.sh:37-51` exits on that failure, and there are no generated experiment artifacts under `outputs/` or `results/` in this checkout.
- This is not just a theoretical reproducibility risk; the included run evidence shows the pipeline did not complete.

#### 3. High: "multi-GPU support" is only independent job scheduling, not actual multi-GPU training
- `scripts/gpu_utils.sh:55-63` and `scripts/gpu_utils.sh:93-143` advertise `torchrun`, `accelerate`, and FSDP helpers.
- `scripts/run_all_experiments.sh:17` computes `TORCHRUN` but never uses it.
- `scripts/run_all_experiments.sh:56-61` and `scripts/run_all_experiments.sh:70-75` pin training jobs to a single GPU via `CUDA_VISIBLE_DEVICES="$GPU0"`.
- The ablation/eval loops in `scripts/run_all_experiments.sh:86-99` and `scripts/run_all_experiments.sh:133-143` merely fan out separate single-GPU subprocesses.
- `scripts/train_spm.py:171-180` and `scripts/train_ppo_integration.py:251-259` each load one model replica onto one device via `device_map={"": device_idx}`.
- I would not count this as true multi-GPU training support for a 9B-model artifact.

#### 4. High: checkpoint/resume support is partial and brittle
- `scripts/train_spm.py:323-335` only saves checkpoints every `checkpoint_interval` sessions, so an interruption can lose substantial progress.
- `scripts/train_spm.py:214-245` restores metadata and the replay reservoir, but not optimizer state, RNG state, or any in-flight session buffer.
- Adapter save/load assumptions are inconsistent:
  - `src/streaming_memory.py:480-489` saves either to `adapters/` or per-adapter subdirectories depending on code path.
  - `scripts/train_spm.py:235-240` resumes by loading from the root `adapters` directory.
  - `scripts/eval_spm.py:158-180` expects per-adapter subdirectories such as `adapters/working` and `adapters/longterm`.
- `scripts/train_ppo_integration.py:301-312` resumes only the PPO policy/optimizer, while recreating a fresh `StreamingParameterMemory` at `scripts/train_ppo_integration.py:268-273`; the environment state itself is not resumed.

#### 5. High: several advertised features are dead or disconnected from the implementation
- `configs/spm_config.yaml:26-57` advertises batch size, gradient accumulation, epoch count, scheduler, `save_steps`, gradient checkpointing, consolidation frequency, replay ratio, and cross-session probes.
- A repository-wide search shows those fields are never consumed outside the config file; in practice `scripts/train_spm.py:261-263` only uses `training.max_seq_length` from the `training` block.
- `scripts/run_all_experiments.sh:66-76` trains a PPO consolidation policy in phase 2, but no code path ever loads `outputs/ppo_policy/consolidation_policy.pt`.
- The README experiment table (`README.md:81-91`) also does not describe that PPO phase, so the documented pipeline and the actual pipeline diverge.

#### 6. Medium: evaluation quality is too weak for paper-grade evidence
- `scripts/eval_streaming.py:67-87` claims "NLI-based semantic persona retention" but actually scores keyword overlap in generated text.
- `scripts/eval_streaming.py:344-352` silently falls back to synthetic sessions if dataset loading fails.
- `scripts/train_spm.py:73-86` likewise falls back to synthetic training sessions.
- `scripts/eval_spm.py:205-247` evaluates retention on five hand-written toy facts rather than benchmark-grounded probes.
- `configs/spm_config.yaml:56` and `README.md:107-112` promise `adaptation_speed`, but `scripts/eval_streaming.py:428-437` never computes or reports that metric.

#### 7. Medium: code quality is acceptable for a prototype, but not artifact-grade
- The repository has real implementation rather than empty placeholders, and the Python/shell sources are syntactically valid.
- However, there are no unit or integration tests for the central claims: gradient isolation, checkpoint round-trips, distributed launch, or end-to-end experiment execution.
- `src/streaming_memory.py:147-163` creates a new `AdamW` optimizer on every single turn update, which is both inefficient and guarantees optimizer state is discarded between turns.

### Overall Judgment By Requested Axis

#### 1. Code Quality and Completeness
- Below artifact bar.
- There is a meaningful prototype here, but it is not complete: major pipeline pieces are disconnected, several configuration knobs are dead, and the main evaluation path does not consume the main training artifact.

#### 2. Multi-GPU Support
- Partial only.
- The repo can schedule separate single-GPU jobs across multiple devices, but it does not implement actual multi-GPU training for the core model.

#### 3. Checkpoint Resume Support
- Partial only.
- There is coarse checkpointing, but it is not robust enough for long expensive runs, and the save/load layout is inconsistent across training and evaluation.

#### 4. Ready to Produce Experimental Results?
- No.
- The strongest blockers are:
  1. the included run evidence shows phase 1 failing before any results are produced,
  2. the main evaluator does not load the phase-1 trained SPM model,
  3. the reported metrics are still too heuristic for NeurIPS-grade claims.

### Actionable Feedback
1. Make phase 3 evaluate the actual phase-1 artifact. Add a `--model_dir`/`--checkpoint` argument to `eval_streaming.py`, load `outputs/spm_training/final`, and assert that the directory exists before evaluation starts.
2. Add a post-setup smoke test that instantiates `config["model"]["base_model"]` and fails fast if the installed `transformers`/`peft` stack cannot load it.
3. Decide whether the project truly supports multi-GPU training. If yes, wire `torchrun`/Accelerate/FSDP into `train_spm.py`; if not, stop advertising multi-GPU support beyond parallel job scheduling.
4. Standardize one checkpoint layout and add a round-trip save/load test that covers both `train_spm.py --resume_from_checkpoint` and `eval_spm.py --model_dir`.
5. Save step-level or much more frequent checkpoints, including optimizer state, RNG state, and any session-local state needed for faithful resume.
6. Either remove the PPO phase from the main pipeline or integrate the learned policy into SPM training/evaluation. Right now it consumes compute without affecting the reported method.
7. Remove silent synthetic-data fallbacks from paper runs. If a dataset is unavailable, fail explicitly or require an opt-in debug flag such as `--allow_synthetic_fallback`.
8. Implement the promised `adaptation_speed` metric and replace keyword-overlap retention checks with a stronger evaluator before claiming NeurIPS-level results.
9. Add minimal automated tests: one-turn train smoke test, checkpoint resume smoke test, phase-3 evaluator smoke test on a fixture, and a small multi-process launch test.

### Bottom Line
This repository is a serious prototype, not a result-ready NeurIPS artifact. The codebase has substantive content, but the current pipeline cannot yet support reliable experimental claims because the main training artifact is not what the main evaluator measures, the logged end-to-end run already failed on model loading, and the multi-GPU and resume stories are both incomplete.
