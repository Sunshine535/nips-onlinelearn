#!/usr/bin/env python3
"""SPM training: two-timescale residual LoRA with behavioral distillation.

For each session: zero-init WM → per-turn NTP updates → session-end KL consolidation.
Tracks retention, forgetting, and adaptation speed."""

import argparse
import glob
import json
import logging
import math
import os
import sys
import time

import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.streaming_memory import StreamingParameterMemory


def load_personachat_sessions(config: dict, tokenizer, max_sessions: int = 200,
                              allow_synthetic: bool = False):
    """Load PersonaChat as multi-turn conversation sessions grouped by persona."""
    session_len = config["streaming"]["session_length"]
    sessions = []

    try:
        ds = load_dataset("bavard/personachat_truecased", split="train", trust_remote_code=True)
        logger.info("Loaded PersonaChat: %d examples", len(ds))

        persona_sessions = {}
        for i, ex in enumerate(ds):
            if i >= 100000:
                break
            personality = ex.get("personality", [])
            persona_key = "|".join(personality[:3]) if personality else f"anon_{i // session_len}"
            history = ex.get("history", [])
            candidates = ex.get("candidates", [])
            reply = candidates[-1] if candidates else ""

            if not history or not reply:
                continue

            if persona_key not in persona_sessions:
                persona_sessions[persona_key] = {
                    "persona": " ".join(personality) if personality else "",
                    "turns": [],
                }
            persona_sessions[persona_key]["turns"].append({
                "user": history[-1] if history else "",
                "assistant": reply,
            })

        for pk, sess_data in persona_sessions.items():
            turns = sess_data["turns"]
            persona = sess_data["persona"]
            for start in range(0, len(turns), session_len):
                chunk = turns[start : start + session_len]
                if len(chunk) >= 3:
                    sessions.append({"persona": persona, "turns": chunk})
            if len(sessions) >= max_sessions:
                break

    except Exception as e:
        logger.warning("Failed to load PersonaChat: %s. Using synthetic data.", e)

    if len(sessions) < 10:
        if not allow_synthetic:
            raise RuntimeError(
                f"Only {len(sessions)} real sessions loaded (need >=10). "
                "Pass --allow-synthetic to use synthetic data instead."
            )
        logger.info("Generating synthetic conversation sessions...")
        for s in range(max_sessions):
            persona = f"I like topic {s % 20}. My name is person_{s}."
            turns = []
            for t in range(session_len):
                turns.append({
                    "user": f"Tell me about topic {s % 20}, aspect {t}.",
                    "assistant": (
                        f"As someone who likes topic {s % 20}, "
                        f"aspect {t} is fascinating because of concept_{s * 10 + t}."
                    ),
                })
            sessions.append({"persona": persona, "turns": turns})

    logger.info("Loaded %d conversation sessions", len(sessions))
    return sessions[:max_sessions]


def format_turn(user_msg: str, assistant_msg: str, persona: str = "") -> str:
    system = persona if persona else "You are a helpful and personalized assistant."
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_msg}<|im_end|>"
    )


@torch.no_grad()
def compute_perplexity(model, tokenizer, texts: list, max_length: int = 1024) -> float:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for text in texts[:50]:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        ids = enc["input_ids"].to(next(model.parameters()).device)
        out = model(input_ids=ids, labels=ids.clone())
        total_loss += out.loss.item() * ids.shape[1]
        total_tokens += ids.shape[1]
    return math.exp(total_loss / max(total_tokens, 1))


@torch.no_grad()
def probe_retention(spm, tokenizer, probe_facts: list) -> float:
    """Probe retention of previously taught facts (semantic check)."""
    device = next(spm.model.parameters()).device
    correct = 0
    for fact in probe_facts:
        prompt = f"<|im_start|>user\n{fact['question']}<|im_end|>\n<|im_start|>assistant\n"
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        try:
            out = spm.generate(ids, max_new_tokens=100, use_longterm=True)
            response = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).lower()
            keywords = fact["answer"].lower().split()[:3]
            if any(kw in response for kw in keywords):
                correct += 1
        except Exception as e:
            logger.warning("probe_retention generate failed: %s", e)
    return correct / max(len(probe_facts), 1)


def main():
    parser = argparse.ArgumentParser(description="Train SPM (Two-Timescale Residual LoRA)")
    parser.add_argument("--config", type=str, default="configs/spm_config.yaml")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_sessions", type=int, default=None)
    parser.add_argument("--probe_interval", type=int, default=10)
    parser.add_argument("--beta", type=float, default=None, help="KL distillation weight")
    parser.add_argument("--gamma", type=float, default=None, help="Fisher trust-region weight")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--allow-synthetic", action="store_true", default=False,
                        help="Allow synthetic data fallback if real data unavailable")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = args.output_dir or config.get("output_dir", "./outputs")
    output_dir = os.path.join(output_dir, "spm_training")
    os.makedirs(output_dir, exist_ok=True)

    base_model_name = config["model"]["base_model"]
    num_sessions = args.num_sessions or config["streaming"]["num_sessions"]
    beta = args.beta if args.beta is not None else config.get("consolidation", {}).get("beta", 1.0)
    gamma = args.gamma if args.gamma is not None else config.get("consolidation", {}).get("gamma", 0.0)

    logger.info("=== SPM Training (Behavioral Distillation Consolidation) ===")
    logger.info("Base model: %s", base_model_name)
    logger.info("Sessions: %d, Turns/session: %d", num_sessions, config["streaming"]["session_length"])
    logger.info("β (KL weight): %.2f, γ (Fisher trust-region): %.2f", beta, gamma)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_idx = int(os.environ.get("LOCAL_RANK", 0))
    base_model = None
    for attn_impl in ["sdpa", "eager"]:
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation=attn_impl,
                device_map={"": device_idx},
            )
            logger.info("Attention implementation: %s", attn_impl)
            break
        except (ImportError, ValueError, RuntimeError) as e:
            logger.warning("attn_implementation=%s failed: %s", attn_impl, e)
    if base_model is None:
        raise RuntimeError(f"Cannot load model {base_model_name}")
    base_model.config.use_cache = False

    spm = StreamingParameterMemory(
        base_model=base_model,
        working_config=config["working_memory"],
        longterm_config=config["long_term_memory"],
        reservoir_size=config["long_term_memory"].get("max_memory_buffer", 5000),
        beta=beta,
        gamma=gamma,
    )

    sessions = load_personachat_sessions(config, tokenizer, num_sessions,
                                         allow_synthetic=args.allow_synthetic)
    device = next(spm.model.parameters()).device

    probe_facts = []
    for sess in sessions[:5]:
        for turn in sess["turns"][:3]:
            probe_facts.append({
                "question": f"What did you say about: {turn['user'][:50]}?",
                "answer": turn["assistant"][:30],
                "taught_session": 0,
            })

    training_log = []
    forgetting_curve = []
    start_session = 0

    if args.resume_from_checkpoint:
        ckpt_search = args.resume_from_checkpoint
        if ckpt_search == "auto":
            ckpts = sorted(
                glob.glob(os.path.join(output_dir, "checkpoint-session-*", "checkpoint_session*.pt")),
                key=os.path.getmtime,
            )
            ckpt_search = ckpts[-1] if ckpts else None
        if ckpt_search and os.path.isfile(ckpt_search):
            logger.info("Resuming from %s", ckpt_search)
            state = torch.load(ckpt_search, map_location="cpu", weights_only=False)
            start_session = state.get("session_idx", 0) + 1
            training_log = state.get("training_log", [])
            forgetting_curve = state.get("forgetting_curve", [])
            probe_facts = state.get("probe_facts", probe_facts)
            import pickle
            reservoir_path = os.path.join(os.path.dirname(ckpt_search), "reservoir.pkl")
            if os.path.isfile(reservoir_path):
                with open(reservoir_path, "rb") as f:
                    spm.reservoir = pickle.load(f)
                logger.info("Restored reservoir (%d items)", len(spm.reservoir))
            adapter_dir = os.path.join(os.path.dirname(ckpt_search), "adapters")
            if os.path.isdir(adapter_dir):
                for adapter_name in ["longterm", "working"]:
                    adapter_path = os.path.join(adapter_dir, adapter_name)
                    if os.path.isdir(adapter_path):
                        try:
                            spm.model.load_adapter(adapter_path, adapter_name=adapter_name)
                            logger.info("Restored %s adapter from %s", adapter_name, adapter_path)
                        except Exception as e:
                            logger.warning("Could not restore %s adapter: %s", adapter_name, e)
            spm.session_id = state.get("session_id", start_session - 1)
            spm.total_turns = state.get("total_turns", 0)
            logger.info("Resuming from session %d (total_turns=%d)", start_session, spm.total_turns)

    start_time = time.time()

    for session_idx, session in enumerate(sessions):
        if session_idx < start_session:
            continue

        spm.start_new_session()
        session_losses = []
        persona = session.get("persona", "")
        turn_latencies = []

        for turn_idx, turn in enumerate(session["turns"]):
            turn_start = time.time()
            text = format_turn(turn["user"], turn["assistant"], persona)
            encoded = tokenizer(
                text, return_tensors="pt", truncation=True,
                max_length=config["training"]["max_seq_length"],
            )
            input_ids = encoded["input_ids"].to(device)
            labels = input_ids.clone()

            result = spm.process_turn(input_ids, labels)
            session_losses.append(result["loss"])
            turn_latencies.append(time.time() - turn_start)

            if turn_idx % 5 == 0:
                logger.info(
                    "Session %d/%d Turn %d/%d | Loss: %.4f | Latency: %.1fms",
                    session_idx + 1, len(sessions), turn_idx + 1, len(session["turns"]),
                    result["loss"], turn_latencies[-1] * 1000,
                )

        # Session-end consolidation
        consolidation_start = time.time()
        consolidation_loss = spm.consolidate_session()
        consolidation_time = time.time() - consolidation_start

        avg_loss = sum(session_losses) / max(len(session_losses), 1)
        avg_turn_latency = sum(turn_latencies) / max(len(turn_latencies), 1)

        retention = None
        if (session_idx + 1) % args.probe_interval == 0 and probe_facts:
            retention = probe_retention(spm, tokenizer, probe_facts)
            forgetting_curve.append({
                "session": session_idx + 1,
                "retention": retention,
                "total_turns": spm.total_turns,
            })
            logger.info("  Retention probe: %.2f%% (%d facts)", retention * 100, len(probe_facts))

            for turn in session["turns"][:2]:
                probe_facts.append({
                    "question": f"What did we discuss about: {turn['user'][:50]}?",
                    "answer": turn["assistant"][:30],
                    "taught_session": session_idx,
                })

        entry = {
            "session": session_idx,
            "avg_loss": avg_loss,
            "consolidation_loss": consolidation_loss,
            "num_turns": len(session["turns"]),
            "retention": retention,
            "reservoir_size": len(spm.reservoir),
            "avg_turn_latency_ms": avg_turn_latency * 1000,
            "consolidation_time_ms": consolidation_time * 1000,
        }
        training_log.append(entry)

        if (session_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            logger.info(
                "=== Session %d/%d | Loss: %.4f | Consol: %.4f | "
                "Reservoir: %d | Elapsed: %.1fs ===",
                session_idx + 1, len(sessions), avg_loss, consolidation_loss,
                len(spm.reservoir), elapsed,
            )

        ckpt_interval = config.get("training", {}).get("checkpoint_interval", 20)
        if (session_idx + 1) % ckpt_interval == 0:
            ckpt_dir = os.path.join(output_dir, f"checkpoint-session-{session_idx + 1}")
            spm.save(ckpt_dir)
            ckpt_meta_path = os.path.join(ckpt_dir, f"checkpoint_session{session_idx + 1}.pt")
            torch.save({
                "session_idx": session_idx,
                "session_id": spm.session_id,
                "training_log": training_log,
                "forgetting_curve": forgetting_curve,
                "probe_facts": probe_facts,
                "total_turns": spm.total_turns,
                "beta": spm.beta,
                "gamma": spm.gamma,
            }, ckpt_meta_path)
            logger.info("Saved checkpoint at session %d -> %s", session_idx + 1, ckpt_dir)

    spm.save(os.path.join(output_dir, "final"))

    with open(os.path.join(output_dir, "training_log.json"), "w") as f:
        json.dump(training_log, f, indent=2)
    with open(os.path.join(output_dir, "forgetting_curve.json"), "w") as f:
        json.dump(forgetting_curve, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if forgetting_curve:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            sessions_x = [p["session"] for p in forgetting_curve]
            retention_y = [p["retention"] for p in forgetting_curve]
            axes[0].plot(sessions_x, retention_y, "b-o", label="Retention")
            axes[0].set_xlabel("Session")
            axes[0].set_ylabel("Retention Rate")
            axes[0].set_title("Knowledge Retention Over Sessions")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            losses = [e["avg_loss"] for e in training_log]
            axes[1].plot(range(len(losses)), losses, "r-", alpha=0.5, label="Session Loss")
            axes[1].set_xlabel("Session")
            axes[1].set_ylabel("Loss")
            axes[1].set_title("Training Loss")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            consol_losses = [e["consolidation_loss"] for e in training_log if e["consolidation_loss"] > 0]
            if consol_losses:
                axes[2].plot(range(len(consol_losses)), consol_losses, "g-o", markersize=3)
                axes[2].set_xlabel("Session")
                axes[2].set_ylabel("Consolidation Loss")
                axes[2].set_title("Consolidation Loss")
                axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "training_curves.pdf"), dpi=150, bbox_inches="tight")
            plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150, bbox_inches="tight")
            plt.close()
    except ImportError:
        pass

    total_time = time.time() - start_time
    logger.info("\n=== Training Complete ===")
    logger.info("Sessions: %d, Total turns: %d", len(sessions), spm.total_turns)
    logger.info("Reservoir size: %d", len(spm.reservoir))
    if forgetting_curve:
        logger.info("Final retention: %.2f%%", forgetting_curve[-1]["retention"] * 100)
    logger.info("Total time: %.1fs (%.1f min)", total_time, total_time / 60)
    logger.info("Saved to %s", output_dir)


if __name__ == "__main__":
    main()
