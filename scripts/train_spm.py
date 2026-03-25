#!/usr/bin/env python3
"""Streaming Parameter Memory training pipeline.
Simulates streaming conversation sessions from PersonaChat.
For each session: update working LoRA → estimate Fisher → consolidate to long-term.
Tracks: response quality, knowledge retention, forgetting rate."""

import argparse
import json
import logging
import math
import os
import sys

import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.streaming_memory import StreamingParameterMemory


def load_personachat_sessions(config: dict, tokenizer, max_sessions: int = 200):
    """Load PersonaChat as multi-turn conversation sessions with personas."""
    session_len = config["streaming"]["session_length"]
    sessions = []

    try:
        ds = load_dataset("bavard/personachat_truecased", split="train")
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
                chunk = turns[start:start + session_len]
                if len(chunk) >= 3:
                    sessions.append({"persona": persona, "turns": chunk})
            if len(sessions) >= max_sessions:
                break

    except Exception as e:
        logger.warning("Failed to load PersonaChat: %s. Using synthetic data.", e)

    if len(sessions) < 10:
        logger.info("Generating synthetic conversation sessions...")
        for s in range(max_sessions):
            persona = f"I like topic {s % 20}. My name is person_{s}."
            turns = []
            for t in range(session_len):
                turns.append({
                    "user": f"Tell me about topic {s % 20}, aspect {t}.",
                    "assistant": (f"As someone who likes topic {s % 20}, "
                                  f"aspect {t} is fascinating because of concept_{s * 10 + t}."),
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
    """Test retention of previously taught facts."""
    device = next(spm.model.parameters()).device
    correct = 0
    for fact in probe_facts:
        prompt = f"<|im_start|>user\n{fact['question']}<|im_end|>\n<|im_start|>assistant\n"
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        out = spm.generate(ids, max_new_tokens=100, use_longterm=True)
        response = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).lower()
        if fact["answer"].lower() in response:
            correct += 1
    return correct / max(len(probe_facts), 1)


def main():
    parser = argparse.ArgumentParser(description="Train Streaming Parameter Memory")
    parser.add_argument("--config", type=str, default="configs/spm_config.yaml")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_sessions", type=int, default=None)
    parser.add_argument("--probe_interval", type=int, default=10,
                        help="Sessions between retention probes")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = args.output_dir or config.get("output_dir", "./outputs")
    output_dir = os.path.join(output_dir, "spm_training")
    os.makedirs(output_dir, exist_ok=True)

    base_model_name = config["model"]["base_model"]
    num_sessions = args.num_sessions or config["streaming"]["num_sessions"]

    logger.info("=== Streaming Parameter Memory Training ===")
    logger.info("Base model: %s", base_model_name)
    logger.info("Sessions: %d, Turns/session: %d", num_sessions,
                config["streaming"]["session_length"])

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_idx = int(os.environ.get("LOCAL_RANK", 0))
    base_model = None
    for attn_impl in ["flash_attention_2", "sdpa", "eager"]:
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                dtype=torch.bfloat16,
                attn_implementation=attn_impl,
                device_map={"": device_idx},
            )
            logger.info("Attention implementation: %s", attn_impl)
            break
        except (ImportError, ValueError) as e:
            logger.warning("attn_implementation=%s failed: %s", attn_impl, e)
    if base_model is None:
        raise RuntimeError(f"Cannot load model {base_model_name}")
    base_model.config.use_cache = False

    spm = StreamingParameterMemory(
        base_model=base_model,
        working_config=config["working_memory"],
        longterm_config=config["long_term_memory"],
        memory_buffer_size=config["long_term_memory"].get("max_memory_buffer", 5000),
    )

    sessions = load_personachat_sessions(config, tokenizer, num_sessions)
    device = next(spm.model.parameters()).device

    # Build retention probes from early sessions
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
    consolidation_count = 0

    for session_idx, session in enumerate(sessions):
        spm.start_new_session()
        session_losses = []
        persona = session.get("persona", "")

        # Teach turns
        for turn_idx, turn in enumerate(session["turns"]):
            text = format_turn(turn["user"], turn["assistant"], persona)
            encoded = tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=config["training"]["max_seq_length"])
            input_ids = encoded["input_ids"].to(device)
            labels = input_ids.clone()

            result = spm.process_turn(input_ids, labels)
            session_losses.append(result["loss"])

            if result["consolidated"]:
                consolidation_count += 1

            if turn_idx % 5 == 0:
                logger.info(
                    "Session %d/%d Turn %d/%d | Loss: %.4f | Consolidated: %s | Buffer: %d",
                    session_idx + 1, len(sessions), turn_idx + 1, len(session["turns"]),
                    result["loss"], result["consolidated"],
                    len(spm.memory_buffer.inputs),
                )

        avg_loss = sum(session_losses) / max(len(session_losses), 1)

        # Periodic retention probe
        retention = None
        if (session_idx + 1) % args.probe_interval == 0 and probe_facts:
            retention = probe_retention(spm, tokenizer, probe_facts)
            forgetting_curve.append({
                "session": session_idx + 1,
                "retention": retention,
                "total_turns": spm.total_turns,
            })
            logger.info("  Retention probe: %.2f%% (%d facts)", retention * 100, len(probe_facts))

            # Add new probe facts from this session
            for turn in session["turns"][:2]:
                probe_facts.append({
                    "question": f"What did we discuss about: {turn['user'][:50]}?",
                    "answer": turn["assistant"][:30],
                    "taught_session": session_idx,
                })

        entry = {
            "session": session_idx,
            "avg_loss": avg_loss,
            "num_turns": len(session["turns"]),
            "consolidations": consolidation_count,
            "retention": retention,
            "buffer_size": len(spm.memory_buffer.inputs),
        }
        training_log.append(entry)

        if (session_idx + 1) % 10 == 0:
            logger.info(
                "=== Session %d/%d | Avg loss: %.4f | Consolidations: %d | Total turns: %d ===",
                session_idx + 1, len(sessions), avg_loss, consolidation_count, spm.total_turns,
            )

        if (session_idx + 1) % 50 == 0:
            ckpt_dir = os.path.join(output_dir, f"checkpoint-session-{session_idx + 1}")
            spm.save(ckpt_dir)

    # Save final model
    spm.save(os.path.join(output_dir, "final"))

    # Save logs
    with open(os.path.join(output_dir, "training_log.json"), "w") as f:
        json.dump(training_log, f, indent=2)

    with open(os.path.join(output_dir, "forgetting_curve.json"), "w") as f:
        json.dump(forgetting_curve, f, indent=2)

    # Plot forgetting curve
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if forgetting_curve:
            sessions_x = [p["session"] for p in forgetting_curve]
            retention_y = [p["retention"] for p in forgetting_curve]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            ax1.plot(sessions_x, retention_y, "b-o", label="Retention")
            ax1.set_xlabel("Session")
            ax1.set_ylabel("Retention Rate")
            ax1.set_title("Knowledge Retention Over Time")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            losses = [e["avg_loss"] for e in training_log]
            ax2.plot(range(len(losses)), losses, "r-", alpha=0.5, label="Session avg loss")
            window = min(10, len(losses))
            if window > 1:
                smoothed = [sum(losses[max(0, i - window):i + 1]) / min(i + 1, window)
                            for i in range(len(losses))]
                ax2.plot(range(len(smoothed)), smoothed, "r-", linewidth=2, label="Smoothed")
            ax2.set_xlabel("Session")
            ax2.set_ylabel("Loss")
            ax2.set_title("Training Loss")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "training_curves.pdf"), dpi=150, bbox_inches="tight")
            plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150, bbox_inches="tight")
            plt.close()
    except ImportError:
        pass

    logger.info("\n=== Training complete ===")
    logger.info("Total sessions: %d, Total turns: %d", len(sessions), spm.total_turns)
    logger.info("Total consolidations: %d", consolidation_count)
    if forgetting_curve:
        logger.info("Final retention: %.2f%%", forgetting_curve[-1]["retention"] * 100)
    logger.info("Saved to %s", output_dir)


if __name__ == "__main__":
    main()
