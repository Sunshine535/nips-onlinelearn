#!/usr/bin/env python3
"""Streaming Parameter Memory: working LoRA (per-turn) + long-term LoRA (consolidated).
Model: Qwen/Qwen3.5-9B. Fisher Information for importance scoring."""

import argparse
import json
import logging
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


def load_conversation_data(config: dict, tokenizer):
    """Load multi-turn conversation data for streaming training."""
    logger.info("Loading conversation data...")
    sessions = []

    try:
        ds = load_dataset("bavard/personachat_truecased", split="train", trust_remote_code=True)
        logger.info("Loaded PersonaChat: %d examples", len(ds))

        current_session = []
        for i, ex in enumerate(ds):
            if i >= 50000:
                break
            history = ex.get("history", [])
            candidates = ex.get("candidates", [])
            reply = candidates[-1] if candidates else ""

            if not history or not reply:
                continue

            turn = {"user": history[-1] if history else "", "assistant": reply}
            current_session.append(turn)

            if len(current_session) >= config["streaming"]["session_length"]:
                sessions.append(current_session)
                current_session = []

        if current_session:
            sessions.append(current_session)

    except Exception as e:
        logger.warning("Failed to load PersonaChat: %s. Generating synthetic data.", e)

    if len(sessions) < 10:
        logger.info("Generating synthetic conversation sessions...")
        for s in range(200):
            session = []
            for t in range(config["streaming"]["session_length"]):
                session.append({
                    "user": f"Tell me about topic {s}, aspect {t}.",
                    "assistant": f"Regarding topic {s}, aspect {t}: This is an interesting area "
                                 f"that connects to concept {s * 10 + t}. Let me elaborate further.",
                })
            sessions.append(session)

    logger.info("Loaded %d conversation sessions", len(sessions))
    return sessions


def format_turn(user_msg: str, assistant_msg: str, persona: str = "") -> str:
    """Format a conversation turn for model input."""
    system = persona or "You are a helpful and personalized assistant."
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_msg}<|im_end|>"
    )


def main():
    parser = argparse.ArgumentParser(description="Train Streaming Parameter Memory")
    parser.add_argument("--config", type=str, default="configs/spm_config.yaml")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_sessions", type=int, default=None)
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
    logger.info("Sessions: %d, Turns/session: %d", num_sessions, config["streaming"]["session_length"])

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        device_map={"": int(os.environ.get("LOCAL_RANK", 0))},
    )
    base_model.config.use_cache = False

    spm = StreamingParameterMemory(
        base_model=base_model,
        working_config=config["working_memory"],
        longterm_config=config["long_term_memory"],
        memory_buffer_size=config["long_term_memory"].get("max_memory_buffer", 5000),
    )

    sessions = load_conversation_data(config, tokenizer)
    sessions = sessions[:num_sessions]

    training_log = []
    device = next(spm.model.parameters()).device

    for session_idx, session in enumerate(sessions):
        spm.start_new_session()
        session_losses = []

        for turn_idx, turn in enumerate(session):
            text = format_turn(turn["user"], turn["assistant"])
            encoded = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=config["training"]["max_seq_length"])
            input_ids = encoded["input_ids"].to(device)
            labels = input_ids.clone()

            result = spm.process_turn(input_ids, labels)
            session_losses.append(result["loss"])

            if turn_idx % 5 == 0:
                logger.info(
                    "Session %d/%d Turn %d/%d | Loss: %.4f | Consolidated: %s",
                    session_idx + 1, len(sessions), turn_idx + 1, len(session),
                    result["loss"], result["consolidated"],
                )

        avg_loss = sum(session_losses) / max(len(session_losses), 1)
        training_log.append({
            "session": session_idx,
            "avg_loss": avg_loss,
            "num_turns": len(session),
            "consolidations": sum(1 for l in session_losses if l < 0.5),
        })

        if (session_idx + 1) % 10 == 0:
            logger.info(
                "=== Session %d/%d complete | Avg loss: %.4f | Total turns: %d ===",
                session_idx + 1, len(sessions), avg_loss, spm.total_turns,
            )

        if (session_idx + 1) % 50 == 0:
            ckpt_dir = os.path.join(output_dir, f"checkpoint-session-{session_idx + 1}")
            spm.save(ckpt_dir)

    # Save final
    spm.save(os.path.join(output_dir, "final"))

    log_path = os.path.join(output_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)

    logger.info("=== Training complete. Log saved to %s ===", log_path)
    logger.info("Total sessions: %d, Total turns: %d", len(sessions), spm.total_turns)


if __name__ == "__main__":
    main()
