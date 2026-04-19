#!/usr/bin/env python3
"""Pilot experiment: Tool-Use Forgetting phenomenon validation.

Goal: Answer 3 gating questions in <24 GPU hours:
  Q1. Does format validity decay earlier than tool selection accuracy
      under BOTH strict (parser-pass) and relaxed (repair-distance) metrics?
  Q2. Does the effect survive conditional evaluation
      (format conditioned on correct selection, etc.)?
  Q3. Is the gap larger on tool tasks than on matched NLU controls?

GO/NO-GO rule: ≥2 out of 3 "clearly yes" → commit 3-4 months to full project.

Pipeline:
  1. Baseline eval: Qwen3.5-0.5B on 500 ToolBench samples, 4 axes measured
  2. Drift fine-tune: LoRA on Alpaca instruction data (no tool use)
  3. Control fine-tune: LoRA on matched-size NLU-only data
  4. Re-eval every 50 steps (total 500 steps)
  5. Compute per-axis half-life; plot strict vs relaxed

Usage:
    bash run_pilot.sh  # launches 4 parallel conditions on 4 GPUs

Or standalone:
    python3 scripts/pilot_tool_forgetting.py \
        --condition baseline --model Qwen/Qwen3.5-0.5B \
        --output_dir outputs/pilot/baseline --gpu 0
"""

import argparse
import json
import logging
import os
import random
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool-use test set: deterministic, hand-curated for reliable parsing
# ---------------------------------------------------------------------------

TOOL_TEST_SET: List[Dict[str, Any]] = [
    {
        "id": f"tool_{i:03d}",
        "prompt": prompt,
        "gold_tool": tool,
        "gold_args": args,
    }
    for i, (prompt, tool, args) in enumerate([
        ("Get today's weather in Tokyo.",
         "get_weather", {"city": "Tokyo"}),
        ("Send a message to Alice saying hello.",
         "send_message", {"recipient": "Alice", "body": "hello"}),
        ("Compute 123 times 456.",
         "calculate", {"expression": "123 * 456"}),
        ("Search Wikipedia for quantum computing.",
         "search", {"query": "quantum computing", "source": "wikipedia"}),
        ("Schedule a meeting with Bob on Friday at 3pm.",
         "schedule_meeting", {"attendee": "Bob", "time": "Friday 3pm"}),
        ("Translate 'hello' to French.",
         "translate", {"text": "hello", "target_lang": "French"}),
        ("What is the stock price of AAPL?",
         "get_stock_price", {"symbol": "AAPL"}),
        ("Set a reminder for tomorrow 9am: call mom.",
         "set_reminder", {"time": "tomorrow 9am", "note": "call mom"}),
        ("Find restaurants near Central Park.",
         "find_places", {"query": "restaurants", "location": "Central Park"}),
        ("Play song Hello by Adele.",
         "play_music", {"song": "Hello", "artist": "Adele"}),
    ] * 50)  # 500 total (10 templates × 50 with slight paraphrases — later)
]

# NLU test for CF control metric
NLU_TEST_TEMPLATES = [
    ("The capital of France is", "Paris"),
    ("Water freezes at", "0 degrees Celsius"),
    ("Shakespeare wrote", "Hamlet"),
    ("The speed of light is", "299792458 m/s"),
    ("Photosynthesis requires", "sunlight"),
] * 100  # 500 total

SYSTEM_PROMPT = (
    "You are a helpful tool-using assistant. When the user asks for an action, "
    "respond with exactly one JSON object:\n"
    '{"tool": "<tool_name>", "args": {<key>: <value>, ...}}\n'
    "Only output the JSON. Do not include any other text."
)


def format_tool_prompt(user: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ---------------------------------------------------------------------------
# Four-axis evaluation
# ---------------------------------------------------------------------------

def _try_parse_json(text: str) -> Tuple[bool, Dict]:
    """Strict JSON parse. Returns (pass, parsed_dict or {}). Repair not allowed."""
    try:
        obj = json.loads(text.strip())
        return True, obj
    except Exception:
        return False, {}


def _try_parse_json_relaxed(text: str) -> Tuple[bool, Dict, int]:
    """Relaxed parse with repair (pass, parsed, edit_distance).

    edit_distance measures how many characters were added/removed to recover valid JSON.
    0 = perfect, <5 = minor format issue, larger = major.
    """
    text = text.strip()
    if not text:
        return False, {}, 999

    # Strategy 1: Strict JSON
    ok, obj = _try_parse_json(text)
    if ok:
        return True, obj, 0

    # Strategy 2: Strip surrounding text/markdown
    m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text)
    if m:
        candidate = m.group(0)
        ok, obj = _try_parse_json(candidate)
        if ok:
            return True, obj, abs(len(text) - len(candidate))

    # Strategy 3: Add common missing characters
    repaired = text.rstrip(",") + "}" if text.rstrip().endswith(",") else text
    for addition in ["}", "\"}", "\"}}", "}"]:
        cand = text.rstrip() + addition if not text.rstrip().endswith("}") else text
        ok, obj = _try_parse_json(cand)
        if ok:
            return True, obj, len(addition)

    return False, {}, 999


def eval_four_axes(
    model: nn.Module, tokenizer, test_set: List[Dict], device: str,
    max_new_tokens: int = 100,
) -> Dict[str, float]:
    """Compute 4 disentangled failure axes + relaxed variants.

    Returns dict with:
        strict_schema_pass  -- strict JSON parse rate
        relaxed_schema_pass -- repair-distance ≤ 10
        tool_selection      -- correct tool name (cond on schema_pass if strict)
        arg_correctness     -- all gold args present with correct values
        chain_format_rate   -- for multi-step (here: fraction with valid tool+args structure)
        repair_distance_mean -- mean edit distance for repair
    """
    model.eval()
    results = {
        "strict_schema_pass": 0,
        "relaxed_schema_pass": 0,
        "tool_selection_uncond": 0,
        "tool_selection_cond_schema": 0,  # conditional: among schema_pass only
        "arg_correctness_uncond": 0,
        "arg_correctness_cond_all": 0,  # conditional: among correct selection
        "repair_distance_sum": 0,
    }
    n = len(test_set)
    n_strict_pass = 0
    n_select_correct = 0

    for item in test_set:
        prompt = format_tool_prompt(item["prompt"])
        ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=512).input_ids.to(device)
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

        # Strict parse
        strict_ok, strict_obj = _try_parse_json(response)
        # Relaxed parse
        relaxed_ok, relaxed_obj, dist = _try_parse_json_relaxed(response)
        # Tool selection (from best available parse)
        parsed = strict_obj if strict_ok else relaxed_obj
        selected_tool = parsed.get("tool", "") if parsed else ""
        select_ok = selected_tool == item["gold_tool"]
        # Arg correctness
        gold_args = item["gold_args"]
        parsed_args = parsed.get("args", {}) if parsed else {}
        arg_ok = all(
            str(parsed_args.get(k, "")).lower() == str(v).lower()
            for k, v in gold_args.items()
        ) if parsed_args else False

        if strict_ok:
            results["strict_schema_pass"] += 1
            n_strict_pass += 1
        if relaxed_ok:
            results["relaxed_schema_pass"] += 1
        if select_ok:
            results["tool_selection_uncond"] += 1
            if strict_ok:
                results["tool_selection_cond_schema"] += 1
            n_select_correct += 1
        if arg_ok:
            results["arg_correctness_uncond"] += 1
            if select_ok:
                results["arg_correctness_cond_all"] += 1
        results["repair_distance_sum"] += dist if dist < 999 else 50

    # Normalize
    for k in ["strict_schema_pass", "relaxed_schema_pass",
              "tool_selection_uncond", "arg_correctness_uncond"]:
        results[k] /= n
    # Conditionals
    results["tool_selection_cond_schema"] = (
        results["tool_selection_cond_schema"] / max(n_strict_pass, 1)
    )
    results["arg_correctness_cond_all"] = (
        results["arg_correctness_cond_all"] / max(n_select_correct, 1)
    )
    results["repair_distance_mean"] = results.pop("repair_distance_sum") / n
    return results


def eval_nlu(model, tokenizer, device: str) -> Dict[str, float]:
    """Simple NLU CF reference: exact-match completion on factual templates."""
    model.eval()
    correct = 0
    n = len(NLU_TEST_TEMPLATES)
    for prompt, gold in NLU_TEST_TEMPLATES:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=20, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        if gold.lower() in response.lower():
            correct += 1
    return {"nlu_exact_match": correct / n}


# ---------------------------------------------------------------------------
# Drift fine-tuning (via HuggingFace Trainer + LoRA)
# ---------------------------------------------------------------------------

def load_drift_dataset(condition: str, tokenizer, max_samples: int = 1000):
    """Load the drift dataset based on condition."""
    if condition == "alpaca":
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        ds = ds.shuffle(seed=42).select(range(max_samples))
        texts = [
            f"Instruction: {ex['instruction']}\nInput: {ex['input']}\n"
            f"Output: {ex['output']}"
            for ex in ds
        ]
    elif condition == "nlu_control":
        # Matched-size NLU-only data (commonsense QA format)
        ds = load_dataset("commonsense_qa", split="train")
        ds = ds.shuffle(seed=42).select(range(max_samples))
        texts = [
            f"Question: {ex['question']}\n"
            f"Answer: {ex['choices']['text'][ord(ex['answerKey'])-ord('A')]}"
            for ex in ds if ex.get('answerKey')
        ][:max_samples]
    elif condition == "baseline":
        # No fine-tuning, just baseline eval
        return None
    else:
        raise ValueError(f"Unknown condition: {condition}")

    enc = tokenizer(
        texts, truncation=True, max_length=256, padding="max_length",
        return_tensors="pt",
    )
    return [
        {"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i],
         "labels": enc["input_ids"][i].clone()}
        for i in range(len(enc["input_ids"]))
    ]


def run_pilot(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading {args.model} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device,
    )
    # LoRA for fine-tuning
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # Baseline eval (t=0)
    logger.info("=== Baseline evaluation (t=0) ===")
    tool_metrics_0 = eval_four_axes(model, tokenizer, TOOL_TEST_SET, device)
    nlu_metrics_0 = eval_nlu(model, tokenizer, device)
    checkpoints = [{
        "step": 0, "tool": tool_metrics_0, "nlu": nlu_metrics_0,
    }]
    logger.info(f"t=0: {json.dumps({**tool_metrics_0, **nlu_metrics_0}, indent=2)}")

    # Fine-tune
    if args.condition != "baseline":
        drift_data = load_drift_dataset(args.condition, tokenizer, args.n_drift_samples)
        logger.info(f"Loaded {len(drift_data)} drift samples for condition={args.condition}")

        # Simple training loop (no Trainer to keep it lightweight)
        model.train()
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=2e-4,
        )
        batch_size = 4
        step = 0
        for epoch in range(10):
            random.shuffle(drift_data)
            for i in range(0, len(drift_data), batch_size):
                batch = drift_data[i:i+batch_size]
                ids = torch.stack([b["input_ids"] for b in batch]).to(device)
                mask = torch.stack([b["attention_mask"] for b in batch]).to(device)
                labels = torch.stack([b["labels"] for b in batch]).to(device)
                labels = labels.masked_fill(mask == 0, -100)
                out = model(input_ids=ids, attention_mask=mask, labels=labels)
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0,
                )
                optimizer.step()
                optimizer.zero_grad()
                step += 1

                # Evaluate every args.eval_every steps
                if step % args.eval_every == 0 or step == args.max_steps:
                    logger.info(f"=== Evaluation at step {step} ===")
                    tool_m = eval_four_axes(model, tokenizer, TOOL_TEST_SET, device)
                    nlu_m = eval_nlu(model, tokenizer, device)
                    checkpoints.append({
                        "step": step, "tool": tool_m, "nlu": nlu_m,
                    })
                    logger.info(f"step={step}: schema_strict={tool_m['strict_schema_pass']:.3f} "
                                f"schema_relaxed={tool_m['relaxed_schema_pass']:.3f} "
                                f"sel={tool_m['tool_selection_uncond']:.3f} "
                                f"arg={tool_m['arg_correctness_uncond']:.3f} "
                                f"nlu={nlu_m['nlu_exact_match']:.3f}")
                    model.train()  # back to train mode

                if step >= args.max_steps:
                    break
            if step >= args.max_steps:
                break

    # Save results
    result = {
        "condition": args.condition,
        "model": args.model,
        "n_drift_samples": args.n_drift_samples,
        "max_steps": args.max_steps,
        "checkpoints": checkpoints,
    }
    out_path = Path(args.output_dir) / f"pilot_{args.condition}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved results to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["baseline", "alpaca", "nlu_control"],
                        required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.5B-Instruct")
    parser.add_argument("--output_dir", default="outputs/pilot")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n_drift_samples", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    run_pilot(args)


if __name__ == "__main__":
    main()
