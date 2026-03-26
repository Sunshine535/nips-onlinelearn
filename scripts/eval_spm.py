#!/usr/bin/env python3
"""Evaluate SPM on multi-turn personalization: PersonaChat/LIGHT.
Measure cross-session memory retention."""

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


def compute_perplexity(model, tokenizer, texts: list, max_length: int = 2048) -> float:
    """Compute perplexity on a list of texts."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for text in texts:
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = encoded["input_ids"].to(next(model.parameters()).device)
        labels = input_ids.clone()

        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels)
            total_loss += outputs.loss.item() * input_ids.shape[1]
            total_tokens += input_ids.shape[1]

    return math.exp(total_loss / max(total_tokens, 1))


def compute_bleu(predictions: list, references: list) -> float:
    """Simple BLEU-4 computation."""
    from collections import Counter

    def ngrams(tokens, n):
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    total_score = 0.0
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        if not pred_tokens:
            continue
        scores = []
        for n in range(1, 5):
            pred_ng = Counter(ngrams(pred_tokens, n))
            ref_ng = Counter(ngrams(ref_tokens, n))
            matched = sum(min(pred_ng[ng], ref_ng[ng]) for ng in pred_ng)
            total_ng = max(sum(pred_ng.values()), 1)
            scores.append(matched / total_ng)
        bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(pred_tokens), 1)))
        bleu = bp * math.exp(sum(math.log(max(s, 1e-10)) for s in scores) / 4)
        total_score += bleu

    return total_score / max(len(predictions), 1)


def evaluate_memory_retention(spm: StreamingParameterMemory, tokenizer, probe_facts: list) -> dict:
    """Test if the model retains facts from previous sessions."""
    device = next(spm.model.parameters()).device
    correct = 0
    total = len(probe_facts)

    for fact in probe_facts:
        question = fact["question"]
        expected = fact["answer"].lower()

        prompt = (
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        try:
            output = spm.generate(input_ids, max_new_tokens=100, use_longterm=True)
            response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).lower()
            if expected in response:
                correct += 1
        except Exception as e:
            logger.warning("evaluate_memory_retention generate failed: %s", e)

    spm.model.set_adapter("working")
    return {"accuracy": correct / max(total, 1), "correct": correct, "total": total}


def load_eval_data(config: dict):
    """Load evaluation datasets."""
    all_data = {}
    for ds_cfg in config["evaluation"]["datasets"]:
        name = ds_cfg["name"]
        try:
            ds = load_dataset(ds_cfg["dataset_id"], split=ds_cfg["split"])
            max_s = ds_cfg.get("max_samples", 1000)
            if len(ds) > max_s:
                ds = ds.shuffle(seed=42).select(range(max_s))
            all_data[name] = ds
            logger.info("Loaded %s: %d samples", name, len(ds))
        except Exception as e:
            logger.warning("Failed to load %s: %s", name, e)
    return all_data


def main():
    parser = argparse.ArgumentParser(description="Evaluate Streaming Parameter Memory")
    parser.add_argument("--config", type=str, default="configs/spm_config.yaml")
    parser.add_argument("--model_dir", type=str, default="outputs/spm_training/final")
    parser.add_argument("--output_dir", type=str, default="outputs/eval")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    base_model_name = config["model"]["base_model"]

    logger.info("=== SPM Evaluation ===")
    logger.info("Loading base model: %s", base_model_name)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_idx = int(os.environ.get("LOCAL_RANK", 0))
    base_model = None
    for attn_impl in ["flash_attention_2", "sdpa", "eager"]:
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_impl,
                device_map={"": device_idx},
            )
            logger.info("Attention implementation: %s", attn_impl)
            break
        except (ImportError, ValueError) as e:
            logger.warning("attn_implementation=%s failed: %s", attn_impl, e)
    if base_model is None:
        raise RuntimeError(f"Cannot load model {base_model_name}")

    spm = StreamingParameterMemory(
        base_model=base_model,
        working_config=config["working_memory"],
        longterm_config=config["long_term_memory"],
    )

    adapter_path = os.path.join(args.model_dir, "adapters")
    if os.path.exists(adapter_path):
        logger.info("Loading saved adapters from %s", adapter_path)
        from peft import set_peft_model_state_dict
        for adapter_name in ["working", "longterm"]:
            adapter_sub = os.path.join(adapter_path, adapter_name)
            if not os.path.isdir(adapter_sub):
                continue
            try:
                safetensors_file = os.path.join(adapter_sub, "adapter_model.safetensors")
                bin_file = os.path.join(adapter_sub, "adapter_model.bin")
                if os.path.exists(safetensors_file):
                    from safetensors.torch import load_file
                    state_dict = load_file(safetensors_file)
                elif os.path.exists(bin_file):
                    state_dict = torch.load(bin_file, map_location="cpu", weights_only=True)
                else:
                    logger.warning("  No weights found for adapter '%s'", adapter_name)
                    continue
                set_peft_model_state_dict(spm.model, state_dict, adapter_name=adapter_name)
                logger.info("  Loaded adapter '%s' from %s", adapter_name, adapter_sub)
            except Exception as e:
                logger.warning("  Failed to load adapter '%s': %s", adapter_name, e)
        spm.model.set_adapter("working")
        logger.info("Adapter loading complete")
    else:
        logger.warning("No saved model found at %s, evaluating initialized model", args.model_dir)

    eval_data = load_eval_data(config)
    all_results = {}

    # 1. Perplexity evaluation
    logger.info("=== Perplexity Evaluation ===")
    for name, ds in eval_data.items():
        texts = []
        for ex in ds:
            history = ex.get("history", ex.get("dialog", []))
            if isinstance(history, list) and history:
                texts.append(" ".join(str(h) for h in history[-3:]))
            elif isinstance(history, str):
                texts.append(history)
        texts = [t for t in texts if len(t) > 10][:200]
        if texts:
            ppl = compute_perplexity(spm.model, tokenizer, texts)
            all_results[f"{name}_perplexity"] = ppl
            logger.info("  %s perplexity: %.2f", name, ppl)

    # 2. Online learning simulation
    logger.info("=== Online Learning Simulation ===")
    teaching_facts = [
        {"fact": "My favorite color is blue.", "question": "What is my favorite color?", "answer": "blue"},
        {"fact": "I live in Tokyo.", "question": "Where do I live?", "answer": "tokyo"},
        {"fact": "My hobby is painting.", "question": "What is my hobby?", "answer": "painting"},
        {"fact": "I have two cats.", "question": "How many cats do I have?", "answer": "two"},
        {"fact": "I work as an engineer.", "question": "What do I do for work?", "answer": "engineer"},
    ]

    device = next(spm.model.parameters()).device
    spm.start_new_session()

    for fact_info in teaching_facts:
        text = (
            f"<|im_start|>user\nRemember this: {fact_info['fact']}<|im_end|>\n"
            f"<|im_start|>assistant\nI'll remember that. {fact_info['fact']}<|im_end|>"
        )
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = encoded["input_ids"].to(device)
        labels = input_ids.clone()
        result = spm.process_turn(input_ids, labels)
        logger.info("  Taught: '%s' (loss: %.4f)", fact_info["fact"], result["loss"])

    # 3. Immediate retention test
    logger.info("=== Immediate Retention Test ===")
    immediate = evaluate_memory_retention(spm, tokenizer, teaching_facts)
    all_results["immediate_retention"] = immediate
    logger.info("  Immediate retention: %.2f%% (%d/%d)", immediate["accuracy"] * 100,
               immediate["correct"], immediate["total"])

    # 4. Cross-session retention test
    logger.info("=== Cross-Session Retention Test ===")
    spm.start_new_session()
    for _ in range(5):
        filler = "<|im_start|>user\nTell me a joke.<|im_end|>\n<|im_start|>assistant\nHere's a joke for you!<|im_end|>"
        encoded = tokenizer(filler, return_tensors="pt", truncation=True, max_length=256)
        input_ids = encoded["input_ids"].to(device)
        spm.process_turn(input_ids, input_ids.clone())

    cross_session = evaluate_memory_retention(spm, tokenizer, teaching_facts)
    all_results["cross_session_retention"] = cross_session
    logger.info("  Cross-session retention: %.2f%% (%d/%d)", cross_session["accuracy"] * 100,
               cross_session["correct"], cross_session["total"])

    # 5. Generation quality
    logger.info("=== Generation Quality ===")
    test_prompts = [
        "What have we discussed in our previous conversations?",
        "Can you tell me something about myself?",
        "What are my preferences?",
    ]
    generation_results = []
    for prompt in test_prompts:
        input_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        try:
            output = spm.generate(input_ids, max_new_tokens=200, use_longterm=True)
            response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
        except Exception as e:
            logger.warning("Generation failed: %s", e)
            response = "[generation failed]"
        generation_results.append({"prompt": prompt, "response": response})
        logger.info("  Q: %s", prompt)
        logger.info("  A: %s", response[:200])
    all_results["generation_samples"] = generation_results

    # Save results
    output_path = os.path.join(args.output_dir, "spm_eval_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    logger.info("\n=== EVALUATION SUMMARY ===")
    for key, val in all_results.items():
        if isinstance(val, (int, float)):
            logger.info("  %s: %.4f", key, val)
        elif isinstance(val, dict) and "accuracy" in val:
            logger.info("  %s: %.2f%%", key, val["accuracy"] * 100)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
