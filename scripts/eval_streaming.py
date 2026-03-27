#!/usr/bin/env python3
"""Streaming evaluation on PersonaChat + LIGHT.
Simulate 50 sessions. Compare: no adaptation, full fine-tune, EWC, SPM (ours).
Metrics: persona consistency, knowledge retention@N, response perplexity, BLEU."""

import argparse
import json
import logging
import math
import os
import sys
from collections import Counter

import torch
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.streaming_memory import StreamingParameterMemory


# ── Metrics ─────────────────────────────────────────────────────────────────


def compute_perplexity(model, tokenizer, texts: list, max_length: int = 1024) -> float:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for text in texts[:100]:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        ids = enc["input_ids"].to(next(model.parameters()).device)
        with torch.no_grad():
            out = model(input_ids=ids, labels=ids.clone())
            total_loss += out.loss.item() * ids.shape[1]
            total_tokens += ids.shape[1]
    return math.exp(total_loss / max(total_tokens, 1))


def compute_bleu(predictions: list, references: list) -> float:
    def ngrams(tokens, n):
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    total = 0.0
    for pred, ref in zip(predictions, references):
        pred_tok = pred.lower().split()
        ref_tok = ref.lower().split()
        if not pred_tok:
            continue
        scores = []
        for n in range(1, 5):
            pred_ng = Counter(ngrams(pred_tok, n))
            ref_ng = Counter(ngrams(ref_tok, n))
            matched = sum(min(pred_ng[ng], ref_ng[ng]) for ng in pred_ng)
            total_ng = max(sum(pred_ng.values()), 1)
            scores.append(matched / total_ng)
        bp = min(1.0, math.exp(1 - len(ref_tok) / max(len(pred_tok), 1)))
        bleu = bp * math.exp(sum(math.log(max(s, 1e-10)) for s in scores) / 4)
        total += bleu
    return total / max(len(predictions), 1)


def compute_persona_consistency(model, tokenizer, persona_facts: list,
                                probe_questions: list, device) -> float:
    """Measure how well the model retains persona-specific information."""
    model.eval()
    prev_cache = getattr(model.config, "use_cache", True)
    model.config.use_cache = True
    correct = 0
    try:
        for fact, question in zip(persona_facts, probe_questions):
            prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                out = model.generate(input_ids=ids, max_new_tokens=100, do_sample=False)
            resp = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).lower()
            if fact.lower() in resp:
                correct += 1
    finally:
        model.config.use_cache = prev_cache
    return correct / max(len(persona_facts), 1)


def knowledge_retention_at_n(retention_scores: list, n: int) -> float:
    """Retention@N: average retention after N new sessions."""
    if len(retention_scores) <= n:
        return retention_scores[-1] if retention_scores else 0.0
    return retention_scores[n]


# ── Methods ─────────────────────────────────────────────────────────────────


def format_turn(user_msg: str, assistant_msg: str, persona: str = "") -> str:
    system = persona if persona else "You are a helpful and personalized assistant."
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_msg}<|im_end|>"
    )


class NoAdaptationBaseline:
    """Baseline: frozen model, no adaptation."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

    def process_turn(self, input_ids, labels):
        return {"loss": 0.0, "consolidated": False}

    def start_session(self):
        pass

    def generate(self, input_ids, max_new_tokens=200):
        prev_cache = getattr(self.model.config, "use_cache", True)
        self.model.config.use_cache = True
        try:
            with torch.no_grad():
                return self.model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=False)
        finally:
            self.model.config.use_cache = prev_cache


class FullFineTuneMethod:
    """LoRA fine-tuning on every turn (no catastrophic forgetting protection)."""

    def __init__(self, model, tokenizer, lr=5e-4):
        lora_cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                              task_type=TaskType.CAUSAL_LM, bias="none")
        self.model = get_peft_model(model, lora_cfg)
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=lr)

    def process_turn(self, input_ids, labels):
        self.model.train()
        outputs = self.model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return {"loss": loss.item(), "consolidated": False}

    def start_session(self):
        pass

    def generate(self, input_ids, max_new_tokens=200):
        self.model.eval()
        prev_cache = getattr(self.model.config, "use_cache", True)
        self.model.config.use_cache = True
        try:
            with torch.no_grad():
                return self.model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=False)
        finally:
            self.model.config.use_cache = prev_cache


class EWCMethod:
    """Elastic Weight Consolidation: single LoRA with EWC penalty."""

    def __init__(self, model, tokenizer, lr=5e-4, ewc_lambda=5000.0):
        lora_cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                              task_type=TaskType.CAUSAL_LM, bias="none")
        self.model = get_peft_model(model, lora_cfg)
        self.tokenizer = tokenizer
        self.lr = lr
        self.ewc_lambda = ewc_lambda
        self.fisher = {}
        self.prev_params = {}
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=lr)

    def _compute_fisher(self, input_ids, labels):
        self.model.train()
        self.model.zero_grad()
        with torch.enable_grad():
            out = self.model(input_ids=input_ids, labels=labels)
            out.loss.backward()
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if name not in self.fisher:
                    self.fisher[name] = param.grad.data.pow(2).clone()
                else:
                    self.fisher[name] = 0.9 * self.fisher[name] + 0.1 * param.grad.data.pow(2)
        self.model.zero_grad()

    def _ewc_loss(self):
        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        for name, param in self.model.named_parameters():
            if name in self.fisher and name in self.prev_params:
                loss += (self.fisher[name] * (param - self.prev_params[name]).pow(2)).sum()
        return self.ewc_lambda * loss

    def process_turn(self, input_ids, labels):
        self.model.train()
        out = self.model(input_ids=input_ids, labels=labels)
        ewc = self._ewc_loss()
        loss = out.loss + ewc
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self._compute_fisher(input_ids, labels)
        return {"loss": out.loss.item(), "consolidated": False}

    def start_session(self):
        self.prev_params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

    def generate(self, input_ids, max_new_tokens=200):
        self.model.eval()
        prev_cache = getattr(self.model.config, "use_cache", True)
        self.model.config.use_cache = True
        try:
            with torch.no_grad():
                return self.model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=False)
        finally:
            self.model.config.use_cache = prev_cache


class SPMMethod:
    """Our Streaming Parameter Memory method."""

    def __init__(self, model, tokenizer, config):
        self.tokenizer = tokenizer
        self.spm = StreamingParameterMemory(
            base_model=model,
            working_config=config["working_memory"],
            longterm_config=config["long_term_memory"],
            memory_buffer_size=config["long_term_memory"].get("max_memory_buffer", 5000),
        )
        self.model = self.spm.model

    def process_turn(self, input_ids, labels):
        return self.spm.process_turn(input_ids, labels)

    def start_session(self):
        self.spm.start_new_session()

    def generate(self, input_ids, max_new_tokens=200):
        return self.spm.generate(input_ids, max_new_tokens=max_new_tokens, use_longterm=True)


# ── Evaluation ──────────────────────────────────────────────────────────────


def load_eval_sessions(config: dict, dataset_name: str, max_sessions: int = 50):
    """Load evaluation sessions from PersonaChat or LIGHT."""
    sessions = []
    session_len = config["streaming"]["session_length"]

    if dataset_name == "personachat":
        try:
            ds = load_dataset("bavard/personachat_truecased", split="validation")
            current = []
            for ex in ds:
                history = ex.get("history", [])
                candidates = ex.get("candidates", [])
                personality = ex.get("personality", [])
                reply = candidates[-1] if candidates else ""
                if history and reply:
                    current.append({
                        "user": history[-1], "assistant": reply,
                        "persona": " ".join(personality) if personality else "",
                    })
                if len(current) >= session_len:
                    sessions.append(current)
                    current = []
                if len(sessions) >= max_sessions:
                    break
        except Exception as e:
            logger.warning("Failed to load PersonaChat val: %s", e)

    elif dataset_name == "light":
        try:
            ds = load_dataset("light_dialog", split="test")
            for ex in ds:
                dialog = ex.get("dialog", [])
                if len(dialog) >= 4:
                    turns = []
                    for k in range(0, len(dialog) - 1, 2):
                        turns.append({"user": str(dialog[k]), "assistant": str(dialog[k + 1]), "persona": ""})
                    if turns:
                        sessions.append(turns[:session_len])
                if len(sessions) >= max_sessions:
                    break
        except Exception as e:
            logger.warning("Failed to load LIGHT: %s", e)

    if len(sessions) < 10:
        for s in range(max_sessions):
            turns = [{"user": f"Session {s} turn {t}", "assistant": f"Response to session {s} turn {t}",
                       "persona": f"I am test persona {s}"} for t in range(session_len)]
            sessions.append(turns)

    return sessions[:max_sessions]


def evaluate_method(method, method_name: str, tokenizer, sessions: list, device):
    """Run streaming evaluation for one method."""
    logger.info("  Evaluating: %s (%d sessions)", method_name, len(sessions))

    all_losses = []
    predictions, references = [], []
    session_retention = []
    taught_facts = []

    for sess_idx, session in enumerate(sessions):
        method.start_session()
        sess_losses = []

        for turn in session:
            text = format_turn(turn["user"], turn["assistant"], turn.get("persona", ""))
            encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
            input_ids = encoded["input_ids"].to(device)
            labels = input_ids.clone()

            result = method.process_turn(input_ids, labels)
            sess_losses.append(result["loss"])

            taught_facts.append({"user": turn["user"][:50], "assistant": turn["assistant"][:50]})

        all_losses.extend(sess_losses)

        # Generate responses for last 3 turns
        for turn in session[-3:]:
            prompt = (f"<|im_start|>user\n{turn['user']}<|im_end|>\n"
                      f"<|im_start|>assistant\n")
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            try:
                out = method.generate(ids, max_new_tokens=100)
                pred = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            except Exception as e:
                logger.warning("generate failed (%s): %s", method_name, e)
                pred = ""
            predictions.append(pred)
            references.append(turn["assistant"])

        # Retention probe
        if taught_facts and sess_idx % 5 == 0:
            correct = 0
            for fact in taught_facts[-10:]:
                q_prompt = (f"<|im_start|>user\nRemind me: {fact['user']}<|im_end|>\n"
                            f"<|im_start|>assistant\n")
                ids = tokenizer(q_prompt, return_tensors="pt").input_ids.to(device)
                try:
                    out = method.generate(ids, max_new_tokens=50)
                    resp = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).lower()
                    if any(w in resp for w in fact["assistant"].lower().split()[:3]):
                        correct += 1
                except Exception as e:
                    logger.warning("retention probe generate failed: %s", e)
            retention = correct / max(len(taught_facts[-10:]), 1)
            session_retention.append({"session": sess_idx, "retention": retention})

        if sess_idx % 10 == 0:
            avg_l = sum(sess_losses) / max(len(sess_losses), 1)
            logger.info("    Session %d/%d | Avg loss: %.4f", sess_idx + 1, len(sessions), avg_l)

    # Compute metrics
    eval_texts = [format_turn(t["user"], t["assistant"]) for s in sessions[-5:] for t in s[:3]]
    ppl = compute_perplexity(method.model, tokenizer, eval_texts) if eval_texts else float("inf")
    bleu = compute_bleu(predictions, references)

    retention_scores = [r["retention"] for r in session_retention] if session_retention else [0.0]

    results = {
        "perplexity": ppl,
        "bleu": bleu,
        "avg_loss": sum(all_losses) / max(len(all_losses), 1) if all_losses else 0.0,
        "retention@10": knowledge_retention_at_n(retention_scores, min(10, len(retention_scores) - 1)),
        "retention@30": knowledge_retention_at_n(retention_scores, min(30, len(retention_scores) - 1)),
        "final_retention": retention_scores[-1] if retention_scores else 0.0,
        "retention_curve": session_retention,
        "num_sessions": len(sessions),
    }

    logger.info("    Results: PPL=%.2f BLEU=%.4f Retention=%.4f",
                results["perplexity"], results["bleu"], results["final_retention"])
    return results


def main():
    parser = argparse.ArgumentParser(description="Streaming evaluation: 4 methods comparison")
    parser.add_argument("--config", type=str, default="configs/spm_config.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs/streaming_eval")
    parser.add_argument("--num_sessions", type=int, default=50)
    parser.add_argument("--methods", nargs="+",
                        default=["no_adapt", "full_ft", "ewc", "spm"],
                        choices=["no_adapt", "full_ft", "ewc", "spm"])
    parser.add_argument("--datasets", nargs="+", default=["personachat", "light"],
                        choices=["personachat", "light"])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    base_model_name = config["model"]["base_model"]

    all_results = {}
    for ds_name in args.datasets:
        logger.info("\n" + "=" * 60)
        logger.info("DATASET: %s", ds_name)
        logger.info("=" * 60)

        sessions = load_eval_sessions(config, ds_name, args.num_sessions)
        ds_results = {}

        for method_name in args.methods:
            logger.info("\n>>> Method: %s on %s", method_name, ds_name)

            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            device_idx = int(os.environ.get("LOCAL_RANK", os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]))
            base_model = None
            for attn_impl in ["flash_attention_2", "sdpa", "eager"]:
                try:
                    base_model = AutoModelForCausalLM.from_pretrained(
                        base_model_name, torch_dtype=torch.bfloat16,
                        attn_implementation=attn_impl,
                        device_map={"": device_idx},
                    )
                    break
                except (ImportError, ValueError):
                    pass
            if base_model is None:
                raise RuntimeError(f"Cannot load model {base_model_name}")
            base_model.config.use_cache = False

            if method_name == "no_adapt":
                method = NoAdaptationBaseline(base_model, tokenizer)
            elif method_name == "full_ft":
                method = FullFineTuneMethod(base_model, tokenizer, lr=5e-5)
            elif method_name == "ewc":
                method = EWCMethod(base_model, tokenizer, ewc_lambda=5000.0)
            elif method_name == "spm":
                method = SPMMethod(base_model, tokenizer, config)
            else:
                continue

            device = next(method.model.parameters()).device
            results = evaluate_method(method, method_name, tokenizer, sessions, device)
            ds_results[method_name] = results

            del method, base_model
            torch.cuda.empty_cache()

        all_results[ds_name] = ds_results

    # Save results
    with open(os.path.join(args.output_dir, "streaming_eval_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)
    header = f"{'Dataset':<15} {'Method':<12} {'PPL':>10} {'BLEU':>10} {'Ret@10':>10} {'Ret@30':>10} {'AvgLoss':>10}"
    logger.info(header)
    logger.info("-" * 80)
    for ds_name, ds_res in all_results.items():
        for method, metrics in ds_res.items():
            logger.info(f"{ds_name:<15} {method:<12} "
                        f"{metrics['perplexity']:>10.2f} {metrics['bleu']:>10.4f} "
                        f"{metrics['retention@10']:>10.4f} {metrics['retention@30']:>10.4f} "
                        f"{metrics['avg_loss']:>10.4f}")

    # Save LaTeX table
    latex = ["\\begin{table}[h]", "\\centering",
             "\\caption{Streaming Evaluation Results}",
             "\\begin{tabular}{llccccc}", "\\toprule",
             "Dataset & Method & PPL $\\downarrow$ & BLEU $\\uparrow$ & Ret@10 $\\uparrow$ & Ret@30 $\\uparrow$ & Loss $\\downarrow$ \\\\",
             "\\midrule"]
    for ds_name, ds_res in all_results.items():
        for method, m in ds_res.items():
            latex.append(f"{ds_name} & {method} & {m['perplexity']:.2f} & {m['bleu']:.4f} & "
                         f"{m['retention@10']:.4f} & {m['retention@30']:.4f} & {m['avg_loss']:.4f} \\\\")
    latex.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    with open(os.path.join(args.output_dir, "results_table.tex"), "w") as f:
        f.write("\n".join(latex))

    # Plot forgetting curves
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for ds_name, ds_res in all_results.items():
            fig, ax = plt.subplots(figsize=(10, 6))
            for method, metrics in ds_res.items():
                curve = metrics.get("retention_curve", [])
                if curve:
                    sessions_x = [c["session"] for c in curve]
                    rets = [c["retention"] for c in curve]
                    ax.plot(sessions_x, rets, "-o", label=method, markersize=4)
            ax.set_xlabel("Session")
            ax.set_ylabel("Knowledge Retention")
            ax.set_title(f"Forgetting Curve - {ds_name}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, f"forgetting_curve_{ds_name}.pdf"),
                        dpi=150, bbox_inches="tight")
            plt.savefig(os.path.join(args.output_dir, f"forgetting_curve_{ds_name}.png"),
                        dpi=150, bbox_inches="tight")
            plt.close()
    except ImportError:
        pass

    logger.info("All results saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
