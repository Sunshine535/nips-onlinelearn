#!/usr/bin/env python3
"""Unified streaming evaluation: 13 methods x {dialogue, classification}.

Methods:
  frozen, single_lora_sgd, single_lora_ewc, single_lora_replay,
  single_lora_ewc_replay, param_matched_lora, dual_lora_ema,
  dual_lora_periodic_avg, dual_lora_heuristic, retrieval_augmented,
  sdft, spm, mirror_lora

Tasks:
  dialogue     — PersonaChat persona-grouped sessions (E6)
  classification — AG News with concept drift (E7)

Metrics:
  Dialogue: Semantic Retention F1, Adaptation Speed, Forgetting Rate, PPL
  Classification: Rolling accuracy, Forgetting, Forward transfer, Dynamic regret
"""

import argparse
import copy
import json
import logging
import math
import os
import random
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as TF
import yaml
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.streaming_memory import StreamingParameterMemory
from src.streaming_mirror_lora import StreamingMirrorLoRA


# ---------------------------------------------------------------------------
# PEFT dual-adapter workaround: set_adapter(["a","b"]) fails on
# ModulesToSaveWrapper layers. This function sets adapters at the LoRA
# layer level only, bypassing that limitation.
# ---------------------------------------------------------------------------
def _set_adapters_layerwise(model, adapter_names):
    """Activate multiple LoRA adapters simultaneously at the layer level.

    PEFT's model.set_adapter(["a", "b"]) crashes on ModulesToSaveWrapper.
    This workaround iterates over LoRA linear layers directly and calls
    their set_adapter method, which *does* support multiple adapters.
    """
    from peft.tuners.tuners_utils import BaseTunerLayer

    if isinstance(adapter_names, str):
        adapter_names = [adapter_names]
    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            if hasattr(module, "merged") and module.merged:
                module.unmerge()
            module.set_adapter(adapter_names)


def _cast_lora_params_to_fp32(model):
    """Cast LoRA adapter parameters to float32.

    When the base model is loaded in fp16/bf16, PEFT creates LoRA params in
    the same dtype. But zero-initialized lora_B in fp16 causes vanishing
    gradients due to underflow. Casting to fp32 fixes this.
    """
    for name, param in model.named_parameters():
        if "lora" in name and param.requires_grad:
            param.data = param.data.to(torch.float32)

def _reset_to_primary_adapter(method):
    """Reset model to its primary adapter before inference.

    Multi-adapter methods (mirror_lora, dual_lora_*, sdft) leave multiple
    adapters active after process_turn(). This causes shape mismatches or
    incorrect outputs during classification evaluation. This function resets
    to the single primary adapter.
    """
    model = getattr(method, "model", None)
    if model is None:
        return
    try:
        if hasattr(model, "peft_config"):
            adapters = list(model.peft_config.keys())
            if "working" in adapters:
                _set_adapters_layerwise(model, ["working"])
            elif "fast" in adapters:
                _set_adapters_layerwise(model, ["fast"])
            elif len(adapters) == 1:
                _set_adapters_layerwise(model, [adapters[0]])
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Optional NLI model for semantic retention
# ---------------------------------------------------------------------------
_NLI_MODEL = None
_NLI_TOKENIZER = None


def _load_nli():
    global _NLI_MODEL, _NLI_TOKENIZER
    if _NLI_MODEL is not None:
        return True
    try:
        from transformers import AutoModelForSequenceClassification

        nli_name = "cross-encoder/nli-deberta-v3-base"
        _NLI_TOKENIZER = AutoTokenizer.from_pretrained(nli_name)
        _NLI_MODEL = AutoModelForSequenceClassification.from_pretrained(nli_name)
        _NLI_MODEL.eval()
        if torch.cuda.is_available():
            _NLI_MODEL = _NLI_MODEL.to("cuda:0")
        logger.info("Loaded NLI model: %s", nli_name)
        return True
    except Exception as exc:
        logger.warning("NLI model unavailable (%s); falling back to ROUGE-L proxy", exc)
        return False


# ===================================================================
# Metrics
# ===================================================================


def _rouge_l_f1(reference: str, hypothesis: str) -> float:
    """ROUGE-L F1 via LCS."""
    ref_tok = reference.lower().split()
    hyp_tok = hypothesis.lower().split()
    if not ref_tok or not hyp_tok:
        return 0.0
    m, n = len(ref_tok), len(hyp_tok)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tok[i - 1] == hyp_tok[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    prec = lcs / n
    rec = lcs / m
    return 2 * prec * rec / (prec + rec)


def _nli_entailment_score(premise: str, hypothesis: str) -> float:
    """Return probability that premise entails hypothesis using the NLI model."""
    if _NLI_MODEL is None:
        return _rouge_l_f1(premise, hypothesis)
    enc = _NLI_TOKENIZER(
        premise, hypothesis, return_tensors="pt", truncation=True, max_length=512
    )
    device = next(_NLI_MODEL.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = _NLI_MODEL(**enc).logits
    probs = torch.softmax(logits, dim=-1)
    # label 0 = entailment for cross-encoder/nli-deberta-v3-base
    return probs[0, 0].item()


def compute_semantic_retention_f1(
    model: nn.Module,
    persona_facts: List[str],
    tokenizer,
    device: torch.device,
    use_nli: bool = True,
) -> float:
    """NLI-based semantic retention F1 with ROUGE-L fallback.

    For each persona fact, generate a response to a probing question and check
    whether the fact is entailed in the response.

    Note: Does NOT modify adapter state — callers are responsible for ensuring
    the model is in the desired inference state. Modifying state here caused
    regressions (Loss=1.5 bug in multiple methods) during multi-adapter runs.
    """
    if use_nli:
        _load_nli()

    model.eval()
    prev_cache = getattr(model.config, "use_cache", True)
    model.config.use_cache = True
    scores: List[float] = []
    try:
        for fact in persona_facts:
            prompt = (
                f"<|im_start|>user\nTell me about yourself regarding: {fact[:80]}"
                f"<|im_end|>\n<|im_start|>assistant\n"
            )
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                out = model.generate(input_ids=ids, max_new_tokens=100, do_sample=False)
            resp = tokenizer.decode(out[0][ids.shape[1] :], skip_special_tokens=True)
            if use_nli and _NLI_MODEL is not None:
                scores.append(_nli_entailment_score(resp, fact))
            else:
                scores.append(_rouge_l_f1(fact, resp))
    finally:
        model.config.use_cache = prev_cache
    return sum(scores) / max(len(scores), 1)


def compute_adaptation_speed(session_retention_curve: List[float], threshold: float = 0.5) -> int:
    """Number of turns until retention F1 >= threshold. Returns len(curve) if never reached."""
    for i, val in enumerate(session_retention_curve):
        if val >= threshold:
            return i + 1
    return len(session_retention_curve)


def compute_forgetting_rate(cross_session_retentions: List[float]) -> float:
    """Average per-session retention decline."""
    if len(cross_session_retentions) < 2:
        return 0.0
    declines = []
    for i in range(1, len(cross_session_retentions)):
        declines.append(max(0.0, cross_session_retentions[i - 1] - cross_session_retentions[i]))
    return sum(declines) / len(declines)


def compute_dynamic_regret_proxy(
    online_losses: List[float], oracle_losses: Optional[List[float]] = None
) -> float:
    """Cumulative online loss minus oracle (if provided)."""
    if oracle_losses is not None:
        return sum(ol - orl for ol, orl in zip(online_losses, oracle_losses))
    return sum(online_losses)


def compute_rolling_accuracy(
    predictions: List[int], labels: List[int], window: int = 50
) -> List[float]:
    """Rolling window accuracy for classification."""
    accs: List[float] = []
    for i in range(len(predictions)):
        start = max(0, i - window + 1)
        correct = sum(1 for p, l in zip(predictions[start : i + 1], labels[start : i + 1]) if p == l)
        accs.append(correct / (i - start + 1))
    return accs


def compute_perplexity(model: nn.Module, tokenizer, texts: List[str], max_length: int = 1024) -> float:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    device = next(model.parameters()).device
    for text in texts[:100]:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        ids = enc["input_ids"].to(device)
        with torch.no_grad():
            out = model(input_ids=ids, labels=ids.clone())
            total_loss += out.loss.item() * ids.shape[1]
            total_tokens += ids.shape[1]
    return math.exp(total_loss / max(total_tokens, 1))


# ===================================================================
# Chat formatting
# ===================================================================


def format_turn(user_msg: str, assistant_msg: str, persona: str = "") -> str:
    system = persona if persona else "You are a helpful and personalized assistant."
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_msg}<|im_end|>"
    )


# ===================================================================
# Model loading
# ===================================================================


def load_base_model(model_name: str, device: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load base model and tokenizer. Tries sdpa then eager attention."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = None
    for attn_impl in ["sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation=attn_impl,
                device_map={"": device},
            )
            break
        except (ImportError, ValueError, RuntimeError) as exc:
            logger.warning("attn_implementation=%s failed: %s", attn_impl, exc)
    if model is None:
        raise RuntimeError(f"Cannot load model {model_name}")
    model.config.use_cache = False
    return model, tokenizer


def clone_base_model(model: AutoModelForCausalLM) -> AutoModelForCausalLM:
    """Deep-copy a base model (before any PEFT wrapping)."""
    return copy.deepcopy(model)


# ===================================================================
# Method wrappers — uniform interface
# ===================================================================

# Each wrapper exposes:
#   .model           — the nn.Module used for eval
#   .process_turn(input_ids, labels) -> dict with "loss"
#   .start_session()
#   .end_session()   — optional; consolidation at session boundary
#   .generate(input_ids, max_new_tokens) -> tensor
#   .classify(input_ids) -> logits tensor  (classification only)


class _BaseWrapper:
    """Mixin supplying default generate / classify / end_session."""

    model: nn.Module

    def end_session(self):
        pass

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 200) -> torch.Tensor:
        self.model.eval()
        prev = getattr(self.model.config, "use_cache", True)
        self.model.config.use_cache = True
        try:
            with torch.no_grad():
                return self.model.generate(
                    input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=False
                )
        finally:
            self.model.config.use_cache = prev

    def classify(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(input_ids=input_ids).logits


class FrozenWrapper(_BaseWrapper):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

    def process_turn(self, input_ids, labels):
        with torch.no_grad():
            out = self.model(input_ids=input_ids, labels=labels)
        return {"loss": out.loss.item()}

    def start_session(self):
        pass


class SingleLoRASGDWrapper(_BaseWrapper):
    def __init__(self, model, tokenizer, lr=5e-4, rank=16):
        cfg = LoraConfig(
            r=rank, lora_alpha=32, target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM, bias="none",
        )
        self.model = get_peft_model(model, cfg)
        _cast_lora_params_to_fp32(self.model)
        self.tokenizer = tokenizer
        self._params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(self._params, lr=lr)

    def process_turn(self, input_ids, labels):
        self.model.train()
        out = self.model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(self._params, 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return {"loss": out.loss.item()}

    def start_session(self):
        pass


class SingleLoRAEWCWrapper(_BaseWrapper):
    def __init__(self, model, tokenizer, lr=5e-4, rank=16, ewc_lambda=5000.0):
        cfg = LoraConfig(
            r=rank, lora_alpha=32, target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM, bias="none",
        )
        self.model = get_peft_model(model, cfg)
        _cast_lora_params_to_fp32(self.model)
        self.tokenizer = tokenizer
        self.ewc_lambda = ewc_lambda
        self._params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(self._params, lr=lr)
        self.fisher: Dict[str, torch.Tensor] = {}
        self.prev_params: Dict[str, torch.Tensor] = {}

    def process_turn(self, input_ids, labels):
        self.model.train()
        out = self.model(input_ids=input_ids, labels=labels)
        ewc_loss = self._ewc_penalty()
        loss = out.loss + ewc_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._params, 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self._update_fisher(input_ids, labels)
        return {"loss": out.loss.item()}

    def _ewc_penalty(self) -> torch.Tensor:
        penalty = torch.tensor(0.0, device=next(self.model.parameters()).device)
        for name, param in self.model.named_parameters():
            if name in self.fisher and name in self.prev_params:
                penalty += (self.fisher[name] * (param - self.prev_params[name]).pow(2)).sum()
        return self.ewc_lambda * penalty

    def _update_fisher(self, input_ids, labels):
        self.model.train()
        self.model.zero_grad()
        with torch.enable_grad():
            out = self.model(input_ids=input_ids, labels=labels)
            out.loss.backward()
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                gsq = param.grad.data.pow(2).clone()
                if name not in self.fisher:
                    self.fisher[name] = gsq
                else:
                    self.fisher[name] = 0.9 * self.fisher[name] + 0.1 * gsq
        self.model.zero_grad()

    def start_session(self):
        self.prev_params = {
            n: p.data.clone() for n, p in self.model.named_parameters() if p.requires_grad
        }


class SingleLoRAReplayWrapper(_BaseWrapper):
    def __init__(self, model, tokenizer, lr=5e-4, rank=16, replay_size=1000):
        cfg = LoraConfig(
            r=rank, lora_alpha=32, target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM, bias="none",
        )
        self.model = get_peft_model(model, cfg)
        _cast_lora_params_to_fp32(self.model)
        self.tokenizer = tokenizer
        self._params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(self._params, lr=lr)
        self.replay_buffer: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.max_replay = replay_size

    def process_turn(self, input_ids, labels):
        self.model.train()
        out = self.model(input_ids=input_ids, labels=labels)
        total_loss = out.loss
        # Replay one sample
        if self.replay_buffer:
            idx = random.randint(0, len(self.replay_buffer) - 1)
            r_ids, r_lab = self.replay_buffer[idx]
            dev = next(self.model.parameters()).device
            r_out = self.model(input_ids=r_ids.to(dev), labels=r_lab.to(dev))
            total_loss = total_loss + 0.5 * r_out.loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._params, 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        # Store
        if len(self.replay_buffer) < self.max_replay:
            self.replay_buffer.append((input_ids.detach().cpu(), labels.detach().cpu()))
        else:
            j = random.randint(0, len(self.replay_buffer) - 1)
            self.replay_buffer[j] = (input_ids.detach().cpu(), labels.detach().cpu())
        return {"loss": out.loss.item()}

    def start_session(self):
        pass


class SingleLoRAEWCReplayWrapper(_BaseWrapper):
    """Combines EWC + experience replay."""

    def __init__(self, model, tokenizer, lr=5e-4, rank=16, ewc_lambda=5000.0, replay_size=1000):
        cfg = LoraConfig(
            r=rank, lora_alpha=32, target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM, bias="none",
        )
        self.model = get_peft_model(model, cfg)
        _cast_lora_params_to_fp32(self.model)
        self.tokenizer = tokenizer
        self.ewc_lambda = ewc_lambda
        self._params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(self._params, lr=lr)
        self.fisher: Dict[str, torch.Tensor] = {}
        self.prev_params: Dict[str, torch.Tensor] = {}
        self.replay_buffer: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.max_replay = replay_size

    def process_turn(self, input_ids, labels):
        self.model.train()
        out = self.model(input_ids=input_ids, labels=labels)
        ewc = self._ewc_penalty()
        total_loss = out.loss + ewc
        if self.replay_buffer:
            idx = random.randint(0, len(self.replay_buffer) - 1)
            r_ids, r_lab = self.replay_buffer[idx]
            dev = next(self.model.parameters()).device
            r_out = self.model(input_ids=r_ids.to(dev), labels=r_lab.to(dev))
            total_loss = total_loss + 0.5 * r_out.loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._params, 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self._update_fisher(input_ids, labels)
        if len(self.replay_buffer) < self.max_replay:
            self.replay_buffer.append((input_ids.detach().cpu(), labels.detach().cpu()))
        else:
            j = random.randint(0, len(self.replay_buffer) - 1)
            self.replay_buffer[j] = (input_ids.detach().cpu(), labels.detach().cpu())
        return {"loss": out.loss.item()}

    def _ewc_penalty(self) -> torch.Tensor:
        penalty = torch.tensor(0.0, device=next(self.model.parameters()).device)
        for name, param in self.model.named_parameters():
            if name in self.fisher and name in self.prev_params:
                penalty += (self.fisher[name] * (param - self.prev_params[name]).pow(2)).sum()
        return self.ewc_lambda * penalty

    def _update_fisher(self, input_ids, labels):
        self.model.train()
        self.model.zero_grad()
        with torch.enable_grad():
            out = self.model(input_ids=input_ids, labels=labels)
            out.loss.backward()
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                gsq = param.grad.data.pow(2).clone()
                if name not in self.fisher:
                    self.fisher[name] = gsq
                else:
                    self.fisher[name] = 0.9 * self.fisher[name] + 0.1 * gsq
        self.model.zero_grad()

    def start_session(self):
        self.prev_params = {
            n: p.data.clone() for n, p in self.model.named_parameters() if p.requires_grad
        }


class ParamMatchedLoRAWrapper(SingleLoRAEWCReplayWrapper):
    """Same as EWC+Replay but with rank scaled to match our method's param count."""

    def __init__(self, model, tokenizer, lr=5e-4, rank=80, ewc_lambda=5000.0, replay_size=1000):
        super().__init__(model, tokenizer, lr=lr, rank=rank, ewc_lambda=ewc_lambda, replay_size=replay_size)


class DualLoRAEMAWrapper(_BaseWrapper):
    """Dual LoRA: fast adapter trains online, slow adapter is EMA of fast."""

    def __init__(self, model, tokenizer, lr=5e-4, rank=16, ema_beta=0.99):
        fast_cfg = LoraConfig(
            r=rank, lora_alpha=32, target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM, bias="none",
        )
        self.model = get_peft_model(model, fast_cfg, adapter_name="fast")
        self.model.add_adapter(
            "slow",
            LoraConfig(r=rank, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                       task_type=TaskType.CAUSAL_LM, bias="none"),
        )
        self.model.set_adapter("fast")
        _cast_lora_params_to_fp32(self.model)
        self.tokenizer = tokenizer
        self.ema_beta = ema_beta
        self._params = [p for n, p in self.model.named_parameters() if p.requires_grad and ".fast." in n]
        self.optimizer = torch.optim.AdamW(self._params, lr=lr)

    def process_turn(self, input_ids, labels):
        self.model.set_adapter("fast")
        self.model.train()
        out = self.model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(self._params, 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self._ema_update()
        return {"loss": out.loss.item()}

    def _ema_update(self):
        # PEFT parameter names: ...lora_A.fast.weight / ...lora_B.fast.weight
        # Match on ".fast." and ".slow." (with dots) for robust adapter filtering
        fast_dict = {n: p for n, p in self.model.named_parameters() if ".fast." in n and "lora" in n}
        slow_dict = {n: p for n, p in self.model.named_parameters() if ".slow." in n and "lora" in n}
        for fn, fp in fast_dict.items():
            sn = fn.replace(".fast.", ".slow.")
            if sn in slow_dict:
                slow_dict[sn].data.mul_(self.ema_beta).add_(fp.data, alpha=1.0 - self.ema_beta)

    def start_session(self):
        pass

    def generate(self, input_ids, max_new_tokens=200):
        _set_adapters_layerwise(self.model, ["fast", "slow"])
        return super().generate(input_ids, max_new_tokens)


class DualLoRAPeriodicAvgWrapper(_BaseWrapper):
    """Dual LoRA: fast adapter trains; slow adapter = periodic snapshot average."""

    def __init__(self, model, tokenizer, lr=5e-4, rank=16, avg_period=20):
        fast_cfg = LoraConfig(
            r=rank, lora_alpha=32, target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM, bias="none",
        )
        self.model = get_peft_model(model, fast_cfg, adapter_name="fast")
        self.model.add_adapter(
            "slow",
            LoraConfig(r=rank, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                       task_type=TaskType.CAUSAL_LM, bias="none"),
        )
        self.model.set_adapter("fast")
        _cast_lora_params_to_fp32(self.model)
        self.tokenizer = tokenizer
        self.avg_period = avg_period
        self._step = 0
        self._accum: Dict[str, torch.Tensor] = {}
        self._accum_n = 0
        self._params = [p for n, p in self.model.named_parameters() if p.requires_grad and ".fast." in n]
        self.optimizer = torch.optim.AdamW(self._params, lr=lr)

    def process_turn(self, input_ids, labels):
        self.model.set_adapter("fast")
        self.model.train()
        out = self.model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(self._params, 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self._step += 1
        # accumulate
        for n, p in self.model.named_parameters():
            if ".fast." in n and "lora" in n:
                if n not in self._accum:
                    self._accum[n] = p.data.clone()
                else:
                    self._accum[n].add_(p.data)
        self._accum_n += 1
        if self._accum_n >= self.avg_period:
            self._flush_avg()
        return {"loss": out.loss.item()}

    def _flush_avg(self):
        slow_dict = {n: p for n, p in self.model.named_parameters() if ".slow." in n and "lora" in n}
        for fn, acc in self._accum.items():
            sn = fn.replace(".fast.", ".slow.")
            if sn in slow_dict:
                slow_dict[sn].data.copy_(acc / self._accum_n)
        self._accum.clear()
        self._accum_n = 0

    def start_session(self):
        pass

    def generate(self, input_ids, max_new_tokens=200):
        _set_adapters_layerwise(self.model, ["fast", "slow"])
        return super().generate(input_ids, max_new_tokens)


class DualLoRAHeuristicWrapper(_BaseWrapper):
    """Dual LoRA: fast adapter trains; slow adapter consolidates when loss plateaus."""

    def __init__(self, model, tokenizer, lr=5e-4, rank=16, plateau_window=5, plateau_threshold=0.01):
        fast_cfg = LoraConfig(
            r=rank, lora_alpha=32, target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM, bias="none",
        )
        self.model = get_peft_model(model, fast_cfg, adapter_name="fast")
        self.model.add_adapter(
            "slow",
            LoraConfig(r=rank, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                       task_type=TaskType.CAUSAL_LM, bias="none"),
        )
        self.model.set_adapter("fast")
        _cast_lora_params_to_fp32(self.model)
        self.tokenizer = tokenizer
        self.plateau_window = plateau_window
        self.plateau_threshold = plateau_threshold
        self._recent_losses: List[float] = []
        self._params = [p for n, p in self.model.named_parameters() if p.requires_grad and ".fast." in n]
        self.optimizer = torch.optim.AdamW(self._params, lr=lr)

    def process_turn(self, input_ids, labels):
        self.model.set_adapter("fast")
        self.model.train()
        out = self.model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(self._params, 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self._recent_losses.append(out.loss.item())
        if len(self._recent_losses) >= self.plateau_window:
            window = self._recent_losses[-self.plateau_window :]
            if max(window) - min(window) < self.plateau_threshold:
                self._consolidate()
                self._recent_losses.clear()
        return {"loss": out.loss.item()}

    def _consolidate(self):
        fast_dict = {n: p for n, p in self.model.named_parameters() if ".fast." in n and "lora" in n}
        slow_dict = {n: p for n, p in self.model.named_parameters() if ".slow." in n and "lora" in n}
        for fn, fp in fast_dict.items():
            sn = fn.replace(".fast.", ".slow.")
            if sn in slow_dict:
                slow_dict[sn].data.copy_(0.5 * slow_dict[sn].data + 0.5 * fp.data)

    def start_session(self):
        self._recent_losses.clear()

    def generate(self, input_ids, max_new_tokens=200):
        _set_adapters_layerwise(self.model, ["fast", "slow"])
        return super().generate(input_ids, max_new_tokens)


class RetrievalAugmentedWrapper(_BaseWrapper):
    """No parameter updates; store history and prepend top-k context at generation."""

    def __init__(self, model, tokenizer, top_k=5):
        self.model = model
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.memory: List[str] = []
        self.model.eval()

    def process_turn(self, input_ids, labels):
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        self.memory.append(text[:200])
        with torch.no_grad():
            out = self.model(input_ids=input_ids, labels=labels)
        return {"loss": out.loss.item()}

    def start_session(self):
        pass

    def generate(self, input_ids, max_new_tokens=200):
        prompt_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        retrieved = self.memory[-self.top_k :] if self.memory else []
        context = "\n".join(retrieved)
        augmented = f"<|im_start|>system\nRelevant history:\n{context}<|im_end|>\n{prompt_text}"
        ids = self.tokenizer(
            augmented, return_tensors="pt", truncation=True, max_length=2048
        ).input_ids.to(next(self.model.parameters()).device)
        self.model.eval()
        prev = getattr(self.model.config, "use_cache", True)
        self.model.config.use_cache = True
        try:
            with torch.no_grad():
                return self.model.generate(input_ids=ids, max_new_tokens=max_new_tokens, do_sample=False)
        finally:
            self.model.config.use_cache = prev


class SDFTWrapper(_BaseWrapper):
    """Selective Dual-LoRA Fine-Tuning: only update LoRA params with top-k gradient magnitude."""

    def __init__(self, model, tokenizer, lr=5e-4, rank=16, top_k_frac=0.3):
        cfg = LoraConfig(
            r=rank, lora_alpha=32, target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM, bias="none",
        )
        self.model = get_peft_model(model, cfg)
        _cast_lora_params_to_fp32(self.model)
        self.tokenizer = tokenizer
        self.top_k_frac = top_k_frac
        self._params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(self._params, lr=lr)

    def process_turn(self, input_ids, labels):
        self.model.train()
        out = self.model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        # Selective masking: zero out bottom gradients
        for p in self._params:
            if p.grad is not None:
                flat = p.grad.data.abs().flatten()
                k = max(1, int(len(flat) * self.top_k_frac))
                threshold = flat.topk(k).values[-1]
                mask = p.grad.data.abs() >= threshold
                p.grad.data.mul_(mask.float())
        torch.nn.utils.clip_grad_norm_(self._params, 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return {"loss": out.loss.item()}

    def start_session(self):
        pass


class SPMWrapper(_BaseWrapper):
    """StreamingParameterMemory (original two-timescale KL distillation)."""

    def __init__(self, model, tokenizer, config):
        self.tokenizer = tokenizer
        beta = config.get("consolidation", {}).get("beta", 1.0)
        gamma = config.get("consolidation", {}).get("gamma", 0.0)
        self._spm = StreamingParameterMemory(
            base_model=model,
            working_config=config["working_memory"],
            longterm_config=config["long_term_memory"],
            reservoir_size=config["long_term_memory"].get("max_memory_buffer", 5000),
            beta=beta,
            gamma=gamma,
        )
        self.model = self._spm.model

    def process_turn(self, input_ids, labels):
        return self._spm.process_turn(input_ids, labels)

    def start_session(self):
        self._spm.start_new_session()

    def end_session(self):
        return self._spm.consolidate_session()

    def generate(self, input_ids, max_new_tokens=200):
        return self._spm.generate(input_ids, max_new_tokens=max_new_tokens, use_longterm=True)


class MirrorLoRAWrapper(_BaseWrapper):
    """StreamingMirrorLoRA (our full method)."""

    def __init__(self, model, tokenizer, config):
        self.tokenizer = tokenizer
        self._mirror = StreamingMirrorLoRA(
            base_model=model,
            working_config=config["working_memory"],
            longterm_config=config["long_term_memory"],
            reservoir_size=config["long_term_memory"].get("max_memory_buffer", 5000),
            beta=config.get("consolidation", {}).get("beta", 1.0),
            gamma=config.get("consolidation", {}).get("gamma", 0.0),
            fisher_beta=0.99,
            consolidation_threshold=1.0,
            invariant_window=3,
            invariant_threshold=0.5,
            use_grassmann=False,
            adaptive_frequency=True,
        )
        self.model = self._mirror.model

    def process_turn(self, input_ids, labels):
        return self._mirror.process_turn(input_ids, labels)

    def start_session(self):
        self._mirror.start_new_session()

    def end_session(self):
        return self._mirror.consolidate()

    def generate(self, input_ids, max_new_tokens=200):
        return self._mirror.generate(input_ids, max_new_tokens=max_new_tokens, use_longterm=True)


# ===================================================================
# Method factory
# ===================================================================

ALL_METHODS = [
    "frozen",
    "single_lora_sgd",
    "single_lora_ewc",
    "single_lora_replay",
    "single_lora_ewc_replay",
    "param_matched_lora",
    "dual_lora_ema",
    "dual_lora_periodic_avg",
    "dual_lora_heuristic",
    "retrieval_augmented",
    "sdft",
    "spm",
    "mirror_lora",
]


def create_method(name: str, model: nn.Module, tokenizer, config: dict):
    """Instantiate a method wrapper by name. `model` is a fresh base model copy."""
    if name == "frozen":
        return FrozenWrapper(model, tokenizer)
    elif name == "single_lora_sgd":
        return SingleLoRASGDWrapper(model, tokenizer)
    elif name == "single_lora_ewc":
        return SingleLoRAEWCWrapper(model, tokenizer)
    elif name == "single_lora_replay":
        return SingleLoRAReplayWrapper(model, tokenizer)
    elif name == "single_lora_ewc_replay":
        return SingleLoRAEWCReplayWrapper(model, tokenizer)
    elif name == "param_matched_lora":
        return ParamMatchedLoRAWrapper(model, tokenizer, rank=80)
    elif name == "dual_lora_ema":
        return DualLoRAEMAWrapper(model, tokenizer)
    elif name == "dual_lora_periodic_avg":
        return DualLoRAPeriodicAvgWrapper(model, tokenizer)
    elif name == "dual_lora_heuristic":
        return DualLoRAHeuristicWrapper(model, tokenizer)
    elif name == "retrieval_augmented":
        return RetrievalAugmentedWrapper(model, tokenizer)
    elif name == "sdft":
        return SDFTWrapper(model, tokenizer)
    elif name == "spm":
        return SPMWrapper(model, tokenizer, config)
    elif name == "mirror_lora":
        return MirrorLoRAWrapper(model, tokenizer, config)
    else:
        raise ValueError(f"Unknown method: {name}")


# ===================================================================
# Data loading
# ===================================================================


def load_dialogue_data(
    max_sessions: int, sessions_per_persona: int, turns_per_session: int
) -> Tuple[List[List[dict]], List[List[str]]]:
    """Load PersonaChat grouped by persona. Returns (sessions, persona_facts_per_session)."""
    sessions: List[List[dict]] = []
    persona_facts_list: List[List[str]] = []

    try:
        ds = load_dataset("bavard/personachat_truecased", split="validation", trust_remote_code=True)
    except Exception as exc:
        logger.warning("PersonaChat load failed (%s); generating synthetic data", exc)
        return _synthetic_dialogue_data(max_sessions, turns_per_session)

    persona_groups: Dict[str, List[dict]] = defaultdict(list)
    persona_fact_map: Dict[str, List[str]] = {}

    for ex in ds:
        history = ex.get("history", [])
        candidates = ex.get("candidates", [])
        personality = ex.get("personality", [])
        reply = candidates[-1] if candidates else ""
        if not history or not reply:
            continue
        persona_key = " ".join(sorted(personality)) if personality else "_no_persona_"
        persona_groups[persona_key].append({
            "user": history[-1],
            "assistant": reply,
            "persona": " ".join(personality) if personality else "",
        })
        if persona_key not in persona_fact_map and personality:
            persona_fact_map[persona_key] = list(personality)

    for persona_key, turns in persona_groups.items():
        if len(sessions) >= max_sessions:
            break
        facts = persona_fact_map.get(persona_key, [])
        for s_idx in range(sessions_per_persona):
            start = s_idx * turns_per_session
            chunk = turns[start : start + turns_per_session]
            if not chunk:
                break
            sessions.append(chunk)
            persona_facts_list.append(facts)
            if len(sessions) >= max_sessions:
                break

    if not sessions:
        return _synthetic_dialogue_data(max_sessions, turns_per_session)

    logger.info("Loaded %d dialogue sessions from PersonaChat", len(sessions))
    return sessions, persona_facts_list


def _synthetic_dialogue_data(
    max_sessions: int, turns_per_session: int
) -> Tuple[List[List[dict]], List[List[str]]]:
    sessions = []
    facts_list = []
    for s in range(max_sessions):
        facts = [f"I am persona number {s}.", f"My favorite color is color-{s % 7}."]
        turns = [
            {
                "user": f"Session {s} turn {t}: tell me about yourself.",
                "assistant": f"I am persona {s}. My favorite color is color-{s % 7}. Turn {t}.",
                "persona": " ".join(facts),
            }
            for t in range(turns_per_session)
        ]
        sessions.append(turns)
        facts_list.append(facts)
    logger.info("Generated %d synthetic dialogue sessions", len(sessions))
    return sessions, facts_list


def load_classification_data(max_sessions: int, turns_per_session: int) -> Dict[str, Any]:
    """Load AG News and create a concept-drift stream.

    Returns dict with keys: stream (list of (text, label)), label_names, drift_points.
    """
    label_names = ["World", "Sports", "Business", "Sci/Tech"]

    try:
        ds = load_dataset("ag_news", split="test", trust_remote_code=True)
    except Exception as exc:
        logger.warning("AG News load failed (%s); generating synthetic", exc)
        return _synthetic_classification_data(max_sessions, turns_per_session, label_names)

    # Group by label
    by_label: Dict[int, List[str]] = defaultdict(list)
    for ex in ds:
        by_label[ex["label"]].append(ex["text"][:512])

    total_turns = max_sessions * turns_per_session
    stream: List[Tuple[str, int]] = []
    drift_points: List[int] = []

    # Phase 1: gradual shift (topics 0,1 dominant -> topics 2,3 dominant)
    phase1_len = int(total_turns * 0.6)
    for t in range(phase1_len):
        frac = t / max(phase1_len - 1, 1)
        # probability of topic 2,3 increases linearly
        if random.random() < frac:
            label = random.choice([2, 3])
        else:
            label = random.choice([0, 1])
        if by_label[label]:
            text = by_label[label][t % len(by_label[label])]
            stream.append((text, label))

    drift_points.append(len(stream))

    # Phase 2: sudden appearance of topic 3 burst
    phase2_len = int(total_turns * 0.15)
    for t in range(phase2_len):
        label = 3
        if by_label[label]:
            text = by_label[label][(phase1_len + t) % len(by_label[label])]
            stream.append((text, label))

    drift_points.append(len(stream))

    # Phase 3: mixed
    phase3_len = total_turns - len(stream)
    for t in range(phase3_len):
        label = random.choice([0, 1, 2, 3])
        if by_label[label]:
            text = by_label[label][(phase1_len + phase2_len + t) % len(by_label[label])]
            stream.append((text, label))

    logger.info("Created classification stream: %d samples, %d drift points", len(stream), len(drift_points))
    return {"stream": stream, "label_names": label_names, "drift_points": drift_points}


def _synthetic_classification_data(
    max_sessions: int, turns_per_session: int, label_names: List[str]
) -> Dict[str, Any]:
    total_turns = max_sessions * turns_per_session
    stream = []
    for t in range(total_turns):
        label = t % len(label_names)
        stream.append((f"Synthetic text about {label_names[label]} topic {t}.", label))
    return {"stream": stream, "label_names": label_names, "drift_points": [total_turns // 2]}


# ===================================================================
# Evaluation loops
# ===================================================================


def evaluate_dialogue(
    method,
    method_name: str,
    tokenizer,
    sessions: List[List[dict]],
    persona_facts_list: List[List[str]],
    device: torch.device,
    use_nli: bool = True,
) -> Dict[str, Any]:
    """Run dialogue streaming evaluation on a single method."""
    logger.info("  [dialogue] Evaluating %s (%d sessions)", method_name, len(sessions))
    t_start_total = time.time()

    all_losses: List[float] = []
    session_retentions: List[float] = []
    session_adaptation_curves: List[List[float]] = []
    turn_latencies: List[float] = []
    predictions: List[str] = []
    references: List[str] = []

    for sess_idx, (session, facts) in enumerate(zip(sessions, persona_facts_list)):
        method.start_session()
        sess_losses: List[float] = []
        turn_ret_curve: List[float] = []

        for turn_idx, turn in enumerate(session):
            t0 = time.time()
            text = format_turn(turn["user"], turn["assistant"], turn.get("persona", ""))
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
            input_ids = enc["input_ids"].to(device)
            labels = input_ids.clone()

            result = method.process_turn(input_ids, labels)
            loss_val = result.get("loss", 0.0)
            sess_losses.append(loss_val)
            turn_latencies.append(time.time() - t0)

            # Per-turn retention probe (every 5 turns)
            if facts and turn_idx > 0 and turn_idx % 5 == 0:
                ret = compute_semantic_retention_f1(method.model, facts[:3], tokenizer, device, use_nli)
                turn_ret_curve.append(ret)

        # End-of-session consolidation
        if hasattr(method, "end_session"):
            try:
                method.end_session()
            except Exception as exc:
                logger.warning("end_session failed for %s: %s", method_name, exc)

        all_losses.extend(sess_losses)
        session_adaptation_curves.append(turn_ret_curve)

        # Cross-session retention: probe with facts from this session
        if facts:
            ret = compute_semantic_retention_f1(method.model, facts[:3], tokenizer, device, use_nli)
            session_retentions.append(ret)

        # Generation quality: last 2 turns
        for turn in session[-2:]:
            prompt = f"<|im_start|>user\n{turn['user']}<|im_end|>\n<|im_start|>assistant\n"
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            try:
                out = method.generate(ids, max_new_tokens=100)
                pred = tokenizer.decode(out[0][ids.shape[1] :], skip_special_tokens=True)
            except Exception:
                pred = ""
            predictions.append(pred)
            references.append(turn["assistant"])

        if (sess_idx + 1) % 10 == 0 or sess_idx == 0:
            avg_l = sum(sess_losses) / max(len(sess_losses), 1)
            logger.info(
                "    Session %d/%d | loss=%.4f | ret=%.4f",
                sess_idx + 1, len(sessions), avg_l,
                session_retentions[-1] if session_retentions else 0.0,
            )

    # Aggregate metrics
    final_ret = session_retentions[-1] if session_retentions else 0.0
    forgetting = compute_forgetting_rate(session_retentions)

    adapt_speeds = []
    for curve in session_adaptation_curves:
        if curve:
            adapt_speeds.append(compute_adaptation_speed(curve))

    eval_texts = [format_turn(t["user"], t["assistant"]) for s in sessions[-3:] for t in s[:3]]
    ppl = float("inf")
    try:
        ppl = compute_perplexity(method.model, tokenizer, eval_texts)
    except Exception:
        pass

    elapsed = time.time() - t_start_total
    results = {
        "semantic_retention_f1": final_ret,
        "adaptation_speed": sum(adapt_speeds) / max(len(adapt_speeds), 1) if adapt_speeds else 0.0,
        "forgetting_rate": forgetting,
        "perplexity": ppl,
        "avg_loss": sum(all_losses) / max(len(all_losses), 1),
        "session_retentions": session_retentions,
        "avg_turn_latency_ms": sum(turn_latencies) / max(len(turn_latencies), 1) * 1000,
        "total_time_s": elapsed,
        "num_sessions": len(sessions),
    }
    logger.info(
        "    %s done: RetF1=%.4f Forget=%.4f AdaptSpeed=%.1f PPL=%.2f (%.1fs)",
        method_name, final_ret, forgetting,
        results["adaptation_speed"], ppl, elapsed,
    )
    return results


def evaluate_classification(
    method,
    method_name: str,
    tokenizer,
    data: Dict[str, Any],
    device: torch.device,
    turns_per_session: int = 15,
) -> Dict[str, Any]:
    """Run classification streaming evaluation on a single method."""
    stream = data["stream"]
    label_names = data["label_names"]
    num_labels = len(label_names)

    logger.info("  [classification] Evaluating %s (%d samples)", method_name, len(stream))
    t_start_total = time.time()

    all_losses: List[float] = []
    all_preds: List[int] = []
    all_labels: List[int] = []
    turn_latencies: List[float] = []
    session_accs: List[float] = []

    sess_correct = 0
    sess_total = 0

    for idx, (text, label) in enumerate(stream):
        if idx > 0 and idx % turns_per_session == 0:
            if hasattr(method, "end_session"):
                try:
                    method.end_session()
                except Exception:
                    pass
            if sess_total > 0:
                session_accs.append(sess_correct / sess_total)
            method.start_session()
            sess_correct = 0
            sess_total = 0

        t0 = time.time()

        # Format as next-token-prediction: "Classify: <text>\nLabel: <label_name>"
        label_text = label_names[label]
        full_text = f"Classify the following text.\nText: {text[:300]}\nLabel: {label_text}"
        enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = enc["input_ids"].to(device)
        labels_tensor = input_ids.clone()

        # Predict before update: get logits for last token
        try:
            # Reset to primary adapter before inference (fixes multi-adapter bug)
            _reset_to_primary_adapter(method)
            method.model.eval()
            with torch.no_grad():
                logits = method.model(input_ids=input_ids).logits
            # Use the logit over label tokens as a proxy for classification
            last_logits = logits[0, -1, :]
            label_token_ids = [
                tokenizer.encode(ln, add_special_tokens=False)[0] for ln in label_names
            ]
            label_logits = last_logits[label_token_ids]
            pred = label_logits.argmax().item()
        except Exception:
            pred = 0

        all_preds.append(pred)
        all_labels.append(label)
        if pred == label:
            sess_correct += 1
        sess_total += 1

        # Update
        result = method.process_turn(input_ids, labels_tensor)
        all_losses.append(result.get("loss", 0.0))
        turn_latencies.append(time.time() - t0)

        if (idx + 1) % 200 == 0:
            recent_acc = sum(
                1 for p, l in zip(all_preds[-200:], all_labels[-200:]) if p == l
            ) / min(200, len(all_preds))
            logger.info(
                "    Step %d/%d | loss=%.4f | recent_acc=%.4f",
                idx + 1, len(stream), all_losses[-1], recent_acc,
            )

    if sess_total > 0:
        session_accs.append(sess_correct / sess_total)

    # Rolling accuracy
    rolling_acc = compute_rolling_accuracy(all_preds, all_labels, window=50)

    # Forgetting: compare first-half session accs to second-half
    forgetting = 0.0
    if len(session_accs) >= 4:
        mid = len(session_accs) // 2
        first_half = sum(session_accs[:mid]) / mid
        second_half = sum(session_accs[mid:]) / (len(session_accs) - mid)
        forgetting = max(0.0, first_half - second_half)

    # Forward transfer: accuracy on new topics after drift
    drift_points = data.get("drift_points", [])
    forward_transfer = 0.0
    if drift_points and len(all_preds) > drift_points[0]:
        dp = drift_points[0]
        window = min(100, len(all_preds) - dp)
        if window > 0:
            forward_transfer = sum(
                1 for p, l in zip(all_preds[dp : dp + window], all_labels[dp : dp + window]) if p == l
            ) / window

    dynamic_regret = compute_dynamic_regret_proxy(all_losses)

    elapsed = time.time() - t_start_total
    results = {
        "rolling_accuracy_final": rolling_acc[-1] if rolling_acc else 0.0,
        "overall_accuracy": sum(1 for p, l in zip(all_preds, all_labels) if p == l) / max(len(all_preds), 1),
        "forgetting": forgetting,
        "forward_transfer": forward_transfer,
        "dynamic_regret": dynamic_regret,
        "avg_loss": sum(all_losses) / max(len(all_losses), 1),
        "session_accs": session_accs,
        "rolling_accuracy_curve": rolling_acc[::max(1, len(rolling_acc) // 200)],  # subsample for JSON
        "avg_turn_latency_ms": sum(turn_latencies) / max(len(turn_latencies), 1) * 1000,
        "total_time_s": elapsed,
        "num_samples": len(stream),
    }
    logger.info(
        "    %s done: acc=%.4f forget=%.4f fwd_transfer=%.4f regret=%.2f (%.1fs)",
        method_name, results["overall_accuracy"], forgetting,
        forward_transfer, dynamic_regret, elapsed,
    )
    return results


# ===================================================================
# Save / aggregate
# ===================================================================


def save_results(results: dict, method_name: str, task: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{task}_{method_name}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("  Saved %s -> %s", method_name, path)


def aggregate_results(output_dir: str, task: str):
    """Read all per-method JSONs and write a comparison table."""
    import glob as globmod

    pattern = os.path.join(output_dir, f"{task}_*.json")
    files = sorted(globmod.glob(pattern))
    if not files:
        logger.warning("No result files found for aggregation")
        return

    table: Dict[str, dict] = {}
    for fpath in files:
        name = os.path.basename(fpath).replace(f"{task}_", "").replace(".json", "")
        with open(fpath) as f:
            table[name] = json.load(f)

    # Print table
    logger.info("")
    logger.info("=" * 100)
    logger.info("AGGREGATED RESULTS: task=%s", task)
    logger.info("=" * 100)

    if task == "dialogue":
        header = f"{'Method':<25s} {'RetF1':>8s} {'Forget':>8s} {'AdaptSpd':>9s} {'PPL':>8s} {'Loss':>8s} {'Latency':>10s}"
        logger.info(header)
        logger.info("-" * 100)
        for name, m in sorted(table.items()):
            logger.info(
                f"{name:<25s} "
                f"{m.get('semantic_retention_f1', 0):>8.4f} "
                f"{m.get('forgetting_rate', 0):>8.4f} "
                f"{m.get('adaptation_speed', 0):>9.1f} "
                f"{m.get('perplexity', 0):>8.2f} "
                f"{m.get('avg_loss', 0):>8.4f} "
                f"{m.get('avg_turn_latency_ms', 0):>9.1f}ms"
            )
    else:
        header = f"{'Method':<25s} {'Acc':>8s} {'Forget':>8s} {'FwdTrans':>9s} {'Regret':>10s} {'Loss':>8s} {'Latency':>10s}"
        logger.info(header)
        logger.info("-" * 100)
        for name, m in sorted(table.items()):
            logger.info(
                f"{name:<25s} "
                f"{m.get('overall_accuracy', 0):>8.4f} "
                f"{m.get('forgetting', 0):>8.4f} "
                f"{m.get('forward_transfer', 0):>9.4f} "
                f"{m.get('dynamic_regret', 0):>10.2f} "
                f"{m.get('avg_loss', 0):>8.4f} "
                f"{m.get('avg_turn_latency_ms', 0):>9.1f}ms"
            )

    # Save aggregated JSON
    agg_path = os.path.join(output_dir, f"{task}_comparison.json")
    with open(agg_path, "w") as f:
        json.dump(table, f, indent=2, default=str)
    logger.info("Aggregated comparison -> %s", agg_path)

    # LaTeX table
    latex_lines = [
        "\\begin{table}[h]",
        "\\centering",
        f"\\caption{{Streaming Evaluation Results ({task})}}",
    ]
    if task == "dialogue":
        latex_lines.extend([
            "\\begin{tabular}{lccccc}",
            "\\toprule",
            "Method & Ret-F1 $\\uparrow$ & Forget $\\downarrow$ & Adapt Speed $\\downarrow$ & PPL $\\downarrow$ & Latency (ms) \\\\",
            "\\midrule",
        ])
        for name, m in sorted(table.items()):
            latex_lines.append(
                f"{name} & {m.get('semantic_retention_f1', 0):.4f} & "
                f"{m.get('forgetting_rate', 0):.4f} & "
                f"{m.get('adaptation_speed', 0):.1f} & "
                f"{m.get('perplexity', 0):.2f} & "
                f"{m.get('avg_turn_latency_ms', 0):.1f} \\\\"
            )
    else:
        latex_lines.extend([
            "\\begin{tabular}{lccccc}",
            "\\toprule",
            "Method & Accuracy $\\uparrow$ & Forget $\\downarrow$ & Fwd Transfer $\\uparrow$ & Dyn Regret $\\downarrow$ & Latency (ms) \\\\",
            "\\midrule",
        ])
        for name, m in sorted(table.items()):
            latex_lines.append(
                f"{name} & {m.get('overall_accuracy', 0):.4f} & "
                f"{m.get('forgetting', 0):.4f} & "
                f"{m.get('forward_transfer', 0):.4f} & "
                f"{m.get('dynamic_regret', 0):.2f} & "
                f"{m.get('avg_turn_latency_ms', 0):.1f} \\\\"
            )

    latex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    tex_path = os.path.join(output_dir, f"{task}_results_table.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(latex_lines))
    logger.info("LaTeX table -> %s", tex_path)


# ===================================================================
# Main
# ===================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Unified streaming evaluation: all methods on dialogue/classification tasks"
    )
    parser.add_argument("--task", choices=["dialogue", "classification"], required=True)
    parser.add_argument("--methods", nargs="+", default=["all"])
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--config", default="configs/spm_config.yaml")
    parser.add_argument("--output_dir", default="outputs/streaming_eval")
    parser.add_argument("--max_sessions", type=int, default=50)
    parser.add_argument("--sessions_per_persona", type=int, default=5)
    parser.add_argument("--turns_per_session", type=int, default=15)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_nli", action="store_true", help="Disable NLI model, use ROUGE-L only")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if not torch.cuda.is_available():
        args.device = "cpu"
        logger.warning("CUDA not available, using CPU")

    # Load config
    config_path = args.config
    if os.path.isfile(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        logger.warning("Config %s not found; using defaults", config_path)
        config = {
            "working_memory": {
                "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "learning_rate": 5e-4, "online_update_steps": 3,
            },
            "long_term_memory": {
                "lora_r": 64, "lora_alpha": 128, "lora_dropout": 0.02,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                "consolidation_lr": 1e-4, "consolidation_epochs": 3, "max_memory_buffer": 5000,
            },
            "consolidation": {"beta": 1.0, "gamma": 0.0},
        }

    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve methods
    methods_to_run = ALL_METHODS if "all" in args.methods else args.methods
    for m in methods_to_run:
        if m not in ALL_METHODS:
            logger.error("Unknown method '%s'. Available: %s", m, ALL_METHODS)
            sys.exit(1)

    logger.info("Task: %s", args.task)
    logger.info("Methods: %s", methods_to_run)
    logger.info("Model: %s", args.model)
    logger.info("Device: %s", args.device)
    logger.info("Output: %s", args.output_dir)

    # Load data once
    if args.task == "dialogue":
        sessions, persona_facts_list = load_dialogue_data(
            args.max_sessions, args.sessions_per_persona, args.turns_per_session
        )
    else:
        classification_data = load_classification_data(args.max_sessions, args.turns_per_session)

    # Load base model once
    logger.info("Loading base model: %s", args.model)
    base_model, tokenizer = load_base_model(args.model, args.device)
    logger.info("Base model loaded. Parameters: %d", sum(p.numel() for p in base_model.parameters()))

    # Evaluate each method
    for method_name in methods_to_run:
        logger.info("")
        logger.info("=" * 70)
        logger.info("METHOD: %s", method_name)
        logger.info("=" * 70)

        try:
            # Clone base model for this method
            model_copy = clone_base_model(base_model)
            method = create_method(method_name, model_copy, tokenizer, config)

            device = next(method.model.parameters()).device

            if args.task == "dialogue":
                results = evaluate_dialogue(
                    method, method_name, tokenizer,
                    sessions, persona_facts_list, device,
                    use_nli=not args.no_nli,
                )
            else:
                results = evaluate_classification(
                    method, method_name, tokenizer,
                    classification_data, device,
                    turns_per_session=args.turns_per_session,
                )

            save_results(results, method_name, args.task, args.output_dir)

        except torch.cuda.OutOfMemoryError:
            logger.error("OOM for method %s -- skipping", method_name)
            torch.cuda.empty_cache()
            save_results(
                {"error": "OOM", "method": method_name},
                method_name, args.task, args.output_dir,
            )
        except Exception as exc:
            logger.error("Failed method %s: %s", method_name, exc, exc_info=True)
            save_results(
                {"error": str(exc), "method": method_name},
                method_name, args.task, args.output_dir,
            )
        finally:
            # Cleanup
            if "model_copy" in dir():
                del model_copy
            if "method" in dir():
                del method
            torch.cuda.empty_cache()

    # Aggregate
    aggregate_results(args.output_dir, args.task)
    logger.info("Done. All results in %s", args.output_dir)


if __name__ == "__main__":
    main()
