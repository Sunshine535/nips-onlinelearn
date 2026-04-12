"""LLM-level baselines for streaming adaptation experiments.

All 11 baselines share the BaselineMethod interface so they can be swapped in
as drop-in comparisons against StreamingParameterMemory and StreamingMirrorLoRA.
"""

import copy
import logging
import math
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

try:
    from src.streaming_memory import FisherEstimator, ReservoirBuffer, SessionBuffer
except ImportError:
    from streaming_memory import FisherEstimator, ReservoirBuffer, SessionBuffer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _default_lora_config(config: dict, rank_override: Optional[int] = None) -> LoraConfig:
    """Build a LoraConfig from a config dict."""
    return LoraConfig(
        r=rank_override or config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.0),
        target_modules=config.get("target_modules", ["q_proj", "v_proj"]),
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )


def _get_lora_params(model: nn.Module) -> List[nn.Parameter]:
    """Return all trainable LoRA parameters."""
    return [p for n, p in model.named_parameters() if p.requires_grad and "lora" in n]


def _online_sgd_update(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    params: List[nn.Parameter],
    steps: int = 3,
) -> float:
    """Run several gradient steps on NTP loss, return average loss."""
    model.train()
    total = 0.0
    for _ in range(steps):
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        optimizer.zero_grad()
        total += loss.item()
    return total / max(steps, 1)


def _ensure_batch(t: torch.Tensor) -> torch.Tensor:
    """Ensure tensor has a batch dimension."""
    if t.dim() == 1:
        return t.unsqueeze(0)
    return t


def _make_fisher_dataloader(buffer: ReservoirBuffer, device: torch.device, n: int = 50, batch_size: int = 4):
    """Build a simple iterable loader from reservoir samples for FisherEstimator."""
    if len(buffer) == 0:
        return None
    inputs, labels_list = buffer.sample(min(n, len(buffer)))
    max_len = max(inp["input_ids"].shape[-1] for inp in inputs)
    padded_ids = torch.zeros(len(inputs), max_len, dtype=torch.long)
    padded_labels = torch.full((len(inputs), max_len), -100, dtype=torch.long)
    for i, (inp, lab) in enumerate(zip(inputs, labels_list)):
        ids = inp["input_ids"].squeeze()
        lb = lab.squeeze()
        padded_ids[i, : ids.shape[-1]] = ids
        padded_labels[i, : lb.shape[-1]] = lb

    class _Loader:
        def __init__(self, ids_t, lab_t, bs):
            self.ids, self.lab, self.bs = ids_t, lab_t, bs

        def __iter__(self):
            idx = 0
            while idx < len(self.ids):
                end = min(idx + self.bs, len(self.ids))
                yield {
                    "input_ids": self.ids[idx:end].to(device),
                    "labels": self.lab[idx:end].to(device),
                    "attention_mask": (self.ids[idx:end] != 0).long().to(device),
                }
                idx = end

    return _Loader(padded_ids, padded_labels, batch_size)


# ===================================================================
# 1. FrozenBaseline
# ===================================================================

class FrozenBaseline:
    """No adaptation at all.  Lower bound on streaming performance."""

    def __init__(self, base_model: nn.Module, config: dict):
        self.model = base_model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def process_turn(self, input_ids: torch.Tensor, labels: torch.Tensor) -> dict:
        with torch.no_grad():
            outputs = self.model(input_ids=_ensure_batch(input_ids), labels=_ensure_batch(labels))
        return {"loss": outputs.loss.item()}

    def start_new_session(self):
        pass

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 256) -> torch.Tensor:
        self.model.eval()
        prev = getattr(self.model.config, "use_cache", True)
        self.model.config.use_cache = True
        try:
            out = self.model.generate(input_ids=_ensure_batch(input_ids), max_new_tokens=max_new_tokens, do_sample=False)
        finally:
            self.model.config.use_cache = prev
        return out


# ===================================================================
# 2. SingleLoRaSGD
# ===================================================================

class SingleLoRaSGD:
    """Single LoRA adapter with per-turn online SGD.  No forgetting protection."""

    def __init__(self, base_model: nn.Module, config: dict):
        lora_cfg = _default_lora_config(config)
        self.model = get_peft_model(base_model, lora_cfg)
        self._params = _get_lora_params(self.model)
        self._lr = config.get("learning_rate", 5e-4)
        self._steps = config.get("update_steps", 3)
        self._optimizer = torch.optim.AdamW(self._params, lr=self._lr)

    def process_turn(self, input_ids: torch.Tensor, labels: torch.Tensor) -> dict:
        loss = _online_sgd_update(
            self.model, _ensure_batch(input_ids), _ensure_batch(labels),
            self._optimizer, self._params, self._steps,
        )
        return {"loss": loss}

    def start_new_session(self):
        pass

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 256) -> torch.Tensor:
        self.model.eval()
        prev = getattr(self.model.config, "use_cache", True)
        self.model.config.use_cache = True
        try:
            out = self.model.generate(input_ids=_ensure_batch(input_ids), max_new_tokens=max_new_tokens, do_sample=False)
        finally:
            self.model.config.use_cache = prev
        return out


# ===================================================================
# 3. SingleLoRaEWC
# ===================================================================

class SingleLoRaEWC:
    """Single LoRA + Elastic Weight Consolidation.

    After each session, compute diagonal Fisher and penalise deviation:
        L = L_ntp + lambda * sum_i F_i * (theta_i - theta_i^old)^2
    """

    def __init__(self, base_model: nn.Module, config: dict):
        lora_cfg = _default_lora_config(config)
        self.model = get_peft_model(base_model, lora_cfg)
        self._params = _get_lora_params(self.model)
        self._lr = config.get("learning_rate", 5e-4)
        self._steps = config.get("update_steps", 3)
        self._ewc_lambda = config.get("ewc_lambda", 1.0)
        self._fisher_samples = config.get("fisher_samples", 200)
        self._optimizer = torch.optim.AdamW(self._params, lr=self._lr)

        self._fisher: Dict[str, torch.Tensor] = {}
        self._prev_params: Dict[str, torch.Tensor] = {}
        self._session_buffer = SessionBuffer()

    def _ewc_penalty(self) -> torch.Tensor:
        device = next(self.model.parameters()).device
        penalty = torch.tensor(0.0, device=device)
        if not self._fisher:
            return penalty
        for name, param in self.model.named_parameters():
            if name in self._fisher and name in self._prev_params:
                penalty = penalty + (self._fisher[name].to(device) * (param - self._prev_params[name].to(device)).pow(2)).sum()
        return penalty

    def process_turn(self, input_ids: torch.Tensor, labels: torch.Tensor) -> dict:
        input_ids = _ensure_batch(input_ids)
        labels = _ensure_batch(labels)
        self._session_buffer.add(input_ids, labels)

        self.model.train()
        total = 0.0
        for _ in range(self._steps):
            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = outputs.loss + self._ewc_lambda * self._ewc_penalty()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._params, 1.0)
            self._optimizer.step()
            self._optimizer.zero_grad()
            total += loss.item()
        return {"loss": total / max(self._steps, 1)}

    def start_new_session(self):
        # Compute Fisher over data from the session that just ended.
        if len(self._session_buffer) > 0:
            device = next(self.model.parameters()).device
            loader = _make_fisher_dataloader(
                self._to_reservoir(), device, n=self._fisher_samples,
            )
            if loader is not None:
                estimator = FisherEstimator(self.model, num_samples=self._fisher_samples)
                new_fisher = estimator.estimate(loader)
                # Accumulate: online EWC sums Fisher diagonals across tasks.
                for name, f in new_fisher.items():
                    if name in self._fisher:
                        self._fisher[name] = self._fisher[name].to(f.device) + f
                    else:
                        self._fisher[name] = f

        # Snapshot current params.
        self._prev_params = {
            n: p.data.clone().cpu()
            for n, p in self.model.named_parameters()
            if p.requires_grad and "lora" in n
        }
        self._session_buffer.clear()

    def _to_reservoir(self) -> ReservoirBuffer:
        """Convert session buffer to a transient reservoir for the Fisher estimator."""
        rb = ReservoirBuffer(max_size=len(self._session_buffer))
        for inp, lab in zip(self._session_buffer.inputs, self._session_buffer.labels):
            rb.inputs.append(inp)
            rb.labels.append(lab)
        rb.total_seen = len(rb.inputs)
        return rb

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 256) -> torch.Tensor:
        self.model.eval()
        prev = getattr(self.model.config, "use_cache", True)
        self.model.config.use_cache = True
        try:
            out = self.model.generate(input_ids=_ensure_batch(input_ids), max_new_tokens=max_new_tokens, do_sample=False)
        finally:
            self.model.config.use_cache = prev
        return out


# ===================================================================
# 4. SingleLoRaReplay
# ===================================================================

class SingleLoRaReplay:
    """Single LoRA + experience replay from a reservoir buffer."""

    def __init__(self, base_model: nn.Module, config: dict):
        lora_cfg = _default_lora_config(config)
        self.model = get_peft_model(base_model, lora_cfg)
        self._params = _get_lora_params(self.model)
        self._lr = config.get("learning_rate", 5e-4)
        self._steps = config.get("update_steps", 3)
        self._replay_ratio = config.get("replay_ratio", 0.5)
        self._reservoir = ReservoirBuffer(max_size=config.get("reservoir_size", 5000))
        self._optimizer = torch.optim.AdamW(self._params, lr=self._lr)

    def process_turn(self, input_ids: torch.Tensor, labels: torch.Tensor) -> dict:
        input_ids = _ensure_batch(input_ids)
        labels = _ensure_batch(labels)
        device = next(self.model.parameters()).device

        self.model.train()
        total = 0.0
        for _ in range(self._steps):
            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            # Mix in replay if buffer is non-empty.
            if len(self._reservoir) > 0 and random.random() < self._replay_ratio:
                r_inputs, r_labels = self._reservoir.sample(1)
                r_ids = r_inputs[0]["input_ids"].to(device)
                r_lab = r_labels[0].to(device)
                r_ids = _ensure_batch(r_ids)
                r_lab = _ensure_batch(r_lab)
                r_out = self.model(input_ids=r_ids, labels=r_lab)
                loss = 0.5 * loss + 0.5 * r_out.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._params, 1.0)
            self._optimizer.step()
            self._optimizer.zero_grad()
            total += loss.item()

        self._reservoir.add(input_ids.squeeze(0), labels.squeeze(0))
        return {"loss": total / max(self._steps, 1)}

    def start_new_session(self):
        pass

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 256) -> torch.Tensor:
        self.model.eval()
        prev = getattr(self.model.config, "use_cache", True)
        self.model.config.use_cache = True
        try:
            out = self.model.generate(input_ids=_ensure_batch(input_ids), max_new_tokens=max_new_tokens, do_sample=False)
        finally:
            self.model.config.use_cache = prev
        return out


# ===================================================================
# 5. SingleLoRaEWCReplay
# ===================================================================

class SingleLoRaEWCReplay:
    """Single LoRA + EWC + experience replay combined."""

    def __init__(self, base_model: nn.Module, config: dict):
        lora_cfg = _default_lora_config(config)
        self.model = get_peft_model(base_model, lora_cfg)
        self._params = _get_lora_params(self.model)
        self._lr = config.get("learning_rate", 5e-4)
        self._steps = config.get("update_steps", 3)
        self._ewc_lambda = config.get("ewc_lambda", 1.0)
        self._replay_ratio = config.get("replay_ratio", 0.5)
        self._fisher_samples = config.get("fisher_samples", 200)
        self._reservoir = ReservoirBuffer(max_size=config.get("reservoir_size", 5000))
        self._optimizer = torch.optim.AdamW(self._params, lr=self._lr)

        self._fisher: Dict[str, torch.Tensor] = {}
        self._prev_params: Dict[str, torch.Tensor] = {}
        self._session_buffer = SessionBuffer()

    def _ewc_penalty(self) -> torch.Tensor:
        device = next(self.model.parameters()).device
        penalty = torch.tensor(0.0, device=device)
        if not self._fisher:
            return penalty
        for name, param in self.model.named_parameters():
            if name in self._fisher and name in self._prev_params:
                penalty = penalty + (self._fisher[name].to(device) * (param - self._prev_params[name].to(device)).pow(2)).sum()
        return penalty

    def process_turn(self, input_ids: torch.Tensor, labels: torch.Tensor) -> dict:
        input_ids = _ensure_batch(input_ids)
        labels = _ensure_batch(labels)
        device = next(self.model.parameters()).device
        self._session_buffer.add(input_ids, labels)

        self.model.train()
        total = 0.0
        for _ in range(self._steps):
            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = outputs.loss + self._ewc_lambda * self._ewc_penalty()

            if len(self._reservoir) > 0 and random.random() < self._replay_ratio:
                r_inputs, r_labels = self._reservoir.sample(1)
                r_ids = _ensure_batch(r_inputs[0]["input_ids"].to(device))
                r_lab = _ensure_batch(r_labels[0].to(device))
                r_out = self.model(input_ids=r_ids, labels=r_lab)
                loss = 0.5 * loss + 0.5 * r_out.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._params, 1.0)
            self._optimizer.step()
            self._optimizer.zero_grad()
            total += loss.item()

        self._reservoir.add(input_ids.squeeze(0), labels.squeeze(0))
        return {"loss": total / max(self._steps, 1)}

    def start_new_session(self):
        if len(self._session_buffer) > 0:
            device = next(self.model.parameters()).device
            rb = ReservoirBuffer(max_size=len(self._session_buffer))
            for inp, lab in zip(self._session_buffer.inputs, self._session_buffer.labels):
                rb.inputs.append(inp)
                rb.labels.append(lab)
            rb.total_seen = len(rb.inputs)
            loader = _make_fisher_dataloader(rb, device, n=self._fisher_samples)
            if loader is not None:
                estimator = FisherEstimator(self.model, num_samples=self._fisher_samples)
                new_fisher = estimator.estimate(loader)
                for name, f in new_fisher.items():
                    if name in self._fisher:
                        self._fisher[name] = self._fisher[name].to(f.device) + f
                    else:
                        self._fisher[name] = f

        self._prev_params = {
            n: p.data.clone().cpu()
            for n, p in self.model.named_parameters()
            if p.requires_grad and "lora" in n
        }
        self._session_buffer.clear()

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 256) -> torch.Tensor:
        self.model.eval()
        prev = getattr(self.model.config, "use_cache", True)
        self.model.config.use_cache = True
        try:
            out = self.model.generate(input_ids=_ensure_batch(input_ids), max_new_tokens=max_new_tokens, do_sample=False)
        finally:
            self.model.config.use_cache = prev
        return out


# ===================================================================
# 6. ParamMatchedLoRA
# ===================================================================

class ParamMatchedLoRA:
    """Single LoRA with rank=80 (~65M params).

    Tests whether dual-timescale gains come purely from extra parameters.
    """

    def __init__(self, base_model: nn.Module, config: dict):
        rank = config.get("matched_rank", 80)
        lora_cfg = _default_lora_config(config, rank_override=rank)
        self.model = get_peft_model(base_model, lora_cfg)
        self._params = _get_lora_params(self.model)
        self._lr = config.get("learning_rate", 5e-4)
        self._steps = config.get("update_steps", 3)
        self._optimizer = torch.optim.AdamW(self._params, lr=self._lr)

        total_p = sum(p.numel() for p in self._params)
        logger.info("ParamMatchedLoRA: rank=%d, trainable_params=%d", rank, total_p)

    def process_turn(self, input_ids: torch.Tensor, labels: torch.Tensor) -> dict:
        loss = _online_sgd_update(
            self.model, _ensure_batch(input_ids), _ensure_batch(labels),
            self._optimizer, self._params, self._steps,
        )
        return {"loss": loss}

    def start_new_session(self):
        pass

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 256) -> torch.Tensor:
        self.model.eval()
        prev = getattr(self.model.config, "use_cache", True)
        self.model.config.use_cache = True
        try:
            out = self.model.generate(input_ids=_ensure_batch(input_ids), max_new_tokens=max_new_tokens, do_sample=False)
        finally:
            self.model.config.use_cache = prev
        return out


# ===================================================================
# 7. DualLoRaEMA
# ===================================================================

class DualLoRaEMA:
    """Two LoRA adapters (fast + slow).  Slow updated via EMA of fast at session end.

    theta_slow = beta * theta_slow + (1-beta) * theta_fast
    """

    def __init__(self, base_model: nn.Module, config: dict):
        fast_cfg = _default_lora_config(config)
        self.model = get_peft_model(base_model, fast_cfg, adapter_name="fast")
        slow_cfg = _default_lora_config(config)
        self.model.add_adapter("slow", slow_cfg)
        self.model.set_adapter("fast")

        self._ema_beta = config.get("ema_beta", 0.9)
        self._lr = config.get("learning_rate", 5e-4)
        self._steps = config.get("update_steps", 3)
        self._fast_params = [p for n, p in self.model.named_parameters() if p.requires_grad and "fast" in n and "lora" in n]
        if not self._fast_params:
            self._fast_params = _get_lora_params(self.model)
        self._optimizer = torch.optim.AdamW(self._fast_params, lr=self._lr)

    def process_turn(self, input_ids: torch.Tensor, labels: torch.Tensor) -> dict:
        self.model.set_adapter("fast")
        try:
            self.model.set_adapter(["fast", "slow"])
        except (TypeError, ValueError):
            self.model.set_adapter("fast")

        loss = _online_sgd_update(
            self.model, _ensure_batch(input_ids), _ensure_batch(labels),
            self._optimizer, self._fast_params, self._steps,
        )
        return {"loss": loss}

    def start_new_session(self):
        """EMA merge at session boundary, then zero-init fast adapter."""
        beta = self._ema_beta
        fast_state = {}
        slow_state = {}
        for name, param in self.model.named_parameters():
            if "lora" not in name:
                continue
            if "fast" in name:
                key = name.replace("fast", "slow")
                fast_state[key] = param.data.clone()
            elif "slow" in name:
                slow_state[name] = param

        for key, slow_param in slow_state.items():
            if key in fast_state:
                slow_param.data.mul_(beta).add_(fast_state[key], alpha=1.0 - beta)

        # Zero-init fast adapter.
        for name, param in self.model.named_parameters():
            if "fast" in name and "lora" in name:
                param.data.zero_()

        # Reset optimizer.
        self._optimizer = torch.optim.AdamW(self._fast_params, lr=self._lr)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 256) -> torch.Tensor:
        self.model.eval()
        try:
            self.model.set_adapter(["fast", "slow"])
        except (TypeError, ValueError):
            self.model.set_adapter("fast")
        prev = getattr(self.model.config, "use_cache", True)
        self.model.config.use_cache = True
        try:
            out = self.model.generate(input_ids=_ensure_batch(input_ids), max_new_tokens=max_new_tokens, do_sample=False)
        finally:
            self.model.config.use_cache = prev
        return out


# ===================================================================
# 8. DualLoRaPeriodicAvg
# ===================================================================

class DualLoRaPeriodicAvg:
    """Two LoRA adapters.  Slow = periodic uniform average of fast.

    Every K turns, average fast weights into slow.
    """

    def __init__(self, base_model: nn.Module, config: dict):
        fast_cfg = _default_lora_config(config)
        self.model = get_peft_model(base_model, fast_cfg, adapter_name="fast")
        slow_cfg = _default_lora_config(config)
        self.model.add_adapter("slow", slow_cfg)
        self.model.set_adapter("fast")

        self._avg_period = config.get("avg_period", 10)
        self._lr = config.get("learning_rate", 5e-4)
        self._steps = config.get("update_steps", 3)
        self._turn_count = 0

        self._fast_params = [p for n, p in self.model.named_parameters() if p.requires_grad and "fast" in n and "lora" in n]
        if not self._fast_params:
            self._fast_params = _get_lora_params(self.model)
        self._optimizer = torch.optim.AdamW(self._fast_params, lr=self._lr)

    def _merge_into_slow(self):
        fast_state = {}
        for name, param in self.model.named_parameters():
            if "fast" in name and "lora" in name:
                key = name.replace("fast", "slow")
                fast_state[key] = param.data.clone()
        for name, param in self.model.named_parameters():
            if "slow" in name and "lora" in name and name in fast_state:
                param.data.copy_(0.5 * param.data + 0.5 * fast_state[name])

    def process_turn(self, input_ids: torch.Tensor, labels: torch.Tensor) -> dict:
        self.model.set_adapter("fast")
        try:
            self.model.set_adapter(["fast", "slow"])
        except (TypeError, ValueError):
            self.model.set_adapter("fast")

        loss = _online_sgd_update(
            self.model, _ensure_batch(input_ids), _ensure_batch(labels),
            self._optimizer, self._fast_params, self._steps,
        )
        self._turn_count += 1
        if self._turn_count % self._avg_period == 0:
            self._merge_into_slow()
        return {"loss": loss}

    def start_new_session(self):
        # Merge remaining and zero-init fast.
        self._merge_into_slow()
        for name, param in self.model.named_parameters():
            if "fast" in name and "lora" in name:
                param.data.zero_()
        self._turn_count = 0
        self._optimizer = torch.optim.AdamW(self._fast_params, lr=self._lr)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 256) -> torch.Tensor:
        self.model.eval()
        try:
            self.model.set_adapter(["fast", "slow"])
        except (TypeError, ValueError):
            self.model.set_adapter("fast")
        prev = getattr(self.model.config, "use_cache", True)
        self.model.config.use_cache = True
        try:
            out = self.model.generate(input_ids=_ensure_batch(input_ids), max_new_tokens=max_new_tokens, do_sample=False)
        finally:
            self.model.config.use_cache = prev
        return out


# ===================================================================
# 9. DualLoRaHeuristic
# ===================================================================

class DualLoRaHeuristic:
    """Two LoRA adapters with heuristic consolidation.

    At session end, merge the top-K parameters (by gradient magnitude) from
    the fast adapter into the slow adapter.
    """

    def __init__(self, base_model: nn.Module, config: dict):
        fast_cfg = _default_lora_config(config)
        self.model = get_peft_model(base_model, fast_cfg, adapter_name="fast")
        slow_cfg = _default_lora_config(config)
        self.model.add_adapter("slow", slow_cfg)
        self.model.set_adapter("fast")

        self._top_k_frac = config.get("top_k_frac", 0.3)
        self._lr = config.get("learning_rate", 5e-4)
        self._steps = config.get("update_steps", 3)

        self._fast_params = [p for n, p in self.model.named_parameters() if p.requires_grad and "fast" in n and "lora" in n]
        if not self._fast_params:
            self._fast_params = _get_lora_params(self.model)
        self._optimizer = torch.optim.AdamW(self._fast_params, lr=self._lr)

        # Accumulate gradient magnitudes over the session.
        self._grad_accum: Dict[str, torch.Tensor] = {}
        self._grad_count = 0

    def process_turn(self, input_ids: torch.Tensor, labels: torch.Tensor) -> dict:
        self.model.set_adapter("fast")
        try:
            self.model.set_adapter(["fast", "slow"])
        except (TypeError, ValueError):
            self.model.set_adapter("fast")

        loss = _online_sgd_update(
            self.model, _ensure_batch(input_ids), _ensure_batch(labels),
            self._optimizer, self._fast_params, self._steps,
        )

        # Accumulate gradient magnitudes.
        for name, param in self.model.named_parameters():
            if "fast" in name and "lora" in name and param.grad is not None:
                if name not in self._grad_accum:
                    self._grad_accum[name] = param.grad.data.abs().clone()
                else:
                    self._grad_accum[name] += param.grad.data.abs()
        self._grad_count += 1

        return {"loss": loss}

    def start_new_session(self):
        """Merge top-K parameters by gradient magnitude from fast into slow."""
        if self._grad_count == 0:
            self._reset_fast()
            return

        # Flatten all gradient magnitudes to find the top-K threshold.
        all_grads = []
        for name, g in self._grad_accum.items():
            avg = g / self._grad_count
            all_grads.append(avg.flatten())
        if not all_grads:
            self._reset_fast()
            return

        all_flat = torch.cat(all_grads)
        k = max(1, int(self._top_k_frac * all_flat.numel()))
        threshold = torch.topk(all_flat, k).values[-1].item()

        # Apply selective merge.
        fast_state = {}
        for name, param in self.model.named_parameters():
            if "fast" in name and "lora" in name:
                key = name.replace("fast", "slow")
                fast_state[key] = (name, param.data.clone())

        for name, param in self.model.named_parameters():
            if "slow" in name and "lora" in name and name in fast_state:
                fast_name, fast_data = fast_state[name]
                if fast_name in self._grad_accum:
                    avg_grad = self._grad_accum[fast_name] / self._grad_count
                    mask = (avg_grad >= threshold).float()
                    param.data.copy_(param.data * (1.0 - mask) + fast_data * mask)

        self._reset_fast()

    def _reset_fast(self):
        for name, param in self.model.named_parameters():
            if "fast" in name and "lora" in name:
                param.data.zero_()
        self._grad_accum.clear()
        self._grad_count = 0
        self._optimizer = torch.optim.AdamW(self._fast_params, lr=self._lr)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 256) -> torch.Tensor:
        self.model.eval()
        try:
            self.model.set_adapter(["fast", "slow"])
        except (TypeError, ValueError):
            self.model.set_adapter("fast")
        prev = getattr(self.model.config, "use_cache", True)
        self.model.config.use_cache = True
        try:
            out = self.model.generate(input_ids=_ensure_batch(input_ids), max_new_tokens=max_new_tokens, do_sample=False)
        finally:
            self.model.config.use_cache = prev
        return out


# ===================================================================
# 10. RetrievalAugmented
# ===================================================================

class RetrievalAugmented:
    """No parameter adaptation.  Prepend top-k similar turns from history via TF-IDF.

    Stores all past (input_ids, labels) in a retrieval buffer and uses simple
    token-frequency overlap as the similarity metric.
    """

    def __init__(self, base_model: nn.Module, config: dict):
        self.model = base_model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self._top_k = config.get("retrieval_k", 3)
        self._max_context = config.get("max_context_tokens", 1024)
        self._buffer: List[torch.Tensor] = []  # list of 1-d input_ids on cpu

    @staticmethod
    def _tfidf_vector(ids: torch.Tensor) -> Counter:
        """Simple term-frequency counter over token ids."""
        return Counter(ids.flatten().tolist())

    @staticmethod
    def _cosine_tf(a: Counter, b: Counter) -> float:
        """Cosine similarity between two Counter-based TF vectors."""
        common = set(a.keys()) & set(b.keys())
        if not common:
            return 0.0
        dot = sum(a[k] * b[k] for k in common)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return dot / (na * nb)

    def _retrieve(self, query_ids: torch.Tensor) -> List[torch.Tensor]:
        """Return top-k most similar past turns by TF-IDF cosine."""
        if not self._buffer:
            return []
        q_tf = self._tfidf_vector(query_ids)
        scored = []
        for idx, buf_ids in enumerate(self._buffer):
            sim = self._cosine_tf(q_tf, self._tfidf_vector(buf_ids))
            scored.append((sim, idx))
        scored.sort(key=lambda x: -x[0])
        results = []
        total_len = 0
        for sim, idx in scored[: self._top_k]:
            ids = self._buffer[idx]
            if total_len + ids.shape[-1] > self._max_context:
                break
            results.append(ids)
            total_len += ids.shape[-1]
        return results

    def process_turn(self, input_ids: torch.Tensor, labels: torch.Tensor) -> dict:
        input_ids = _ensure_batch(input_ids)
        labels = _ensure_batch(labels)
        device = next(self.model.parameters()).device

        # Retrieve and prepend context.
        retrieved = self._retrieve(input_ids)
        if retrieved:
            ctx = torch.cat([r.to(device) for r in retrieved], dim=-1)
            ctx = _ensure_batch(ctx)
            aug_ids = torch.cat([ctx, input_ids.to(device)], dim=-1)
            # Build labels: -100 for context, real labels for the turn.
            ctx_labels = torch.full((1, ctx.shape[-1]), -100, dtype=torch.long, device=device)
            aug_labels = torch.cat([ctx_labels, labels.to(device)], dim=-1)
        else:
            aug_ids = input_ids.to(device)
            aug_labels = labels.to(device)

        with torch.no_grad():
            outputs = self.model(input_ids=aug_ids, labels=aug_labels)

        # Store turn in buffer.
        self._buffer.append(input_ids.detach().cpu().squeeze(0))
        return {"loss": outputs.loss.item()}

    def start_new_session(self):
        # Keep the retrieval buffer across sessions.
        pass

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 256) -> torch.Tensor:
        self.model.eval()
        input_ids = _ensure_batch(input_ids)
        device = next(self.model.parameters()).device

        retrieved = self._retrieve(input_ids)
        if retrieved:
            ctx = torch.cat([r.to(device) for r in retrieved], dim=-1)
            ctx = _ensure_batch(ctx)
            aug_ids = torch.cat([ctx, input_ids.to(device)], dim=-1)
        else:
            aug_ids = input_ids.to(device)

        prev = getattr(self.model.config, "use_cache", True)
        self.model.config.use_cache = True
        try:
            out = self.model.generate(input_ids=aug_ids, max_new_tokens=max_new_tokens, do_sample=False)
        finally:
            self.model.config.use_cache = prev
        return out


# ===================================================================
# 11. SDFTBaseline
# ===================================================================

class SDFTBaseline:
    """Self-Distillation Fine-Tuning approximation.

    Single LoRA.  At session end, use the full session data as an in-context
    demonstration to produce teacher logits, then distill from the
    demo-conditioned model into the adapter via KL divergence.
    """

    def __init__(self, base_model: nn.Module, config: dict):
        lora_cfg = _default_lora_config(config)
        self.model = get_peft_model(base_model, lora_cfg)
        self._params = _get_lora_params(self.model)
        self._lr = config.get("learning_rate", 5e-4)
        self._steps = config.get("update_steps", 3)
        self._distill_epochs = config.get("distill_epochs", 2)
        self._distill_lr = config.get("distill_lr", 1e-4)
        self._max_demo_tokens = config.get("max_demo_tokens", 512)
        self._optimizer = torch.optim.AdamW(self._params, lr=self._lr)
        self._session_buffer = SessionBuffer()

    def process_turn(self, input_ids: torch.Tensor, labels: torch.Tensor) -> dict:
        input_ids = _ensure_batch(input_ids)
        labels = _ensure_batch(labels)
        self._session_buffer.add(input_ids, labels)

        loss = _online_sgd_update(
            self.model, input_ids, labels,
            self._optimizer, self._params, self._steps,
        )
        return {"loss": loss}

    def start_new_session(self):
        """Self-distillation: use session data as demo context to produce teacher logits."""
        if len(self._session_buffer) == 0:
            return

        device = next(self.model.parameters()).device
        inputs, labels_list = self._session_buffer.get_all()

        # Build a demonstration prefix from session data (truncated).
        demo_pieces = []
        demo_len = 0
        for inp in inputs:
            ids = inp["input_ids"].squeeze()
            if demo_len + ids.shape[-1] > self._max_demo_tokens:
                break
            demo_pieces.append(ids)
            demo_len += ids.shape[-1]

        if not demo_pieces:
            self._session_buffer.clear()
            return

        demo_prefix = torch.cat(demo_pieces, dim=-1).to(device).unsqueeze(0)

        # Cache teacher logits: demo-conditioned model on each turn.
        teacher_cache = []
        self.model.eval()
        with torch.no_grad():
            for inp in inputs:
                turn_ids = inp["input_ids"].to(device)
                turn_ids = _ensure_batch(turn_ids)
                aug_ids = torch.cat([demo_prefix, turn_ids], dim=-1)
                out = self.model(input_ids=aug_ids)
                # Teacher logits over the turn portion only.
                turn_len = turn_ids.shape[-1]
                teacher_logits = out.logits[:, -turn_len:, :]
                teacher_cache.append({"input_ids": turn_ids.cpu(), "teacher_logits": teacher_logits.cpu()})

        # Distillation: train adapter to match teacher logits (without demo prefix).
        self.model.train()
        distill_opt = torch.optim.AdamW(self._params, lr=self._distill_lr)
        for _ in range(self._distill_epochs):
            for cached in teacher_cache:
                turn_ids = cached["input_ids"].to(device)
                t_logits = cached["teacher_logits"].to(device)

                s_out = self.model(input_ids=turn_ids)
                s_logits = s_out.logits
                min_len = min(s_logits.shape[1], t_logits.shape[1])
                kl = F.kl_div(
                    F.log_softmax(s_logits[:, :min_len, :], dim=-1),
                    F.softmax(t_logits[:, :min_len, :], dim=-1),
                    reduction="batchmean",
                )
                kl.backward()
                torch.nn.utils.clip_grad_norm_(self._params, 1.0)
                distill_opt.step()
                distill_opt.zero_grad()

        # Reset per-session state.  Keep adapter weights.
        self._session_buffer.clear()
        self._optimizer = torch.optim.AdamW(self._params, lr=self._lr)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 256) -> torch.Tensor:
        self.model.eval()
        prev = getattr(self.model.config, "use_cache", True)
        self.model.config.use_cache = True
        try:
            out = self.model.generate(input_ids=_ensure_batch(input_ids), max_new_tokens=max_new_tokens, do_sample=False)
        finally:
            self.model.config.use_cache = prev
        return out


# ===================================================================
# Factory
# ===================================================================

_REGISTRY: Dict[str, type] = {
    "frozen": FrozenBaseline,
    "single_lora_sgd": SingleLoRaSGD,
    "single_lora_ewc": SingleLoRaEWC,
    "single_lora_replay": SingleLoRaReplay,
    "single_lora_ewc_replay": SingleLoRaEWCReplay,
    "param_matched_lora": ParamMatchedLoRA,
    "dual_lora_ema": DualLoRaEMA,
    "dual_lora_periodic_avg": DualLoRaPeriodicAvg,
    "dual_lora_heuristic": DualLoRaHeuristic,
    "retrieval_augmented": RetrievalAugmented,
    "sdft": SDFTBaseline,
}


def create_baseline(name: str, base_model: nn.Module, config: dict):
    """Factory for creating baseline methods by name.

    Args:
        name: One of: frozen, single_lora_sgd, single_lora_ewc,
              single_lora_replay, single_lora_ewc_replay, param_matched_lora,
              dual_lora_ema, dual_lora_periodic_avg, dual_lora_heuristic,
              retrieval_augmented, sdft.
        base_model: The pretrained LLM (before any PEFT adapters).
        config: Dict with LoRA and method-specific hyperparameters.

    Returns:
        An instance satisfying the BaselineMethod interface.
    """
    key = name.lower().replace("-", "_")
    if key not in _REGISTRY:
        raise ValueError(f"Unknown baseline '{name}'. Available: {sorted(_REGISTRY.keys())}")
    return _REGISTRY[key](base_model, config)
