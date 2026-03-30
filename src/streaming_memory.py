"""Two-Timescale Residual LoRA with Behavioral Distillation Consolidation.

Core method: zero-init WM-LoRA (residual over persistent LT-LoRA) with session-end
KL-distillation consolidation. Split objective: KL on current session data (behavior
transfer) + NTP replay on reservoir (retention)."""

import copy
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

logger = logging.getLogger(__name__)


@dataclass
class ReservoirBuffer:
    """Reservoir-sampled replay buffer with uniform replacement."""

    inputs: List[Dict[str, torch.Tensor]] = field(default_factory=list)
    labels: List[torch.Tensor] = field(default_factory=list)
    max_size: int = 5000
    total_seen: int = 0

    def add(self, input_ids: torch.Tensor, labels: torch.Tensor):
        self.total_seen += 1
        item_input = {"input_ids": input_ids.detach().cpu()}
        item_label = labels.detach().cpu()

        if len(self.inputs) < self.max_size:
            self.inputs.append(item_input)
            self.labels.append(item_label)
        else:
            idx = random.randint(0, self.total_seen - 1)
            if idx < self.max_size:
                self.inputs[idx] = item_input
                self.labels[idx] = item_label

    def sample(self, n: int) -> Tuple[List[Dict], List[torch.Tensor]]:
        if len(self.inputs) == 0:
            return [], []
        indices = random.choices(range(len(self.inputs)), k=min(n, len(self.inputs)))
        return [self.inputs[i] for i in indices], [self.labels[i] for i in indices]

    def merge_session(self, session_buffer: "SessionBuffer"):
        for inp, lab in zip(session_buffer.inputs, session_buffer.labels):
            self.add(inp["input_ids"], lab)

    def __len__(self):
        return len(self.inputs)


@dataclass
class SessionBuffer:
    """Current-session data buffer for KL distillation consolidation."""

    inputs: List[Dict[str, torch.Tensor]] = field(default_factory=list)
    labels: List[torch.Tensor] = field(default_factory=list)

    def add(self, input_ids: torch.Tensor, labels: torch.Tensor):
        self.inputs.append({"input_ids": input_ids.detach().cpu()})
        self.labels.append(labels.detach().cpu())

    def get_all(self) -> Tuple[List[Dict], List[torch.Tensor]]:
        return self.inputs, self.labels

    def clear(self):
        self.inputs.clear()
        self.labels.clear()

    def __len__(self):
        return len(self.inputs)


class FisherEstimator:
    """Diagonal Fisher for optional trust-region stabilizer."""

    def __init__(self, model: nn.Module, num_samples: int = 200):
        self.model = model
        self.num_samples = num_samples

    def estimate(self, dataloader) -> Dict[str, torch.Tensor]:
        fisher = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and "lora" in name:
                fisher[name] = torch.zeros_like(param.data)

        self.model.eval()
        count = 0
        for batch in dataloader:
            if count >= self.num_samples:
                break
            batch = {k: v.to(next(self.model.parameters()).device) for k, v in batch.items()}
            self.model.zero_grad()
            with torch.enable_grad():
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                loss.backward()

            for name, param in self.model.named_parameters():
                if name in fisher and param.grad is not None:
                    fisher[name] += param.grad.data.pow(2)
            count += 1

        for name in fisher:
            fisher[name] /= max(count, 1)
        return fisher


class WorkingMemoryLoRA:
    """Fast-updating residual LoRA, zero-initialized each session."""

    def __init__(self, config: dict):
        self.config = config
        self.lora_config = LoraConfig(
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules=config["target_modules"],
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        self.update_count = 0
        self.max_turns = config.get("max_turns_before_consolidation", 10)

    def needs_consolidation(self) -> bool:
        return self.update_count >= self.max_turns

    def zero_init(self, model: PeftModel):
        """Zero-initialize WM-LoRA for new session (residual design)."""
        for name, param in model.named_parameters():
            if "working" in name and "lora" in name:
                param.data.zero_()
        self.update_count = 0

    def online_update(
        self, model: PeftModel, input_ids: torch.Tensor, labels: torch.Tensor,
        lr: float = 5e-4, steps: int = 3,
    ) -> float:
        """Per-turn NTP gradient steps on WM-LoRA only."""
        model.train()
        wm_params = [p for n, p in model.named_parameters() if p.requires_grad and "working" in n and "lora" in n]
        if not wm_params:
            wm_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora" in n]
        optimizer = torch.optim.AdamW(wm_params, lr=lr)

        total_loss = 0.0
        for _ in range(steps):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(wm_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        self.update_count += 1
        return total_loss / steps

    def reset(self):
        self.update_count = 0


class LongTermMemoryLoRA:
    """Persistent LoRA consolidating across sessions via behavioral distillation."""

    def __init__(self, config: dict):
        self.config = config
        self.lora_config = LoraConfig(
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules=config["target_modules"],
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        self.fisher: Dict[str, torch.Tensor] = {}
        self.prev_params: Dict[str, torch.Tensor] = {}

    def store_checkpoint(self, model: nn.Module):
        self.prev_params = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad and "longterm" in name and "lora" in name
        }

    def consolidate(
        self,
        model: PeftModel,
        session_buffer: SessionBuffer,
        reservoir: ReservoirBuffer,
        beta: float = 1.0,
        gamma: float = 0.0,
        epochs: int = 3,
        lr: float = 1e-4,
        fisher_estimator: Optional[FisherEstimator] = None,
    ) -> float:
        """Behavioral distillation consolidation.

        L = L_ntp(θ_LT'; B_all) + β · D_KL(p_{LT+WM} || p_{LT'}) + γ · Fisher_trust_region
        """
        device = next(model.parameters()).device

        if gamma > 0 and fisher_estimator is not None:
            dataloader = self._make_dataloader(reservoir, device, batch_size=4)
            if dataloader:
                self.fisher = fisher_estimator.estimate(dataloader)
        self.store_checkpoint(model)

        teacher_logits_cache = self._cache_teacher_logits(model, session_buffer, device)

        model.set_adapter("longterm")
        model.train()

        lt_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora" in n]
        if not lt_params:
            logger.warning("No trainable LT params found")
            return 0.0
        optimizer = torch.optim.AdamW(lt_params, lr=lr)

        total_loss = 0.0
        steps = 0

        for epoch in range(epochs):
            # --- Retention: NTP on reservoir ---
            if len(reservoir) > 0:
                replay_inputs, replay_labels = reservoir.sample(min(32, len(reservoir)))
                for inp, lab in zip(replay_inputs, replay_labels):
                    input_ids = inp["input_ids"].to(device)
                    if input_ids.dim() == 1:
                        input_ids = input_ids.unsqueeze(0)
                    labels_t = lab.to(device)
                    if labels_t.dim() == 1:
                        labels_t = labels_t.unsqueeze(0)

                    outputs = model(input_ids=input_ids, labels=labels_t)
                    ntp_loss = outputs.loss

                    trust_region = torch.tensor(0.0, device=device)
                    if gamma > 0:
                        trust_region = self._fisher_trust_region(model, gamma)

                    loss = ntp_loss + trust_region
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(lt_params, 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    total_loss += loss.item()
                    steps += 1

            # --- Behavior transfer: KL on session data ---
            if teacher_logits_cache:
                for cached in teacher_logits_cache:
                    input_ids = cached["input_ids"].to(device)
                    teacher_log = cached["teacher_logits"].to(device)

                    with torch.no_grad():
                        pass
                    student_out = model(input_ids=input_ids)
                    student_logits = student_out.logits

                    min_len = min(student_logits.shape[1], teacher_log.shape[1])
                    s_logits = student_logits[:, :min_len, :]
                    t_logits = teacher_log[:, :min_len, :]

                    kl_loss = F.kl_div(
                        F.log_softmax(s_logits, dim=-1),
                        F.softmax(t_logits, dim=-1),
                        reduction="batchmean",
                    )

                    loss = beta * kl_loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(lt_params, 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    total_loss += loss.item()
                    steps += 1

        return total_loss / max(steps, 1)

    def _cache_teacher_logits(
        self, model: PeftModel, session_buffer: SessionBuffer, device: torch.device
    ) -> list:
        """Cache teacher (LT+WM) logits on session data before switching to LT-only."""
        cache = []
        if len(session_buffer) == 0:
            return cache

        model.eval()
        try:
            model.set_adapter(["working", "longterm"])
        except (TypeError, ValueError):
            model.set_adapter("working")

        inputs, labels = session_buffer.get_all()
        for inp, lab in zip(inputs, labels):
            input_ids = inp["input_ids"].to(device)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)

            with torch.no_grad():
                outputs = model(input_ids=input_ids)
                cache.append({
                    "input_ids": input_ids.cpu(),
                    "teacher_logits": outputs.logits.cpu(),
                })

        return cache

    def _fisher_trust_region(self, model: nn.Module, gamma: float) -> torch.Tensor:
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if name in self.fisher and name in self.prev_params:
                loss += (self.fisher[name] * (param - self.prev_params[name]).pow(2)).sum()
        return gamma * loss

    def _make_dataloader(self, reservoir: ReservoirBuffer, device, batch_size=4):
        if len(reservoir) == 0:
            return None
        inputs, labels = reservoir.sample(min(50, len(reservoir)))
        max_len = max(inp["input_ids"].shape[-1] for inp in inputs)
        padded_ids = torch.zeros(len(inputs), max_len, dtype=torch.long)
        padded_labels = torch.full((len(inputs), max_len), -100, dtype=torch.long)
        for i, (inp, lab) in enumerate(zip(inputs, labels)):
            ids = inp["input_ids"].squeeze()
            l = lab.squeeze()
            padded_ids[i, : ids.shape[-1]] = ids
            padded_labels[i, : l.shape[-1]] = l

        class SimpleLoader:
            def __init__(self, ids_t, labels_t, bs):
                self.ids = ids_t
                self.labels = labels_t
                self.bs = bs
                self.idx = 0

            def __iter__(self):
                self.idx = 0
                return self

            def __next__(self):
                if self.idx >= len(self.ids):
                    raise StopIteration
                end = min(self.idx + self.bs, len(self.ids))
                batch = {
                    "input_ids": self.ids[self.idx:end].to(device),
                    "labels": self.labels[self.idx:end].to(device),
                    "attention_mask": (self.ids[self.idx:end] != 0).long().to(device),
                }
                self.idx = end
                return batch

        return SimpleLoader(padded_ids, padded_labels, batch_size)


class StreamingParameterMemory:
    """Two-timescale residual LoRA with behavioral distillation consolidation."""

    def __init__(
        self,
        base_model: nn.Module,
        working_config: dict,
        longterm_config: dict,
        reservoir_size: int = 5000,
        beta: float = 1.0,
        gamma: float = 0.0,
    ):
        self.base_model = base_model
        self.working = WorkingMemoryLoRA(working_config)
        self.longterm = LongTermMemoryLoRA(longterm_config)
        self.reservoir = ReservoirBuffer(max_size=reservoir_size)
        self.session_buffer = SessionBuffer()
        self.beta = beta
        self.gamma = gamma
        self.session_id = 0
        self.total_turns = 0

        self.model = get_peft_model(base_model, self.working.lora_config, adapter_name="working")
        self.model.add_adapter("longterm", self.longterm.lora_config)
        self.model.set_adapter("working")

    def process_turn(self, input_ids: torch.Tensor, labels: torch.Tensor) -> dict:
        """Process one conversation turn: update WM, store in session buffer."""
        self.model.set_adapter("working")
        try:
            self.model.set_adapter(["working", "longterm"])
        except (TypeError, ValueError):
            self.model.set_adapter("working")

        loss = self.working.online_update(
            self.model, input_ids, labels,
            lr=self.working.config.get("learning_rate", 5e-4),
            steps=self.working.config.get("online_update_steps", 3),
        )

        self.session_buffer.add(input_ids, labels)
        self.total_turns += 1

        return {"loss": loss, "total_turns": self.total_turns, "consolidated": False}

    def consolidate_session(self) -> float:
        """Session-end consolidation via behavioral distillation."""
        if len(self.session_buffer) == 0:
            logger.info("No session data to consolidate")
            return 0.0

        logger.info(
            "Consolidating session %d (%d turns, reservoir %d)",
            self.session_id, len(self.session_buffer), len(self.reservoir),
        )

        fisher_est = None
        if self.gamma > 0:
            fisher_est = FisherEstimator(self.model, num_samples=50)

        loss = self.longterm.consolidate(
            model=self.model,
            session_buffer=self.session_buffer,
            reservoir=self.reservoir,
            beta=self.beta,
            gamma=self.gamma,
            epochs=self.longterm.config.get("consolidation_epochs", 3),
            lr=self.longterm.config.get("consolidation_lr", 1e-4),
            fisher_estimator=fisher_est,
        )

        self.reservoir.merge_session(self.session_buffer)
        self.session_buffer.clear()

        self.model.set_adapter("working")
        logger.info("Consolidation complete (loss=%.4f)", loss)
        return loss

    def start_new_session(self):
        """Begin new session: zero-init WM, clear session buffer."""
        self.session_id += 1
        self.session_buffer.clear()
        self.working.zero_init(self.model)
        try:
            self.model.set_adapter(["working", "longterm"])
        except (TypeError, ValueError):
            self.model.set_adapter("working")
        logger.info("Started session %d (residual WM zero-init)", self.session_id)

    @torch.no_grad()
    def generate(
        self, input_ids: torch.Tensor, max_new_tokens: int = 256, use_longterm: bool = True,
    ) -> torch.Tensor:
        """Generate with combined WM + LT adapters."""
        self.model.eval()
        if use_longterm:
            try:
                self.model.set_adapter(["working", "longterm"])
            except (TypeError, ValueError):
                self.model.set_adapter("longterm")
        else:
            self.model.set_adapter("working")

        prev_use_cache = getattr(self.model.config, "use_cache", True)
        self.model.config.use_cache = True
        try:
            output = self.model.generate(
                input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=False,
            )
        finally:
            self.model.config.use_cache = prev_use_cache
        return output

    def save(self, output_dir: str):
        """Save adapters, reservoir, and session metadata."""
        import pickle

        os.makedirs(output_dir, exist_ok=True)
        adapter_dir = os.path.join(output_dir, "adapters")
        try:
            self.model.save_pretrained(adapter_dir)
        except Exception as e:
            logger.warning("save_pretrained failed (%s), saving adapters individually", e)
            for name in ["working", "longterm"]:
                try:
                    self.model.set_adapter(name)
                    self.model.save_pretrained(os.path.join(adapter_dir, name))
                except Exception:
                    pass
            self.model.set_adapter("working")

        with open(os.path.join(output_dir, "reservoir.pkl"), "wb") as f:
            pickle.dump(self.reservoir, f)

        metadata = {
            "session_id": self.session_id,
            "total_turns": self.total_turns,
            "reservoir_size": len(self.reservoir),
            "beta": self.beta,
            "gamma": self.gamma,
        }
        import json

        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Saved SPM state to %s", output_dir)
