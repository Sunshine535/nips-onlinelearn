"""Dual-LoRA architecture: working memory (fast update) + long-term memory (consolidated).
Uses Fisher Information for importance-weighted consolidation."""

import copy
import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

logger = logging.getLogger(__name__)


@dataclass
class MemoryBuffer:
    """Experience replay buffer for memory consolidation."""
    inputs: List[Dict[str, torch.Tensor]] = field(default_factory=list)
    labels: List[torch.Tensor] = field(default_factory=list)
    importance: List[float] = field(default_factory=list)
    max_size: int = 5000

    def add(self, input_ids: torch.Tensor, labels: torch.Tensor, importance: float = 1.0):
        if len(self.inputs) >= self.max_size:
            min_idx = min(range(len(self.importance)), key=lambda i: self.importance[i])
            if importance > self.importance[min_idx]:
                self.inputs[min_idx] = {"input_ids": input_ids.cpu()}
                self.labels[min_idx] = labels.cpu()
                self.importance[min_idx] = importance
        else:
            self.inputs.append({"input_ids": input_ids.cpu()})
            self.labels.append(labels.cpu())
            self.importance.append(importance)

    def sample(self, n: int) -> Tuple[List[Dict], List[torch.Tensor]]:
        import random
        if len(self.inputs) == 0:
            return [], []
        weights = [imp / sum(self.importance) for imp in self.importance]
        indices = random.choices(range(len(self.inputs)), weights=weights, k=min(n, len(self.inputs)))
        return [self.inputs[i] for i in indices], [self.labels[i] for i in indices]


class FisherEstimator:
    """Estimate diagonal Fisher Information Matrix for parameter importance."""

    def __init__(self, model: nn.Module, num_samples: int = 200):
        self.model = model
        self.num_samples = num_samples

    def estimate(self, dataloader) -> Dict[str, torch.Tensor]:
        """Estimate diagonal Fisher using empirical gradients."""
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
    """Fast-updating working LoRA for current session context."""

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

    def online_update(self, model: PeftModel, input_ids: torch.Tensor, labels: torch.Tensor,
                      lr: float = 5e-4, steps: int = 5) -> float:
        """Perform quick online gradient steps on working LoRA."""
        model.train()
        optimizer = torch.optim.AdamW(
            [p for n, p in model.named_parameters() if p.requires_grad and "lora" in n],
            lr=lr,
        )
        total_loss = 0.0
        for _ in range(steps):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        self.update_count += 1
        return total_loss / steps

    def reset(self):
        self.update_count = 0


class LongTermMemoryLoRA:
    """Slowly-consolidating long-term LoRA with EWC protection."""

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
        """Store current parameters as anchor for EWC."""
        self.prev_params = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad and "lora" in name
        }

    def ewc_loss(self, model: nn.Module, ewc_lambda: float = 5000.0) -> torch.Tensor:
        """Elastic Weight Consolidation penalty."""
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if name in self.fisher and name in self.prev_params:
                loss += (self.fisher[name] * (param - self.prev_params[name]).pow(2)).sum()
        return ewc_lambda * loss

    def consolidate(self, model: PeftModel, memory_buffer: MemoryBuffer,
                    fisher_estimator: FisherEstimator, dataloader,
                    epochs: int = 3, lr: float = 1e-4, ewc_lambda: float = 5000.0) -> float:
        """Consolidate working memory into long-term by distillation with EWC."""
        self.fisher = fisher_estimator.estimate(dataloader)
        self.store_checkpoint(model)

        model.train()
        optimizer = torch.optim.AdamW(
            [p for n, p in model.named_parameters() if p.requires_grad and "lora" in n],
            lr=lr,
        )

        total_loss = 0.0
        steps = 0
        for epoch in range(epochs):
            replay_inputs, replay_labels = memory_buffer.sample(32)
            for inp, lab in zip(replay_inputs, replay_labels):
                device = next(model.parameters()).device
                input_ids = inp["input_ids"].to(device).unsqueeze(0) if inp["input_ids"].dim() == 1 else inp["input_ids"].to(device)
                labels = lab.to(device).unsqueeze(0) if lab.dim() == 1 else lab.to(device)

                outputs = model(input_ids=input_ids, labels=labels)
                task_loss = outputs.loss
                ewc = self.ewc_loss(model, ewc_lambda)
                loss = task_loss + ewc

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
                steps += 1

        return total_loss / max(steps, 1)


class StreamingParameterMemory:
    """Main controller: coordinates working LoRA ↔ long-term LoRA lifecycle."""

    def __init__(self, base_model: nn.Module, working_config: dict, longterm_config: dict,
                 memory_buffer_size: int = 5000):
        self.base_model = base_model
        self.working = WorkingMemoryLoRA(working_config)
        self.longterm = LongTermMemoryLoRA(longterm_config)
        self.memory_buffer = MemoryBuffer(max_size=memory_buffer_size)
        self.session_id = 0
        self.total_turns = 0

        self.model = get_peft_model(base_model, self.working.lora_config, adapter_name="working")
        self.model.add_adapter("longterm", self.longterm.lora_config)
        self.model.set_adapter("working")

    def process_turn(self, input_ids: torch.Tensor, labels: torch.Tensor) -> dict:
        """Process a single conversation turn: update working memory, maybe consolidate."""
        self.model.set_adapter("working")
        loss = self.working.online_update(
            self.model, input_ids, labels,
            lr=self.working.config.get("learning_rate", 5e-4),
            steps=self.working.config.get("online_update_steps", 5),
        )

        self.memory_buffer.add(input_ids, labels, importance=1.0 / max(loss, 0.1))
        self.total_turns += 1

        consolidated = False
        if self.working.needs_consolidation():
            self._consolidate()
            consolidated = True

        return {"loss": loss, "consolidated": consolidated, "total_turns": self.total_turns}

    def _consolidate(self):
        """Transfer important knowledge from working LoRA to long-term LoRA."""
        logger.info("Consolidating working memory → long-term (session %d, turn %d)",
                    self.session_id, self.total_turns)

        self.model.set_adapter("longterm")
        from torch.utils.data import DataLoader, TensorDataset

        replay_inputs, replay_labels = self.memory_buffer.sample(
            self.longterm.config.get("fisher_samples", 200)
        )
        if replay_inputs:
            max_len = max(inp["input_ids"].shape[-1] for inp in replay_inputs)
            padded_ids = torch.zeros(len(replay_inputs), max_len, dtype=torch.long)
            padded_labels = torch.full((len(replay_inputs), max_len), -100, dtype=torch.long)
            for i, (inp, lab) in enumerate(zip(replay_inputs, replay_labels)):
                ids = inp["input_ids"].squeeze()
                l = lab.squeeze()
                padded_ids[i, :ids.shape[-1]] = ids
                padded_labels[i, :l.shape[-1]] = l

            ds = TensorDataset(padded_ids, padded_labels)

            class SimpleLoader:
                def __init__(self, dataset, batch_size=4):
                    self.dataset = dataset
                    self.batch_size = batch_size
                    self.idx = 0
                def __iter__(self):
                    self.idx = 0
                    return self
                def __next__(self):
                    if self.idx >= len(self.dataset):
                        raise StopIteration
                    end = min(self.idx + self.batch_size, len(self.dataset))
                    batch_ids = torch.stack([self.dataset[j][0] for j in range(self.idx, end)])
                    batch_labels = torch.stack([self.dataset[j][1] for j in range(self.idx, end)])
                    self.idx = end
                    return {"input_ids": batch_ids, "labels": batch_labels, "attention_mask": (batch_ids != 0).long()}

            loader = SimpleLoader(ds)
            fisher_est = FisherEstimator(self.model, num_samples=50)
            self.longterm.consolidate(
                self.model, self.memory_buffer, fisher_est, loader,
                epochs=self.longterm.config.get("consolidation_epochs", 3),
                lr=self.longterm.config.get("consolidation_lr", 1e-4),
                ewc_lambda=self.longterm.config.get("ewc_lambda", 5000.0),
            )

        self.working.reset()
        self.model.set_adapter("working")
        logger.info("Consolidation complete")

    def start_new_session(self):
        """Begin new conversation session: reset working memory."""
        self.session_id += 1
        self.working.reset()
        self.model.set_adapter("working")
        # Re-initialize working LoRA weights
        for name, param in self.model.named_parameters():
            if "working" in name and "lora" in name:
                if "lora_A" in name:
                    nn.init.kaiming_uniform_(param)
                elif "lora_B" in name:
                    nn.init.zeros_(param)
        logger.info("Started new session %d", self.session_id)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 256, use_longterm: bool = True) -> torch.Tensor:
        """Generate with combined working + long-term memory."""
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
        """Save both adapters and memory buffer."""
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
                    self.model.save_pretrained(adapter_dir)
                except Exception:
                    pass
            self.model.set_adapter("working")
        with open(os.path.join(output_dir, "memory_buffer.pkl"), "wb") as f:
            pickle.dump(self.memory_buffer, f)
        logger.info("Saved SPM state to %s", output_dir)
