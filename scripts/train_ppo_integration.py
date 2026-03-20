#!/usr/bin/env python3
"""PPO policy for adaptive consolidation decisions.
Trains an MLP policy network that decides when to consolidate working LoRA → long-term.
State: Fisher stats + buffer size + session length + recent loss trend.
Action: consolidate now / wait.
Reward: response_quality - forgetting_penalty."""

import argparse
import json
import logging
import math
import os
import sys
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.streaming_memory import StreamingParameterMemory, FisherEstimator


class ConsolidationPolicy(nn.Module):
    """MLP policy: state → P(consolidate)."""

    def __init__(self, state_dim: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.value_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor):
        logits = self.net(state)
        value = self.value_head(state)
        return logits, value

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        logits, value = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            action = torch.multinomial(probs, 1).squeeze(-1)
        log_prob = F.log_softmax(logits, dim=-1).gather(1, action.unsqueeze(-1)).squeeze(-1)
        return action, log_prob, value.squeeze(-1), probs


class PPOTrainer:
    """Proximal Policy Optimization for consolidation policy."""

    def __init__(self, policy: ConsolidationPolicy, lr: float = 3e-4,
                 clip_ratio: float = 0.2, gamma: float = 0.99, gae_lambda: float = 0.95,
                 entropy_coef: float = 0.01, value_coef: float = 0.5):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.clip_ratio = clip_ratio
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            next_val = values[t + 1] if t + 1 < len(values) else 0
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        returns = [a + v for a, v in zip(advantages, values[:len(advantages)])]
        return advantages, returns

    def update(self, trajectories: list, epochs: int = 4, batch_size: int = 64):
        states = torch.stack([t["state"] for t in trajectories])
        actions = torch.tensor([t["action"] for t in trajectories])
        old_log_probs = torch.tensor([t["log_prob"] for t in trajectories])
        rewards = [t["reward"] for t in trajectories]
        values = [t["value"] for t in trajectories]
        dones = [t.get("done", False) for t in trajectories]

        advantages, returns = self.compute_gae(rewards, values, dones)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss_val = 0.0
        for _ in range(epochs):
            for i in range(0, len(trajectories), batch_size):
                end = min(i + batch_size, len(trajectories))
                b_states = states[i:end]
                b_actions = actions[i:end]
                b_old_lp = old_log_probs[i:end]
                b_adv = advantages[i:end]
                b_ret = returns[i:end]

                logits, values_pred = self.policy(b_states)
                log_probs = F.log_softmax(logits, dim=-1)
                new_lp = log_probs.gather(1, b_actions.unsqueeze(-1)).squeeze(-1)

                ratio = (new_lp - b_old_lp).exp()
                surr1 = ratio * b_adv
                surr2 = ratio.clamp(1 - self.clip_ratio, 1 + self.clip_ratio) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values_pred.squeeze(-1), b_ret)

                entropy = -(F.softmax(logits, dim=-1) * log_probs).sum(dim=-1).mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                total_loss_val += loss.item()

        return total_loss_val


def build_state_vector(spm, recent_losses: list, session_turn: int,
                       buffer_size: int, session_idx: int) -> torch.Tensor:
    """Build state observation for the policy."""
    fisher_mag = 0.0
    if spm.longterm.fisher:
        fisher_mag = sum(f.abs().mean().item() for f in spm.longterm.fisher.values()) / \
                     max(len(spm.longterm.fisher), 1)

    avg_recent_loss = sum(recent_losses[-5:]) / max(len(recent_losses[-5:]), 1) if recent_losses else 5.0
    loss_trend = (recent_losses[-1] - recent_losses[-3]) if len(recent_losses) >= 3 else 0.0
    turns_since_consolidation = spm.working.update_count
    normalized_buffer = min(buffer_size / 5000.0, 1.0)

    state = torch.tensor([
        fisher_mag,
        normalized_buffer,
        session_turn / 20.0,
        avg_recent_loss,
        loss_trend,
        turns_since_consolidation / 10.0,
        session_idx / 100.0,
        spm.total_turns / 1000.0,
    ], dtype=torch.float32)
    return state


def compute_reward(loss_before: float, loss_after: float,
                   retention_before: float, retention_after: float,
                   did_consolidate: bool, consolidation_cost: float = 0.1) -> float:
    """Reward = quality improvement - forgetting penalty - consolidation cost."""
    quality = max(0, loss_before - loss_after) * 2.0
    forgetting = max(0, retention_before - retention_after) * 5.0
    cost = consolidation_cost if did_consolidate else 0.0
    return quality - forgetting - cost


def format_turn(user_msg: str, assistant_msg: str, persona: str = "") -> str:
    system = persona if persona else "You are a helpful and personalized assistant."
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_msg}<|im_end|>"
    )


@torch.no_grad()
def quick_retention(spm, tokenizer, facts: list) -> float:
    if not facts:
        return 1.0
    device = next(spm.model.parameters()).device
    correct = 0
    for f in facts[-10:]:
        prompt = f"<|im_start|>user\n{f['q']}<|im_end|>\n<|im_start|>assistant\n"
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        out = spm.generate(ids, max_new_tokens=50, use_longterm=True)
        resp = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).lower()
        if f["a"].lower() in resp:
            correct += 1
    return correct / max(len(facts[-10:]), 1)


def main():
    parser = argparse.ArgumentParser(description="Train PPO consolidation policy")
    parser.add_argument("--config", type=str, default="configs/spm_config.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs/ppo_policy")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--turns_per_episode", type=int, default=40)
    parser.add_argument("--policy_lr", type=float, default=3e-4)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    base_model_name = config["model"]["base_model"]

    logger.info("=== PPO Consolidation Policy Training ===")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
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

    policy = ConsolidationPolicy(state_dim=8, hidden_dim=64)
    ppo = PPOTrainer(policy, lr=args.policy_lr)

    # Load training conversations
    try:
        ds = load_dataset("bavard/personachat_truecased", split="train", trust_remote_code=True)
        conversations = []
        for i, ex in enumerate(ds):
            if i >= 10000:
                break
            history = ex.get("history", [])
            candidates = ex.get("candidates", [])
            reply = candidates[-1] if candidates else ""
            if history and reply:
                conversations.append({"user": history[-1], "assistant": reply})
    except Exception:
        conversations = [
            {"user": f"Tell me about topic {i % 50}.", "assistant": f"Topic {i % 50} is about concept {i}."}
            for i in range(5000)
        ]

    device = next(spm.model.parameters()).device
    episode_rewards = []
    training_log = []

    for episode in range(args.num_episodes):
        logger.info(">>> Episode %d/%d", episode + 1, args.num_episodes)
        spm.start_new_session()

        trajectories = []
        recent_losses = deque(maxlen=20)
        taught_facts = []
        episode_reward = 0.0

        import random
        ep_convs = random.sample(conversations, min(args.turns_per_episode, len(conversations)))

        retention_before = quick_retention(spm, tokenizer, taught_facts)

        for turn_idx, conv in enumerate(ep_convs):
            text = format_turn(conv["user"], conv["assistant"])
            encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
            input_ids = encoded["input_ids"].to(device)
            labels = input_ids.clone()

            state = build_state_vector(
                spm, list(recent_losses), turn_idx,
                len(spm.memory_buffer.inputs), episode)

            action, log_prob, value, probs = policy.get_action(state.unsqueeze(0))
            action = action.item()
            should_consolidate = (action == 1)

            override = spm.working.needs_consolidation()
            if should_consolidate or override:
                spm.working.max_turns = spm.working.update_count

            result = spm.process_turn(input_ids, labels)
            recent_losses.append(result["loss"])

            taught_facts.append({"q": conv["user"][:50], "a": conv["assistant"][:30]})

            retention_after = quick_retention(spm, tokenizer, taught_facts) if turn_idx % 10 == 0 else retention_before

            reward = compute_reward(
                loss_before=recent_losses[-2] if len(recent_losses) >= 2 else 5.0,
                loss_after=result["loss"],
                retention_before=retention_before,
                retention_after=retention_after,
                did_consolidate=result["consolidated"],
            )
            episode_reward += reward
            retention_before = retention_after

            trajectories.append({
                "state": state,
                "action": action,
                "log_prob": log_prob.item(),
                "value": value.item(),
                "reward": reward,
                "done": (turn_idx == len(ep_convs) - 1),
            })

            spm.working.max_turns = config["working_memory"].get("max_turns_before_consolidation", 10)

        if len(trajectories) >= 10:
            ppo_loss = ppo.update(trajectories, epochs=args.ppo_epochs)
        else:
            ppo_loss = 0.0

        episode_rewards.append(episode_reward)
        consolidation_rate = sum(1 for t in trajectories if t["action"] == 1) / max(len(trajectories), 1)

        training_log.append({
            "episode": episode,
            "total_reward": episode_reward,
            "avg_reward": episode_reward / max(len(trajectories), 1),
            "ppo_loss": ppo_loss,
            "consolidation_rate": consolidation_rate,
            "num_turns": len(trajectories),
        })

        logger.info("  Reward: %.4f | PPO loss: %.4f | Consol rate: %.2f",
                     episode_reward, ppo_loss, consolidation_rate)

    # Save policy
    torch.save(policy.state_dict(), os.path.join(args.output_dir, "consolidation_policy.pt"))
    with open(os.path.join(args.output_dir, "ppo_training_log.json"), "w") as f:
        json.dump(training_log, f, indent=2)

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        rewards = [e["total_reward"] for e in training_log]
        axes[0].plot(rewards, "b-o", markersize=3)
        axes[0].set_title("Episode Reward")
        axes[0].set_xlabel("Episode")
        axes[0].grid(True, alpha=0.3)

        rates = [e["consolidation_rate"] for e in training_log]
        axes[1].plot(rates, "g-o", markersize=3)
        axes[1].set_title("Consolidation Rate")
        axes[1].set_xlabel("Episode")
        axes[1].grid(True, alpha=0.3)

        losses = [e["ppo_loss"] for e in training_log if e["ppo_loss"] > 0]
        axes[2].plot(losses, "r-o", markersize=3)
        axes[2].set_title("PPO Loss")
        axes[2].set_xlabel("Update")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "ppo_training.pdf"), dpi=150, bbox_inches="tight")
        plt.savefig(os.path.join(args.output_dir, "ppo_training.png"), dpi=150, bbox_inches="tight")
        plt.close()
    except ImportError:
        pass

    logger.info("\n=== PPO Training Complete ===")
    logger.info("Mean episode reward: %.4f", sum(episode_rewards) / max(len(episode_rewards), 1))
    logger.info("Saved policy to %s", args.output_dir)


if __name__ == "__main__":
    main()
