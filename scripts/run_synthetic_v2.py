#!/usr/bin/env python3
"""Synthetic experiments V2 — fixes for Round 1 reviewer feedback.

Key changes:
1. K_T = O(sqrt(T)) experiment: jumps become rarer over time -> validates O(sqrt(T)) claim
2. Properly separated timescales: rho_s << rho_f
3. Aligned ablations: Fisher-precision merge and gradient-stability selectivity
"""

import json
import math
import os
import sys
import time
from typing import Dict, List, Tuple

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.synthetic_benchmark import (
    TwoRateDriftStream,
    OnlineLowRankLearner,
    DualTimescaleLearner,
    EMALearner,
    PeriodicAveragingLearner,
    SubspaceTrackingLearner,
    _pre_generate,
    _regret_step,
    _init_subspace,
    _slow_U_update,
)


# ============================================================
# Fix 1: K_T = O(sqrt(T)) stream — jumps become rarer
# ============================================================

class DecreasingJumpStream:
    """Stream where jump frequency DECREASES over time.

    Jump at step t if floor(sqrt(t)) > floor(sqrt(t-1)).
    This gives K_T = O(sqrt(T)) jumps total.
    """
    def __init__(self, d=100, r=8, rho_s=0.0, jump_size=2.0, noise_std=0.1, seed=42):
        self.d, self.r = d, r
        self.rho_s = rho_s
        self.jump_size = jump_size
        self.noise_std = noise_std
        self._rng = torch.Generator()
        self._rng.manual_seed(seed)
        self.t = 0

        Z = torch.randn(d, r, generator=self._rng)
        self.U, _ = torch.linalg.qr(Z)
        self.a = torch.randn(r, generator=self._rng)
        self.a = self.a / self.a.norm() * jump_size

        self._U0 = self.U.clone()
        self._a0 = self.a.clone()
        self._rng_state0 = self._rng.get_state()
        self._jump_count = 0

    def step(self):
        theta_star = self.U @ self.a
        x = torch.randn(self.d, generator=self._rng)
        noise = torch.randn(1, generator=self._rng).item() * self.noise_std
        y = (theta_star @ x).item() + noise

        # Slow subspace evolution
        if self.rho_s > 0:
            Z = torch.randn(self.d, self.r, generator=self._rng)
            Z = Z - self.U @ (self.U.T @ Z)
            self.U = self.U + self.rho_s * Z
            self.U, _ = torch.linalg.qr(self.U)

        # Coefficient jump: occurs when floor(sqrt(t+1)) > floor(sqrt(t))
        self.t += 1
        if int(math.sqrt(self.t)) > int(math.sqrt(self.t - 1)):
            direction = torch.randn(self.r, generator=self._rng)
            direction = direction / direction.norm()
            self.a = self.a + self.jump_size * direction
            self._jump_count += 1

        return x, y, theta_star

    def reset(self):
        self.t = 0
        self._jump_count = 0
        self.U = self._U0.clone()
        self.a = self._a0.clone()
        self._rng.set_state(self._rng_state0)

    def get_optimal_subspace(self):
        return self.U.clone()

    @property
    def jump_count(self):
        return self._jump_count


# ============================================================
# Fix 2: Properly separated full-drift
# ============================================================

class SeparatedDriftStream:
    """Stream with WELL-SEPARATED slow and fast timescales.

    Slow: rho_s = 0.001 (very slow subspace rotation)
    Fast: jumps every 20 steps (very fast coefficient changes)
    """
    def __init__(self, d=100, r=8, rho_s=0.001, jump_interval=20,
                 jump_size=2.0, noise_std=0.1, seed=42):
        self.d, self.r = d, r
        self.rho_s = rho_s
        self.jump_interval = jump_interval
        self.jump_size = jump_size
        self.noise_std = noise_std
        self._rng = torch.Generator()
        self._rng.manual_seed(seed)
        self.t = 0

        Z = torch.randn(d, r, generator=self._rng)
        self.U, _ = torch.linalg.qr(Z)
        self.a = torch.randn(r, generator=self._rng)
        self.a = self.a / self.a.norm() * jump_size

        self._U0 = self.U.clone()
        self._a0 = self.a.clone()
        self._rng_state0 = self._rng.get_state()

    def step(self):
        theta_star = self.U @ self.a
        x = torch.randn(self.d, generator=self._rng)
        noise = torch.randn(1, generator=self._rng).item() * self.noise_std
        y = (theta_star @ x).item() + noise

        # Very slow subspace rotation
        Z = torch.randn(self.d, self.r, generator=self._rng)
        Z = Z - self.U @ (self.U.T @ Z)
        self.U = self.U + self.rho_s * Z
        self.U, _ = torch.linalg.qr(self.U)

        # Fast coefficient jumps
        self.t += 1
        if self.t % self.jump_interval == 0:
            direction = torch.randn(self.r, generator=self._rng)
            direction = direction / direction.norm()
            self.a = self.a + self.jump_size * direction

        return x, y, theta_star

    def reset(self):
        self.t = 0
        self.U = self._U0.clone()
        self.a = self._a0.clone()
        self._rng.set_state(self._rng_state0)

    def get_optimal_subspace(self):
        return self.U.clone()


def run_experiment_generic(stream, learners, T):
    """Run all learners on same stream data."""
    # Pre-generate data
    stream.reset()
    data = []
    for _ in range(T):
        data.append(stream.step())

    results = {}
    for name, learner in learners.items():
        for x, y, ts in data:
            learner.update(x, y, ts)
        results[name] = learner.get_regret()
    return results


def make_learners(d, r, U_init):
    return {
        "single_timescale": OnlineLowRankLearner(d, r, lr=0.5, U_init=U_init.clone()),
        "dual_timescale": DualTimescaleLearner(
            d, r, lr_fast=0.5, lr_slow=0.01,
            consolidation_period=20, window_size=30, U_init=U_init.clone()
        ),
        "ema": EMALearner(d, r, lr=0.5, beta=0.99, U_init=U_init.clone()),
        "periodic_avg": PeriodicAveragingLearner(
            d, r, lr=0.5, avg_period=50, U_init=U_init.clone()
        ),
    }


def main():
    output_dir = "outputs/synthetic_v2"
    os.makedirs(output_dir, exist_ok=True)

    d, r = 100, 8
    all_results = {}

    # ============================================================
    # Experiment A: K_T = O(sqrt(T)) — validates O(sqrt(T)) claim
    # ============================================================
    print("=" * 60)
    print("Exp A: K_T = O(sqrt(T)) decreasing jump frequency")
    print("=" * 60)

    for T in [1000, 5000, 10000, 20000]:
        stream = DecreasingJumpStream(d=d, r=r, rho_s=0.0, jump_size=2.0, seed=42)
        U0 = stream.get_optimal_subspace()
        learners = make_learners(d, r, U0)
        results = run_experiment_generic(stream, learners, T)

        K_T = int(math.sqrt(T))
        row = {name: regret[-1] for name, regret in results.items()}
        all_results[f"sqrt_jump_T{T}"] = {**row, "K_T": K_T}

        s = row["single_timescale"]
        dual = row["dual_timescale"]
        # Check scaling: dual should be O(sqrt(T)), single should be Omega(T)
        print(f"  T={T:>5d} K_T={K_T:>3d} | single={s:>10.1f} ({s/T:.2f}/step) | dual={dual:>10.1f} ({dual/T:.4f}/step) | ratio={s/max(dual,1):.1f}x")

    # ============================================================
    # Experiment B: Separated full drift
    # ============================================================
    print("\n" + "=" * 60)
    print("Exp B: Separated timescales (rho_s=0.001, jump_interval=20)")
    print("=" * 60)

    for T in [1000, 5000, 10000]:
        stream = SeparatedDriftStream(d=d, r=r, rho_s=0.001, jump_interval=20, jump_size=2.0, seed=42)
        U0 = stream.get_optimal_subspace()
        learners = make_learners(d, r, U0)
        results = run_experiment_generic(stream, learners, T)

        row = {name: regret[-1] for name, regret in results.items()}
        all_results[f"separated_drift_T{T}"] = row

        s = row["single_timescale"]
        dual = row["dual_timescale"]
        print(f"  T={T:>5d} | single={s:>10.1f} | dual={dual:>10.1f} | ratio={s/max(dual,1):.1f}x")

    # ============================================================
    # Experiment C: Timescale separation sweep
    # ============================================================
    print("\n" + "=" * 60)
    print("Exp C: Timescale separation ratio sweep")
    print("=" * 60)

    T = 5000
    for rho_s in [0.0, 0.0001, 0.001, 0.005, 0.01, 0.05]:
        for ji in [10, 20, 50, 100]:
            stream = SeparatedDriftStream(d=d, r=r, rho_s=rho_s, jump_interval=ji, jump_size=2.0, seed=42)
            U0 = stream.get_optimal_subspace()
            learners = {
                "single": OnlineLowRankLearner(d, r, lr=0.5, U_init=U0.clone()),
                "dual": DualTimescaleLearner(d, r, lr_fast=0.5, lr_slow=0.01,
                    consolidation_period=20, window_size=30, U_init=U0.clone()),
            }
            results = run_experiment_generic(stream, learners, T)
            s = results["single"][-1]
            dual = results["dual"][-1]
            ratio = s / max(dual, 1)
            sep = ji * rho_s  # separation metric
            all_results[f"sweep_rhos{rho_s}_ji{ji}"] = {
                "single": s, "dual": dual, "ratio": ratio,
                "rho_s": rho_s, "jump_interval": ji, "separation": sep
            }
            marker = "✓" if ratio > 1 else "✗"
            print(f"  rho_s={rho_s:.4f} ji={ji:>3d} sep={sep:.3f} | ratio={ratio:>5.1f}x {marker}")

    # Save all results
    with open(os.path.join(output_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to {output_dir}/all_results.json")


if __name__ == "__main__":
    main()
