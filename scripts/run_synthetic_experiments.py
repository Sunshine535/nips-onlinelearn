#!/usr/bin/env python3
"""Run all synthetic experiments (E1-E5) for the separation theorem validation.

Usage:
    python3 scripts/run_synthetic_experiments.py [--output_dir outputs/synthetic]
"""

import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.synthetic_benchmark import (
    TwoRateDriftStream,
    OnlineLowRankLearner,
    DualTimescaleLearner,
    EMALearner,
    PeriodicAveragingLearner,
    SubspaceTrackingLearner,
    run_experiment,
    run_phase_diagram,
    run_frequency_sweep,
    run_selectivity_test,
    run_merge_rule_test,
    _pre_generate,
)


def run_e1_separation(output_dir: str):
    """E1: Controlled Two-Rate Latent Drift — separation theorem validation."""
    print("=" * 60)
    print("E1: Separation Theorem Validation")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    # Test across multiple horizons
    for T in [500, 1000, 2000, 5000, 10000, 20000]:
        # Coefficient-only (fixed subspace) — cleanest separation
        stream = TwoRateDriftStream(
            d=100, r=8, rho_s=0.0, jump_interval=50,
            jump_size=2.0, noise_std=0.1, seed=42
        )
        true_U = stream.get_optimal_subspace()

        learners = {
            "single_timescale": OnlineLowRankLearner(100, 8, lr=0.5, U_init=true_U),
            "dual_timescale": DualTimescaleLearner(
                100, 8, lr_fast=0.5, lr_slow=0.01,
                consolidation_period=20, window_size=30, U_init=true_U
            ),
            "ema": EMALearner(100, 8, lr=0.5, beta=0.99, U_init=true_U),
            "periodic_avg": PeriodicAveragingLearner(
                100, 8, lr=0.5, avg_period=50, U_init=true_U
            ),
            "subspace_tracking": SubspaceTrackingLearner(100, 8, lr_oja=0.05, U_init=true_U),
        }

        exp_results = run_experiment(stream, learners, T)
        row = {}
        for name, regret_list in exp_results.items():
            row[name] = regret_list[-1] if regret_list else 0.0
        results[f"coeff_only_T{T}"] = row
        dual_r = row.get("dual_timescale", 1)
        single_r = row.get("single_timescale", 1)
        ratio = single_r / max(dual_r, 1)
        print(f"  T={T:>5d} | single={single_r:>10.1f} | dual={dual_r:>10.1f} | ratio={ratio:.1f}x")

    # Full two-rate drift (subspace + coefficients)
    print("\n  --- Full two-rate drift ---")
    for T in [1000, 5000, 10000]:
        stream = TwoRateDriftStream(
            d=100, r=8, rho_s=0.01, jump_interval=50,
            jump_size=2.0, noise_std=0.1, seed=42
        )
        true_U = stream.get_optimal_subspace()

        learners = {
            "single_timescale": OnlineLowRankLearner(100, 8, lr=0.5, U_init=true_U),
            "dual_timescale": DualTimescaleLearner(
                100, 8, lr_fast=0.5, lr_slow=0.01,
                consolidation_period=20, window_size=30, U_init=true_U
            ),
            "ema": EMALearner(100, 8, lr=0.5, beta=0.99, U_init=true_U),
            "periodic_avg": PeriodicAveragingLearner(
                100, 8, lr=0.5, avg_period=50, U_init=true_U
            ),
            "subspace_tracking": SubspaceTrackingLearner(100, 8, lr_oja=0.05, U_init=true_U),
        }

        exp_results = run_experiment(stream, learners, T)
        row = {}
        for name, regret_list in exp_results.items():
            row[name] = regret_list[-1] if regret_list else 0.0
        results[f"full_drift_T{T}"] = row
        dual_r = row.get("dual_timescale", 1)
        single_r = row.get("single_timescale", 1)
        ratio = single_r / max(dual_r, 1)
        print(f"  T={T:>5d} | single={single_r:>10.1f} | dual={dual_r:>10.1f} | ratio={ratio:.1f}x")

    with open(os.path.join(output_dir, "e1_separation.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {output_dir}/e1_separation.json\n")
    return results


def run_e2_phase_diagram(output_dir: str):
    """E2: Phase Transition Diagram — (rho_s, rho_f) grid."""
    print("=" * 60)
    print("E2: Phase Transition Diagram")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    rho_s_range = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    rho_f_range = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

    results = run_phase_diagram(rho_s_range, rho_f_range, T=3000, d=100, r=8)

    with open(os.path.join(output_dir, "e2_phase_diagram.json"), "w") as f:
        json.dump({
            "rho_s_range": rho_s_range,
            "rho_f_range": rho_f_range,
            **{k: v if not isinstance(v, list) else
               [[float(x) for x in row] if isinstance(row, list) else float(row) for row in v]
               for k, v in results.items()}
        }, f, indent=2)
    print(f"  Saved to {output_dir}/e2_phase_diagram.json\n")
    return results


def run_e3_frequency_sweep(output_dir: str):
    """E3: Consolidation Frequency Sweep — validate B* formula."""
    print("=" * 60)
    print("E3: Consolidation Frequency Sweep")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    stream = TwoRateDriftStream(
        d=100, r=8, rho_s=0.01, jump_interval=50,
        jump_size=2.0, noise_std=0.1, seed=42
    )

    B_range = [2, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500]
    raw = run_frequency_sweep(B_range, stream, T=5000)

    # raw is {"B_values": [...], "final_regret": [...], "regret_curves": {...}}
    results = dict(zip([str(b) for b in raw["B_values"]], raw["final_regret"]))

    with open(os.path.join(output_dir, "e3_frequency_sweep.json"), "w") as f:
        json.dump({"B_values": raw["B_values"], "final_regret": raw["final_regret"]}, f, indent=2)

    # Find empirical optimum
    best_idx = raw["final_regret"].index(min(raw["final_regret"]))
    best_B = raw["B_values"][best_idx]
    best_reg = raw["final_regret"][best_idx]
    print(f"  Empirical B* = {best_B} (regret={best_reg:.1f})")
    for b, r in zip(raw["B_values"], raw["final_regret"]):
        print(f"    B={b:>4d}: regret={r:.1f}")
    print(f"  Saved to {output_dir}/e3_frequency_sweep.json\n")
    return results


def run_e4_selectivity(output_dir: str):
    """E4: Invariant-Selective Consolidation Test."""
    print("=" * 60)
    print("E4: Invariant-Selective Consolidation")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    stream = TwoRateDriftStream(
        d=100, r=8, rho_s=0.01, jump_interval=50,
        jump_size=2.0, noise_std=0.1, seed=42
    )

    raw = run_selectivity_test(stream, T=5000)

    results = {k: v[-1] if isinstance(v, list) and v else v for k, v in raw.items()}

    with open(os.path.join(output_dir, "e4_selectivity.json"), "w") as f:
        json.dump(results, f, indent=2)

    for k, v in results.items():
        print(f"  {k}: regret={v:.1f}")
    print(f"  Saved to {output_dir}/e4_selectivity.json\n")
    return results


def run_e5_merge_rules(output_dir: str):
    """E5: Merge Rule Comparison."""
    print("=" * 60)
    print("E5: Merge Rule Comparison")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    stream = TwoRateDriftStream(
        d=100, r=8, rho_s=0.01, jump_interval=50,
        jump_size=2.0, noise_std=0.1, seed=42
    )

    raw = run_merge_rule_test(stream, T=5000)

    results = {k: v[-1] if isinstance(v, list) and v else v for k, v in raw.items()}

    with open(os.path.join(output_dir, "e5_merge_rules.json"), "w") as f:
        json.dump(results, f, indent=2)

    for k, v in results.items():
        print(f"  {k}: regret={v:.1f}")
    print(f"  Saved to {output_dir}/e5_merge_rules.json\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run synthetic experiments E1-E5")
    parser.add_argument("--output_dir", default="outputs/synthetic", help="Output directory")
    parser.add_argument("--experiments", nargs="+", default=["e1", "e2", "e3", "e4", "e5"],
                        help="Which experiments to run")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()

    all_results = {}

    if "e1" in args.experiments:
        all_results["e1"] = run_e1_separation(args.output_dir)

    if "e2" in args.experiments:
        all_results["e2"] = run_e2_phase_diagram(args.output_dir)

    if "e3" in args.experiments:
        all_results["e3"] = run_e3_frequency_sweep(args.output_dir)

    if "e4" in args.experiments:
        all_results["e4"] = run_e4_selectivity(args.output_dir)

    if "e5" in args.experiments:
        all_results["e5"] = run_e5_merge_rules(args.output_dir)

    elapsed = time.time() - t0
    print(f"\nAll synthetic experiments complete in {elapsed:.0f}s")
    print(f"Results in {args.output_dir}/")

    # Write summary
    summary = {
        "experiments_run": args.experiments,
        "total_time_seconds": elapsed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
