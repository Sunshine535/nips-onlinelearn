#!/usr/bin/env python3
"""Analyze pilot results and return GO/NO-GO decision.

Answers 3 gating questions:
  Q1: Does format validity decay earlier than tool selection
      under BOTH strict AND relaxed metrics?
  Q2: Does the effect survive conditional evaluation?
  Q3: Is the gap on tool tasks larger than on matched NLU control?

Decision rule: ≥2 "clearly yes" → GO (commit 3-4 months)
              1 "clearly yes" + good signal → INVESTIGATE
              0 or all negative → NO-GO (re-scope)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List


def compute_half_life(steps: List[int], values: List[float], baseline: float) -> float:
    """Step at which value crosses 0.5 * baseline. Returns max_step if never crossed."""
    target = 0.5 * baseline
    for step, v in zip(steps, values):
        if v <= target:
            return float(step)
    return float(steps[-1])  # never crossed


def analyze_condition(path: Path) -> Dict:
    with open(path) as f:
        data = json.load(f)
    cps = data["checkpoints"]
    steps = [c["step"] for c in cps]

    # Extract trajectories
    strict = [c["tool"]["strict_schema_pass"] for c in cps]
    relaxed = [c["tool"]["relaxed_schema_pass"] for c in cps]
    sel_uncond = [c["tool"]["tool_selection_uncond"] for c in cps]
    sel_cond = [c["tool"]["tool_selection_cond_schema"] for c in cps]
    arg_uncond = [c["tool"]["arg_correctness_uncond"] for c in cps]
    arg_cond = [c["tool"]["arg_correctness_cond_all"] for c in cps]
    nlu = [c["nlu"]["nlu_exact_match"] for c in cps]

    b_strict, b_relaxed = strict[0], relaxed[0]
    b_sel, b_arg, b_nlu = sel_uncond[0], arg_uncond[0], nlu[0]

    return {
        "condition": data["condition"],
        "steps": steps,
        "strict_schema": strict,
        "relaxed_schema": relaxed,
        "sel_uncond": sel_uncond,
        "sel_cond": sel_cond,
        "arg_uncond": arg_uncond,
        "arg_cond": arg_cond,
        "nlu": nlu,
        "half_life": {
            "strict_schema": compute_half_life(steps, strict, b_strict),
            "relaxed_schema": compute_half_life(steps, relaxed, b_relaxed),
            "sel_uncond": compute_half_life(steps, sel_uncond, b_sel),
            "arg_uncond": compute_half_life(steps, arg_uncond, b_arg),
            "nlu": compute_half_life(steps, nlu, b_nlu),
        },
        "baseline": {
            "strict_schema": b_strict,
            "relaxed_schema": b_relaxed,
            "sel_uncond": b_sel,
            "arg_uncond": b_arg,
            "nlu": b_nlu,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="outputs/pilot")
    args = parser.parse_args()

    d = Path(args.input_dir)
    conditions = {}
    for name in ["baseline", "alpaca", "nlu_control"]:
        p = d / f"pilot_{name}.json"
        if p.exists():
            conditions[name] = analyze_condition(p)
        else:
            print(f"[WARN] Missing {p}")

    if "alpaca" not in conditions:
        print("[ERROR] Cannot analyze without alpaca (drift) condition")
        return

    drift = conditions["alpaca"]
    ctrl = conditions.get("nlu_control")
    baseline = conditions.get("baseline")

    print("=" * 70)
    print("Pilot Analysis: Tool-Use Forgetting")
    print("=" * 70)

    # Print baseline stats
    if baseline:
        print(f"\nBaseline (no fine-tune):")
        b = baseline["baseline"]
        print(f"  strict_schema  = {b['strict_schema']:.3f}")
        print(f"  relaxed_schema = {b['relaxed_schema']:.3f}")
        print(f"  sel_uncond     = {b['sel_uncond']:.3f}")
        print(f"  arg_uncond     = {b['arg_uncond']:.3f}")
        print(f"  nlu            = {b['nlu']:.3f}")

    # Drift (alpaca) half-lives
    print(f"\nDrift condition (alpaca fine-tune) half-lives (steps to 50% of baseline):")
    hl = drift["half_life"]
    for k, v in hl.items():
        print(f"  {k:18s}: {v:.0f}")

    # NLU control
    if ctrl:
        print(f"\nControl condition (nlu fine-tune) half-lives:")
        for k, v in ctrl["half_life"].items():
            print(f"  {k:18s}: {v:.0f}")

    # ----- Q1: Format vs Selection half-life comparison -----
    print("\n" + "-" * 70)
    print("Q1: Does format decay EARLIER than selection under BOTH metrics?")
    print("-" * 70)
    hl_strict = hl["strict_schema"]
    hl_relaxed = hl["relaxed_schema"]
    hl_sel = hl["sel_uncond"]
    ratio_strict = hl_sel / max(hl_strict, 1)
    ratio_relaxed = hl_sel / max(hl_relaxed, 1)
    print(f"  ratio_strict  = sel_halflife / strict_schema_halflife  = {ratio_strict:.2f}x")
    print(f"  ratio_relaxed = sel_halflife / relaxed_schema_halflife = {ratio_relaxed:.2f}x")
    q1_strict = ratio_strict >= 1.5
    q1_relaxed = ratio_relaxed >= 1.5
    q1 = q1_strict and q1_relaxed
    print(f"  Q1 verdict: {'YES' if q1 else ('MIXED' if (q1_strict or q1_relaxed) else 'NO')}")
    print(f"    - strict ≥ 1.5x: {q1_strict}")
    print(f"    - relaxed ≥ 1.5x: {q1_relaxed}")

    # ----- Q2: Conditional evaluation survival -----
    print("\n" + "-" * 70)
    print("Q2: Does effect survive CONDITIONAL evaluation?")
    print("-" * 70)
    hl_sel_cond = hl["sel_uncond"]  # we already compute unconditional
    # Conditional sel = sel | strict_schema_pass; halflife computed from drift['sel_cond']
    hl_sel_cond_strict = compute_half_life(
        drift["steps"], drift["sel_cond"], drift["sel_cond"][0] if drift["sel_cond"][0] > 0 else 1.0
    )
    # Compare with unconditional
    ratio_cond = hl_sel_cond_strict / max(hl_strict, 1)
    print(f"  conditional sel halflife (cond on strict pass) = {hl_sel_cond_strict:.0f}")
    print(f"  strict schema halflife                          = {hl_strict:.0f}")
    print(f"  ratio (sel|pass) / schema = {ratio_cond:.2f}x")
    q2 = ratio_cond >= 1.3  # conditional sel should still outlive schema
    print(f"  Q2 verdict: {'YES' if q2 else 'NO'}")

    # ----- Q3: Tool gap vs NLU gap -----
    print("\n" + "-" * 70)
    print("Q3: Is gap on TOOL tasks larger than on MATCHED NLU control?")
    print("-" * 70)
    # Measure total drop from baseline to final step
    drop_tool = drift["baseline"]["strict_schema"] - drift["strict_schema"][-1]
    drop_nlu = drift["baseline"]["nlu"] - drift["nlu"][-1]
    print(f"  Tool strict_schema drop (drift) = {drop_tool:+.3f}")
    print(f"  NLU drop (drift)                = {drop_nlu:+.3f}")
    q3 = drop_tool >= 2 * drop_nlu
    print(f"  Q3 verdict: {'YES' if q3 else 'NO'} (tool drop ≥ 2x NLU drop)")

    # ----- Final decision -----
    yes_count = sum([q1, q2, q3])
    print("\n" + "=" * 70)
    print("GO/NO-GO DECISION")
    print("=" * 70)
    print(f"  Q1 (format earlier than selection): {'YES' if q1 else 'NO'}")
    print(f"  Q2 (survives conditional eval):     {'YES' if q2 else 'NO'}")
    print(f"  Q3 (tool gap > NLU gap):            {'YES' if q3 else 'NO'}")
    print(f"  Total: {yes_count}/3 YES")
    print()
    if yes_count >= 2:
        print("  ✅ GO — commit 3-4 months to full B1 project")
    elif yes_count == 1:
        print("  ⚠️  INVESTIGATE — one signal present, re-scope before committing")
    else:
        print("  ❌ NO-GO — phenomenon not present, pick different direction")

    # Save machine-readable verdict
    verdict = {
        "q1": q1, "q1_strict": q1_strict, "q1_relaxed": q1_relaxed,
        "q2": q2, "q3": q3, "yes_count": yes_count,
        "decision": "GO" if yes_count >= 2 else ("INVESTIGATE" if yes_count == 1 else "NO-GO"),
        "ratios": {
            "ratio_strict": ratio_strict, "ratio_relaxed": ratio_relaxed,
            "ratio_cond": ratio_cond,
            "tool_drop": drop_tool, "nlu_drop": drop_nlu,
        },
    }
    out_path = Path(args.input_dir) / "pilot_verdict.json"
    with open(out_path, "w") as f:
        json.dump(verdict, f, indent=2)
    print(f"\nVerdict saved to {out_path}")


if __name__ == "__main__":
    main()
