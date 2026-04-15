#!/usr/bin/env python3
"""Collect all experiment results into a single summary file."""
import json, glob, os

out = []

for f in sorted(glob.glob("outputs/streaming_eval/*/dialogue_comparison.json")):
    tag = os.path.basename(os.path.dirname(f))
    data = json.load(open(f))
    out.append(f"\n=== {tag} (dialogue) ===")
    out.append(f"{'Method':<30} {'RetF1':>8} {'PPL':>8} {'Loss':>8} {'Forget':>8} {'Speed':>6}")
    out.append("-" * 75)
    for name, m in sorted(data.items()):
        out.append(f"{name:<30} {m.get('semantic_retention_f1',0):>8.4f} {m.get('perplexity',0):>8.2f} {m.get('avg_loss',0):>8.4f} {m.get('forgetting_rate',0):>8.4f} {m.get('adaptation_speed',0):>6.2f}")

for f in sorted(glob.glob("outputs/streaming_eval/*/classification_comparison.json")):
    tag = os.path.basename(os.path.dirname(f))
    data = json.load(open(f))
    out.append(f"\n=== {tag} (classification) ===")
    out.append(f"{'Method':<30} {'Acc':>8} {'Forget':>8} {'FwdTfr':>8} {'Regret':>10}")
    out.append("-" * 70)
    for name, m in sorted(data.items()):
        out.append(f"{name:<30} {m.get('overall_accuracy',0):>8.4f} {m.get('forgetting',0):>8.4f} {m.get('forward_transfer',0):>8.4f} {m.get('dynamic_regret',0):>10.2f}")

for f in sorted(glob.glob("outputs/synthetic_v2/all_results.json")):
    data = json.load(open(f))
    out.append(f"\n=== synthetic_v2 ===")
    for k, v in data.items():
        out.append(f"{k}: {json.dumps(v)}")

txt = "\n".join(out)
with open("RESULTS_SUMMARY.txt", "w") as fh:
    fh.write(txt)
print(txt)
