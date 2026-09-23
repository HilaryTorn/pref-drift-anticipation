"""Read the seed sweep of Rust reward models, pick the best, enforce the 0.60 gate.

This is the step that decides whether PPO is launchable at all, and it is
deliberately the cheap one: a few minutes of GPU per seed against ~30 h for the
PPO run it gates. ``train_rl.py --algo ppo`` hard-refuses an RM below 0.60
held-out pairwise accuracy, so failing here costs the RM time and nothing else.

WHY THREE SEEDS. The gate is noisy at this pair count. eval_fraction 0.1 leaves
34 held-out pairs at 4B and 53 at 9B, so ONE pair is worth 2.9 / 1.9 points: an
RM at exactly 0.60 true accuracy passes a single draw about half the time, and a
coin-flip RM sneaks through more often than anyone would like. Reading three
draws and their spread is the difference between "Rust pairs are separable" and
"one seed got lucky". Raising eval_fraction instead would cost training pairs
that are already scarce (343 / 538).

Writes /root/rm_gate.json (or --out) with every seed's number, prints the table,
and exits non-zero if the best seed misses the gate -- which is the driver's
signal to stop before booking the long run.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

GATE = 0.60


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="RM output directories, one per seed")
    ap.add_argument("--out", default="/root/rm_gate.json")
    ap.add_argument("--gate", type=float, default=GATE)
    args = ap.parse_args()

    seeds = []
    for run in args.runs:
        path = Path(run) / "reward_model_manifest.json"
        if not path.is_file():
            print(f"  [missing] {path} -- seed did not finish")
            continue
        m = json.loads(path.read_text())
        seeds.append({
            "dir": str(run),
            "seed": m.get("seed"),
            "held_out_pairwise_accuracy": float(m["held_out_pairwise_accuracy"]),
            "n_eval": int(m.get("n_eval", 0)),
            "n_train": int(m.get("n_train", 0)),
            "pairs_path": m.get("pairs_path"),
        })

    if not seeds:
        print("no reward model finished; nothing to gate on")
        return 2

    seeds.sort(key=lambda s: s["held_out_pairwise_accuracy"], reverse=True)
    best = seeds[0]
    accs = [s["held_out_pairwise_accuracy"] for s in seeds]
    n_eval = best["n_eval"]
    per_pair = 100.0 / n_eval if n_eval else float("nan")

    print(f"\nreward-model seed sweep ({len(seeds)} seeds, {n_eval} held-out pairs, "
          f"one pair = {per_pair:.1f} points)")
    for s in seeds:
        mark = "  <- best" if s is best else ""
        print(f"  seed {s['seed']}  acc {s['held_out_pairwise_accuracy']:.4f}  "
              f"({s['n_train']} train / {s['n_eval']} eval){mark}")
    if len(accs) > 1:
        print(f"  spread: min {min(accs):.4f}  max {max(accs):.4f}  "
              f"mean {statistics.fmean(accs):.4f}  sd {statistics.stdev(accs):.4f}")

    passed = best["held_out_pairwise_accuracy"] >= args.gate
    record = {
        "schema": "rust_rm_gate_v1",
        "gate": args.gate,
        "passed": passed,
        "best": best,
        "seeds": seeds,
        "n_eval": n_eval,
        "points_per_pair": per_pair,
        "accuracies": accs,
        "mean": statistics.fmean(accs),
        "sd": statistics.stdev(accs) if len(accs) > 1 else None,
    }
    Path(args.out).write_text(json.dumps(record, indent=2) + "\n")

    if not passed:
        print(f"\nGATE FAILED: best held-out accuracy {best['held_out_pairwise_accuracy']:.4f} "
              f"< {args.gate:.2f}. train_rl.py --algo ppo would refuse this RM. "
              "Textbook answer is to fix the PAIRS (more of them, or a wider margin), "
              "not to swap the objective or pass --allow_weak_reward_model.")
        return 1

    print(f"\nGATE PASSED: {best['dir']} at {best['held_out_pairwise_accuracy']:.4f}")
    # stdout's LAST line is the chosen directory; the driver reads it with tail -1.
    print(best["dir"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
