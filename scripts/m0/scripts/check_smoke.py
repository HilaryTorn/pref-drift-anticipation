#!/usr/bin/env python3
"""Inspect a finished M0 run and report whether the plumbing is sound.

Point it at a run directory after the smoke run:

    python m0/scripts/check_smoke.py --run runs/qwen4b-m0-smoke

"The smoke run completed without crashing" is a much weaker statement than it appears. Every
failure below produces a run that finishes cleanly, writes all its files, and trains on a signal
that is wrong -- so the only way to catch them is to look at the artifacts on purpose.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="run directory (save_dir/run_name)")
    parser.add_argument("--show", type=int, default=1, help="sample completions to print")
    args = parser.parse_args()

    run = Path(args.run)
    if not run.exists():
        raise SystemExit(f"no such run directory: {run}")

    problems: list[str] = []
    notes: list[str] = []

    # --- 0. record which attention path this run used ---------------------------------------------
    # Reported, not enforced. The fused kernels would be faster, but they are unavailable on this
    # stack: flash-linear-attention needs Triton >= 3.3.0 and torch 2.5.1 pins Triton 3.1.0, while
    # causal-conv1d's setup.py needs nvcc, which the DLAMI does not ship (driver only, no toolkit).
    # Both were tried on 2026-07-21; fla installed but raised
    # `TypeError: Autotuner.__init__() got an unexpected keyword argument 'do_bench'` on import,
    # which is worse than absent -- transformers sees the package, takes the fast-path branch, and
    # then crashes. It was uninstalled.
    #
    # What matters for validity is that the smoke and the real run use the SAME path. Since neither
    # can use the kernels, the fallback is the validated configuration and its absence is not a
    # failure. If the pins ever move, install them BEFORE the smoke, never between smoke and run.
    kernels = {}
    for module, package in (("fla", "flash-linear-attention"), ("causal_conv1d", "causal-conv1d")):
        try:
            __import__(module)
            kernels[package] = True
        except Exception:  # noqa: BLE001
            kernels[package] = False
    if all(kernels.values()):
        notes.append("fused hybrid-attention kernels active")
    else:
        absent = [name for name, ok in kernels.items() if not ok]
        notes.append(f"reference PyTorch attention path ({', '.join(absent)} unavailable) -- "
                     f"slower, and the same path the real run will take, so the comparison holds")

    # --- 0b. wall-clock, so speed regressions are visible rather than inferred -------------------
    state_files = sorted(run.glob("checkpoint-*/trainer_state.json"))
    if state_files:
        state = json.loads(state_files[-1].read_text())
        runtime = next((entry.get("train_runtime") for entry in reversed(state.get("log_history", []))
                        if entry.get("train_runtime")), None)
        steps = state.get("global_step")
        if runtime and steps:
            notes.append(f"wall clock: {runtime:.0f}s for {steps} steps "
                         f"({runtime / steps:.0f}s/step) -- compare across runs to see whether a "
                         f"stack change actually helped")

    rollouts = []
    for path in sorted(run.glob("rollout_samples.rank*.jsonl")):
        rollouts += load_jsonl(path)
    format_log = load_jsonl(run / "format_log.jsonl")
    checkpoints = sorted(p.name for p in run.glob("checkpoint-*"))

    # --- 1. artifacts exist ----------------------------------------------------------------------
    if not checkpoints:
        problems.append("no checkpoint-* directories: nothing was saved")
    if not rollouts:
        problems.append("no rollout_samples.rank*.jsonl: log_rollouts is off, or the reward never ran")
    if not format_log:
        problems.append("no format_log.jsonl: FormatCadenceCallback never fired")

    if not rollouts:
        print("cannot check further without rollout samples")
        for problem in problems:
            print(f"  FAIL  {problem}")
        return 1

    # --- 2. THE critical check: are the think tags still in the decoded text? ---------------------
    # If TRL strips </think>, split_completion() treats the whole completion as visible content and
    # the band term measures total length instead of reasoning length. The reward curve looks
    # completely normal while optimizing the wrong quantity, so nothing else in this run would
    # reveal it. This is the single reason the smoke run has to be on 4B.
    with_close = sum("</think>" in row["completion"] for row in rollouts)
    share = with_close / len(rollouts)
    if with_close == 0:
        problems.append(
            "NO completion contains '</think>'. Either the model is not reasoning natively, or TRL "
            "stripped the special tokens. If the latter, the band term is silently measuring total "
            "length instead of reasoning length. Do not start the real run until this is resolved."
        )
    elif share < 0.25:
        notes.append(
            f"only {share:.0%} of completions closed a think block. Low, but that may just be an "
            f"untrained model rambling past the limit -- check n_reasoning_tokens below."
        )
    else:
        notes.append(f"{share:.0%} of completions contain '</think>' -- tags survive decoding")

    # --- 3. is there a learning signal? ----------------------------------------------------------
    # GRPO learns from WITHIN-GROUP reward spread. If every completion for a prompt scores the same,
    # its advantages are all zero and that prompt teaches nothing -- the run burns GPU on no
    # gradient. Uniformly zero spread usually means the reward is saturated or degenerate.
    stds = [row.get("group_std", 0.0) for row in rollouts]
    zero_groups = sum(1 for s in stds if s < 1e-6) / len(stds)
    if zero_groups > 0.9:
        problems.append(
            f"{zero_groups:.0%} of rollouts sit in groups with zero reward spread. GRPO's advantage "
            f"is the within-group deviation, so those prompts produce no gradient at all."
        )
    else:
        notes.append(f"reward spread present ({zero_groups:.0%} of rollouts in zero-variance groups)")

    # --- 4. reward term breakdown ----------------------------------------------------------------
    n = len(rollouts)
    commit = sum(row.get("commit", 0) for row in rollouts) / n
    over = sum(bool(row.get("over_budget")) for row in rollouts) / n
    lengths = sorted(row.get("n_reasoning_tokens", 0) for row in rollouts)
    mean_reward = sum(row["reward"] for row in rollouts) / n
    notes.append(f"rollouts={n} mean_reward={mean_reward:+.3f} commit_rate={commit:.2f} "
                 f"over_budget={over:.2f} reasoning_tokens median={lengths[n // 2]} "
                 f"p95={lengths[int(n * 0.95)]}")

    if commit == 0:
        notes.append("commit rate is 0 at smoke scale -- expected for an untrained model, and "
                     "exactly the problem M0 exists to fix. Not a failure here.")

    # --- 5. baseline present? --------------------------------------------------------------------
    steps = [entry["step"] for entry in format_log]
    if 0 not in steps:
        problems.append(
            "format_log.jsonl has no step-0 entry, so there is no untrained baseline to read the "
            "checkpoints against (on_train_begin did not fire)."
        )
    else:
        base = next(e for e in format_log if e["step"] == 0)
        notes.append(f"step-0 baseline: commit_rate={base['commit_rate']:.3f} "
                     f"mean_reasoning_tokens={base['mean_reasoning_tokens']:.0f}")

    # --- report -----------------------------------------------------------------------------------
    print(f"run: {run}")
    print(f"checkpoints: {checkpoints or 'none'}")
    print(f"format_log steps: {steps or 'none'}\n")
    for note in notes:
        print(f"  ok    {note}")
    for problem in problems:
        print(f"  FAIL  {problem}")

    if args.show:
        print("\n--- sample completion(s) ---")
        for row in rollouts[: args.show]:
            print(f"\n[reward={row['reward']:+.3f} commit={row.get('commit')} "
                  f"band={row.get('band')} n_reasoning={row.get('n_reasoning_tokens')}]")
            print(row["completion"][:1500])

    print()
    if problems:
        print(f"NOT READY: {len(problems)} problem(s) above must be resolved before the real run.")
        return 1
    print("plumbing looks sound. Safe to start the real run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
