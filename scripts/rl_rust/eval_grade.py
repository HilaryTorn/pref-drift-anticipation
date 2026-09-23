"""Grade offline held-out generations with the Rust verifier. Needs rustc, no GPU.

Consumes eval_generate.py's completions JSONL and emits a row in exactly the
schema DriftCadenceCallback wrote to reward_log.jsonl, so offline points drop
into the same curve as any in-training ones:

    {"step", "eval_reward", "eval_reward_given_termination", "termination_rate",
     "n_generations", "n_truncated", "max_new_tokens"}

``eval_reward`` keeps its original meaning (truncated generations count as 0 and
are averaged in) because it is the PRODUCT of termination and quality; read it
next to ``eval_reward_given_termination``, which conditions on the model having
actually finished. Rust truncates 15-30%, so the two numbers routinely tell
different stories and the conditioned one is what survives truncation-matching.

    venv/bin/python rl-rust/eval_grade.py \
        --completions /root/eval/4b-step125.completions.jsonl \
        --prompts rl-rust/out/heldout-500-clean.jsonl \
        --step 125 --max_new_tokens 2048 --out /root/eval/reward_log.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "rl-rust"))

EVAL_MAX_TESTS = 32  # matches train_rust_grpo.EVAL_MAX_TESTS: richer than training's 12, affordable


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--completions", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--label", default=None, help="arm label recorded in the row")
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--max_tests", type=int, default=EVAL_MAX_TESTS)
    ap.add_argument("--timeout", type=float, default=5.0)
    args = ap.parse_args()

    from scripts.rl_rust.rl_rust.rewards import rust_toolchain_available, score_candidates

    if not rust_toolchain_available():
        raise SystemExit("rustc not on PATH: every generation would score 0.0 and look like a real null")

    verifiers = {r["record_id"]: r["verifier"] for r in
                 (json.loads(l) for l in open(args.prompts) if l.strip())}
    by_id: dict[str, list[dict]] = {}
    for line in open(args.completions):
        if not line.strip():
            continue
        row = json.loads(line)
        by_id.setdefault(row["record_id"], []).append(row)

    total, count = 0.0, 0
    finished_total, finished_count = 0.0, 0
    per_prompt = []
    for rid, cands in by_id.items():
        scored = score_candidates(
            [c["text"] for c in cands], verifiers[rid],
            timeout=args.timeout, max_tests=args.max_tests,
            finish_reasons=[c["finish_reason"] for c in cands], language="rust",
        )
        for s in scored:
            # Unconditional reward: a truncated generation is a 0, matching the
            # original callback so the two are the same measurement.
            reward = s["reward"] if s["reward"] is not None else 0.0
            total += reward
            count += 1
            if s["reward"] is not None:
                finished_total += s["reward"]
                finished_count += 1
        per_prompt.append(sum((s["reward"] or 0.0) for s in scored) / len(scored))

    row = {
        "step": args.step,
        "label": args.label,
        "eval_reward": total / max(count, 1),
        "eval_reward_given_termination": finished_total / finished_count if finished_count else 0.0,
        "termination_rate": finished_count / max(count, 1),
        "n_generations": count,
        "n_truncated": count - finished_count,
        "n_prompts": len(by_id),
        "max_new_tokens": args.max_new_tokens,
        "max_tests": args.max_tests,
        # Per-prompt spread: the standard error on the mean, so two arms can be
        # compared without pretending a 250-prompt sample is exact.
        "sem_per_prompt": (statistics.stdev(per_prompt) / (len(per_prompt) ** 0.5)) if len(per_prompt) > 1 else None,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as fh:
        fh.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(json.dumps(row, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
