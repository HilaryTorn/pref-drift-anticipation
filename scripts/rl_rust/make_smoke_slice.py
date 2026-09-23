"""Cut the exact-size slices the Rust GRPO smoke config requires.

train_rl.load_config hard-fails unless the training set has exactly
``expected_unique_prompts`` rows, so the slice size is not a preference.

Takes rows from the FRONT of the frozen cohort deterministically. This is a
plumbing shakedown, not a measurement, so representativeness does not matter --
reproducibility does.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

DEFAULT_COHORT = (
    REPO / "data/rl/v1/cohorts"
    / "prism-drift-qwen35-9b-m0-v4--rev-8f3d499236d7--data-fulltests-v2-registry-repaired-v1-shared-n1000-s0"
    / "train.jsonl"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", default=str(DEFAULT_COHORT))
    parser.add_argument("--train_rows", type=int, default=24, help="must equal expected_unique_prompts")
    parser.add_argument("--eval_rows", type=int, default=4)
    parser.add_argument("--out_dir", default=str(REPO / "rl-rust/out/smoke"))
    args = parser.parse_args()

    cohort = Path(args.cohort)
    if not cohort.is_file():
        print(f"cohort not found: {cohort}")
        return 1

    rows = [json.loads(line) for line in cohort.open()]
    print(f"cohort {cohort.name}: {len(rows)} rows")

    need = args.train_rows + args.eval_rows
    if len(rows) < need:
        print(f"cohort too small: need {need}, have {len(rows)}")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = rows[: args.train_rows]
    # Held-out slice must not overlap the training slice, or eval_reward is
    # measuring memorization of the 24 prompts it just trained on.
    evalset = rows[args.train_rows : args.train_rows + args.eval_rows]

    for name, subset in (("train.jsonl", train), ("eval.jsonl", evalset)):
        path = out_dir / name
        with path.open("w") as handle:
            for row in subset:
                handle.write(json.dumps(row) + "\n")
        print(f"  wrote {path}  ({len(subset)} rows)")

    train_ids = {r.get("record_id") for r in train}
    eval_ids = {r.get("record_id") for r in evalset}
    assert not (train_ids & eval_ids), "train/eval overlap"
    print(f"  no overlap between train and eval ({len(train_ids)} / {len(eval_ids)} ids)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
