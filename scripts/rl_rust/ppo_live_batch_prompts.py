#!/usr/bin/env python3
"""Stage A of the PPO example_datapoints live batch: write the pinned prompts file. No GPU.

WHY A LIVE BATCH EXISTS AT ALL. GRPO's example_datapoints are assembled from `rollout_samples.rank0.jsonl`, which its trainer logged during training. PPO logs no such file: `ppo_rust.yaml` sets `num_sample_generations: 0` and `train_rust_ppo.py` adds no rollout writer, so after a completed PPO run there is no recorded artifact showing what one PPO training datapoint contains. The battery's `formal_sample_rule` forbids synthetic examples, so the honest substitute is to generate from the trained policy and score with the same calibrated reward model PPO used -- a real program the PPO policy wrote, carrying a real score from the real scorer.

WHY THESE TWO PROBLEMS. `build_rust_rl_stimuli.py` pins the same record ids for all three RL arms so the three options differ only in the shape of the feedback, not in which problems appear. Those ids are already written into the stimuli file's PPO provenance block; this script reads them from there rather than re-deriving them, so the pinning cannot drift from what GRPO and DPO actually show.

Output is the row schema `eval_generate.py` expects, so generation reuses the merged-model vLLM path that already exists instead of a second, less-tested generator.

Usage (Mac or pod, no GPU):

    python3 rl-rust/ppo_live_batch_prompts.py --size 4b --out rl-rust/out/ppo-live-batch/4b.prompts.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def pinned_ids(size: str) -> list[str]:
    """The record ids GRPO and DPO already illustrate, read from the stimuli file."""
    stimuli_path = ROOT / "data" / "source" / f"coding_training_stimuli_v3_rust_rl_{size}.json"
    items = {item["id"]: item for item in json.loads(stimuli_path.read_text())}
    provenance = items["coding.ppo.rust"]["example_datapoints_provenance"]
    ids = provenance.get("pinned_record_ids")
    if not ids:
        raise SystemExit(f"{stimuli_path}: coding.ppo.rust carries no pinned_record_ids")

    # The pinning is only meaningful if the other two arms really show these problems.
    for other in ("coding.grpo.rust", "coding.dpo.rust"):
        theirs = items[other]["example_datapoints_provenance"].get("selected_record_ids")
        if theirs != ids:
            raise SystemExit(
                f"pinned ids disagree: coding.ppo.rust has {ids}, {other} shows {theirs}. "
                "Rebuild the stimuli file before generating the live batch."
            )
    return ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", required=True, choices=("4b", "9b"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ids = pinned_ids(args.size)
    cohort = ROOT / "rl-rust" / "out" / f"cohort-{args.size}-clean-n992" / "train.jsonl"
    if not cohort.exists():
        raise SystemExit(f"missing cohort file {cohort}")

    # The cohort file PPO trained on, by sha256 in its run_metadata -- so the prompt text here is
    # the same string the policy saw at training time, not a lookalike from another build.
    by_id = {}
    for line in cohort.open():
        row = json.loads(line)
        by_id[row["record_id"]] = row

    missing = [rid for rid in ids if rid not in by_id]
    if missing:
        raise SystemExit(f"pinned ids absent from {cohort.name}: {missing}")

    rows = [by_id[rid] for rid in ids]
    out = args.out or ROOT / "rl-rust" / "out" / "ppo-live-batch" / f"{args.size}.prompts.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"wrote {out}: {len(rows)} pinned prompts")
    for row in rows:
        print(f"  {row['record_id']}  {row['prompt'][:70]!r}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
