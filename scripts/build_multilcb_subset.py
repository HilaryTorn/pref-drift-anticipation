#!/usr/bin/env python3
"""Stage a filtered multi-LCB problem set as its own release tag.

Why this exists: the v6 window is 175 problems of which 147 are solved by neither the base nor any
arm, so the paired McNemar that decides every capability claim runs on ~43 live problems. The 80
hard problems contributed ONE discordant pair across both sizes. Adding easy+medium problems from
earlier shards is the cheapest way to buy power -- see private/docs/runpod-l40s-rust-rl-eval-runbook.md.

Upstream loads `DATASET_PATH` with `load_dataset(path, split="test")`, which globs `test*.jsonl`
in the directory, so a subset is just a directory with one `test.jsonl` in it. `RELEASE_VERSION`
is a free-form string that names the output folder, so the subset gets its own namespace and can
never be graded against, or written over, a v6 run.

**Contamination is the whole reason this script emits a manifest.** The Rust SFT arms were trained
on test-verified solutions to v1-v5 and held out on v6 (config.yaml, the LCB arm block). Joined by
`lcb_question_id`, 175 of the 222 easy+medium problems in v3-v5 are in that training pool. So:

  - RL arms (GRPO/DPO/PPO) are CLEAN on every shard. Their cohort is `open-r1/codeforces` via
    Nemotron; LCB v3-v6 are atcoder+leetcode only, so the platforms do not even intersect.
  - SFT arms are CONTAMINATED on 175 of 222. Scoring them there measures train-set recall, not
    capability. That is a legitimate diagnostic and a useful ceiling line, but it must never share
    a column with the RL arms. The manifest marks every problem so grading can split the two.

    python3 scripts/build_multilcb_subset.py --shards v3 v4 v5 --difficulty easy medium --tag v345em
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "data" / "reference" / "multilcb"
SFT_POOL = ROOT / "data" / "training" / "coding.write.rust_lcb" / "pool.jsonl"

# Where each shard's problems live once fetch_multilcb_dataset.py has staged them.
SHARD_FILES = {
    "v1": "release_v5/test.jsonl",
    "v2": "release_v5/test2.jsonl",
    "v3": "release_v5/test3.jsonl",
    "v4": "release_v5/test4.jsonl",
    "v5": "release_v5/test5.jsonl",
    "v6": "v6/test6.jsonl",
}


def sft_pool_ids() -> set[str]:
    """question_ids the Rust SFT arms trained on. Empty set if the pool is absent."""
    if not SFT_POOL.is_file():
        print(f"WARNING: {SFT_POOL} missing; cannot flag SFT contamination")
        return set()
    ids = set()
    for line in SFT_POOL.open():
        qid = json.loads(line).get("lcb_question_id")
        if qid:
            ids.add(qid)
    return ids


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shards", nargs="+", default=["v3", "v4", "v5"])
    p.add_argument("--difficulty", nargs="+", default=["easy", "medium"],
                   help="Hard problems are ~99%% both-fail on Rust at 4B/9B and cost a third of the "
                        "GPU bill for one discordant pair. Excluded by default on purpose.")
    p.add_argument("--tag", default="v345em", help="Release tag; names the staged dir AND upstream's output dir")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    dest = STAGE / args.tag
    if dest.exists() and not args.force:
        raise SystemExit(f"{dest} exists; pass --force to rebuild")
    dest.mkdir(parents=True, exist_ok=True)

    pool = sft_pool_ids()
    kept, rows, per_shard = [], [], {}
    for shard in args.shards:
        src = STAGE / SHARD_FILES[shard]
        if not src.is_file():
            raise SystemExit(f"missing {src}; run scripts/fetch_multilcb_dataset.py --release release_v5")
        n_all = n_kept = n_contam = 0
        for line in src.open():
            r = json.loads(line)
            n_all += 1
            if r.get("difficulty") not in args.difficulty:
                continue
            n_kept += 1
            qid = r.get("question_id")
            contaminated = qid in pool
            n_contam += contaminated
            rows.append(line)
            kept.append({
                "question_id": qid,
                "shard": shard,
                "difficulty": r.get("difficulty"),
                "platform": r.get("platform"),
                "contest_date": r.get("contest_date"),
                "in_rust_sft_pool": contaminated,
            })
        per_shard[shard] = {"problems": n_all, "kept": n_kept, "sft_contaminated": n_contam}
        print(f"  {shard}: kept {n_kept}/{n_all}  ({n_contam} in the Rust SFT training pool)")

    out = dest / "test.jsonl"
    out.write_text("".join(rows))

    n_contam = sum(k["in_rust_sft_pool"] for k in kept)
    manifest = {
        "schema": "multilcb_subset_v1",
        "tag": args.tag,
        "shards": args.shards,
        "difficulty": args.difficulty,
        "n_problems": len(kept),
        "n_sft_contaminated": n_contam,
        "n_sft_clean": len(kept) - n_contam,
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "problems": kept,
        "notes": {
            "rl_arms": "CLEAN on all shards: the RL cohort is open-r1/codeforces; these shards are "
                       "atcoder+leetcode only, so the platforms do not intersect.",
            "sft_arms": "CONTAMINATED where in_rust_sft_pool is true -- those problems are the SFT "
                        "arms' own training data. Score them as train-set recall, never as capability, "
                        "and never in the same column as the RL arms.",
            "comparability": "This tag is its own instrument. Nothing here pools with the stored v6 "
                             "numbers (0.0914 / 0.120 / 0.223); the base is re-measured in the same run.",
        },
    }
    (dest / "subset_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nStaged {len(kept)} problems -> {out}")
    print(f"  SFT-contaminated: {n_contam}   SFT-clean: {len(kept) - n_contam}   (RL arms: all {len(kept)} clean)")
    print(f"  manifest: {dest / 'subset_manifest.json'}")
    print(f"\nUse with:  --release_version {args.tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
