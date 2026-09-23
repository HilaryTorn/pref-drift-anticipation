#!/usr/bin/env python3
"""Add held-out rows to validation.jsonl WITHOUT re-drawing the sample that produced pool.jsonl.

The obvious way to get a bigger held-out set -- re-run build_sft_pools.py with a larger
--validation_count -- is unsafe, and check_pool_prefix_stability.py demonstrates why on this corpus:
build_sft_pools draws pool and validation from ONE `rng.sample(usable, pool_count + validation_count)`
call, and CPython's random.sample picks its algorithm from a threshold that depends on k
(`setsize = 21 + 4**ceil(log(k*3, 4))`; below it a partial shuffle, above it a selection set).

    k=1064 -> setsize 4117      k=1400 -> setsize 16405

python (4499 usable rows) and java (4146) sit ABOVE the first threshold and BELOW the second, so
raising the count flips the algorithm and rewrites their pool.jsonl entirely. rust (3952) and cpp
(3502) sit below both and are unaffected. A rebuild would therefore have silently replaced the
training data behind the python and java Phase 1.1 checkpoints while leaving rust and cpp intact --
meaning a spot-check on rust would have reported success.

This script sidesteps the problem instead of fighting it. The original 1064-row draw is reproduced
and treated as immovable; the new held-out rows are drawn from the rows that draw did NOT take, so
pool.jsonl is never rewritten and the existing 64 validation records are copied through byte-for-byte
from disk rather than regenerated. Disjointness from the training pool is guaranteed by construction,
which is the property the whole held-out arm depends on.

Existing validation ids (magicoder_validation_00000..00063) keep their meaning, so capability
problems already built against them stay valid; new rows continue from 00064.

Usage:
    python3 scripts/extend_validation_split.py --target_validation_count 400 --dry_run
    python3 scripts/extend_validation_split.py --target_validation_count 400
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_sft_pools import (  # noqa: E402
    LANG_TO_LANGUAGE_ID,
    SOURCE_LICENSE,
    SOURCE_NAME,
    load_jsonl,
    write_jsonl,
)
from check_pool_prefix_stability import usable_rows  # noqa: E402


def to_record(row: dict, intervention_id: str, language_id: str, split: str, index: int) -> dict:
    """Byte-compatible with build_sft_pools.build()'s record shape, including key order."""
    return {
        "intervention_id": intervention_id,
        "task": "write",
        "language_id": language_id,
        "messages": [
            {"role": "user", "content": (row.get("problem") or "").strip()},
            {"role": "assistant", "content": (row.get("solution") or "").strip()},
        ],
        "source": SOURCE_NAME,
        "license": SOURCE_LICENSE,
        "id": f"{intervention_id}.magicoder_{split}_{index:05d}",
        "split": split,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--pool_count", type=int, default=1000)
    parser.add_argument("--current_validation_count", type=int, default=64)
    parser.add_argument("--target_validation_count", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42, help="Must match the seed the pools were built with.")
    parser.add_argument("--extra_seed", type=int, default=20260821, help="Separate stream for the new rows; never reuses the original draw's state.")
    parser.add_argument("--languages", default=",".join(LANG_TO_LANGUAGE_ID))
    parser.add_argument("--dry_run", action="store_true", help="Report what would change; write nothing.")
    args = parser.parse_args()

    languages = [lang.strip() for lang in args.languages.split(",") if lang.strip()]
    n_new = args.target_validation_count - args.current_validation_count
    if n_new <= 0:
        raise SystemExit(f"--target_validation_count must exceed --current_validation_count ({args.current_validation_count})")

    for language in languages:
        language_id = LANG_TO_LANGUAGE_ID[language]
        intervention_id = f"coding.write.{language_id}"
        out_dir = ROOT / "data" / "training" / intervention_id
        usable = usable_rows(language)
        n = len(usable)
        original_k = args.pool_count + args.current_validation_count

        # Reproduce the original draw as INDICES. Verified against the object-level draw below
        # rather than assumed -- an index/object mismatch here would hand training rows to the
        # held-out set, which is the one failure this script exists to make impossible.
        rng = random.Random(args.seed)
        taken_idx = rng.sample(range(n), original_k)
        rng = random.Random(args.seed)
        taken_obj = rng.sample(usable, original_k)
        if [usable[i] for i in taken_idx] != taken_obj:
            raise SystemExit(f"{language}: index-level replay diverges from object-level draw; refusing to touch {out_dir}")

        # Sanity-check against what is actually committed, so a corpus or filter change is caught
        # here instead of quietly producing an overlapping "held-out" set.
        committed_pool = load_jsonl(out_dir / "pool.jsonl")
        replay_pool = [to_record(usable[i], intervention_id, language_id, "pool", j) for j, i in enumerate(taken_idx[: args.pool_count])]
        if [r["messages"] for r in replay_pool] != [r["messages"] for r in committed_pool]:
            raise SystemExit(f"{language}: replay does not reproduce committed pool.jsonl; refusing to touch {out_dir}")

        # Exclude BOTH the original draw and anything a previous extension already added, or a
        # second run re-picks rows that are already in validation.jsonl: `remaining` is derived from
        # the original 1064-row draw only, and rng.sample is prefix-stable, so re-sampling it with
        # the same extra_seed reproduces the earlier picks and duplicates them into the file.
        existing_messages = {
            json.dumps(record["messages"], sort_keys=True)
            for record in load_jsonl(out_dir / "validation.jsonl")
        }
        taken = set(taken_idx)
        remaining = [
            i
            for i in range(n)
            if i not in taken
            and json.dumps(
                to_record(usable[i], intervention_id, language_id, "validation", 0)["messages"],
                sort_keys=True,
            )
            not in existing_messages
        ]
        n_new = args.target_validation_count - len(existing_messages)
        if n_new <= 0:
            print(f"  {language:7s} already at {len(existing_messages)} held-out rows; nothing to add")
            continue
        if len(remaining) < n_new:
            print(f"  {language:7s} SKIPPED: needs {n_new} unused rows, only {len(remaining)} remain")
            continue

        extra_idx = random.Random(args.extra_seed).sample(remaining, n_new)
        assert not (set(extra_idx) & taken), "new held-out rows overlap the training draw"

        existing_validation = load_jsonl(out_dir / "validation.jsonl")
        new_records = [
            to_record(usable[i], intervention_id, language_id, "validation", len(existing_validation) + j)
            for j, i in enumerate(extra_idx)
        ]
        combined = existing_validation + new_records

        print(
            f"  {language:7s} usable={n:5d} unused={len(remaining):5d}  "
            f"validation {len(existing_validation)} -> {len(combined)}  (+{len(new_records)}, pool untouched)"
        )
        if not args.dry_run:
            write_jsonl(out_dir / "validation.jsonl", combined)

    if args.dry_run:
        print("\nDry run: nothing written.")
    else:
        print(f"\nWrote extended validation splits. pool.jsonl files were not opened for writing.")
        print("Rebuild the capability bank to pick up the new held-out candidates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
