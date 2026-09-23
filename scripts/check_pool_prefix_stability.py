#!/usr/bin/env python3
"""Prove that raising --validation_count leaves pool.jsonl untouched, BEFORE rebuilding anything.

Why this has to run first: build_sft_pools.build() draws one sample for both splits --

    selected = rng.sample(usable, pool_count + validation_count)
    pool, validation = selected[:pool_count], selected[pool_count:]

-- so changing validation_count changes `k`. If random.sample is not prefix-stable in k, a rebuild
silently rewrites pool.jsonl, and every Phase 1.1 checkpoint is then trained on data that no longer
exists on disk. That would invalidate the entire trained set with no error message anywhere.

CPython's sample is expected to be prefix-stable (both its selection-set and partial-shuffle paths
draw in a k-independent order), but "expected" is not good enough when the downside is losing the
training runs, so this measures it against the real corpus instead of trusting the implementation.

The script is read-only and self-validating. It reconstructs the builder's `usable` list using
build_sft_pools' OWN filter helpers, then checks in two stages:

  1. Fidelity: replay at the CURRENT counts (1000/64) and require byte-identical reproduction of the
     committed pool.jsonl and validation.jsonl. If this fails, the replication is wrong and stage 2
     means nothing -- so it is a hard error, not a warning.
  2. Stability: replay at the proposed counts and check that pool.jsonl is unchanged and the
     existing validation rows survive as a prefix.

Usage:
    python3 scripts/check_pool_prefix_stability.py --validation_count 400
    python3 scripts/check_pool_prefix_stability.py --validation_count 400 --languages rust
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
    MAX_CHARS,
    SOURCE_PATH,
    load_jsonl,
    solution_matches_language,
)
from validate_sft_data import elicitation_snippets, normalize  # noqa: E402


def usable_rows(language: str) -> list[dict]:
    """Reproduce build_sft_pools.build()'s filter loop exactly, in the same order.

    Order matters as much as membership: rng.sample indexes into this list, so a differently
    ordered but identically-membered list yields a different draw. Stage 1 in main() is what
    actually proves this function stayed faithful to the builder.
    """
    rows = [row for row in load_jsonl(SOURCE_PATH) if row.get("lang") == language]
    banned = elicitation_snippets()
    seen: set[str] = set()
    usable: list[dict] = []
    for row in rows:
        problem = (row.get("problem") or "").strip()
        solution = (row.get("solution") or "").strip()
        if not problem or not solution:
            continue
        if not solution_matches_language(solution, language):
            continue
        if len(problem) + len(solution) > MAX_CHARS:
            continue
        key = normalize(problem)
        if key in seen:
            continue
        if any(snippet in normalize(f"{problem}\n{solution}") for snippet in banned):
            continue
        seen.add(key)
        usable.append(row)
    return usable


def committed_messages(path: Path) -> list[list[dict]]:
    """The `messages` payload of each committed record -- the part a rebuild must not change. `id`
    and `split` are assigned by position at write time, so comparing them would beg the question."""
    return [record["messages"] for record in load_jsonl(path)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--pool_count", type=int, default=1000)
    parser.add_argument("--current_validation_count", type=int, default=64)
    parser.add_argument("--validation_count", type=int, default=400, help="Proposed new count.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--languages", default=",".join(LANG_TO_LANGUAGE_ID))
    args = parser.parse_args()

    languages = [lang.strip() for lang in args.languages.split(",") if lang.strip()]
    failures = []

    for language in languages:
        intervention_id = f"coding.write.{LANG_TO_LANGUAGE_ID[language]}"
        out_dir = ROOT / "data" / "training" / intervention_id
        usable = usable_rows(language)

        # --- Stage 1: is the replication faithful? -------------------------------------------
        rng = random.Random(args.seed)
        current_k = args.pool_count + args.current_validation_count
        replayed = rng.sample(usable, current_k)
        replay_pool = [row["messages"] if "messages" in row else None for row in replayed]
        # The builder wraps each source row into a chat record; compare on the user/assistant text.
        replay_pool = [
            [
                {"role": "user", "content": (row.get("problem") or "").strip()},
                {"role": "assistant", "content": (row.get("solution") or "").strip()},
            ]
            for row in replayed
        ]
        committed_pool = committed_messages(out_dir / "pool.jsonl")
        committed_val = committed_messages(out_dir / "validation.jsonl")

        if replay_pool[: args.pool_count] != committed_pool:
            print(f"  {language:7s} STAGE 1 FAILED: replay does not reproduce committed pool.jsonl")
            failures.append((language, "replication-infidelity"))
            continue
        if replay_pool[args.pool_count : current_k] != committed_val:
            print(f"  {language:7s} STAGE 1 FAILED: replay does not reproduce committed validation.jsonl")
            failures.append((language, "replication-infidelity"))
            continue

        # --- Stage 2: does a bigger k disturb the prefix? ------------------------------------
        rng = random.Random(args.seed)
        new_k = args.pool_count + args.validation_count
        if new_k > len(usable):
            print(f"  {language:7s} SKIPPED: needs {new_k} rows, only {len(usable)} usable")
            failures.append((language, "insufficient-rows"))
            continue
        bigger = rng.sample(usable, new_k)
        bigger_msgs = [
            [
                {"role": "user", "content": (row.get("problem") or "").strip()},
                {"role": "assistant", "content": (row.get("solution") or "").strip()},
            ]
            for row in bigger
        ]

        pool_stable = bigger_msgs[: args.pool_count] == committed_pool
        val_prefix_stable = bigger_msgs[args.pool_count : current_k] == committed_val
        gained = args.validation_count - args.current_validation_count

        status = "SAFE" if (pool_stable and val_prefix_stable) else "UNSAFE"
        print(
            f"  {language:7s} usable={len(usable):5d}  pool_unchanged={pool_stable}  "
            f"existing_validation_preserved={val_prefix_stable}  +{gained} new held-out rows  -> {status}"
        )
        if not pool_stable:
            failures.append((language, "pool-would-change"))
        elif not val_prefix_stable:
            failures.append((language, "validation-prefix-would-change"))

    print()
    if failures:
        print("DO NOT REBUILD. Failures:", failures, file=sys.stderr)
        print(
            "A 'pool-would-change' result means raising validation_count rewrites the training data "
            "the Phase 1.1 checkpoints were trained on. Rebuild validation into a SEPARATE file "
            "instead of resampling the shared draw.",
            file=sys.stderr,
        )
        return 1
    print(f"All languages safe: pool.jsonl is untouched by raising validation_count to {args.validation_count}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
