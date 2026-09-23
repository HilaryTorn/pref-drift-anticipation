"""Build the exact-size clean cohorts the Rust GRPO runs train on.

Two things happen here, and both are checked rather than assumed:

**Drop scraped-SAMPLE contamination.** Some Nemotron test inputs end with the
literal token `SAMPLE`, swept in from the HackerEarth problem page during
scraping. A correct solution CANNOT pass such a problem, and the broken test sits
at index 0 -- which `_subsample_indices` always includes in the scored subset. So
these problems inject pure noise into the reward and quietly reward defensive
parsing over correctness. See private/nemotron-contamination/.

The registry-repaired-v1 cohorts already fixed ~97% of it (82 -> 2 at 9B,
79 -> 1 at 4B). This removes the remainder.

**Size to a multiple of the step batch.** train_rl.load_config hard-fails unless
dataset rows == expected_unique_prompts == max_steps * unique_prompts_per_step.
992 = 124 steps x 8 prompts, and it is reachable from both cohorts after
cleaning, so 4B and 9B train on the same number of prompts.

Order is preserved (no reshuffle), so the selection is deterministic and the
cohort remains a prefix-stable subset of the frozen one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

COHORTS = {
    "9b": REPO / "data/rl/v1/cohorts/prism-drift-qwen35-9b-m0-v4--rev-8f3d499236d7--data-fulltests-v2-registry-repaired-v1-shared-n1000-s0/train.jsonl",
    "4b": REPO / "data/rl/v1/cohorts/prism-drift-qwen35-4b-m0-v4--rev-dca63300371b--data-fulltests-v2-registry-repaired-v1-shared-n1000-s0/train.jsonl",
}


def is_contaminated(row: dict) -> bool:
    """The doc's exact test. Do NOT widen to 'INPUT' -- 'END OF INPUT' is a
    legitimate sentinel some problems specify, and flagging it gives false
    positives (a mistake already made and corrected once in this project)."""
    return any(
        (t or "").rstrip().endswith("SAMPLE")
        for t in (row.get("verifier", {}).get("test_inputs") or [])
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=sorted(COHORTS), required=True)
    ap.add_argument("--n", type=int, default=992, help="must equal max_steps * 8")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    src = COHORTS[args.size]
    rows = [json.loads(l) for l in src.open()]
    print(f"source: {src.name}  ({len(rows)} rows)")

    dropped = {"contaminated": [], "no_tests": [], "bad_type": []}
    keep = []
    for r in rows:
        v = r.get("verifier", {})
        if v.get("type") not in {"io_tests", "reference_io_tests"}:
            dropped["bad_type"].append(r["record_id"]); continue
        ins, outs = v.get("test_inputs") or [], v.get("test_outputs") or []
        if not ins or not outs or len(ins) != len(outs):
            dropped["no_tests"].append(r["record_id"]); continue
        if is_contaminated(r):
            dropped["contaminated"].append(r["record_id"]); continue
        keep.append(r)

    for reason, ids in dropped.items():
        if ids:
            print(f"  dropped {len(ids):3d} ({reason}): {ids[:4]}{' ...' if len(ids) > 4 else ''}")
    print(f"  clean pool: {len(keep)}")

    if len(keep) < args.n:
        print(f"ERROR: only {len(keep)} clean rows, need {args.n}")
        return 1
    keep = keep[: args.n]

    out_dir = Path(args.out_dir or (REPO / f"rl-rust/out/cohort-{args.size}-clean-n{args.n}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "train.jsonl"
    with out.open("w") as fh:
        for r in keep:
            fh.write(json.dumps(r) + "\n")

    # --- verify the artifact we just wrote, not the plan -----------------------
    back = [json.loads(l) for l in out.open()]
    assert len(back) == args.n, f"wrote {len(back)}, wanted {args.n}"
    assert not any(is_contaminated(r) for r in back), "contamination survived"
    assert all((r["verifier"].get("test_inputs") or []) for r in back), "row with no tests"
    ids = [r["record_id"] for r in back]
    assert len(set(ids)) == len(ids), "duplicate record_id"

    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    manifest = {
        "schema": "rust_rl_clean_cohort_v1",
        "size": args.size,
        "source": str(src.relative_to(REPO)),
        "source_rows": len(rows),
        "dropped": {k: len(v) for k, v in dropped.items()},
        "dropped_ids": dropped,
        "rows": len(back),
        "max_steps": args.n // 8,
        "unique_prompts_per_step": 8,
        "sha256": digest,
        "order": "source order preserved; first N clean rows",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))

    print(f"  wrote {out}")
    print(f"  rows {len(back)}  -> max_steps {args.n // 8} x 8 prompts")
    print(f"  sha256 {digest[:16]}...")
    print("  VERIFIED: 0 contaminated, no empty test suites, no duplicate ids")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
