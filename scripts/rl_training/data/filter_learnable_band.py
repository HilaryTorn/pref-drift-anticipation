#!/usr/bin/env python3
"""Filter a canonical coding split to prompts with useful verifier signal.

The score file is produced by ``training/build_dpo_pairs.py --scores_out``.
Default behavior keeps prompts where sampled completions have a meaningful
spread around the pass-rate midpoint. Tighten to ``--min_max_score 1.0
--max_min_score 0.0`` if you only want clean full-pass vs full-fail prompts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.rl_training.data.schema import read_coding_jsonl, read_prompt_scores_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Canonical coding_task_v1 JSONL")
    parser.add_argument("--scores", required=True, help="prompt_score_v1 JSONL")
    parser.add_argument("--out", required=True, help="Filtered coding_task_v1 JSONL")
    parser.add_argument("--manifest_out", default=None)
    parser.add_argument("--min_max_score", type=float, default=0.5, help="Require max score >= this")
    parser.add_argument("--max_min_score", type=float, default=0.5, help="Require min score <= this")
    parser.add_argument("--min_margin", type=float, default=0.5, help="Require max-min >= this")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_coding_jsonl(args.dataset)
    scores_by_id = read_prompt_scores_jsonl(args.scores)

    kept, missing = [], 0
    for row in rows:
        record_id = row.get("record_id") or row.get("hash_id")
        score_row = scores_by_id.get(record_id)
        if score_row is None:
            missing += 1
            continue
        min_score = float(score_row.get("min_score", min(score_row["scores"])))
        max_score = float(score_row.get("max_score", max(score_row["scores"])))
        if (
            max_score >= args.min_max_score
            and min_score <= args.max_min_score
            and max_score - min_score >= args.min_margin
        ):
            kept.append(row)

    write_jsonl(args.out, kept)
    manifest = {
        "schema": "learnable_band_manifest_v1",
        "dataset": args.dataset,
        "scores": args.scores,
        "out": args.out,
        "input_rows": len(rows),
        "score_rows": len(scores_by_id),
        "missing_score_rows": missing,
        "kept_rows": len(kept),
        "criteria": {
            "min_max_score": args.min_max_score,
            "max_min_score": args.max_min_score,
            "min_margin": args.min_margin,
        },
    }
    manifest_path = Path(args.manifest_out) if args.manifest_out else Path(args.out).with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n")
    print(f"[band] kept {len(kept)}/{len(rows)} rows -> {args.out}")
    print(f"[band] wrote manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
