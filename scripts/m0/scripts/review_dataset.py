#!/usr/bin/env python3
"""Render the M0 prompt set for human review.

The build script's automated checks cover what is checkable -- label balance, position, the length
heuristic, banned wording. What they cannot cover is judgment: whether a neutral item really is a
third-party judgment rather than a first-person one, and whether a factual item's answer is as
unambiguous as its numbers claim. Those need eyes, and JSONL is not readable by eye.

    python m0/scripts/review_dataset.py --neutral          # the items that need judgment
    python m0/scripts/review_dataset.py --family binary_factual --limit 20
    python m0/scripts/review_dataset.py --sources          # the source tables, deduplicated
    python m0/scripts/review_dataset.py --out review.md    # write instead of print

--sources is usually the right place to start: the 1,750 rows are only ~150 distinct pieces of
content wrapped in different carriers and label schemes, so reviewing the sources covers the
judgment calls in a fraction of the reading.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DIR = ROOT / "data" / "rl" / "m0_format"


def load(split_dir: Path, splits: list[str]) -> list[dict]:
    rows = []
    for split in splits:
        path = split_dir / f"{split}.jsonl"
        if not path.exists():
            raise SystemExit(f"missing {path}; run: python -m m0.data.build_dataset --out {split_dir}")
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                row["_split"] = split
                rows.append(row)
    return rows


def render_sources() -> list[str]:
    """The source content, not the rendered prompts -- the unit at which judgment applies."""
    from scripts.m0.data import facts

    out = ["# M0 source content", ""]
    out += ["Review these rather than the 1,750 rendered rows: the rows are these items wrapped in",
            "different carriers and label schemes, so the content decisions all live here.", ""]

    out += ["## Neutral binary items (no ground truth, format-only scoring)", "",
            "CHECK: is each a third-party or artifact-level judgment? Nothing about the model's own",
            "experience, body, senses, identity, or future.", ""]
    for i, item in enumerate(facts.NEUTRAL_BINARY_ITEMS, 1):
        out += [f"{i:2d}. _{item['premise']}_",
                f"    **{item['question']}**",
                f"    - A: {item['option_a']}",
                f"    - B: {item['option_b']}", ""]

    out += ["## Neutral ternary items (no ground truth)", "",
            "CHECK: same framing rule, and that the answer really is a judgment call rather than a",
            "fact with a knowable answer.", ""]
    for i, item in enumerate(facts.NEUTRAL_TERNARY_ITEMS, 1):
        out += [f"{i:2d}. _{item['change']}_", f"    -> {item['quantity']}", ""]

    out += ["## Factual ternary items (ground truth)", "",
            "CHECK: is the stated answer unambiguously right? A debatable item trains the model to",
            "guess and makes the correctness term reward noise.", ""]
    for i, item in enumerate(facts.TERNARY_FACTUAL_ITEMS, 1):
        out += [f"{i:2d}. _{item['change']}_",
                f"    -> {item['quantity']}  **{item['answer']}**", ""]

    out += ["## Factual binary dimensions (ground truth from the numbers)", "",
            "CHECK: are the values right, and is every emitted pair genuinely unambiguous? Pairs are",
            "only emitted when the values differ by at least the separation shown.", ""]
    for dim in facts.FACTUAL_DIMENSIONS:
        sep = (f"ratio >= {dim['min_ratio']}" if "min_ratio" in dim
               else f"gap >= {dim['min_gap']}")
        out += [f"### {dim['key']} — \"{dim['question']}\" ({dim['unit']}, {sep}, "
                f"{'larger' if dim['larger_wins'] else 'smaller'} wins)", ""]
        for name, value in sorted(dim["items"], key=lambda x: x[1], reverse=True):
            out.append(f"- {value:>10,g}  {name}")
        out.append("")
    return out


def render_rows(rows: list[dict], limit: int | None) -> list[str]:
    out = []
    by_family: dict[str, list[dict]] = {}
    for row in rows:
        by_family.setdefault(row["family"], []).append(row)
    for family, family_rows in sorted(by_family.items()):
        out += [f"# {family} ({len(family_rows)} rows)", ""]
        for row in family_rows[:limit]:
            answer = row["answer"] or "(no ground truth)"
            out += [f"### {row['record_id']} · carrier={row['carrier']} · labels={'/'.join(row['labels'])} "
                    f"· answer={answer}", "", "```", row["prompt"], "```", ""]
        if limit and len(family_rows) > limit:
            out += [f"_...{len(family_rows) - limit} more; raise --limit to see them._", ""]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default=str(DEFAULT_DIR))
    parser.add_argument("--splits", nargs="+", default=["train"],
                        choices=["train", "dev", "heldout"])
    parser.add_argument("--family", default=None,
                        choices=["binary_factual", "ternary_factual", "binary_neutral",
                                 "ternary_neutral"])
    parser.add_argument("--neutral", action="store_true",
                        help="Only the no-ground-truth families -- the ones needing judgment")
    parser.add_argument("--sources", action="store_true",
                        help="The source tables instead of rendered rows (start here)")
    parser.add_argument("--limit", type=int, default=10, help="Rows per family (0 = all)")
    parser.add_argument("--out", default=None, help="Write markdown here instead of printing")
    args = parser.parse_args()

    if args.sources:
        lines = render_sources()
    else:
        rows = load(Path(args.dir), args.splits)
        if args.neutral:
            rows = [r for r in rows if not r["verifiable"]]
        if args.family:
            rows = [r for r in rows if r["family"] == args.family]
        if not rows:
            raise SystemExit("no rows matched those filters")
        lines = render_rows(rows, args.limit or None)

    text = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(text + "\n")
        print(f"wrote {args.out} ({len(lines)} lines)")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
