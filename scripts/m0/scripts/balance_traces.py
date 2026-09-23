"""Measure — and optionally fix — answer skew in a warm-start SFT trace set.

The dataset the traces are generated from is answer-balanced by construction, but the warm-start's survival filters are not: whether a generation commits, closes its block, and lands the right answer varies with the answer itself, so the KEPT set drifts. Measured on the v2 sets, the drift is real — the 4B kept 6 A vs 13 B in its A/B cell, 23 C vs 13 D, and 9 MORE / 14 LESS / 18 SAME in ternary — and SFT on a skewed set installs a label prior inside that cell, which is exactly what M0 exists to avoid. (The methodology's "no LESS prior" check was measured on the 9B's traces; it never covered the 4B.)

Run this on the box between build_warmstart.py and train_warmstart.py:

    python m0/scripts/balance_traces.py results/m0_warmstart_4b/sft_traces.jsonl                # report only
    python m0/scripts/balance_traces.py ...jsonl --out ...balanced.jsonl                        # report + write cleaned set

Report mode exits non-zero if any cell exceeds --tolerance, so it can gate a runbook step. Clean mode trims each (family x label-scheme) cell down to answer balance by dropping surplus traces from over-represented answers — dropping, never duplicating, so every remaining trace is still a real generation. Within an answer group the traces farthest from --target_tokens are dropped first (the warm-start optimises reasoning length, so the trims cost the least-typical traces), with record_id as the deterministic tie-break. Trimming can only REMOVE surplus: if a cell is short of an answer outright, the fix is regeneration (build_warmstart.py --per_cell_target), not cleaning, and the report says so rather than pretending balance was achieved.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def cell_key(rec: dict) -> str:
    return f"{rec['family']}|{'/'.join(rec['labels'])}"


def load_traces(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for field in ("family", "labels", "prompt", "completion"):
                if field not in rec:
                    raise SystemExit(f"{path}:{line_no}: not a warm-start trace row (missing {field!r})")
            records.append(rec)
    return records


def analyze(records: list[dict], tolerance: float) -> tuple[dict, list[str]]:
    """Per-cell answer distributions for verifiable traces, plus the violations against `tolerance`."""
    cells: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    neutral_counts: dict[str, int] = defaultdict(int)
    for rec in records:
        if rec.get("verifiable") and rec.get("answer"):
            cells[cell_key(rec)][rec["answer"]].append(rec)
        else:
            neutral_counts[cell_key(rec)] += 1

    report: dict = {"tolerance": tolerance, "cells": {}, "neutral_cells": dict(sorted(neutral_counts.items()))}
    violations: list[str] = []
    for key in sorted(cells):
        by_answer = cells[key]
        labels = key.split("|")[1].split("/")
        total = sum(len(v) for v in by_answer.values())
        fair = 1.0 / len(labels)
        counts = {label: len(by_answer.get(label, [])) for label in labels}
        shares = {label: count / total for label, count in counts.items()}
        worst = max(abs(share - fair) for share in shares.values())
        report["cells"][key] = {"counts": counts, "shares": {k: round(v, 3) for k, v in shares.items()},
                               "worst_abs_deviation": round(worst, 3)}
        if worst > tolerance:
            violations.append(
                f"{key}: answer shares {shares} deviate {worst:.1%} from uniform (tolerance {tolerance:.0%})"
            )
        missing = [label for label, count in counts.items() if count == 0]
        if missing:
            violations.append(
                f"{key}: no kept trace answers {missing} at all — trimming cannot fix an absence; "
                f"regenerate with build_warmstart.py --per_cell_target instead"
            )
    return report, violations


def trim_to_balance(records: list[dict], tolerance: float, target_tokens: int) -> tuple[list[dict], list[dict]]:
    """Drop the minimum surplus so every cell's answer shares sit within `tolerance` of uniform.

    Within each cell, cap every answer at floor((1 + tolerance-margin) x the smallest achievable fair share) by iteratively dropping from the most over-represented answer until the worst deviation clears. Dropping shrinks the cell, which RAISES the other answers' shares, so this is done as a loop against the live distribution rather than a one-shot cap.
    """
    keep = {id(rec): rec for rec in records}
    cells: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        if rec.get("verifiable") and rec.get("answer"):
            cells[cell_key(rec)][rec["answer"]].append(rec)

    dropped: list[dict] = []
    for key in sorted(cells):
        by_answer = cells[key]
        labels = key.split("|")[1].split("/")
        fair = 1.0 / len(labels)
        # Drop least-typical first: farthest reasoning length from the target, then record_id.
        for group in by_answer.values():
            group.sort(
                key=lambda r: (-abs(r.get("n_reasoning_tokens", target_tokens) - target_tokens),
                               r.get("record_id", "")),
            )
        # Drop from the largest answer group until EVERY share is within tolerance of uniform —
        # including the under-represented side: an answer that is scarce can only be brought back
        # toward fair share by shrinking the cell around it. Whenever any share is out of band,
        # the largest group is necessarily above fair (shares sum to 1), so dropping from it always
        # makes progress, and the loop bottoms out at exact balance (min count each) if it must.
        while True:
            total = sum(len(v) for v in by_answer.values())
            if total == 0:
                break
            shares = {label: len(by_answer.get(label, [])) / total for label in labels}
            if max(abs(share - fair) for share in shares.values()) <= tolerance + 1e-9:
                break
            worst_label = max(shares, key=lambda label: shares[label])
            victim = by_answer[worst_label].pop(0)
            dropped.append(victim)
            del keep[id(victim)]

    return [rec for rec in records if id(rec) in keep], dropped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("traces", help="warm-start trace JSONL (sft_traces.jsonl or *_clean.jsonl)")
    parser.add_argument("--tolerance", type=float, default=0.10,
                        help="max allowed absolute deviation of any answer's share from uniform, per cell. "
                             "Looser than the dataset build's 0.08 because trace cells are far smaller than "
                             "dataset cells; tighten it if --per_cell_target has made the cells large.")
    parser.add_argument("--target_tokens", type=int, default=150,
                        help="when trimming, drop traces farthest from this reasoning length first — "
                             "match build_warmstart's --target_trace_tokens")
    parser.add_argument("--out", default=None,
                        help="write the trimmed-to-balance trace set here (report-only without it). "
                             "Refuses to overwrite the input.")
    parser.add_argument("--report", default=None,
                        help="also write the JSON report here (default: <traces>_balance_report.json "
                             "next to the input)")
    args = parser.parse_args()

    in_path = Path(args.traces)
    records = load_traces(in_path)
    report, violations = analyze(records, args.tolerance)
    print(f"[balance] {in_path}: {len(records)} traces, {len(report['cells'])} verifiable cells")
    for key, cell in report["cells"].items():
        flag = "  <-- SKEWED" if cell["worst_abs_deviation"] > args.tolerance else ""
        print(f"[balance]   {key:45s} {cell['counts']}  worst dev {cell['worst_abs_deviation']:.1%}{flag}")

    if args.out:
        out_path = Path(args.out)
        if out_path.resolve() == in_path.resolve():
            raise SystemExit("--out must differ from the input; keep the raw kept set inspectable")
        kept, dropped = trim_to_balance(records, args.tolerance, args.target_tokens)
        with out_path.open("w") as f:
            for rec in kept:
                f.write(json.dumps(rec, ensure_ascii=True) + "\n")
        print(f"[balance] wrote {len(kept)} traces to {out_path} ({len(dropped)} dropped to restore balance)")
        report["cleaned"] = {"out": str(out_path), "kept": len(kept), "dropped": len(dropped),
                             "dropped_record_ids": sorted(r.get("record_id", "?") for r in dropped)}
        post_report, post_violations = analyze(kept, args.tolerance)
        report["after"] = post_report["cells"]
        # Absences survive trimming by definition; only they should remain.
        remaining = [v for v in post_violations if "no kept trace" in v]
        if len(post_violations) != len(remaining):
            raise SystemExit("[balance] BUG: trimming left share violations behind; do not train on the output")
        violations = remaining

    report_path = Path(args.report) if args.report else in_path.with_name(in_path.stem + "_balance_report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[balance] report -> {report_path}")

    if violations:
        print(f"\n[balance] {len(violations)} problem(s):")
        for violation in violations:
            print(f"  - {violation}")
        return 1
    print("[balance] all cells within tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
