#!/usr/bin/env python3
"""Re-grade the stored completions of a past capability run and check the verdicts still match.

This is NOT a re-score. It generates nothing and cannot change a pass rate -- it replays each saved
`extracted_code` string through the same executors the live scorer uses and compares the result to
the `passed`/`total` recorded at the time. Two things it establishes, both without a GPU:

  1. Grading is deterministic and environment-portable. A problem that graded 5/5 on the run box
     should grade 5/5 here; a mismatch means the verdict depends on the machine (toolchain version,
     locale, timing) and every cross-run comparison in the study is on sand.
  2. Edits to the scorer did not disturb the execution path. Run it before and after touching
     score_coding_capability.py.

Use it as the cheap pre-flight before booking GPU time for a real re-score.

Two limits, both structural. Only languages whose toolchain is on PATH can be replayed (rust needs
rustc, java needs javac, cpp needs g++); missing ones are reported, never silently passed. And runs
saved before 2026-08-21 stored only post-extraction text, so a completion that had a code fence had
its reasoning discarded -- those runs can be re-graded but never re-extracted. Runs from the current
scorer also write raw_responses_coding_capability_*.jsonl, which does support re-extraction.

Usage:
    python3 scripts/verify_capability_harness.py results/coding-capability/qwen35-4b-m0-v4/coding_capability/*.json
    python3 scripts/verify_capability_harness.py --languages python,cpp results/coding-capability/*/coding_capability/*.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from score_coding_capability import (  # noqa: E402
    DEFAULT_PROBLEMS_PATH,
    LANGUAGES,
    score_problem,
    toolchain_missing,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("results", nargs="+", type=Path, help="coding_capability_*.json files to replay")
    parser.add_argument("--problems_path", default=str(DEFAULT_PROBLEMS_PATH))
    parser.add_argument("--languages", default=None, help="Comma-separated subset; default every language present that has a toolchain.")
    parser.add_argument("--exec_timeout", type=float, default=10.0)
    parser.add_argument("--max_problems", type=int, default=None, help="Per language per file, for a quick smoke.")
    args = parser.parse_args()

    problems = {p["id"]: p for p in json.loads(Path(args.problems_path).read_text())["problems"]}
    requested = [lang.strip() for lang in args.languages.split(",")] if args.languages else LANGUAGES

    usable, skipped = [], {}
    for lang in requested:
        missing = toolchain_missing(lang)
        if missing:
            skipped[lang] = missing
        else:
            usable.append(lang)
    for lang, missing in skipped.items():
        print(f"!! {lang}: skipped, toolchain missing on PATH ({', '.join(missing)})")
    if not usable:
        raise SystemExit("No requested language has its toolchain installed.")

    total_checked = total_mismatched = total_absent = 0

    for path in args.results:
        payload = json.loads(path.read_text())
        label = payload.get("model_key") or path.parent.parent.name
        for lang, res in payload["results"].items():
            if lang not in usable:
                continue
            problem_list = res["problems"]
            if args.max_problems:
                problem_list = problem_list[: args.max_problems]
            checked = mismatched = 0
            for recorded in problem_list:
                problem = problems.get(recorded["id"])
                if problem is None:
                    # The bank moved on since this run: not a harness fault, but it does mean the
                    # old numbers cannot be compared problem-for-problem against a new bank.
                    total_absent += 1
                    continue
                for sample in recorded["samples"]:
                    code = sample.get("extracted_code") or ""
                    outcome = score_problem(code, problem, args.exec_timeout)
                    checked += 1
                    if (outcome["passed"], outcome["total"]) != (sample["passed"], sample["total"]):
                        mismatched += 1
                        if mismatched <= 3:
                            print(
                                f"   MISMATCH {label} {recorded['id']}: "
                                f"recorded {sample['passed']}/{sample['total']} -> replay "
                                f"{outcome['passed']}/{outcome['total']}"
                            )
            total_checked += checked
            total_mismatched += mismatched
            verdict = "OK" if mismatched == 0 else f"{mismatched} MISMATCHED"
            print(f"  {label:34s} {lang:7s} {checked:4d} samples replayed  -> {verdict}")

    print(f"\n{total_checked} samples replayed, {total_mismatched} mismatched, {total_absent} problems no longer in the bank")
    if total_mismatched:
        print("Grading is NOT reproducible across machines -- fix before trusting any cross-run delta.", file=sys.stderr)
        return 1
    print("Grading reproduced exactly. Execution path is deterministic on this machine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
