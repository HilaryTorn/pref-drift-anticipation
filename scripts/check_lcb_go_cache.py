#!/usr/bin/env python3
"""Preflight a LiveCodeBench teacher cache before Linux grading.

Step 2.5 of docs/lcb-sft-runbook.md. Verification (`score_multilcb.py --evaluate_only`) only runs
on x86 Linux with the conda toolchain, so the first time anyone finds out the cache is malformed is
normally after a box has been launched and an environment built. This script reproduces every check
upstream's `eval_plang` makes while LOADING the cache -- and nothing that needs a compiler -- so a
format problem surfaces here, in seconds, on any machine.

What it checks, each mapped to the upstream behaviour it would otherwise trip:

  1. Cache exists at the exact path upstream derives from (release_version, k, debug). A mismatched
     --k or --release_version silently points at a different filename.
  2. Every benchmark question_id has a cache entry. Upstream prints "ERROR missing generations for N
     problems. Skipping missing problems." and grades a smaller set than you think.
  3. No entry has an all-empty output_list. Upstream's --continue_existing treats those as "not yet
     generated" and tries to CALL AN ENDPOINT during --evaluate_only, which on the grading box has
     none -- the run dies confusingly.
  4. output_list and code_list are both length k on every entry, and aligned. Upstream zips them
     against the benchmark positionally; a length mismatch grades the wrong code for the wrong
     problem.
  5. `prompt` parses as a JSON messages list with a user turn -- what build_lcb_go_pool.py reads to
     build the training row.
  6. Reports how many problems carry extractable code, by difficulty. This is the ceiling on the
     eventual pool: a problem with no code cannot pass, so if this is already tiny the run is not
     worth grading.

Usage:
    python3 scripts/check_lcb_go_cache.py --language php --k 2
    python3 scripts/check_lcb_go_cache.py --language csharp --k 1 --limit 3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_lcb_go_teacher import (  # noqa: E402
    CACHE_TEMPERATURE,
    CACHE_TOP_P,
    DEFAULT_LANGUAGE,
    LANGUAGES,
    MULTILCB_ROOT,
    load_upstream,
    teacher_name,
)
from lcb_languages import markdown_fence, upstream_language  # noqa: E402


def has_exact_output_shape(text: str, language: str) -> bool:
    fence = re.escape(markdown_fence(language))
    return re.fullmatch(
        rf"\s*```{fence}[ \t]*\r?\n(?:(?!```)[\s\S])+```\s*", text or ""
    ) is not None


def cache_path(release_version: str, k: int, limit: int, language: str) -> Path:
    folder = f"{release_version}_{'debug_' if limit else ''}{teacher_name(language)}_cot"
    upstream = upstream_language(language)
    return MULTILCB_ROOT / "output" / folder / f"codegeneration_{upstream}_{k}_{CACHE_TEMPERATURE}_{CACHE_TOP_P}_cot.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release_version", default="release_v5")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE, choices=LANGUAGES)
    parser.add_argument("--k", type=int, default=2, help="Must match the teacher run; it is in the cache filename.")
    parser.add_argument("--limit", type=int, default=0, help="Smoke: check the debug_ cache.")
    parser.add_argument("--dataset_path", default=None)
    args = parser.parse_args()

    path = cache_path(args.release_version, args.k, args.limit, args.language)
    if not path.exists():
        print(f"FAIL: no cache at {path}")
        print("      Check --k and --release_version match the teacher run (both are in the filename).")
        return 1

    dataset_path = Path(args.dataset_path) if args.dataset_path else ROOT / "data" / "reference" / "multilcb" / args.release_version
    print(f"Loading {args.release_version} benchmark to compare against ...")
    benchmark, _, _ = load_upstream(args.release_version, dataset_path, args.limit, args.language)
    expected = {problem.question_id: problem for problem in benchmark}

    records = json.loads(path.read_text())
    by_id = {record.get("question_id"): record for record in records}
    print(f"Cache: {len(records)} entries in {path.relative_to(ROOT)}")
    print(f"Benchmark: {len(benchmark)} problems\n")

    failures: list[str] = []
    warnings: list[str] = []

    missing = sorted(set(expected) - set(by_id))
    if missing:
        failures.append(
            f"{len(missing)} benchmark problems have no cache entry (upstream would skip them silently). "
            f"First few: {missing[:5]}"
        )
    extra = sorted(set(by_id) - set(expected))
    if extra:
        warnings.append(f"{len(extra)} cache entries are not in the benchmark; upstream ignores them. First few: {extra[:5]}")

    all_empty: list[str] = []
    bad_lengths: list[str] = []
    bad_prompt: list[str] = []
    bad_output_shape: list[str] = []
    with_code = 0
    code_by_difficulty: Counter = Counter()
    total_by_difficulty: Counter = Counter()

    for qid, record in by_id.items():
        if qid not in expected:
            continue
        outputs = record.get("output_list") or []
        codes = record.get("code_list") or []
        difficulty = record.get("difficulty", "?")
        total_by_difficulty[difficulty] += 1

        if not any((text or "").strip() for text in outputs):
            all_empty.append(qid)
        if len(outputs) != args.k or len(codes) != args.k:
            bad_lengths.append(f"{qid} (outputs={len(outputs)}, codes={len(codes)}, expected {args.k})")

        try:
            messages = json.loads(record.get("prompt") or "")
            if not any(m.get("role") == "user" for m in messages):
                bad_prompt.append(f"{qid} (no user turn)")
        except (json.JSONDecodeError, TypeError, AttributeError):
            bad_prompt.append(f"{qid} (prompt is not a JSON messages list)")

        if any((code or "").strip() for code in codes):
            with_code += 1
            code_by_difficulty[difficulty] += 1
        for index, (output, code) in enumerate(zip(outputs, codes)):
            if (code or "").strip() and not has_exact_output_shape(output or "", args.language):
                bad_output_shape.append(f"{qid}[{index}]")

    if all_empty:
        failures.append(
            f"{len(all_empty)} entries have an ALL-EMPTY output_list. Upstream's --continue_existing treats "
            f"these as ungenerated and will try to contact an endpoint during --evaluate_only, which the "
            f"grading box does not have. First few: {all_empty[:5]}"
        )
    if bad_lengths:
        failures.append(f"{len(bad_lengths)} entries have wrong list lengths: {bad_lengths[:5]}")
    if bad_prompt:
        failures.append(f"{len(bad_prompt)} entries have an unreadable prompt (build_lcb_go_pool.py needs it): {bad_prompt[:5]}")
    if bad_output_shape:
        failures.append(
            f"{len(bad_output_shape)} extractable solutions violate the required single fenced "
            f"{markdown_fence(args.language)!r} block with no surrounding prose. First few: "
            f"{bad_output_shape[:5]}"
        )

    print(f"Extractable {args.language} code, by difficulty (this is the CEILING on the eventual pool):")
    for level in ("easy", "medium", "hard"):
        total = total_by_difficulty.get(level, 0)
        if total:
            got = code_by_difficulty.get(level, 0)
            print(f"  {level:7s} {got:4d}/{total:4d}  ({got / total:.0%})")
    print(f"  {'TOTAL':7s} {with_code:4d}/{len(by_id):4d}  ({with_code / max(len(by_id), 1):.0%})\n")

    if with_code < 0.5 * len(by_id):
        warnings.append(
            f"Only {with_code}/{len(by_id)} problems produced any fenced {args.language} code. Check the teacher's "
            f"stop_reason histogram -- a wave of incomplete:max_output_tokens means --max_tokens is too low "
            f"or --effort too high."
        )

    for warning in warnings:
        print(f"WARN: {warning}")
    for failure in failures:
        print(f"FAIL: {failure}")

    if failures:
        print(f"\n{len(failures)} blocking problem(s). Do NOT ship this to the grading box yet.")
        return 1

    print("PASS: cache is well-formed and complete. Safe to grade on the Linux box:")
    print(
        f"  python3 scripts/score_multilcb.py --model_key {teacher_name(args.language)} --languages {args.language} "
        f"--release_version {args.release_version} --n {args.k} --evaluate_only --diagnose"
        + (f" --debug --debug_size {args.limit}" if args.limit else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
