#!/usr/bin/env python3
"""Adjudicate two independent sources against the stored answer key and write a corrected bank.

The capability bank's expected values were captured by executing each Magicoder row's own reference
solution -- unvetted 2023 GPT-3.5 output. Where that reference was wrong, the key encodes its bug and
a CORRECT model answer is marked wrong. `rearrange_string` is the clearest case found: the prompt asks
for non-alphabetic characters moved to the end, and the stored key holds the *reversed* string,
because the reference solved a different problem entirely.

Two independent sources (Claude Opus 5 and GPT-5.6 Luna -- different families, different providers)
solved the same prompts the eval poses. For each problem where BOTH disagree with the stored key,
this derives each source's actual outputs and classifies every input:

    correct   both sources produce the SAME value, differing from the key
              -> the key is wrong and the right answer is known. Fix the value, KEEP the test.
    ambiguous the two sources differ from EACH OTHER
              -> no defensible answer exists; the question is underdetermined. Drop that input.
    domain    a source errors on the input
              -> outside the function's domain (the bank probes every list[int] problem with [],
                 [-1,-2,-3], [0,0,0] regardless of what its prompt defines). Drop that input.

Correcting rather than deleting is deliberate: dropping every disputed input would preferentially
remove the DISCRIMINATING tests -- the ones where a better model does better -- and flatten exactly
the signal the eval exists to measure. Only genuine ambiguity is dropped.

Problems where both sources AGREE with the key are untouched, and so are the ones where the sources
split (one agrees, one does not): a split cannot distinguish "key is wrong" from "one source failed",
and Luna is the cost-optimized tier, so its lone disagreement is weak evidence. Those are reported,
never edited.

Writes a new file by default; the live bank is only overwritten with --apply. Every change is logged
to the audit report with old value, new value, and the reason.

SECURITY: compiles and runs model-generated code, as the live scorer does. Throwaway box.

Usage:
    python3 scripts/apply_answer_key_corrections.py --dry_run
    python3 scripts/apply_answer_key_corrections.py --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_coding_capability_pool_eval import derive_expected  # noqa: E402
from score_coding_capability import DEFAULT_PROBLEMS_PATH, toolchain_missing  # noqa: E402


def load_reports(paths: list[str]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for path in paths:
        report = json.loads(Path(path).read_text())
        for bucket in ("agrees", "disagrees"):
            for entry in report["buckets"][bucket]:
                merged[entry["id"]] = {**entry, "_agreed": bucket == "agrees"}
    return merged


def outputs(code: str, problem: dict[str, Any]) -> dict[str, Any] | None:
    """Run one candidate solution over the problem's own test inputs and capture what it returns.

    class_name is hardcoded to Solution because that is the class the java harness writes and the
    prompt demands; derive_expected only consults it for java call rendering."""
    sig = {
        "name": problem["entry_point"],
        "param_names": problem["param_names"],
        "param_types": problem["param_types"],
        "return_type": problem["return_type"],
        "class_name": "Solution",
    }
    try:
        kept, expected, error = derive_expected(
            code, sig, [args for args, _ in problem["tests"]], problem["language_id"]
        )
    except Exception:  # noqa: BLE001 - a broken candidate must not abort the sweep
        return None
    if error or kept is None:
        return None
    return {json.dumps(a): e for a, e in zip(kept, expected)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--problems_path", default=str(DEFAULT_PROBLEMS_PATH))
    parser.add_argument("--opus_reports", nargs="+", default=[
        str(ROOT / "results/answer_key/answer_key_v2_report_opus.json"),
        str(ROOT / "results/answer_key/answer_key_v3_opus_delta.json"),
    ])
    parser.add_argument("--openai_reports", nargs="+", default=[
        str(ROOT / "results/answer_key/answer_key_v2_report_openai.json"),
        str(ROOT / "results/answer_key/answer_key_v3_openai_delta.json"),
    ])
    parser.add_argument("--min_tests", type=int, default=3, help="Drop a problem outright if fewer test cases survive; a 2-test problem is passable by luck.")
    parser.add_argument("--out_path", default=str(ROOT / "results/answer_key/coding_capability_problems_corrected.json"))
    parser.add_argument("--audit_path", default=str(ROOT / "results/answer_key/answer_key_audit.json"))
    parser.add_argument("--exec_timeout", type=float, default=10.0)
    parser.add_argument("--apply", action="store_true", help="Overwrite the live bank instead of writing --out_path.")
    parser.add_argument("--dry_run", action="store_true", help="Report only; write nothing at all.")
    args = parser.parse_args()

    bank = json.loads(Path(args.problems_path).read_text())
    problems = {p["id"]: p for p in bank["problems"]}
    opus = load_reports(args.opus_reports)
    luna = load_reports(args.openai_reports)

    covered = [i for i in problems if i in opus and i in luna]
    both_disagree = [i for i in covered if not opus[i]["_agreed"] and not luna[i]["_agreed"]]
    split = [i for i in covered if opus[i]["_agreed"] != luna[i]["_agreed"]]

    audit: dict[str, Any] = {
        "schema": "answer_key_audit_v1",
        "sources": {"a": args.opus_reports, "b": args.openai_reports},
        "problems_in_bank": len(bank["problems"]),
        "covered_by_both_sources": len(covered),
        "flagged_split_untouched": split,
        "corrections": [],
        "dropped_inputs": [],
        "dropped_problems": [],
        "underivable_problems": [],
    }

    corrected_bank: list[dict[str, Any]] = []
    for problem in bank["problems"]:
        pid = problem["id"]
        if pid not in both_disagree or toolchain_missing(problem["language_id"]):
            corrected_bank.append(problem)
            continue
        a = outputs(opus[pid].get("candidate_solution") or "", problem)
        b = outputs(luna[pid].get("candidate_solution") or "", problem)
        if a is None or b is None:
            # Cannot adjudicate without both sources' outputs; leaving the problem untouched is the
            # conservative choice -- an unadjudicated key is no worse than before this ran.
            audit["underivable_problems"].append(pid)
            corrected_bank.append(problem)
            continue

        new_tests: list[list[Any]] = []
        for args_, stored in problem["tests"]:
            key = json.dumps(args_)
            if key not in a or key not in b:
                audit["dropped_inputs"].append({"id": pid, "input": args_, "reason": "out_of_domain"})
                continue
            if a[key] != b[key]:
                audit["dropped_inputs"].append({
                    "id": pid, "input": args_, "reason": "ambiguous",
                    "source_a": a[key], "source_b": b[key], "key": stored,
                })
                continue
            if a[key] != stored:
                audit["corrections"].append({
                    "id": pid, "language_id": problem["language_id"], "input": args_,
                    "old": stored, "new": a[key],
                })
                new_tests.append([args_, a[key]])
            else:
                new_tests.append([args_, stored])

        if len(new_tests) < args.min_tests:
            audit["dropped_problems"].append({
                "id": pid, "language_id": problem["language_id"], "split": problem["split"],
                "tests_remaining": len(new_tests),
            })
            continue
        updated = dict(problem)
        updated["tests"] = new_tests
        updated["answer_key_corroborated"] = True
        corrected_bank.append(updated)

    bank["problems"] = corrected_bank
    from collections import Counter

    counts = Counter((p["language_id"], p["split"]) for p in corrected_bank)
    print(f"  problems: {audit['problems_in_bank']} -> {len(corrected_bank)}")
    print(f"  key corrections applied : {len(audit['corrections'])}")
    print(f"  inputs dropped          : {len(audit['dropped_inputs'])}")
    print(f"  problems dropped        : {len(audit['dropped_problems'])} (below --min_tests {args.min_tests})")
    print(f"  split, left untouched   : {len(split)}")
    print(f"  underivable, untouched  : {len(audit['underivable_problems'])}")
    print()
    for lang in ("python", "java", "rust", "cpp"):
        print(f"  {lang:7s} pool={counts[(lang, 'pool')]:3d}  held-out={counts[(lang, 'validation')]:3d}")
    print(f"  held-out total: {sum(v for k, v in counts.items() if k[1] == 'validation')}")

    if args.dry_run:
        print("\nDry run: nothing written.")
        return 0
    Path(args.audit_path).write_text(json.dumps(audit, indent=2) + "\n")
    target = Path(args.problems_path if args.apply else args.out_path)
    target.write_text(json.dumps(bank, indent=2) + "\n")
    print(f"\nBank  -> {target}")
    print(f"Audit -> {args.audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
