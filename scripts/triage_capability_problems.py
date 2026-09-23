#!/usr/bin/env python3
"""Find capability problems that no correct answer could pass, by grading their own reference solution.

Every `typed_function` problem in the bank was built by executing the source row's own reference
solution to capture expected values, so that reference IS the known-good answer. Feeding it back
through the live scoring harness therefore separates two things a pass-rate of 0.00 cannot:

    reference FAILS -> the harness/driver/dependencies are broken. No model could ever pass.
                       Remove the problem.
    reference PASSES -> the problem works and is simply hard. KEEP it -- deleting problems because
                       models fail them selects on the outcome and shrinks the eval to what the
                       models already do.

WHAT THIS CANNOT DO, and it matters. Expected values were *derived by running* the reference, so a
pass here is close to tautological for the values themselves; it genuinely tests the driver,
compilation and dependency path (which is where the known cpp failures live), not the correctness of
the expected outputs. A reference that compiles, runs deterministically and computes the WRONG thing
produces a problem whose expected values encode that bug, and a correct model answer will fail it.
This script reports such problems as healthy. They surface as "harness fine, every model wrong, no
error recorded" and need a human to read them -- use --explain to list those candidates.

Intended second use: the same check belongs in build_coding_capability_pool_eval.py as an admission
gate, which would have caught the two broken cpp drivers before they ever entered the bank.

Usage:
    python3 scripts/triage_capability_problems.py --results 'results/coding-capability/*/coding_capability/*.json'
    python3 scripts/triage_capability_problems.py --languages rust --explain
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from score_coding_capability import (  # noqa: E402
    DEFAULT_PROBLEMS_PATH,
    LANGUAGES,
    score_problem,
    toolchain_missing,
)

FENCE_BLOCK_RE = re.compile(r"```([A-Za-z+#0-9]*)\n(.*?)```", re.DOTALL)
FENCE_ALIASES = {
    "python": {"python", "py", "python3"},
    "java": {"java"},
    "rust": {"rust", "rs"},
    "cpp": {"cpp", "c++", "cxx"},
}


def reference_code(row: dict, language: str) -> str | None:
    """The assistant turn's language-tagged fenced blocks, joined. Multi-block solutions are common
    (imports shown apart from the function), so taking only the last block would drop real code."""
    solution = ""
    for message in row.get("messages", []):
        if message.get("role") == "assistant":
            solution = message.get("content") or ""
    blocks = [code for tag, code in FENCE_BLOCK_RE.findall(solution) if tag.lower() in FENCE_ALIASES[language]]
    return "\n\n".join(b.strip() for b in blocks) if blocks else None


def load_source_rows(language_id: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    base = ROOT / "data" / "training" / f"coding.write.{language_id}"
    for name in ("pool.jsonl", "validation.jsonl"):
        path = base / name
        if not path.exists():
            continue
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    rows[record["id"]] = record
    return rows


def never_passed(result_globs: list[str]) -> set[str]:
    """Problem ids whose full_pass_rate was 0.0 in every run that scored them."""
    seen: dict[str, list[float]] = defaultdict(list)
    for pattern in result_globs:
        for path in glob.glob(pattern):
            payload = json.loads(Path(path).read_text())
            for res in payload.get("results", {}).values():
                for problem in res["problems"]:
                    seen[problem["id"]].append(problem["full_pass_rate"])
    return {pid for pid, rates in seen.items() if rates and all(r == 0.0 for r in rates)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--problems_path", default=str(DEFAULT_PROBLEMS_PATH))
    parser.add_argument("--languages", default=",".join(LANGUAGES))
    parser.add_argument("--results", action="append", default=None, help="Glob of past result JSONs; flags which problems never passed. Repeatable.")
    parser.add_argument("--exec_timeout", type=float, default=10.0)
    parser.add_argument("--explain", action="store_true", help="List problems needing human review (harness healthy, nobody passed).")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "answer_key" / "coding_capability_triage_report.json")
    args = parser.parse_args()

    requested = [lang.strip() for lang in args.languages.split(",") if lang.strip()]
    usable = []
    for lang in requested:
        missing = toolchain_missing(lang)
        if missing:
            print(f"!! {lang}: skipped, toolchain missing on PATH ({', '.join(missing)})")
        else:
            usable.append(lang)
    if not usable:
        raise SystemExit("No requested language has its toolchain installed.")

    problems = json.loads(Path(args.problems_path).read_text())["problems"]
    dead = never_passed(args.results) if args.results else set()

    report: dict[str, list] = {
        "broken": [],
        "reference_unusable_problem_works": [],
        "healthy_but_unpassed": [],
        "healthy": [],
        "no_reference": [],
    }

    for lang in usable:
        rows = load_source_rows(lang)
        lang_problems = [p for p in problems if p["language_id"] == lang]
        broken = unpassed = unusable = 0
        for problem in lang_problems:
            row = rows.get(problem["source_row_id"])
            code = reference_code(row, lang) if row else None
            if code is None:
                report["no_reference"].append(problem["id"])
                continue
            outcome = score_problem(code, problem, args.exec_timeout)
            passed = outcome["total"] > 0 and outcome["passed"] == outcome["total"]
            entry = {
                "id": problem["id"],
                "kind": problem["kind"],
                "split": problem["split"],
                "reference_passed": passed,
                "error": (outcome["error"] or "")[:200] or None,
                "never_passed_by_any_model": problem["id"] in dead,
            }
            if not passed and not entry["never_passed_by_any_model"]:
                # A model passed it, so the problem is demonstrably solvable and stays. The
                # reference simply is not standalone -- typically `mod foo;` pointing at a file that
                # was never part of the row, or an ABI/target the local machine doesn't support.
                # Removing these would delete working problems on the strength of a bad reference.
                report["reference_unusable_problem_works"].append(entry)
                unusable += 1
            elif not passed:
                report["broken"].append(entry)
                broken += 1
            elif entry["never_passed_by_any_model"]:
                report["healthy_but_unpassed"].append(entry)
                unpassed += 1
            else:
                report["healthy"].append(entry)
        healthy = len(lang_problems) - broken - unpassed - unusable
        print(
            f"  {lang:7s} {len(lang_problems):3d} problems -> {broken} REMOVE, {unusable} reference-unusable (keep), "
            f"{unpassed} need human review, {healthy} healthy"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"\nRemove (reference fails AND no model ever passed): {len(report['broken'])}")
    print(f"Keep, reference unusable but a model passed anyway: {len(report['reference_unusable_problem_works'])}")
    print(f"Keep, but a human must read (harness fine, no model ever passed): {len(report['healthy_but_unpassed'])}")
    print(f"Healthy: {len(report['healthy'])}   No reference found: {len(report['no_reference'])}")
    print(f"Report -> {args.out}")

    if args.explain and report["healthy_but_unpassed"]:
        print("\nThese pass with their own reference solution, yet no model ever passed them.")
        print("That is the signature of expected values that encode a buggy reference -- read each one:")
        for entry in report["healthy_but_unpassed"]:
            print(f"  {entry['id']}  ({entry['split']}, {entry['kind']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
