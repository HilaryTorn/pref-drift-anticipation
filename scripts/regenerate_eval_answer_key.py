#!/usr/bin/env python3
"""Corroborate the capability bank's answer key with a second, independent solution (pass 1).

The bank's expected values were produced by *executing the Magicoder row's own reference solution*
(build_coding_capability_pool_eval.py). Those solutions are unvetted GPT-3.5 output from 2023 -- on
the rust pool, roughly half of the rows that should compile standalone don't -- so a problem whose
key encodes a buggy-but-deterministic reference marks a CORRECT model answer wrong, and no mechanical
check inside that single source can detect it. See the 2026-08-21 triage: of 8 problems no model ever
passed, 7 had a broken key or a prompt/grader mismatch.

This asks a strong model the *same question the eval asks* (the prompt is built by the scorer's own
build_prompt, so the teacher solves exactly the problem the eval poses, not a paraphrase) and grades
its answer against the stored key:

    agrees   -> two independent sources produced the same outputs. Key corroborated; keep.
    disagrees-> one of them is wrong, OR the prompt is ambiguous and both readings are defensible.
                NOT auto-resolved here -- see below.
    no code  -> teacher produced nothing gradeable; retry candidate.

**Disagreement is deliberately not resolved in this pass.** Overwriting the key with the new answer
would swap one unvetted source for another, and it would silently "fix" problems whose real defect is
an ambiguous question (the triangle problem returning -1 for negative input; path_join('','') -> '/')
where each answer is right under a different reading. Pass 2 escalates only the disagreements: two
further independent solutions, then majority (old key wrong -> replace, logged with both values) or
mutual disagreement (question is ambiguous -> drop the problem, it cannot fairly grade anyone). Pass 1
exists first because its disagreement RATE is what sizes pass 2.

Cost: measured 369 input tokens/problem (count_tokens over a seeded sample of the real pools) plus a
short system prompt; output dominates, and thinking bills as output, so --effort is the cost dial.
Roughly 2-6.5 cents/problem at list price, halved by the Batch API -- which is why this uses batches:
nothing here is latency-sensitive.

SECURITY: `collect` compiles and runs model-generated code to grade it, exactly as the live scorer
does. Run it on a throwaway box, per the note in score_coding_capability.py.

Usage:
    # 20% shakedown across all four languages
    python3 scripts/regenerate_eval_answer_key.py submit --sample_frac 0.2
    python3 scripts/regenerate_eval_answer_key.py collect

    # full run once the shakedown looks sane
    python3 scripts/regenerate_eval_answer_key.py submit
    python3 scripts/regenerate_eval_answer_key.py collect
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from score_coding_capability import (  # noqa: E402
    DEFAULT_PROBLEMS_PATH,
    LANGUAGES,
    build_prompt,
    extract_code,
    score_problem,
    toolchain_missing,
)

DEFAULT_STATE_PATH = ROOT / "results" / "answer_key" / "answer_key_regen_batch.json"
DEFAULT_REPORT_PATH = ROOT / "results" / "answer_key" / "answer_key_regen_report.json"

# The teacher is being asked for a REFERENCE solution, not a chat answer. It must obey the same
# signature contract the eval imposes on the model under test, or a correct solution scores as a
# disagreement for a purely mechanical reason and pollutes the rate this pass exists to measure.
SYSTEM_PROMPT = (
    "You are producing reference solutions used to validate a coding benchmark's answer key. "
    "Write the single most standard, correct implementation of exactly what is asked. "
    "Match the requested signature and language exactly. Handle edge cases the way the problem "
    "statement specifies -- do not invent behaviour it does not describe, and do not add error "
    "sentinels, logging, or extra parameters. Return only the code, in one fenced code block."
)


def load_api_key() -> None:
    """Anthropic SDK reads ANTHROPIC_API_KEY from the environment. Fall back to the repo's key file
    (gitignored) so the script works without the caller exporting it first. Never logged or stored."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    key_path = ROOT / "api_keys" / "api_key_anthropic.txt"
    if key_path.exists():
        os.environ["ANTHROPIC_API_KEY"] = key_path.read_text().strip()


def select_problems(args: argparse.Namespace) -> list[dict[str, Any]]:
    problems = json.loads(Path(args.problems_path).read_text())["problems"]
    requested = [l.strip() for l in args.languages.split(",") if l.strip()]
    problems = [p for p in problems if p["language_id"] in requested]
    # Rebuilding the bank re-derives every key from the same references, but problem ids are stable
    # across rebuilds (they are formed from the source row), so a prior run's results stay valid for
    # any problem that survived. Skipping those makes corroboration incremental instead of a full
    # re-pay each time the bank changes -- the 2026-08-21 rebuild carried 195 of 206 problems over.
    for report_path in getattr(args, "skip_ids_from", None) or []:
        path = Path(report_path)
        if not path.exists():
            raise SystemExit(f"--skip_ids_from: no such report {path}")
        report = json.loads(path.read_text())
        seen = {e["id"] for entries in report["buckets"].values() for e in entries}
        before = len(problems)
        problems = [p for p in problems if p["id"] not in seen]
        print(f"  skipping {before - len(problems)} problems already covered by {path.name}")
    if args.sample_frac < 1.0:
        # Stratify by language: a flat sample would under-represent java/cpp, and the shakedown's
        # whole purpose is to exercise every toolchain before spending on the full run.
        rng = random.Random(args.seed)
        sampled: list[dict[str, Any]] = []
        for lang in requested:
            in_lang = [p for p in problems if p["language_id"] == lang]
            take = max(1, round(len(in_lang) * args.sample_frac))
            sampled.extend(rng.sample(in_lang, min(take, len(in_lang))))
        problems = sampled
    return problems


def cmd_submit(args: argparse.Namespace) -> int:
    from anthropic import Anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    load_api_key()
    problems = select_problems(args)
    if not problems:
        raise SystemExit("No problems selected.")

    # custom_id must match ^[a-zA-Z0-9_-]{1,64}$ -- problem ids carry dots and run past 64 chars
    # ("coding.write.rust.magicoder_pool_00202.capability"), so index into problem_ids instead and
    # map back on collect. The state file is what makes the mapping recoverable.
    requests = [
        Request(
            custom_id=f"p{index:05d}",
            params=MessageCreateParamsNonStreaming(
                model=args.model,
                max_tokens=args.max_tokens,
                system=SYSTEM_PROMPT,
                output_config={"effort": args.effort},
                messages=[{"role": "user", "content": build_prompt(problem)}],
            ),
        )
        for index, problem in enumerate(problems)
    ]

    batch = Anthropic().messages.batches.create(requests=requests)
    state = {
        "batch_id": batch.id,
        "model": args.model,
        "effort": args.effort,
        "submitted_at": batch.created_at.isoformat() if hasattr(batch.created_at, "isoformat") else str(batch.created_at),
        "sample_frac": args.sample_frac,
        "seed": args.seed,
        "problems_path": str(args.problems_path),
        "problem_ids": [p["id"] for p in problems],
    }
    Path(args.state_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.state_path).write_text(json.dumps(state, indent=2) + "\n")

    by_lang: dict[str, int] = {}
    for p in problems:
        by_lang[p["language_id"]] = by_lang.get(p["language_id"], 0) + 1
    print(f"Submitted batch {batch.id}: {len(requests)} problems {by_lang}")
    print(f"State -> {args.state_path}")
    print(f"Poll with: python3 {Path(__file__).name} collect")
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    from anthropic import Anthropic

    load_api_key()
    state = json.loads(Path(args.state_path).read_text())
    client = Anthropic()

    batch = client.messages.batches.retrieve(state["batch_id"])
    while batch.processing_status != "ended":
        if not args.wait:
            print(f"Batch {batch.id} status={batch.processing_status} counts={batch.request_counts}")
            print("Not finished. Re-run with --wait to block until it ends.")
            return 0
        print(f"  status={batch.processing_status} counts={batch.request_counts}")
        time.sleep(args.poll_interval)
        batch = client.messages.batches.retrieve(state["batch_id"])

    by_id = {p["id"]: p for p in json.loads(Path(state["problems_path"]).read_text())["problems"]}
    # custom_id is "p<index>" into the submitted order; recover the real problem id through state.
    problems = {
        f"p{index:05d}": by_id[pid]
        for index, pid in enumerate(state["problem_ids"])
        if pid in by_id
    }
    real_id = {f"p{index:05d}": pid for index, pid in enumerate(state["problem_ids"])}
    skipped_langs = {lang for lang in LANGUAGES if toolchain_missing(lang)}
    for lang in sorted(skipped_langs):
        print(f"!! {lang}: toolchain unusable here -- its problems are reported as 'ungraded', not as disagreements")

    buckets: dict[str, list[dict[str, Any]]] = {
        "agrees": [],
        "disagrees": [],
        "no_code": [],
        "ungraded_no_toolchain": [],
        "request_failed": [],
    }

    for result in client.messages.batches.results(state["batch_id"]):
        problem = problems.get(result.custom_id)
        if problem is None:
            buckets["request_failed"].append({"id": real_id.get(result.custom_id, result.custom_id), "error": "problem no longer in bank"})
            continue
        if result.result.type != "succeeded":
            buckets["request_failed"].append({"id": real_id.get(result.custom_id, result.custom_id), "error": result.result.type})
            continue

        message = result.result.message
        completion = "".join(b.text for b in message.content if b.type == "text")
        lang = problem["language_id"]
        code, status = extract_code(completion, lang)

        entry = {
            "id": real_id[result.custom_id],
            "language_id": lang,
            "split": problem["split"],
            "extraction_status": status,
            "output_tokens": message.usage.output_tokens,
        }
        if status == "no_code":
            buckets["no_code"].append(entry)
            continue
        if lang in skipped_langs:
            buckets["ungraded_no_toolchain"].append(entry)
            continue

        outcome = score_problem(code, problem, args.exec_timeout)
        agreed = outcome["total"] > 0 and outcome["passed"] == outcome["total"]
        entry["passed"] = outcome["passed"]
        entry["total"] = outcome["total"]
        entry["error"] = (outcome["error"] or "")[:300] or None
        # Recorded for the disagreements only: pass 2 needs the candidate solution to compare against
        # the two tiebreak solutions, and re-generating it would cost tokens for no reason.
        if not agreed:
            entry["candidate_solution"] = code
        buckets["agrees" if agreed else "disagrees"].append(entry)

    graded = len(buckets["agrees"]) + len(buckets["disagrees"])
    report = {
        "schema": "answer_key_regen_pass1_v1",
        "batch_id": state["batch_id"],
        "model": state["model"],
        "effort": state["effort"],
        "sample_frac": state["sample_frac"],
        "counts": {k: len(v) for k, v in buckets.items()},
        "agreement_rate": (len(buckets["agrees"]) / graded) if graded else None,
        "total_output_tokens": sum(e.get("output_tokens", 0) for v in buckets.values() for e in v),
        "buckets": buckets,
    }
    Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_path).write_text(json.dumps(report, indent=2) + "\n")

    print()
    for name, entries in buckets.items():
        print(f"  {name:24s} {len(entries)}")
    if graded:
        print(f"\nAgreement rate: {len(buckets['agrees'])}/{graded} = {report['agreement_rate']:.1%}")
        print("That rate sizes pass 2: only the disagreements get the two tiebreak solutions.")
    by_lang: dict[str, list[int]] = {}
    for e in buckets["agrees"]:
        by_lang.setdefault(e["language_id"], [0, 0])[0] += 1
    for e in buckets["disagrees"]:
        by_lang.setdefault(e["language_id"], [0, 0])[1] += 1
    if by_lang:
        print("\nPer language (agree/disagree):")
        for lang, (a, d) in sorted(by_lang.items()):
            print(f"  {lang:7s} {a}/{a + d}")
    print(f"\nReport -> {args.report_path}")
    print("NOTE: nothing was written to the problem bank. Disagreements are recorded, never applied.")
    return 0


def cmd_run_openai(args: argparse.Namespace) -> int:
    """Second, cross-family source for the answer key, on the caller's own OpenAI key.

    Independence is the point: the incumbent key came from GPT-3.5 in 2023, and pass 1's second
    opinion came from Claude. A third source from the GPT family costs almost nothing and makes
    "all three agree" a much stronger statement than two samples of one model could be.

    Touches none of the Azure/RL machinery -- reads api_keys/api_key_openai.txt directly, imports
    nothing from rl_training/, and sets no AZURE_OPENAI_* variables. Runs synchronously with a small
    thread pool rather than through a batch queue: 68 calls at Luna's rates come to about three
    cents, so a 50% batch discount saves less than two cents and would cost a second implementation.

    On reading the results, tier matters. Luna is the cost-optimized variant ($0.20/$1.20 per MTok),
    so its AGREEMENT is strong evidence -- two different models would have to make the same mistake
    to agree on a wrong non-obvious output -- while its DISAGREEMENT is weak, since it may simply
    have failed the problem. Merge logic should therefore correct a key only when both sources agree
    against it, and flag (never silently drop) the cases where the two sources disagree with each
    other. Deleting a test on a budget model's say-so is how a genuinely discriminating case gets
    lost.
    """
    from concurrent.futures import ThreadPoolExecutor
    from openai import OpenAI

    key_path = ROOT / "api_keys" / "api_key_openai.txt"
    api_key = os.environ.get("OPENAI_API_KEY") or (key_path.read_text().strip() if key_path.exists() else None)
    if not api_key:
        raise SystemExit(f"No OpenAI key: set OPENAI_API_KEY or provide {key_path}")
    client = OpenAI(api_key=api_key)

    problems = select_problems(args)
    if args.limit:
        problems = problems[: args.limit]
    skipped_langs = {lang for lang in LANGUAGES if toolchain_missing(lang)}

    def generate(problem: dict[str, Any]) -> tuple[dict[str, Any], str | None, str | None]:
        try:
            kwargs: dict[str, Any] = {
                "model": args.model,
                "instructions": SYSTEM_PROMPT,
                "input": build_prompt(problem),
                "max_output_tokens": args.max_tokens,
            }
            if args.reasoning_effort:
                kwargs["reasoning"] = {"effort": args.reasoning_effort}
            response = client.responses.create(**kwargs)
            return problem, response.output_text, None
        except Exception as exc:  # noqa: BLE001 - surfaced per problem, never aborts the sweep
            return problem, None, f"{type(exc).__name__}: {exc}"[:300]

    print(f"Generating {len(problems)} solutions with {args.model} ...")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        generated = list(pool.map(generate, problems))

    buckets: dict[str, list[dict[str, Any]]] = {
        "agrees": [], "disagrees": [], "no_code": [], "ungraded_no_toolchain": [], "request_failed": [],
    }
    for problem, completion, error in generated:
        if error is not None:
            buckets["request_failed"].append({"id": problem["id"], "error": error})
            continue
        lang = problem["language_id"]
        code, status = extract_code(completion or "", lang)
        entry = {"id": problem["id"], "language_id": lang, "split": problem["split"], "extraction_status": status}
        if status == "no_code":
            buckets["no_code"].append(entry)
            continue
        if lang in skipped_langs:
            buckets["ungraded_no_toolchain"].append(entry)
            continue
        outcome = score_problem(code, problem, args.exec_timeout)
        agreed = outcome["total"] > 0 and outcome["passed"] == outcome["total"]
        entry.update({"passed": outcome["passed"], "total": outcome["total"], "error": (outcome["error"] or "")[:300] or None})
        # Kept for every problem, not just disputes: the merge step needs each source's actual
        # solution to correct a key, and re-generating the agreeing ones later would be waste.
        entry["candidate_solution"] = code
        buckets["agrees" if agreed else "disagrees"].append(entry)

    graded = len(buckets["agrees"]) + len(buckets["disagrees"])
    report = {
        "schema": "answer_key_regen_pass1_v1",
        "source": "openai",
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "counts": {k: len(v) for k, v in buckets.items()},
        "agreement_rate": (len(buckets["agrees"]) / graded) if graded else None,
        "buckets": buckets,
    }
    Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_path).write_text(json.dumps(report, indent=2) + "\n")

    print()
    for name, entries in buckets.items():
        print(f"  {name:24s} {len(entries)}")
    if graded:
        print(f"\nAgreement with stored key: {len(buckets['agrees'])}/{graded} = {report['agreement_rate']:.1%}")
    print(f"\nReport -> {args.report_path}")
    print("NOTE: nothing was written to the problem bank.")
    return 0


def cmd_prune(args: argparse.Namespace) -> int:
    """Drop the individual test cases an independent solution disputes.

    Rationale for dropping rather than adjudicating: a test on which two competent independent
    sources disagree cannot fairly grade a third party, whichever one is "right". The 20% shakedown
    showed why the disputes cluster where they do -- the bank's test inputs are generic canned probes
    (every list[int] problem is asked about [], [1], [-1,-2,-3], [0,0,0]; every int problem about 0,
    -1, -17), so each function is interrogated outside the domain its prompt defines, and the stored
    expected value is just whatever the 2023 reference happened to print there. countWaysToReachTop
    maps -1 to -1 and -17 to -17; calculateAverage (which divides by n-2) is asked for the average of
    [] and [1]. Pruning removes exactly those cases and keeps the in-domain ones.

    `custom_driver` problems carry a hand-authored driver instead of a test list, so a disagreement
    cannot be localized to one input -- they are reported for review, never auto-modified.
    """
    report = json.loads(Path(args.report_path).read_text())
    bank = json.loads(Path(args.problems_path).read_text())
    problems = {p["id"]: p for p in bank["problems"]}

    disputed_by_problem: dict[str, list[int]] = {}
    undecidable: list[str] = []

    for entry in report["buckets"]["disagrees"]:
        problem = problems.get(entry["id"])
        code = entry.get("candidate_solution")
        if problem is None or not code:
            continue
        if problem["kind"] != "typed_function":
            undecidable.append(entry["id"])
            continue
        if toolchain_missing(problem["language_id"]):
            undecidable.append(entry["id"])
            continue
        # Re-score one test at a time to localize the disagreement. score_problem reads
        # problem["tests"], so a shallow copy carrying a single case isolates that case exactly --
        # no new execution path, and no second implementation of the harness to keep in sync.
        disputed = []
        for index, case in enumerate(problem["tests"]):
            single = dict(problem)
            single["tests"] = [case]
            outcome = score_problem(code, single, args.exec_timeout)
            if not (outcome["total"] > 0 and outcome["passed"] == outcome["total"]):
                disputed.append(index)
        if disputed:
            disputed_by_problem[entry["id"]] = disputed

    kept, pruned, dropped = [], [], []
    for problem in bank["problems"]:
        disputed = disputed_by_problem.get(problem["id"])
        if not disputed:
            kept.append(problem)
            continue
        survivors = [c for i, c in enumerate(problem["tests"]) if i not in set(disputed)]
        record = {
            "id": problem["id"],
            "language_id": problem["language_id"],
            "disputed_inputs": [problem["tests"][i][0] for i in disputed],
            "tests_before": len(problem["tests"]),
            "tests_after": len(survivors),
        }
        if len(survivors) < args.min_tests:
            # Too little left to grade with; keeping it would mean a problem that a single lucky
            # guess passes. Dropping is the honest outcome, and it is recorded, not silent.
            dropped.append(record)
            continue
        new_problem = dict(problem)
        new_problem["tests"] = survivors
        new_problem["pruned_disputed_inputs"] = record["disputed_inputs"]
        pruned.append(record)
        kept.append(new_problem)

    summary = {
        "schema": "answer_key_prune_v1",
        "source_report": str(args.report_path),
        "min_tests": args.min_tests,
        "problems_in": len(bank["problems"]),
        "problems_out": len(kept),
        "problems_pruned": pruned,
        "problems_dropped": dropped,
        "undecidable_custom_driver": undecidable,
    }
    Path(args.prune_report_path).write_text(json.dumps(summary, indent=2) + "\n")

    total_before = sum(len(p.get("tests") or []) for p in bank["problems"])
    total_after = sum(len(p.get("tests") or []) for p in kept)
    print(f"  problems: {len(bank['problems'])} -> {len(kept)}  ({len(dropped)} dropped below --min_tests {args.min_tests})")
    print(f"  test cases: {total_before} -> {total_after}  ({len(pruned)} problems pruned)")
    if undecidable:
        print(f"  {len(undecidable)} custom_driver disagreements cannot be localized -- review by hand:")
        for pid in undecidable:
            print(f"     {pid}")
    print(f"\nPrune report -> {args.prune_report_path}")

    if args.apply:
        bank["problems"] = kept
        Path(args.out_path).write_text(json.dumps(bank, indent=2) + "\n")
        print(f"Wrote pruned bank -> {args.out_path}")
    else:
        print("Dry run: no bank written. Re-run with --apply to write it.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--problems_path", default=str(DEFAULT_PROBLEMS_PATH))
    common.add_argument("--state_path", default=str(DEFAULT_STATE_PATH))

    s = sub.add_parser("submit", parents=[common], help="Build and submit the batch.")
    s.add_argument("--languages", default=",".join(LANGUAGES))
    s.add_argument("--skip_ids_from", action="append", default=None, help="Prior report(s) whose problems are already corroborated; repeatable.")
    s.add_argument("--sample_frac", type=float, default=1.0, help="Fraction to submit, stratified per language. Use 0.2 for a shakedown.")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--model", default="claude-opus-5")
    s.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"], help="Thinking bills as output tokens, so this is the cost dial.")
    s.add_argument("--max_tokens", type=int, default=8000, help="Must cover thinking plus the solution; a truncated answer reads as a disagreement.")
    s.set_defaults(func=cmd_submit)

    c = sub.add_parser("collect", parents=[common], help="Fetch results and grade them against the stored key.")
    c.add_argument("--report_path", default=str(DEFAULT_REPORT_PATH))
    c.add_argument("--exec_timeout", type=float, default=10.0)
    c.add_argument("--wait", action="store_true", help="Block until the batch ends instead of reporting status and exiting.")
    c.add_argument("--poll_interval", type=float, default=60.0)
    c.set_defaults(func=cmd_collect)

    o = sub.add_parser("run-openai", parents=[common], help="Second cross-family source, synchronous, on the caller's own OpenAI key.")
    o.add_argument("--languages", default=",".join(LANGUAGES))
    o.add_argument("--skip_ids_from", action="append", default=None, help="Prior report(s) whose problems are already corroborated; repeatable.")
    o.add_argument("--sample_frac", type=float, default=1.0)
    o.add_argument("--seed", type=int, default=42)
    o.add_argument("--limit", type=int, default=None, help="Cap the number of problems; use a small value to smoke-test the call shape first.")
    o.add_argument("--model", default="gpt-5.6-luna")
    o.add_argument("--reasoning_effort", default=None, help="Omitted by default -- sending it to a model that does not accept it is a 400.")
    o.add_argument("--max_tokens", type=int, default=8000)
    o.add_argument("--workers", type=int, default=8)
    o.add_argument("--exec_timeout", type=float, default=10.0)
    o.add_argument("--report_path", default=str(ROOT / "results" / "answer_key" / "answer_key_regen_report_openai.json"))
    o.set_defaults(func=cmd_run_openai)

    p = sub.add_parser("prune", parents=[common], help="Drop the individual test cases the independent solution disputed.")
    p.add_argument("--report_path", default=str(DEFAULT_REPORT_PATH))
    p.add_argument("--prune_report_path", default=str(ROOT / "results" / "answer_key" / "coding_capability_prune_report.json"))
    p.add_argument("--out_path", default=str(ROOT / "results" / "answer_key" / "coding_capability_problems_pruned.json"))
    p.add_argument("--min_tests", type=int, default=3, help="Drop a problem outright if fewer than this many test cases survive.")
    p.add_argument("--exec_timeout", type=float, default=10.0)
    p.add_argument("--apply", action="store_true", help="Write the pruned bank; default is a dry run.")
    p.set_defaults(func=cmd_prune)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))
