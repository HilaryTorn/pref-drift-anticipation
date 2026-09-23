#!/usr/bin/env python3
"""Multi-LCB coding-capability instrument -- thin adapter over the vendored upstream harness.

This is the *second* coding-capability instrument, alongside `score_coding_capability.py`. They
answer different questions and neither replaces the other:

    score_coding_capability.py  problems drawn from our own coding-SFT corpus, so pool-vs-held-out
                                measures MEMORIZATION. Pass rates sit at 0.68-0.79, near ceiling.
    score_multilcb.py (this)    LiveCodeBench competitive-programming problems, outside our corpus
                                by construction, so there is no memorization gap -- what it gives
                                is HEADROOM, at a difficulty that actually resolves a change.

Upstream (`multi-lcb/`, pinned at d80be9f) is vendored UNMODIFIED and must stay that way. Its
prompt, its LeetCode-functional-to-stdin test conversion, and its stdout comparator are one
artifact: the published per-language numbers are a property of that exact code, so a local "fix"
silently makes our numbers incomparable to theirs. Everything this script does is outside that
boundary -- resolve a model, set env vars, invoke `lcb_runner.runner.main`, and re-shape the output
into the summary schema the rest of `results/` uses.

Grading is whole-program: the model is asked to read stdin and write stdout, and the program is
compiled and run against each test with a timeout. That is a different grading path from the typed
function + generated driver in `score_coding_capability.py`, which is why this wraps upstream
rather than extending our own harness.

SECURITY: this executes model-generated code. Run it on a throwaway box, per the same caution in
`score_coding_capability.py` and `rl_training/rewards.py`.

ENVIRONMENT: two, because generation and evaluation have opposite needs.
  generation  repo `venv/` + `requirements-multilcb-gen.txt`. No compilers, no conda. Works under
              our `datasets==5.0.0` because scripts/fetch_multilcb_dataset.py stages the LCB shards
              as plain jsonl, sidestepping upstream's remote-code script dataset and its 3.6.0 pin.
  evaluation  `multi_lcb_env` conda env + `requirements-multilcb.txt`, on x86 Linux. go/php/ruby/
              scala/csharp toolchains cannot come from pip, and macOS rejects upstream's RLIMIT_AS.
Full runbook: docs/multilcb-eval.md.

Usage:
    # smoke: 10 problems, python only, validates the whole chain for a couple of dollars
    python3 scripts/score_multilcb.py --model_key qwen35-4b-m0-v4-aws --languages python --debug

    # M0 baseline, v6 window (175 problems, 2025-01-04 -> 2025-04-06)
    python3 scripts/score_multilcb.py --model_key qwen35-4b-m0-v4-aws

    # split generation (GPU box) from evaluation (CPU box)
    python3 scripts/score_multilcb.py --model_key qwen35-9b-m0-v4-aws --generate_only
    python3 scripts/score_multilcb.py --model_key qwen35-9b-m0-v4-aws --evaluate_only

    # re-read a finished run with the integration check, free -- no model, no execution
    python3 scripts/score_multilcb.py --model_key qwen35-4b-m0-v4-aws --summarize_only --diagnose
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compute_utilities.naming import default_results_dir, timestamped  # noqa: E402
from scripts.lcb_languages import LANGUAGES as LCB_LANGUAGES, upstream_language  # noqa: E402

DEFAULT_MULTILCB_ROOT = ROOT / "multi-lcb"

# Prompt-side KV allowance per sequence. LCB statements plus the format block measured ~620 tokens
# on the v6 window (from vLLM's prompt-throughput bursts on 2026-08-24); 2048 leaves headroom for
# the long tail without being so large it forces an over-conservative batch size.
KV_PROMPT_ALLOWANCE = 2048

# The KV cache of the box the first real run used (a ~24 GB card: model 8.61 GiB + KV 9.34 GiB at
# --gpu-memory-utilization 0.92). Deliberately the SMALL value: guessing high would re-create the
# failure it exists to prevent. A g6e.xlarge/L40S reports roughly 1,000,000 -- pass the real number
# via --kv_cache_tokens, taken from the vLLM startup line "GPU KV cache size: N tokens".
DEFAULT_KV_CACHE_TOKENS = 287_464

# The six languages this study scores by default. go/rust/scala/csharp are the predicted-weak
# candidates and php the mid-range contrast; csharp and scala add same-platform pairs (CLR, JVM)
# that separate "this language is hard for the model" from "this runtime is".
#
# python is the calibration arm. It is the one language with an external expectation -- upstream
# validates its python subset against official LCB v4-v6 -- so it catches the one failure the
# failure_modes histogram cannot: a fault that depresses every language uniformly while still
# producing well-formed wrong answers. It costs ~17% of the run, which is cheap for the
# only external check available.
#
# ruby is supported and installed but not in this list; pass it via --languages if wanted.
DEFAULT_LANGUAGES = ["python", "rust", "go", "csharp", "scala", "php"]

# Binaries a language needs on PATH before upstream can grade it. Generation needs none of these --
# only evaluation does -- so --generate_only skips the check entirely.
TOOLCHAIN_BINARIES = {
    "python": (),
    "go": ("go",),
    "rust": ("rustc",),
    "php": ("php",),
    "ruby": ("ruby",),
    "scala": ("scalac", "scala", "java"),
    # Both come from the conda-forge `mono` package: mcs compiles, mono runs the .exe.
    "csharp": ("mcs", "mono"),
}


class UserInputError(ValueError):
    """Raised for CLI/data validation errors that should print without a traceback."""


# --- Model resolution ----------------------------------------------------------


def resolve_endpoint(model_key: str) -> tuple[str, str]:
    """Map a config.yaml key to (base_url, served_model_name).

    Reads config.yaml directly rather than going through compute_utilities.create_agent: upstream
    drives its own OpenAI client, so all we need out of the config is the endpoint, and importing
    the agent stack would pull the heavy inference dependencies into an env that deliberately
    does not have them.
    """
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    entry = config.get(model_key)
    if entry is None:
        raise UserInputError(f"Model key {model_key!r} not found in config.yaml")

    model_type = entry.get("model_type")
    if model_type != "vllm_endpoint":
        raise UserInputError(
            f"{model_key!r} has model_type {model_type!r}; this instrument drives upstream's "
            "OpenAI-compatible client, so it only supports 'vllm_endpoint'. Stand the endpoint up "
            "via docs/serving-vllm-aws.md."
        )

    base_url = entry.get("base_url")
    model_name = entry.get("model_name")
    if not base_url or not model_name:
        raise UserInputError(f"{model_key!r} is missing base_url or model_name in config.yaml")
    return base_url, model_name


def endpoint_api_key() -> str:
    """The vLLM endpoint's key, or a placeholder.

    Upstream reads OPENAI_KEY at *import* time and indexes os.environ directly, so this must always
    resolve to something -- an unset variable is a KeyError before any argument is parsed.
    """
    key_path = ROOT / "api_keys" / "api_key_vllm_endpoint.txt"
    if key_path.exists():
        text = key_path.read_text().strip()
        if text:
            return text
    return "dummy"


# --- Upstream invocation -------------------------------------------------------


def upstream_output_dir(model_name: str, release_version: str, cot: bool, debug: bool) -> str:
    """Mirror of upstream `get_output_folder_name` for a VLLMAsync run.

    Replicated rather than imported because computing it needs a fully-parsed upstream config, which
    needs the upstream env already set. Kept honest by `locate_outputs`, which reports what actually
    landed in output/ when this guess misses.
    """
    # Order matters and is upstream's, not the obvious one: the "_cot" suffix, then the "debug_"
    # prefix, then the DATASET_NAME prefix outermost -- so a debug run is "v6_debug_<model>_cot".
    # We set DATASET_NAME to the release tag so two windows of the same model never share a
    # directory and silently pool.
    folder = model_name
    if cot:
        folder += "_cot"
    if debug:
        folder = "debug_" + folder
    return f"{release_version}_{folder}"


def check_kv_budget(args: argparse.Namespace) -> None:
    """Refuse a concurrency setting that cannot fit in the endpoint's KV cache.

    This is the check whose absence cost a 2h40m run on 2026-08-24. vLLM admits every request the
    client sends, so `--batch_size` concurrent sequences each generating up to `--max_tokens` must
    fit in the GPU's KV cache. When they do not, vLLM preempts continuously: aggregate throughput
    collapsed 630 -> 210 tok/s, each request then needed ~27 min to reach max_tokens, the client's
    1200s timeout fired first, and upstream re-sent all 40 requests every 20 minutes until it hit
    its 3-retry limit and died -- having written nothing, because upstream only saves a language's
    generations once that whole language finishes.

    Read `--kv_cache_tokens` off the vLLM startup log line "GPU KV cache size: N tokens".
    """
    per_seq = args.max_tokens + KV_PROMPT_ALLOWANCE
    demanded = args.batch_size * per_seq
    safe_batch = max(1, args.kv_cache_tokens // per_seq)

    print(
        f"KV budget: {args.batch_size} concurrent x {per_seq:,} tokens = {demanded:,} needed, "
        f"{args.kv_cache_tokens:,} available (--kv_cache_tokens)"
    )
    if demanded <= args.kv_cache_tokens:
        return

    message = (
        f"--batch_size {args.batch_size} x --max_tokens {args.max_tokens} demands {demanded:,} KV "
        f"tokens but the endpoint has {args.kv_cache_tokens:,} ({demanded / args.kv_cache_tokens:.1f}x "
        f"oversubscribed). vLLM will preempt continuously, throughput will collapse, and requests "
        f"will time out before they finish. Use --batch_size {safe_batch} or lower, reduce "
        f"--max_tokens, or serve on a bigger card. Set --kv_cache_tokens from the vLLM log line "
        f"'GPU KV cache size: N tokens' if the default is wrong for this box."
    )
    if args.allow_kv_oversubscribe:
        print(f"  !! {message}\n  (proceeding anyway: --allow_kv_oversubscribe)")
    else:
        raise UserInputError(message)


def preflight(args: argparse.Namespace, base_url: str, model_name: str, language: str) -> None:
    """Measure what a real request actually costs, before committing hours to the full run.

    Every cost and wall-clock estimate for this instrument rests on one number -- how many tokens a
    completion takes -- and until this runs, that number is a guess. On 2026-08-24 the guess was
    ~3,000 and the run behaved as if it were 16,384, which is the whole difference between a 1-hour
    job and one that never finishes.

    Sends a handful of real prompts at the real sampling settings and reports the completion-token
    distribution and finish_reason. `length` finish_reasons mean the model is hitting --max_tokens
    rather than choosing to stop: the run will take the worst-case time on every request, and the
    completions are truncated mid-reasoning, so they would be graded as failures anyway.
    """
    import statistics
    from openai import OpenAI  # local: only the preflight needs a synchronous client

    if str(Path(args.multilcb_root).resolve()) not in sys.path:
        sys.path.insert(0, str(Path(args.multilcb_root).resolve()))
    os.environ.update(upstream_env(args, base_url))

    saved_argv = sys.argv
    sys.argv = [
        "main", "--model", "VLLMAsync", "--local_model_path", model_name,
        "--plangs", upstream_language(language), "--num_process", "1", "--n", "1",
    ] + (["--cot_code_execution"] if args.reasoning else [])
    try:
        from lcb_runner.runner.parser import get_args as upstream_get_args
        from lcb_runner.runner.scenario_router import build_prompt_benchmark
        from lcb_runner.lm_styles.models_store import LMStyle

        upstream_args = upstream_get_args()
        benchmark, format_prompt = build_prompt_benchmark(upstream_args)
    finally:
        sys.argv = saved_argv

    problems = benchmark[: args.preflight]
    client = OpenAI(api_key=endpoint_api_key(), base_url=base_url, timeout=args.gen_timeout)
    print(
        f"\nPreflight: {len(problems)} {language} problems at temperature={args.temperature}, "
        f"max_tokens={args.max_tokens}, reasoning={args.reasoning}"
    )

    def ask(problem):
        messages = format_prompt(problem, LMStyle.VLLMAsync, upstream_language(language))
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            n=1,
            extra_body={"chat_template_kwargs": upstream_args.chat_template_kwargs},
        )
        choice = response.choices[0]
        return (
            response.usage.completion_tokens if response.usage else -1,
            choice.finish_reason or "unknown",
            "```" in (choice.message.content or ""),
        )

    # Concurrently, not one at a time. A single request decodes at ~74 tok/s, so at a 32k cap each
    # one takes ~7 minutes -- sequentially that is an hour for 8 samples, which defeats the point of
    # a preflight. In parallel the whole thing costs about as long as its slowest single request.
    lengths: list[int] = []
    finishes: dict[str, int] = {}
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=len(problems)) as pool:
        futures = {pool.submit(ask, p): i for i, p in enumerate(problems, start=1)}
        done = 0
        for future in as_completed(futures):
            done += 1
            try:
                used, reason, has_code = future.result()
            except Exception as exc:  # one bad sample must not lose the others
                print(f"  [{done}/{len(problems)}] request failed: {type(exc).__name__}: {exc}")
                continue
            lengths.append(used)
            finishes[reason] = finishes.get(reason, 0) + 1
            print(
                f"  [{done}/{len(problems)}] {used:6,} completion tokens  "
                f"finish={reason:<10} fenced_code={has_code}",
                flush=True,
            )

    if not lengths:
        return

    lengths.sort()
    hit_cap = finishes.get("length", 0)
    print(
        f"\n  completion tokens: min={lengths[0]:,} median={statistics.median(lengths):,.0f} "
        f"max={lengths[-1]:,}   finish_reason={finishes}"
    )

    per_seq = args.max_tokens + KV_PROMPT_ALLOWANCE
    safe_batch = max(1, args.kv_cache_tokens // per_seq)
    print(
        f"  safe --batch_size for this endpoint: {safe_batch} "
        f"({args.kv_cache_tokens:,} KV tokens / {per_seq:,} per sequence)"
    )
    if hit_cap:
        print(
            f"\n  !! {hit_cap}/{len(lengths)} completions hit --max_tokens ({args.max_tokens:,}) rather "
            f"than stopping. Every such request costs the worst case AND is truncated mid-reasoning, "
            f"so it would be graded as a failure. Do not start the full run: decide --max_tokens "
            f"deliberately first, then freeze it -- scores at different values are not comparable."
        )
    else:
        print("\n  All completions stopped on their own. Sizing from the median above is sound.")


def build_upstream_command(args: argparse.Namespace, model_name: str, plangs: list[str]) -> list[str]:
    cmd = [
        sys.executable, "-m", "lcb_runner.runner.main",
        "--model", "VLLMAsync",
        "--local_model_path", model_name,
        "--plangs", ",".join(upstream_language(lang) for lang in plangs),
        "--num_process", str(args.num_process),
        "--n", str(args.n),
        "--temperature", str(args.temperature),
        "--top_p", str(args.top_p),
        "--max_tokens", str(args.max_tokens),
        "--max_seq_length", str(args.max_seq_length),
        "--batch_size", str(args.batch_size),
        "--openai_timeout", str(args.gen_timeout),
        "--num_process_evaluate", str(args.num_process_evaluate),
        "--eval_timeout", str(args.eval_timeout),
        "--eval_restarts", str(args.eval_restarts),
        # Resuming is always on. Generation is the expensive half and upstream keys its cache by
        # question_id, so a re-run after an interrupted evaluation costs nothing.
        "--continue_existing",
        "--continue_existing_eval",
    ]
    if args.reasoning:
        cmd.append("--cot_code_execution")
    if args.start_date:
        cmd += ["--start_date", args.start_date]
    if args.end_date:
        cmd += ["--end_date", args.end_date]
    if args.debug:
        cmd += ["--debug", "--debug_size", str(args.debug_size)]
    if not args.generate_only:
        cmd.append("--evaluate")
    return cmd


def upstream_env(args: argparse.Namespace, base_url: str) -> dict[str, str]:
    env = dict(os.environ)
    env["OPENAI_BASE_URL"] = base_url
    env["OPENAI_KEY"] = endpoint_api_key()
    env["RELEASE_VERSION"] = args.release_version
    env["DATASET_NAME"] = args.release_version
    # Upstream reads these unconditionally; "" means "load from the Hub", which needs
    # datasets==3.6.0 because the LCB loader is a remote-code script dataset. A staged local path
    # loads as plain jsonl instead, which works under our datasets==5.0.0 -- see
    # scripts/fetch_multilcb_dataset.py for why that matters to the bill.
    env["DATASET_PATH"] = str(Path(args.dataset_path).resolve()) if args.dataset_path else ""
    env["DATASET_CACHE"] = args.dataset_cache or ""
    return env


def default_dataset_path(release_version: str) -> str | None:
    """Use a staged local copy of this release if one exists, else fall back to the Hub."""
    staged = ROOT / "data" / "reference" / "multilcb" / release_version
    return str(staged) if staged.is_dir() else None


# Probe that proves a toolchain can actually RUN, not merely that a file sits on PATH. Conda ships
# `scalac` as a wrapper that execs "$CONDA_PREFIX/libexec/scala3/bin/scalac"; with the env on PATH
# but not activated, CONDA_PREFIX is empty, the wrapper execs "/libexec/..." and dies. On 2026-08-26
# that scored scala 0.000 across 60 problems and looked exactly like a model that cannot write
# Scala. `which` passed the whole time.
TOOLCHAIN_PROBE = {
    "rust": ["rustc", "--version"],
    "go": ["go", "version"],
    "php": ["php", "--version"],
    "ruby": ["ruby", "--version"],
    "scala": ["scalac", "-version"],
    "csharp": ["mcs", "--version"],
}


def toolchain_missing(lang: str) -> list[str]:
    from shutil import which

    missing = [binary for binary in TOOLCHAIN_BINARIES[lang] if which(binary) is None]
    if missing:
        return missing

    probe = TOOLCHAIN_PROBE.get(lang)
    if not probe:
        return []
    try:
        done = subprocess.run(probe, capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        return [f"{probe[0]} (on PATH but unusable: {type(exc).__name__})"]
    if done.returncode != 0:
        detail = (done.stderr or done.stdout or b"").decode("utf-8", "replace").strip().splitlines()
        head = detail[0][:120] if detail else f"exit {done.returncode}"
        return [f"{probe[0]} (on PATH but fails to run: {head})"]
    return []


# --- Result normalisation ------------------------------------------------------


def eval_all_path(out_dir: Path, args: argparse.Namespace, plang: str) -> Path:
    """Mirror of upstream `get_eval_all_output_path`."""
    cot_suffix = "_cot" if args.reasoning else ""
    stem = f"codegeneration_{plang}_{args.n}_{args.temperature}_{args.top_p}{cot_suffix}.json"
    return out_dir / f"eval_all_{stem}"


def sample_metadata(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten upstream's per-problem list-of-per-sample metadata dicts."""
    flat: list[dict[str, Any]] = []
    for record in records:
        for meta in record.get("metadata") or []:
            # Upstream json-dumps each per-sample metadata dict, so these arrive as STRINGS, not
            # dicts. Filtering on isinstance(dict) silently produced an empty failure_modes
            # histogram on the first real run -- exactly when the diagnostic was needed most.
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except json.JSONDecodeError:
                    continue
            if isinstance(meta, dict):
                flat.append(meta)
    return flat


def failure_modes(records: list[dict[str, Any]]) -> dict[str, int]:
    """Histogram of upstream's per-sample failure statuses.

    This is the instrument's own self-check, and it is why a low pass rate is not automatically an
    uninterpretable one. Upstream records a `Status` per sample -- EmptyCode, BuildFailed,
    SyntaxError, TimeoutExpired, OutOfMemory, WrongAnswer -- and the shape of that histogram
    separates causes that a bare pass@1 cannot:

        BuildFailed / SyntaxError dominant   the toolchain or environment is wrong, not the model
        EmptyCode dominant                    generation or extraction broke
        TimeoutExpired dominant               the eval box is too slow, or --eval_timeout too tight
        WrongAnswer dominant                  the model actually ran and got it wrong

    Read this before concluding anything about a language.
    """
    counts: dict[str, int] = {}
    for meta in sample_metadata(records):
        key = "Success" if meta.get("success") else str(meta.get("error", "Unknown"))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def diagnose_language(records: list[dict[str, Any]], n_examples: int = 3) -> None:
    """Print the detail behind `failure_modes` -- what broke, in upstream's own words.

    Two things worth looking at by eye. Repeated identical compiler errors across unrelated problems
    mean the environment is wrong. Wrong answers whose output is *nearly* the expected one -- right
    value, different spacing or separator -- mean the stdin/stdout conversion or the comparator is
    off, not the model.
    """
    metas = sample_metadata(records)

    messages: dict[str, int] = {}
    for meta in metas:
        if meta.get("success"):
            continue
        message = (meta.get("error_message") or "").strip()
        if not message:
            continue
        # First line only: compiler errors carry a per-file path after it that would fragment the count.
        head = message.splitlines()[0][:160]
        messages[head] = messages.get(head, 0) + 1

    if messages:
        print("    most common error messages:")
        for message, count in sorted(messages.items(), key=lambda kv: -kv[1])[:n_examples]:
            print(f"      [{count:4d}x] {message}")

    wrong = [m for m in metas if str(m.get("error")) == "Status.WrongAnswer" or m.get("error") == "WrongAnswer"]
    if wrong:
        print("    sample wrong answers (check for near-misses -- a formatting mismatch is OUR bug):")
        for meta in wrong[:n_examples]:
            print(f"      in={meta.get('inputs', '')!r}")
            print(f"        got={meta.get('output', '')!r}")
            print(f"        exp={meta.get('expected', '')!r}")


def summarize_language(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-language summary from upstream's per-problem eval records.

    Reads `eval_all_*.json` rather than upstream's combined CSV so the difficulty and no-code
    breakdowns come from the same pass, and so the numbers we report are traceable to individual
    problems rather than to a row someone has to re-derive.
    """
    if not records:
        return {"n_problems": 0}

    def pass_at_1(subset: list[dict[str, Any]]) -> float | None:
        if not subset:
            return None
        return sum(r["pass@1"] for r in subset) / len(subset)

    # An empty extracted code string is upstream's "the model never produced a fenced block".
    # This is the analogue of `likely_truncated_rate` in the other instrument, and it must be read
    # BEFORE the pass rate: a score that fell because nothing compiled is a different finding from
    # one that fell because the code was wrong.
    samples = [code for r in records for code in r.get("code_list", [])]
    no_code = sum(1 for code in samples if not (code or "").strip())

    by_difficulty = {
        level: pass_at_1([r for r in records if r.get("difficulty") == level])
        for level in ("easy", "medium", "hard")
    }
    by_platform = {
        platform: pass_at_1([r for r in records if r.get("platform") == platform])
        for platform in ("leetcode", "atcoder", "codeforces")
    }

    # Two numbers, never one. A problem whose generation produced no extractable code
    # scores 0 exactly like a wrong answer, so the headline pass@1 is the PRODUCT of
    # "did the model emit a usable program" and "was that program right". Those move
    # independently: a checkpoint that merely learned to terminate sooner raises the
    # first factor and posts a better score without being better at coding. Report
    # pass@1 restricted to problems where every sample yielded code, alongside how
    # many problems that excludes.
    with_code = [
        r for r in records
        if r.get("code_list") and all((code or "").strip() for code in r["code_list"])
    ]

    dates = sorted(r["contest_date"][:10] for r in records if r.get("contest_date"))
    return {
        "n_problems": len(records),
        "n_samples": len(samples),
        "pass@1": pass_at_1(records),
        "pass@1_given_code": pass_at_1(with_code),
        "n_problems_with_code": len(with_code),
        "code_emission_rate": (len(with_code) / len(records)) if records else 0.0,
        "failure_modes": failure_modes(records),
        "pass@1_by_difficulty": {k: v for k, v in by_difficulty.items() if v is not None},
        "pass@1_by_platform": {k: v for k, v in by_platform.items() if v is not None},
        "difficulty_counts": {
            level: sum(1 for r in records if r.get("difficulty") == level)
            for level in ("easy", "medium", "hard")
        },
        "no_code_rate": (no_code / len(samples)) if samples else 0.0,
        "date_range": [dates[0], dates[-1]] if dates else None,
    }


def collect_results(
    out_dir: Path, args: argparse.Namespace, languages: list[str]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_language: dict[str, Any] = {}
    raw_records: list[dict[str, Any]] = []

    for lang in languages:
        path = eval_all_path(out_dir, args, upstream_language(lang))
        if not path.exists():
            per_language[lang] = {"error": f"no evaluation output at {path}"}
            print(f"  {lang}: MISSING -- {path}")
            continue

        records = json.loads(path.read_text())
        per_language[lang] = summarize_language(records)
        stats = per_language[lang]
        print(
            f"  {lang}: pass@1={stats['pass@1']:.3f} over {stats['n_problems']} problems "
            f"(easy/med/hard "
            f"{_fmt(stats['pass@1_by_difficulty'].get('easy'))}/"
            f"{_fmt(stats['pass@1_by_difficulty'].get('medium'))}/"
            f"{_fmt(stats['pass@1_by_difficulty'].get('hard'))}) "
            f"no_code={stats['no_code_rate']:.1%}"
        )
        if stats["failure_modes"]:
            modes = " ".join(f"{k}={v}" for k, v in stats["failure_modes"].items())
            print(f"    failure_modes: {modes}")
        if stats["no_code_rate"] > 0.05:
            # These are responses truncated at --max_tokens, not refusals. Raising the cap is NOT the
            # fix and this line used to advise it: measured on the 4B v6 baselines 2026-09-04, the
            # length distribution is bimodal -- answers land under ~1k tokens or run to the cap, with
            # essentially nothing in between (C#: 51 responses under 1k, 4 between 1k and 4k, zero
            # from 4k to 16k, 120 at the cap). A response that spirals at 16384 spirals at 32768 too,
            # so a raise doubles generation cost and moves the wall. Report the pair instead.
            print(
                f"    !! {stats['no_code_rate']:.1%} of {lang} samples produced no fenced code -- "
                f"truncated at --max_tokens ({args.max_tokens}), not refusals. Read pass@1 beside "
                f"pass@1_given_code ({_fmt(stats.get('pass@1_given_code'))}); never either alone."
            )
        # A language that essentially never got as far as running is measuring the environment, not
        # the model. Say so here rather than leaving it to be noticed in the JSON.
        broke = sum(
            count for mode, count in stats["failure_modes"].items()
            if any(bad in mode for bad in ("BuildFailed", "SyntaxError", "BuildTimeOut", "NPMFailed"))
        )
        if stats["n_samples"] and broke / stats["n_samples"] > 0.5:
            print(
                f"    !! {broke}/{stats['n_samples']} {lang} samples failed to BUILD. That is an "
                f"environment/toolchain signal, not a capability one -- run --diagnose before "
                f"reporting this language."
            )
        if args.diagnose:
            diagnose_language(records)

        for record in records:
            raw_records.append(
                {
                    "language": lang,
                    "question_id": record.get("question_id"),
                    "platform": record.get("platform"),
                    "difficulty": record.get("difficulty"),
                    "contest_date": record.get("contest_date"),
                    "graded_list": record.get("graded_list"),
                    "pass@1": record.get("pass@1"),
                    "code_list": record.get("code_list"),
                    "output_list": record.get("output_list"),
                    "metadata": record.get("metadata"),
                }
            )

    return per_language, raw_records


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


# --- Orchestration -------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    multilcb_root = Path(args.multilcb_root).resolve()
    if not (multilcb_root / "lcb_runner").is_dir():
        raise UserInputError(
            f"No vendored Multi-LCB at {multilcb_root} (expected a lcb_runner/ inside it)."
        )

    languages = [lang.strip() for lang in args.languages.split(",") if lang.strip()]
    unknown = [lang for lang in languages if lang not in LCB_LANGUAGES]
    if unknown:
        raise UserInputError(f"Unknown language(s) {unknown}; known: {sorted(LCB_LANGUAGES)}")

    if args.generate_only and args.evaluate_only:
        raise UserInputError("--generate_only and --evaluate_only are mutually exclusive.")

    # Only evaluation needs compilers. Generating, preflighting and re-reading a finished run all
    # execute nothing, so none of them is gated on them.
    if not args.generate_only and not args.summarize_only and not args.preflight:
        missing = {
            lang: toolchain_missing(lang)
            for lang in languages
            if lang in TOOLCHAIN_BINARIES and toolchain_missing(lang)
        }
        if missing:
            detail = "; ".join(f"{lang} needs {', '.join(bins)}" for lang, bins in missing.items())
            raise UserInputError(
                f"Missing toolchains on PATH: {detail}. Activate multi_lcb_env "
                "(see requirements-multilcb.txt), or pass --generate_only to defer evaluation."
            )

    if args.dataset_path is None:
        args.dataset_path = default_dataset_path(args.release_version)
        if args.dataset_path:
            print(f"Using staged dataset at {args.dataset_path}")
        else:
            print(
                f"No staged dataset for {args.release_version!r}; upstream will download from the "
                "Hub, which needs datasets==3.6.0. Run scripts/fetch_multilcb_dataset.py to avoid that."
            )

    base_url, model_name = resolve_endpoint(args.model_key)
    out_dir = multilcb_root / "output" / upstream_output_dir(
        model_name, args.release_version, args.reasoning, args.debug
    )

    if args.preflight:
        preflight(args, base_url, model_name, languages[0])
        return

    if not args.summarize_only:
        if not args.evaluate_only:
            check_kv_budget(args)
        if args.evaluate_only:
            # Generation resumes from cache and finds nothing left to do, so no endpoint is
            # contacted -- but OPENAI_BASE_URL still has to be set, because upstream reads it at
            # module import time.
            print("Evaluate-only: upstream will resume generation from cache and evaluate.")
        print(f"Endpoint: {base_url} (served model {model_name!r})")

        # One upstream invocation PER LANGUAGE, in the order given, rather than one invocation for
        # all of them. Two reasons, both learned the hard way:
        #   1. Upstream writes a language's generations only after that entire language finishes, so
        #      a crash in a single-invocation run destroys every hour spent on it. Per-language calls
        #      checkpoint as they go.
        #   2. Upstream reorders --plangs into its own enum order (c++, c#, python, ...), so asking
        #      for python first does not run python first. Calling once per language restores our
        #      order, which matters because python is the calibration arm.
        batches = [[lang] for lang in languages] if args.per_language else [languages]
        for index, batch in enumerate(batches, start=1):
            label = ",".join(batch)
            cmd = build_upstream_command(args, model_name, batch)
            print(f"\n--- [{index}/{len(batches)}] {label} ---")
            print(f"Running: {' '.join(cmd)}\n  cwd={multilcb_root}\n")
            completed = subprocess.run(cmd, cwd=multilcb_root, env=upstream_env(args, base_url))
            if completed.returncode != 0:
                raise UserInputError(
                    f"Upstream lcb_runner exited {completed.returncode} on {label}. Its output "
                    f"above is the authoritative error. Languages completed before this one are "
                    f"saved under {out_dir} and will be reused on re-run (--continue_existing)."
                )

    if args.generate_only:
        print(f"\nGeneration complete. Generations under {out_dir}. "
              "Re-run with --evaluate_only on a box with the toolchains.")
        return

    if not out_dir.is_dir():
        existing = sorted(p.name for p in (multilcb_root / "output").glob("*")) or ["<none>"]
        raise UserInputError(
            f"Expected upstream output at {out_dir}, which does not exist. "
            f"Directories present: {', '.join(existing)}"
        )

    print(f"\nSummarising {out_dir}")
    per_language, raw_records = collect_results(out_dir, args, languages)

    save_dir = Path(args.save_dir or default_results_dir(args.model_key, "multilcb"))
    save_dir.mkdir(parents=True, exist_ok=True)
    run_base = timestamped(args.save_suffix, enabled=not args.no_timestamp)

    summary = {
        "instrument": "multi_lcb",
        "instrument_version": {
            "upstream_commit": args.upstream_commit,
            "release_version": args.release_version,
            "start_date": args.start_date,
            "end_date": args.end_date,
        },
        "model_key": args.model_key,
        "served_model_name": model_name,
        "base_url": base_url,
        "n": args.n,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "reasoning": args.reasoning,
        # A --debug run is a SUBSAMPLE, not a smoke test, whenever debug_size is set deliberately:
        # upstream takes benchmark[:N] in dataset order, which is difficulty-representative
        # (measured: 40% commit at N=40-80 against 36% for the full 175). Record the size, because
        # a pass rate over 60 problems is not comparable to one over 175.
        "debug": args.debug,
        "debug_size": args.debug_size if args.debug else None,
        "languages": languages,
        "upstream_output_dir": str(out_dir),
        "results": per_language,
    }
    out_path = save_dir / f"multilcb_{run_base}.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nWrote {out_path}")

    raw_path = save_dir / f"raw_responses_multilcb_{run_base}.jsonl"
    with raw_path.open("w") as handle:
        for record in raw_records:
            handle.write(json.dumps(record) + "\n")
    print(f"Wrote {raw_path} ({len(raw_records)} problem records)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model_key", required=True,
        help="Key in config.yaml; must be model_type: vllm_endpoint (e.g. qwen35-4b-m0-v4-aws)",
    )
    parser.add_argument(
        "--languages", default=",".join(DEFAULT_LANGUAGES),
        help=f"Comma-separated subset of upstream's languages. Default: {','.join(DEFAULT_LANGUAGES)}. "
        "Also available: python, ruby (installed, just not in the default run). Use the key "
        "'csharp', not 'c#' -- our keys are shell-safe, and the adapter translates.",
    )
    parser.add_argument(
        "--release_version", default="v6",
        help="LCB dataset tag. 'v6' is exactly the 175 problems v6 ADDED (2025-01-04 to "
        "2025-04-06). 'release_v6' is the cumulative ~1055 problems from May 2023 -- pair that "
        "with --start_date or the run is 6x larger than intended.",
    )
    parser.add_argument("--start_date", default=None, help="Filter problems on/after YYYY-MM-DD")
    parser.add_argument("--end_date", default=None, help="Filter problems on/before YYYY-MM-DD")
    parser.add_argument("--n", type=int, default=2, help="Samples per problem per language")
    parser.add_argument("--temperature", type=float, default=0.2, help="LCB default")
    parser.add_argument("--top_p", type=float, default=0.95, help="LCB default")
    parser.add_argument(
        "--max_tokens", type=int, default=16384,
        help="Max generation tokens. Competitive-programming problems with reasoning on run long; "
        "read no_code_rate before reading any pass rate. Scores taken at different values of this "
        "flag are NOT comparable -- re-score every cell together after changing it.",
    )
    parser.add_argument("--max_seq_length", type=int, default=32768)
    parser.add_argument(
        "--reasoning", dest="reasoning", action="store_true", default=True,
        help="Enable thinking (upstream --cot_code_execution). Default on: our checkpoints are "
        "M0 reason-then-commit warm-starts, so scoring them reasoning-off measures a different model.",
    )
    parser.add_argument("--no-reasoning", dest="reasoning", action="store_false")
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Client-side async queue depth = concurrent requests on the endpoint. Was 40 (from "
        "upstream's run.sh, which assumes a much bigger card) until 2026-08-24, when 40 x 16384 "
        "tokens oversubscribed a 24 GB box's KV cache 2.4x and the run collapsed. 8 is safe on a "
        "24 GB card; raise it with --kv_cache_tokens set from the real endpoint.",
    )
    parser.add_argument(
        "--kv_cache_tokens", type=int, default=DEFAULT_KV_CACHE_TOKENS,
        help="The endpoint's KV cache size in tokens, from the vLLM startup log line 'GPU KV cache "
        f"size: N tokens'. Used to reject an unrunnable --batch_size. Default {DEFAULT_KV_CACHE_TOKENS:,} "
        "(a ~24 GB card); an L40S reports roughly 1,000,000.",
    )
    parser.add_argument(
        "--allow_kv_oversubscribe", action="store_true",
        help="Downgrade the KV-budget check from an error to a warning. You almost certainly do not "
        "want this -- oversubscribing does not fail fast, it fails after hours.",
    )
    parser.add_argument(
        "--per_language", dest="per_language", action="store_true", default=True,
        help="Invoke upstream once per language, in the order given (default). Checkpoints each "
        "language's generations before starting the next, and defeats upstream's reordering of "
        "--plangs into its own enum order.",
    )
    parser.add_argument(
        "--single_pass", dest="per_language", action="store_false",
        help="Hand every language to one upstream invocation. Faster startup, but a crash loses "
        "every language's generations and upstream picks the run order.",
    )
    parser.add_argument(
        "--num_process", type=int, default=1,
        help="Parallelism for DATA LOADING only (not generation, not evaluation). Default 1 "
        "deliberately: upstream defaults to os.cpu_count() and feeds whole problems -- each "
        "carrying a multi-MB compressed test blob -- through a multiprocessing Pool, which hangs "
        "for >10 min on macOS spawn. Single-process load+convert of the 175-problem v6 window "
        "takes 4s, so there is nothing to win here.",
    )
    parser.add_argument(
        "--gen_timeout", type=int, default=3600,
        help="Per-request timeout (s). Was 1200 until 2026-08-24, when requests routinely needed "
        "longer than that and upstream re-sent all of them every 20 minutes, throwing away the "
        "partial work each time, until its 3-retry limit killed the run. A too-long timeout only "
        "makes a stall slow to surface; a too-short one destroys the run.",
    )
    parser.add_argument(
        "--num_process_evaluate", type=int, default=min(os.cpu_count() or 4, 60),
        help="Evaluation parallelism. Too high causes spurious TimeoutErrors; upstream uses <=60.",
    )
    parser.add_argument("--eval_timeout", type=int, default=6, help="Per-test timeout (s)")
    parser.add_argument(
        "--eval_restarts", type=int, default=3,
        help="Re-run timed-out evaluations this many times with decreasing parallelism.",
    )
    parser.add_argument(
        "--generate_only", action="store_true",
        help="Generate and stop -- for the GPU box, where the toolchains are not installed.",
    )
    parser.add_argument(
        "--evaluate_only", action="store_true",
        help="Skip generation (resumed from cache) and evaluate -- for a CPU box with the toolchains.",
    )
    parser.add_argument(
        "--summarize_only", action="store_true",
        help="Re-read a finished run's upstream output and rewrite our summary. No model, no execution.",
    )
    parser.add_argument(
        "--diagnose", action="store_true",
        help="Print the detail behind the failure-mode histogram: the most common compiler/runtime "
        "error messages, and sample wrong answers with their inputs and expected output. This is "
        "the cheap integration check -- repeated identical build errors mean a broken environment, "
        "and near-miss wrong answers mean a formatting or conversion bug on our side. Pairs well "
        "with --summarize_only, which re-reads a finished run for free.",
    )
    parser.add_argument(
        "--preflight", type=int, default=0, metavar="N",
        help="Before anything else, send N real prompts to the endpoint and report the "
        "completion-token distribution and finish_reason, then exit. This is the only way to know "
        "what the run will actually cost -- run it FIRST, on every new box. N=5 takes minutes.",
    )
    parser.add_argument("--debug", action="store_true", help="Smoke run on --debug_size problems")
    parser.add_argument("--debug_size", type=int, default=10)
    parser.add_argument(
        "--dataset_path", default=None,
        help="Directory of staged LCB jsonl shards (see scripts/fetch_multilcb_dataset.py). "
        "Defaults to data/reference/multilcb/<release_version> when that exists, otherwise the "
        "Hub -- which downloads 4.5GB for release_v6 and requires datasets==3.6.0.",
    )
    parser.add_argument("--dataset_cache", default=None, help="HF datasets cache dir")
    parser.add_argument("--multilcb_root", default=str(DEFAULT_MULTILCB_ROOT))
    parser.add_argument(
        "--upstream_commit", default="d80be9f24268e9e5c6e1463e26c9f05417fe06a0",
        help="Recorded in the summary so a number can be traced to the harness that produced it.",
    )
    parser.add_argument(
        "--save_dir", default=None, help="Output dir. Default: results/<model_key>/multilcb",
    )
    parser.add_argument("--save_suffix", default="multilcb")
    parser.add_argument(
        "--no-timestamp", dest="no_timestamp", action="store_true",
        help="Write to a fixed filename with no UTC timestamp; a re-run then overwrites it.",
    )
    return parser


if __name__ == "__main__":
    cli_args = build_parser().parse_args()
    try:
        run(cli_args)
    except UserInputError as exc:
        raise SystemExit(str(exc)) from exc
