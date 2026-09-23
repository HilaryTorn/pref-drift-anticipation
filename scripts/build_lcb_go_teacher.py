#!/usr/bin/env python3
"""Generate teacher solutions for LiveCodeBench v1-v5 in Multi-LCB's cache format.

This is step 1 of the LiveCodeBench SFT arms (docs/lcb-sft-runbook.md). The Magicoder arms did not move the model because the model could already produce their targets; the fix is a training target it cannot currently produce and a training window that is DISJOINT from the eval window, so v6 stays the untouched held-out test.

What it does:
  1. Loads the LCB problems for --release_version through the vendored upstream loader (same code path, same stdin/stdout conversion, same problem ordering as scripts/score_multilcb.py).
  2. Holds out --validation_problems problems BY PROBLEM ID (never by row: several solutions to one problem must not straddle the split). The split is written to <workdir>/split.json and read by build_lcb_go_pool.py.
  3. Asks a hosted teacher model for --k solutions per problem, using upstream's own language prompt so the training user prompt is byte-identical to what the eval sends.
  4. Writes the completions into upstream's generation-cache file under multi-lcb/output/, so verification is just `score_multilcb.py --evaluate_only` against a config.yaml key -- upstream's compiler, runner and comparator, unmodified. Nothing here re-implements grading.

It does NOT verify anything. Verification needs the Linux toolchain box (docs/multilcb-eval.md, "CPU box"); this script runs anywhere with the repo environment and the selected provider key.

Three teacher providers, same output either way:

    --provider openrouter  (default) One key, any frontier model, OpenAI-compatible
                           chat/completions. The --model id is validated against the LIVE
                           catalogue before a single request is sent (a typo prints near-matches
                           instead of failing 880 times), and the run is priced from OpenRouter's
                           real rate card rather than a hardcoded table. Key: OPENROUTER_API_KEY.
    --provider azure       Azure OpenAI Responses API, funded by the project's Azure credit.
                           Same api-key header and hard timeout as the RL SFT builder.
    --provider anthropic   Message Batches API at 50% of standard prices, paid per token. The
                           batch id is saved to <workdir>/batch_state.json, so re-running polls
                           instead of re-submitting. --sync sends requests directly (smoke only).

openrouter and azure share rl_training/openai_responses_worker.py -- a generic "POST this JSON,
return that JSON" subprocess with a hard timeout -- differing only in auth header and response
shape. Both run concurrently and append every completed request to <workdir>/<provider>_progress.jsonl,
so an interrupted run resumes for free and retries only what failed. Nothing is ever paid twice.

Cost rule (CLAUDE.md): a full run of any provider prints its estimate and stops until --yes; the
anthropic path additionally refuses a USD estimate above --max_cost_usd.

Usage:
    # smoke: 3 problems, k=1, on the SAME provider the real run will use
    python3 scripts/build_lcb_go_teacher.py --limit 3 --k 1

    # real run (OpenRouter): estimate first, then commit
    python3 scripts/build_lcb_go_teacher.py --k 1
    python3 scripts/build_lcb_go_teacher.py --k 1 --yes
    python3 scripts/build_lcb_go_teacher.py --k 1 --model openai/gpt-5.6-sol --yes

    # real run (Anthropic, paid): submit a batch, re-run the same command to resume polling
    python3 scripts/build_lcb_go_teacher.py --k 1 --provider anthropic --model claude-sonnet-5 --yes

OpenRouter environment -- put it in the repo's gitignored .env (same file as HF_TOKEN), or in
api_keys/api_key_openrouter.txt. An exported variable wins over .env:

    OPENROUTER_API_KEY=sk-or-...

Azure environment (same variables as rl_training/README.md, read from the environment only --
never passed as arguments, never written to any artifact):

    export AZURE_OPENAI_API_KEY='<set locally, never commit>'
    export AZURE_OPENAI_ENDPOINT='https://<resource>.openai.azure.com/openai/v1'
    export AZURE_OPENAI_DEPLOYMENT='<azure-deployment-name>'
    export AZURE_OPENAI_API_VERSION='<api version, provenance only>'
    export AZURE_OPENAI_MODEL_VERSION='<immutable deployed model version, provenance only>'
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MULTILCB_ROOT = ROOT / "multi-lcb"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.lcb_languages import upstream_language  # noqa: E402

DEFAULT_LANGUAGE = "go"

# Appended to upstream's system message, never to the user message: the user message is the
# training prompt and must stay identical to what the eval sends. The addendum only pins the
# output shape so the answer extracts cleanly as a single program.
#
# One entry per language this builder can generate for. Adding a language means adding its addendum
# here AND a matching `lcb-<language>-teacher` grading handle in config.yaml -- see teacher_name().
# The language must also be one upstream can render prompts for and grade (multi-lcb/tests/tests_plangs/).
TEACHER_ADDENDA = {
    "go": (
        "Respond with exactly one fenced ```go code block containing a complete, self-contained Go "
        "program (package main, func main) that reads from stdin and writes to stdout. No prose before "
        "or after the block. Use only the Go standard library. Prefer bufio for I/O; inputs can be "
        "large."
    ),
    "rust": (
        "Respond with exactly one fenced ```rust code block containing a complete, self-contained Rust "
        "program (fn main) that reads from stdin and writes to stdout. No prose before or after the "
        "block. Use only the Rust standard library -- no external crates, since the grader compiles "
        "with rustc and no Cargo manifest. Prefer a locked BufWriter and a single read_to_string for "
        "I/O; inputs can be large."
    ),
    "php": (
        "Respond with exactly one fenced ```php code block containing a complete, self-contained PHP "
        "program that reads from stdin and writes to stdout. No prose before or after the block. Use "
        "only the PHP standard library -- no Composer packages, since the grader runs bare PHP. Inputs "
        "can be large."
    ),
    "csharp": (
        "Respond with exactly one fenced ```csharp code block containing a complete, self-contained C# "
        "program with a static Main entry point that reads from stdin and writes to stdout. No prose "
        "before or after the block. Use only the .NET/Mono standard library -- no NuGet packages, since "
        "the grader compiles with mcs and runs with mono. Inputs can be large."
    ),
}

LANGUAGES = tuple(TEACHER_ADDENDA)


def teacher_name(language: str) -> str:
    """The config.yaml key (and therefore upstream output folder) the cache is graded under.

    score_multilcb.py resolves this key to a served_model_name; upstream names its output folder
    after that name, so the two MUST agree: <release>_<teacher_name>_cot/. A language added to
    TEACHER_ADDENDA without the matching config.yaml key will generate fine and then fail to grade.
    """
    return f"lcb-{language}-teacher"


def default_workdir(language: str) -> Path:
    return ROOT / "data" / "training" / f"lcb_{language}"

# USD per million tokens (input, output), first-party API, from the claude-api skill table.
# Batch mode halves both.
PRICES = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

# Sonnet 5 / Opus 5 reject sampling params (temperature/top_p); Haiku 4.5 accepts them. We send
# none to any of them, so the k samples per problem differ only through the model's own
# nondeterminism. The cache FILENAME still carries upstream's default 0.2/0.95 because that is what
# score_multilcb.py will look for; the numbers are a filename convention here, not a setting.
CACHE_TEMPERATURE = 0.2
CACHE_TOP_P = 0.95

# Azure OpenAI, matching rl_training/build_sft_dataset_azure.py exactly so there is one Azure
# contract in this repo rather than two. Keys are read from the environment only.
AZURE_KEY_ENV = "AZURE_OPENAI_API_KEY"
AZURE_ENDPOINT_ENV = "AZURE_OPENAI_ENDPOINT"
AZURE_DEPLOYMENT_ENV = "AZURE_OPENAI_DEPLOYMENT"
AZURE_API_VERSION_ENV = "AZURE_OPENAI_API_VERSION"
AZURE_MODEL_VERSION_ENV = "AZURE_OPENAI_MODEL_VERSION"
# The env var name the worker subprocess reads the key from; it never appears in argv.
AZURE_WORKER_KEY_ENV = "LCB_GO_TEACHER_WORKER_KEY"
AZURE_RETRY_CODES = {408, 409, 429, 500, 502, 503, 504}

# OpenRouter: one key, every frontier model, OpenAI-compatible chat/completions. Same worker
# subprocess as Azure -- it is a generic "POST this JSON, return that JSON" with a hard timeout;
# only the auth header (bearer vs api-key) and the response shape differ.
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
# Chosen 2026-08-28. The teacher bar is set by the LARGEST student, not the smallest: the study may
# run 4B, 9B and 27B off this one dataset, and a teacher only modestly above Qwen3.5-27B reproduces
# the Magicoder null at 27B (targets the student can already produce => near-zero loss => nothing
# learned). Sol is the flagship reasoning+coding model, which buys margin over every student size.
# Deliberately NOT a same-family model (qwen3.5-397b): its solutions look most like what the student
# already writes, which is the worst property for clearing the bar.
# The id is validated against the live catalogue before any spend, so a stale default cannot cost
# 880 failed requests.
DEFAULT_OPENROUTER_MODEL = "openai/gpt-5.6-sol"


# --- Upstream loading ------------------------------------------------------------


def load_upstream(release_version: str, dataset_path: Path, limit: int, language: str) -> tuple[list[Any], Any, Any]:
    """Load the benchmark and prompt formatter exactly as score_multilcb.py / upstream do.

    Mirrors the env + argv dance in score_multilcb.preflight(). --debug/--debug_size is used for
    --limit so that the subset is upstream's own benchmark[:N] in its own sort order, which is what
    the verification run (also --debug --debug_size N) will load.
    """
    if str(MULTILCB_ROOT) not in sys.path:
        sys.path.insert(0, str(MULTILCB_ROOT))
    if not (MULTILCB_ROOT / "lcb_runner").is_dir():
        raise SystemExit(f"No vendored Multi-LCB at {MULTILCB_ROOT}")
    if not dataset_path.is_dir():
        raise SystemExit(
            f"No staged dataset at {dataset_path}. Run: python3 scripts/fetch_multilcb_dataset.py "
            f"--release {release_version}"
        )

    os.environ["OPENAI_BASE_URL"] = "http://localhost:9/v1"  # never contacted; upstream reads it at import
    os.environ["OPENAI_KEY"] = "dummy"
    os.environ["RELEASE_VERSION"] = release_version
    os.environ["DATASET_NAME"] = release_version
    os.environ["DATASET_PATH"] = str(dataset_path.resolve())
    os.environ["DATASET_CACHE"] = ""

    argv = [
        "main", "--model", "VLLMAsync", "--local_model_path", teacher_name(language),
        "--plangs", upstream_language(language), "--num_process", "1", "--cot_code_execution",
    ]
    if limit:
        argv += ["--debug", "--debug_size", str(limit)]

    saved_argv = sys.argv
    sys.argv = argv
    try:
        from lcb_runner.runner.parser import get_args as upstream_get_args
        from lcb_runner.runner.scenario_router import build_prompt_benchmark

        upstream_args = upstream_get_args()
        benchmark, format_prompt = build_prompt_benchmark(upstream_args)
    finally:
        sys.argv = saved_argv

    # Same two filters upstream's main() applies before generating.
    benchmark = [b for b in benchmark if not b.is_functional()]
    if limit:
        benchmark = benchmark[:limit]
    return benchmark, format_prompt, upstream_args


def cache_path(upstream_args: Any, k: int, language: str) -> Path:
    """Upstream's generation-cache path for this run: output/<dir>/codegeneration_<lang>_<k>_0.2_0.95_cot.json."""
    return (
        MULTILCB_ROOT / "output" / upstream_args.output_dir
        / f"codegeneration_{upstream_language(language)}_{k}_{CACHE_TEMPERATURE}_{CACHE_TOP_P}_cot.json"
    )


# --- Split ---------------------------------------------------------------------


def make_split(benchmark: list[Any], validation_problems: int, seed: int) -> dict[str, Any]:
    qids = sorted(b.question_id for b in benchmark)
    if validation_problems >= len(qids):
        raise SystemExit(f"--validation_problems {validation_problems} >= {len(qids)} problems loaded")
    rng = random.Random(seed)
    validation = sorted(rng.sample(qids, validation_problems))
    validation_set = set(validation)
    return {
        "seed": seed,
        "validation_question_ids": validation,
        "train_question_ids": [q for q in qids if q not in validation_set],
    }


# --- Teacher requests ------------------------------------------------------------


def load_env_files() -> None:
    """Make the repo's .env visible, the way compute_utilities/llm_agent.py already does.

    Resolution order for every Azure setting is: process environment first (so an existing
    `export AZURE_OPENAI_*` session, e.g. the RL builder's, keeps working unchanged), then .env,
    then -- for the key only -- api_keys/api_key_azure_openai.txt. .env and api_keys/*.txt are both
    gitignored, so a personal key never rides along in a commit and never has to be shared.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env")


def azure_settings(deployment_override: str | None) -> dict[str, str]:
    load_env_files()

    key = os.environ.get(AZURE_KEY_ENV, "").strip()
    if not key:
        key_path = ROOT / "api_keys" / "api_key_azure_openai.txt"
        if key_path.exists():
            key = key_path.read_text().strip()

    endpoint = os.environ.get(AZURE_ENDPOINT_ENV, "").strip().rstrip("/")
    deployment = (deployment_override or os.environ.get(AZURE_DEPLOYMENT_ENV, "")).strip()

    missing = [
        name
        for name, value in [(AZURE_KEY_ENV, key), (AZURE_ENDPOINT_ENV, endpoint), (AZURE_DEPLOYMENT_ENV, deployment)]
        if not value
    ]
    if missing:
        raise SystemExit(
            "Missing Azure settings: " + ", ".join(missing) + ".\n"
            "Put them in the repo's gitignored .env (same file as ANTHROPIC_API_KEY / HF_TOKEN), one per line:\n"
            f"  {AZURE_KEY_ENV}=<your own key>\n"
            f"  {AZURE_ENDPOINT_ENV}=https://<your-resource>.openai.azure.com/openai/v1\n"
            f"  {AZURE_DEPLOYMENT_ENV}=<your deployment name>\n"
            f"  {AZURE_API_VERSION_ENV}=<api version>          # provenance only\n"
            f"  {AZURE_MODEL_VERSION_ENV}=<deployed model version>  # provenance only\n"
            f"The key may instead go in api_keys/api_key_azure_openai.txt. Exported environment "
            "variables win over .env, so an existing RL-builder session is unaffected."
        )
    if not endpoint.startswith("https://"):
        raise SystemExit(f"{AZURE_ENDPOINT_ENV} must be an https:// URL, got {endpoint!r}")

    return {
        "key": key,
        "endpoint": endpoint,
        "responses_url": endpoint if endpoint.endswith("/responses") else f"{endpoint}/responses",
        "deployment": deployment,
        "api_version": os.environ.get(AZURE_API_VERSION_ENV, "").strip(),
        "model_version": os.environ.get(AZURE_MODEL_VERSION_ENV, "").strip(),
    }


def post_via_worker(body: dict[str, Any], url: str, key: str, auth_header: str, args: argparse.Namespace) -> tuple[str, Any]:
    """One HTTP POST through rl_training/openai_responses_worker.py, with a hard timeout.

    The worker is provider-agnostic -- it posts JSON and returns the parsed JSON body -- so Azure
    (api-key header, Responses API) and OpenRouter (bearer, chat/completions) share it. The key
    travels in the child's environment under a private variable name, never in argv.
    """
    payload = json.dumps(body, ensure_ascii=True).encode("utf-8")
    env = os.environ.copy()
    env[AZURE_WORKER_KEY_ENV] = key
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    try:
        completed = subprocess.run(
            [
                sys.executable, "-m", "rl_training.openai_responses_worker",
                "--timeout", str(args.request_timeout),
                "--api_key_env", AZURE_WORKER_KEY_ENV,
                "--responses_url", url,
                "--auth_header", auth_header,
            ],
            input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, cwd=str(ROOT), timeout=args.request_timeout + 10, check=False,
        )
    except subprocess.TimeoutExpired:
        return "timeout_error", f"timeout after {args.request_timeout}s"

    stdout = completed.stdout.decode("utf-8", "replace")
    stderr = completed.stderr.decode("utf-8", "replace")
    if completed.returncode != 0 and not stdout.strip():
        return "worker_error", f"worker exit {completed.returncode}: {stderr[:300]}"
    try:
        row = json.loads(stdout)
    except json.JSONDecodeError:
        return "worker_error", f"worker produced invalid JSON: {stderr[:300]}"
    return row.get("status"), row.get("result")


def call_with_retries(custom_id: str, body: dict[str, Any], url: str, key: str, auth_header: str, parse, args: argparse.Namespace) -> dict[str, Any]:
    """Retry an API call on 408/409/429/5xx and timeouts, then hand the body to `parse`."""
    last_error = "unknown_error"
    for attempt in range(1, args.api_retries + 1):
        status, result = post_via_worker(body, url, key, auth_header, args)
        if status == "ok":
            return parse(custom_id, result)
        if status == "http_error":
            code = result.get("code")
            last_error = f"http_{code}: {str(result.get('body'))[:300]}"
            if code not in AZURE_RETRY_CODES or attempt == args.api_retries:
                break
        else:
            last_error = f"{status}: {str(result)[:300]}"
            if attempt == args.api_retries:
                break
        time.sleep(args.retry_sleep * attempt)
    return {"custom_id": custom_id, "stop_reason": "error", "input_tokens": 0, "output_tokens": 0, "text": "", "error": last_error}


def openrouter_catalogue(key: str) -> list[dict[str, Any]]:
    """The live model list. Used to validate --model and to price the run from real rates."""
    import urllib.request

    request = urllib.request.Request(OPENROUTER_MODELS_URL, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8")).get("data") or []


def openrouter_settings(model: str) -> dict[str, Any]:
    """Resolve the key, then validate the model id against the live catalogue BEFORE any spend.

    A mistyped id is the obvious way to lose an afternoon here -- OpenRouter's ids are
    `vendor/model` and the catalogue moves -- so this fails immediately with near-matches rather
    than after N failed requests.
    """
    load_env_files()
    key = os.environ.get(OPENROUTER_KEY_ENV, "").strip()
    if not key:
        key_path = ROOT / "api_keys" / "api_key_openrouter.txt"
        if key_path.exists():
            key = key_path.read_text().strip()
    if not key:
        raise SystemExit(
            f"Missing {OPENROUTER_KEY_ENV}.\n"
            "Put it in the repo's gitignored .env (same file as HF_TOKEN), as one line:\n"
            f"  {OPENROUTER_KEY_ENV}=sk-or-...\n"
            "or in api_keys/api_key_openrouter.txt. An exported environment variable wins over .env."
        )

    try:
        catalogue = openrouter_catalogue(key)
    except Exception as exc:  # network/auth problems must be legible, not a traceback
        raise SystemExit(f"Could not reach OpenRouter's model catalogue ({type(exc).__name__}: {exc}). Check the key and connectivity.")

    entry = next((item for item in catalogue if item.get("id") == model), None)
    if entry is None:
        needle = model.split("/")[-1].lower()
        close = [item["id"] for item in catalogue if needle[:12] in item.get("id", "").lower()][:12]
        raise SystemExit(
            f"Model {model!r} is not in OpenRouter's catalogue.\n"
            + (f"Close matches:\n  " + "\n  ".join(close) if close else "No close matches; browse https://openrouter.ai/models")
            + "\nPass the exact id with --model."
        )

    pricing = entry.get("pricing") or {}
    return {
        "key": key,
        "model": model,
        "name": entry.get("name") or model,
        # OpenRouter quotes USD per token as strings.
        "price_in": float(pricing.get("prompt") or 0.0),
        "price_out": float(pricing.get("completion") or 0.0),
        "context_length": entry.get("context_length"),
    }


def call_azure_once(custom_id: str, system: str, user: str, settings: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    """One Azure OpenAI Responses-API call, matching rl_training/build_sft_dataset_azure.py."""
    body: dict[str, Any] = {
        "model": settings["deployment"],
        "instructions": system,
        "input": user,
        "max_output_tokens": args.max_tokens,
        "store": False,
    }
    if args.effort and args.effort != "none":
        body["reasoning"] = {"effort": args.effort}
    return call_with_retries(custom_id, body, settings["responses_url"], settings["key"], "api-key", azure_record, args)


def call_openrouter_once(custom_id: str, system: str, user: str, settings: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    """One OpenRouter chat/completions call."""
    body: dict[str, Any] = {
        "model": settings["model"],
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": args.max_tokens,
    }
    if args.effort and args.effort != "none":
        # OpenRouter's unified reasoning parameter; ignored by non-reasoning models rather than
        # rejected, so it is safe to send either way.
        body["reasoning"] = {"effort": args.effort}
    return call_with_retries(custom_id, body, OPENROUTER_URL, settings["key"], "bearer", openrouter_record, args)


def openrouter_record(custom_id: str, response: dict[str, Any]) -> dict[str, Any]:
    """Normalise an OpenAI-style chat completion onto the shared record shape.

    OpenRouter can also return a 200 whose body is an {"error": ...} envelope (upstream provider
    refused, no credit, model unavailable), so that is checked before reading choices.
    """
    if isinstance(response.get("error"), dict):
        message = str(response["error"].get("message"))[:300]
        return {"custom_id": custom_id, "stop_reason": "error", "input_tokens": 0, "output_tokens": 0, "text": "", "error": f"openrouter: {message}"}

    choices = response.get("choices") or []
    text, finish = "", "unknown"
    if choices:
        text = (choices[0].get("message") or {}).get("content") or ""
        finish = choices[0].get("finish_reason") or "unknown"

    usage = response.get("usage") or {}
    return {
        "custom_id": custom_id,
        # "length" is the chat-completions spelling of "spent the whole budget and never finished";
        # named to match the Azure path so one stop_reason histogram reads the same either way.
        "stop_reason": {"stop": "end_turn", "length": "incomplete:max_tokens"}.get(finish, finish),
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "text": text,
        "provider_model": response.get("model"),
    }


def azure_record(custom_id: str, response: dict[str, Any]) -> dict[str, Any]:
    """Normalise a Responses-API result onto the same record shape the Anthropic path returns."""
    text = response.get("output_text")
    if not isinstance(text, str) or not text:
        chunks: list[str] = []
        for item in response.get("output") or []:
            if item.get("type") != "message":
                continue
            for content in item.get("content") or []:
                if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                    chunks.append(content["text"])
        text = "".join(chunks)

    status = response.get("status")
    if status == "completed":
        stop_reason = "end_turn"
    elif status == "incomplete":
        # Almost always max_output_tokens: the model spent the whole budget reasoning. Named
        # distinctly so it shows up in the stop_reason histogram instead of hiding as a failure.
        stop_reason = f"incomplete:{((response.get('incomplete_details') or {}).get('reason')) or 'unknown'}"
    else:
        stop_reason = str(status)

    usage = response.get("usage") or {}
    return {
        "custom_id": custom_id,
        "stop_reason": stop_reason,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "text": text,
    }


def run_remote(
    jobs: list[tuple[str, str, str]],
    call_one,
    label: str,
    args: argparse.Namespace,
    progress_path: Path,
    contract_sha256: str,
) -> dict[str, dict[str, Any]]:
    """Concurrent, resumable. Every completed request is appended to progress_path as it lands.

    A ~1,800-request run takes hours; anything that has to start over from zero after a dropped
    laptop lid will be started over from zero. On re-run, completed ids are skipped -- errors are
    NOT skipped, so a re-run also retries exactly the failures.
    """
    done: dict[str, dict[str, Any]] = {}
    if progress_path.exists():
        with progress_path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                cached_contract = record.get("generation_contract_sha256")
                if cached_contract != contract_sha256:
                    raise SystemExit(
                        f"Refusing to resume {progress_path}: cached response {record.get('custom_id')!r} "
                        "was generated under a different or unrecorded teacher contract. Use a fresh "
                        "--workdir; never mix teacher models, effort levels, prompt policies, or benchmark "
                        "orders in one cache."
                    )
                if record.get("stop_reason") != "error":
                    done[record["custom_id"]] = record
        print(f"Resuming: {len(done)} completed requests already in {progress_path.name}")

    todo = [job for job in jobs if job[0] not in done]
    if not todo:
        print("Nothing left to request.")
        return done
    print(f"Requesting {len(todo)} of {len(jobs)} from {label} with {args.workers} workers ...")

    lock = threading.Lock()
    handle = progress_path.open("a")
    completed = 0
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(call_one, cid, system, user): cid for cid, system, user in todo}
            for future in as_completed(futures):
                record = future.result()
                record["generation_contract_sha256"] = contract_sha256
                completed += 1
                with lock:
                    done[record["custom_id"]] = record
                    handle.write(json.dumps(record) + "\n")
                    handle.flush()
                if completed % 25 == 0 or completed == len(todo) or record["stop_reason"] == "error":
                    note = f"  ERROR {record.get('error', '')[:120]}" if record["stop_reason"] == "error" else ""
                    print(f"  [{completed}/{len(todo)}] {record['custom_id']} {record['stop_reason']} {record['output_tokens']:,} out tokens{note}", flush=True)
    finally:
        handle.close()

    errors = sum(1 for r in done.values() if r.get("stop_reason") == "error")
    if errors:
        print(f"!! {errors} requests failed. Re-run the same command to retry only those (completed ones are cached).")
    return done


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    temporary.replace(path)


def establish_generation_contract(
    args: argparse.Namespace,
    settings: dict[str, Any],
    benchmark: list[Any],
    dataset_path: Path,
    workdir: Path,
) -> tuple[dict[str, Any], str, Path]:
    """Pin every setting that could change the meaning of a resumed custom_id.

    ``k`` and ``--only_unsolved`` are deliberately absent: the supported top-up reuses sample 0
    and adds sample 1 under the same teacher contract. Operational settings such as workers,
    retries and timeouts are also absent because they do not change a successful response.
    """
    teacher = {
        "openrouter": settings.get("model"),
        "azure": settings.get("deployment"),
        "anthropic": args.model or "claude-sonnet-5",
    }[args.provider]
    question_ids = [problem.question_id for problem in benchmark]
    dataset_manifest_path = dataset_path / "source_manifest.json"
    if not dataset_manifest_path.exists():
        raise SystemExit(
            f"Missing {dataset_manifest_path}. Re-run scripts/fetch_multilcb_dataset.py so the "
            "paid teacher cache is tied to an immutable dataset revision and byte hashes."
        )
    dataset_manifest = json.loads(dataset_manifest_path.read_text())
    contract = {
        "schema_version": 1,
        "provider": args.provider,
        "teacher": teacher,
        "reasoning_effort": args.effort,
        "max_tokens": args.max_tokens,
        "language": args.language,
        "upstream_language": upstream_language(args.language),
        "release_version": args.release_version,
        "dataset_source_manifest_sha256": hashlib.sha256(dataset_manifest_path.read_bytes()).hexdigest(),
        "dataset_revision": dataset_manifest.get("revision"),
        "benchmark_question_ids_sha256": sha256_json(question_ids),
        "benchmark_problem_count": len(question_ids),
        "teacher_addendum_sha256": hashlib.sha256(TEACHER_ADDENDA[args.language].encode()).hexdigest(),
        "split_seed": args.seed,
        "validation_problems": (0 if len(benchmark) == 1 else 1) if args.limit else args.validation_problems,
        "limit": args.limit,
    }
    digest = sha256_json(contract)
    path = workdir / ("teacher_contract_smoke.json" if args.limit else "teacher_contract.json")
    if path.exists():
        existing = json.loads(path.read_text())
        existing_contract = existing.get("contract")
        if existing_contract != contract or existing.get("sha256") != digest:
            changed = sorted(
                key for key in set(existing_contract or {}) | set(contract)
                if (existing_contract or {}).get(key) != contract.get(key)
            )
            raise SystemExit(
                f"Refusing to reuse {workdir}: immutable teacher contract changed "
                f"({', '.join(changed) or 'digest'}). Use a fresh --workdir."
            )
    else:
        write_json_atomic(path, {"contract": contract, "sha256": digest})
    return contract, digest, path


def build_params(model: str, system: str, user: str, effort: str, max_tokens: int) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    # Effort is only meaningful on adaptive-thinking models; Haiku 4.5 rejects output_config.effort.
    if model in ("claude-opus-5", "claude-sonnet-5") and effort != "none":
        params["thinking"] = {"type": "adaptive"}
        params["output_config"] = {"effort": effort}
    return params


def message_text(message: Any) -> str:
    return "".join(block.text for block in message.content if getattr(block, "type", "") == "text")


def response_record(custom_id: str, message: Any) -> dict[str, Any]:
    text = message_text(message)
    record = {
        "custom_id": custom_id,
        "stop_reason": message.stop_reason,
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
        "text": text,
    }
    if message.stop_reason == "refusal":
        details = getattr(message, "stop_details", None)
        record["refusal_category"] = getattr(details, "category", None)
    return record


def run_sync(client: Any, requests: list[tuple[str, dict[str, Any]]], workers: int) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}

    def one(custom_id: str, params: dict[str, Any]) -> dict[str, Any]:
        message = client.messages.create(**params)
        return response_record(custom_id, message)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, cid, p): cid for cid, p in requests}
        done = 0
        for future in as_completed(futures):
            cid = futures[future]
            done += 1
            try:
                out[cid] = future.result()
                print(f"  [{done}/{len(requests)}] {cid}: {out[cid]['output_tokens']:,} out tokens, {out[cid]['stop_reason']}", flush=True)
            except Exception as exc:  # one failure must not lose the rest
                print(f"  [{done}/{len(requests)}] {cid}: FAILED {type(exc).__name__}: {exc}", flush=True)
                out[cid] = {"custom_id": cid, "stop_reason": "error", "input_tokens": 0, "output_tokens": 0, "text": "", "error": str(exc)}
    return out


def run_batch(client: Any, requests: list[tuple[str, dict[str, Any]]], state_path: Path, poll_seconds: int) -> dict[str, dict[str, Any]]:
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    if state_path.exists():
        state = json.loads(state_path.read_text())
        batch_id = state["batch_id"]
        print(f"Resuming batch {batch_id} from {state_path} (delete that file to submit a fresh batch)")
    else:
        batch = client.messages.batches.create(
            requests=[Request(custom_id=cid, params=MessageCreateParamsNonStreaming(**p)) for cid, p in requests]
        )
        batch_id = batch.id
        state_path.write_text(json.dumps({"batch_id": batch_id, "n_requests": len(requests)}, indent=2) + "\n")
        print(f"Submitted batch {batch_id} ({len(requests)} requests); state saved to {state_path}")

    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(
            f"  {time.strftime('%H:%M:%S')} status={batch.processing_status} processing={counts.processing} "
            f"succeeded={counts.succeeded} errored={counts.errored} expired={counts.expired}",
            flush=True,
        )
        if batch.processing_status == "ended":
            break
        time.sleep(poll_seconds)

    out: dict[str, dict[str, Any]] = {}
    for result in client.messages.batches.results(batch_id):
        cid = result.custom_id
        kind = result.result.type
        if kind == "succeeded":
            out[cid] = response_record(cid, result.result.message)
        else:
            detail = getattr(getattr(result.result, "error", None), "type", kind)
            out[cid] = {"custom_id": cid, "stop_reason": kind, "input_tokens": 0, "output_tokens": 0, "text": "", "error": str(detail)}
    return out


def estimate_cost(model: str, n_requests: int, batch: bool, assumed_output_tokens: int, assumed_input_tokens: int = 900) -> float:
    price_in, price_out = PRICES.get(model, (5.0, 25.0))
    scale = 0.5 if batch else 1.0
    return scale * n_requests * (assumed_input_tokens * price_in + assumed_output_tokens * price_out) / 1e6


def actual_cost(model: str, records: dict[str, dict[str, Any]], batch: bool) -> float:
    price_in, price_out = PRICES.get(model, (5.0, 25.0))
    scale = 0.5 if batch else 1.0
    tokens_in = sum(r.get("input_tokens", 0) for r in records.values())
    tokens_out = sum(r.get("output_tokens", 0) for r in records.values())
    return scale * (tokens_in * price_in + tokens_out * price_out) / 1e6


# --- Main ----------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release_version", default="release_v5", help="Staged LCB release to draw training problems from. Must NOT include v6.")
    parser.add_argument("--dataset_path", default=None, help="Defaults to data/reference/multilcb/<release_version>")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE, choices=LANGUAGES, help=f"Target language for the SFT arm (default {DEFAULT_LANGUAGE!r}). Selects upstream's prompt renderer, the cache filename, the `lcb-<language>-teacher` grading handle, and the default --workdir. Each language is a SEPARATE arm with its own teacher spend, cache and pool -- this flag does not add a language to an existing run.")
    parser.add_argument("--workdir", default=None, help="Where split.json, teacher_raw.jsonl and batch_state.json go. Defaults to data/training/lcb_<language>.")
    parser.add_argument("--validation_problems", type=int, default=150, help="Problems held out BY ID for validation.jsonl. 64 was the Magicoder-era default and was never reasoned; 150 (~17%% of v1-v5) is enough that one flipped problem moves the rate <1 point.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=2, help="Teacher samples per problem. At most ONE passing sample per problem ends up in the pool (the validator rejects duplicate user prompts), so k buys pass PROBABILITY, not pool rows. 2 is the sweet spot.")
    parser.add_argument("--provider", default="openrouter", choices=["openrouter", "azure", "anthropic"], help="openrouter (default): one key, any frontier model, priced from OpenRouter's live rate card. azure: Azure OpenAI Responses API, funded by the Azure credit. anthropic: Message Batches API at 50%%, paid per token.")
    parser.add_argument("--deployment", default=None, help="--provider azure only: deployment name; defaults to $AZURE_OPENAI_DEPLOYMENT.")
    parser.add_argument("--model", default=None, help=f"Model id. OpenRouter uses vendor/model ids (default {DEFAULT_OPENROUTER_MODEL!r}); the id is validated against the live catalogue before any spend, and near-matches are printed if it is wrong. For --provider anthropic: claude-sonnet-5 (default), claude-opus-5, claude-haiku-4-5.")
    parser.add_argument("--effort", default="high", choices=["none", "low", "medium", "high", "xhigh", "max"], help="Reasoning effort. Default high to match the completed Go/Rust reference arms; changing it creates a different intervention dataset. Azure reasoning deployments accept low/medium/high; xhigh/max are Claude-only. Use 'none' for a NON-reasoning Azure deployment (gpt-4o-class) -- sending a reasoning block to one is an error.")
    parser.add_argument("--max_tokens", type=int, default=16000, help="Output cap per request (Azure: max_output_tokens, which reasoning tokens also count against). Prompts are at most ~1.9k input tokens, so this is headroom, not a squeeze.")
    parser.add_argument("--limit", type=int, default=0, help="Smoke: first N problems in upstream order. Writes to the debug_ output folder, so verify with --debug --debug_size N too.")
    parser.add_argument("--only_unsolved", default=None, metavar="EVAL_ALL_JSON", help="Topping up an existing run to a higher --k: path to the previous k's graded eval_all_codegeneration_<lang>_<k>_*.json. Extra samples are then requested ONLY for problems no sample has solved, instead of for all of them -- a second attempt at an already-solved problem buys nothing and costs the same as a first. Roughly halves the top-up bill.")
    parser.add_argument("--sync", action="store_true", help="--provider anthropic only: direct requests instead of a batch. Smoke runs only: full price, and not resumable.")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent requests (openrouter/azure, and anthropic --sync). Lower it on sustained 429s.")
    parser.add_argument("--request_timeout", type=float, default=600.0, help="Per-request hard timeout, seconds (openrouter/azure).")
    parser.add_argument("--api_retries", type=int, default=3, help="Retries per request on 408/409/429/5xx and timeouts (openrouter/azure).")
    parser.add_argument("--retry_sleep", type=float, default=5.0, help="Backoff base, multiplied by attempt number (openrouter/azure).")
    parser.add_argument("--poll_seconds", type=int, default=60)
    parser.add_argument("--assumed_output_tokens", type=int, default=1000, help="For the pre-submit cost estimate only. The completed high-effort Go/Rust reference calls averaged roughly 800-900 billed output tokens; 1000 is a conservative planning value. The final log reports actual token cost.")
    parser.add_argument("--max_cost_usd", type=float, default=10.0, help="Refuse to submit above this estimate without --yes (CLAUDE.md cost rule).")
    parser.add_argument("--yes", action="store_true", help="Confirm a run whose estimate exceeds --max_cost_usd.")
    parser.add_argument("--allow_test_window", action="store_true", help="Permit --release_version v6 for a --limit smoke run only (the 128 MB v6 shard is a quick way to exercise the mechanics before the 4.35 GB training window has downloaded). Never for a real run: v6 is the held-out test.")
    args = parser.parse_args()

    if "v6" in args.release_version:
        if not (args.allow_test_window and args.limit):
            raise SystemExit("Refusing: the training window must not include v6, which is the held-out test. (--allow_test_window with --limit is allowed for a mechanics smoke.)")
        print("!! SMOKE ON THE TEST WINDOW (v6). Mechanics only; nothing from this run may be trained on.")

    dataset_path = Path(args.dataset_path) if args.dataset_path else ROOT / "data" / "reference" / "multilcb" / args.release_version
    workdir = Path(args.workdir) if args.workdir else default_workdir(args.language)
    workdir.mkdir(parents=True, exist_ok=True)

    print(f"Language: {args.language} (grading handle {teacher_name(args.language)}, workdir {workdir})")
    print(f"Loading {args.release_version} from {dataset_path} via upstream loader ...")
    benchmark, format_prompt, upstream_args = load_upstream(args.release_version, dataset_path, args.limit, args.language)
    print(f"{len(benchmark)} problems after upstream's functional-test filter" + (f" (limited to first {args.limit})" if args.limit else ""))

    split_path = workdir / ("split_smoke.json" if args.limit else "split.json")
    if args.limit:
        # A one-request smoke cannot hold out one of one problems. Larger smokes hold out one so
        # both builder paths are exercised without pretending the tiny validation is statistical.
        split = make_split(benchmark, 0 if len(benchmark) == 1 else 1, args.seed)
    else:
        split = make_split(benchmark, args.validation_problems, args.seed)
    split.update({"release_version": args.release_version, "n_problems": len(benchmark)})
    write_json_atomic(split_path, split)
    print(f"Split: {len(split['train_question_ids'])} train / {len(split['validation_question_ids'])} validation problems -> {split_path}")

    from lcb_runner.lm_styles.models_store import LMStyle

    # Raising --k re-emits sample 0 (cached, so free) plus a NEW sample for every problem -- including
    # the ones already solved, where a second attempt buys nothing. --only_unsolved reads the graded
    # output of the previous k and restricts the extra samples to problems no sample has solved yet,
    # which is the whole point of topping up. Sample 0 is always emitted so resume stays intact.
    solved: set[str] = set()
    if args.only_unsolved:
        graded_path = Path(args.only_unsolved)
        if not graded_path.exists():
            raise SystemExit(f"--only_unsolved: no graded output at {graded_path}. Grade the previous k first.")
        for record in json.loads(graded_path.read_text()):
            if any(ok and (code or "").strip()
                   for code, ok in zip(record.get("code_list", []), record.get("graded_list", []))):
                solved.add(record["question_id"])
        print(f"--only_unsolved: {len(solved)} problems already solved in {graded_path.name}; extra samples restricted to the remaining {len(benchmark) - len(solved)}")

    # Provider-agnostic job list: (custom_id, system, user). Each provider builds its own payload.
    jobs: list[tuple[str, str, str]] = []
    prompts: dict[str, list[dict[str, str]]] = {}
    for idx, problem in enumerate(benchmark):
        messages = format_prompt(problem, LMStyle.VLLMAsync, upstream_language(args.language))
        system = messages[0]["content"] + "\n\n" + TEACHER_ADDENDA[args.language]
        user = messages[1]["content"]
        prompts[problem.question_id] = messages
        n_samples = 1 if (problem.question_id in solved) else args.k
        for sample in range(n_samples):
            # custom_id must be <=64 chars of [A-Za-z0-9_-]; question_ids are not guaranteed to be.
            jobs.append((f"p{idx:05d}_s{sample}", system, user))

    settings: dict[str, Any] = {}
    if args.provider == "openrouter":
        model = args.model or DEFAULT_OPENROUTER_MODEL
        settings = openrouter_settings(model)
        # Priced from OpenRouter's live rate card, not an assumption: the only guess left is how
        # many output tokens a reasoning model spends, which --assumed_output_tokens states openly.
        est_in = len(jobs) * 900
        est_out = len(jobs) * args.assumed_output_tokens
        est = est_in * settings["price_in"] + est_out * settings["price_out"]
        print(
            f"\n{len(jobs)} requests to {settings['name']} ({settings['model']}, effort={args.effort}).\n"
            f"OpenRouter rate card: ${settings['price_in'] * 1e6:.2f}/M input, ${settings['price_out'] * 1e6:.2f}/M output.\n"
            f"Estimated cost ~${est:.2f} at ~{args.assumed_output_tokens:,} output tokens/request "
            f"(~{est_out / 1e6:.1f}M output tokens total)."
        )
        if not args.limit and not args.yes:
            raise SystemExit(f"Full run: estimated ~${est:.2f}. Confirm with --yes (a --limit smoke needs no confirmation).")
    elif args.provider == "azure":
        settings = azure_settings(args.deployment)
        est_tokens = len(jobs) * args.assumed_output_tokens
        print(
            f"\n{len(jobs)} requests to Azure deployment {settings['deployment']!r} (effort={args.effort}). "
            f"Estimated burn ~{est_tokens / 1e6:.1f}M output tokens at ~{args.assumed_output_tokens:,} each, "
            f"against the project's Azure credit (CLAUDE.md: the credit is for exactly this kind of >$10 run)."
        )
        if not args.limit and not args.yes:
            raise SystemExit("Full run: confirm with --yes (a --limit smoke needs no confirmation).")
    else:
        anthropic_model = args.model or "claude-sonnet-5"
        est = estimate_cost(anthropic_model, len(jobs), not args.sync, args.assumed_output_tokens)
        print(
            f"\n{len(jobs)} requests to {anthropic_model} (effort={args.effort}, {'sync' if args.sync else 'batch @ 50%'}). "
            f"PAID per token -- estimated cost ~${est:.2f} at ~{args.assumed_output_tokens:,} output tokens/request."
        )
        if est > args.max_cost_usd and not args.yes:
            raise SystemExit(
                f"Estimate ${est:.2f} exceeds --max_cost_usd {args.max_cost_usd:.2f}. Get sign-off (CLAUDE.md cost rule), then re-run with --yes."
            )

    contract, contract_sha256, contract_path = establish_generation_contract(
        args, settings, benchmark, dataset_path, workdir
    )
    print(f"Generation contract: {contract_sha256} -> {contract_path}")

    if args.provider in ("openrouter", "azure"):
        stem = "openrouter" if args.provider == "openrouter" else "azure"
        progress_path = workdir / (f"{stem}_progress_smoke.jsonl" if args.limit else f"{stem}_progress.jsonl")
        if args.provider == "openrouter":
            call_one = lambda cid, system, user: call_openrouter_once(cid, system, user, settings, args)  # noqa: E731
            label = f"OpenRouter ({settings['model']})"
        else:
            call_one = lambda cid, system, user: call_azure_once(cid, system, user, settings, args)  # noqa: E731
            label = f"Azure deployment {settings['deployment']!r}"
        records = run_remote(jobs, call_one, label, args, progress_path, contract_sha256)
    else:
        import anthropic

        key_path = ROOT / "api_keys" / "api_key_anthropic.txt"
        api_key = key_path.read_text().strip() if key_path.exists() else None
        client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        requests = [(cid, build_params(anthropic_model, system, user, args.effort, args.max_tokens)) for cid, system, user in jobs]
        if args.sync:
            records = run_sync(client, requests, args.workers)
        else:
            state_path = workdir / ("batch_state_smoke.json" if args.limit else "batch_state.json")
            records = run_batch(client, requests, state_path, args.poll_seconds)

    # --- Write upstream's cache -------------------------------------------------
    from lcb_runner.utils import extract_code

    save_results = []
    n_empty = 0
    n_skipped = 0
    for idx, problem in enumerate(benchmark):
        output_list, code_list = [], []
        for sample in range(args.k):
            rec = records.get(f"p{idx:05d}_s{sample}") or {"text": "", "stop_reason": "missing"}
            text = rec.get("text") or ""
            # Upstream's --continue_existing treats an instance whose outputs are ALL empty as
            # "not generated yet" and would try to call an endpoint for it during --evaluate_only.
            # A non-empty placeholder with no fenced block grades as a plain failure instead.
            #
            # Under --only_unsolved the extra samples of an already-solved problem were deliberately
            # never requested, so they are not failures and must not be counted as empty -- the
            # problem keeps its passing sample at index 0 and the pool builder finds it there.
            # Counting them would report hundreds of "empty" samples on every top-up round.
            if not text.strip():
                if problem.question_id in solved and sample >= 1:
                    text = "[not requested: already solved by an earlier sample]"
                    n_skipped += 1
                else:
                    text = f"[no teacher output: {rec.get('stop_reason')}]"
                    n_empty += 1
            output_list.append(text)
            code_list.append(extract_code(text))
        save_results.append(problem.insert_output(output_list, code_list, prompts[problem.question_id]))
    save_results.sort(key=lambda r: r["question_id"])

    out_path = cache_path(upstream_args, args.k, args.language)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out_path, save_results)

    raw_path = workdir / ("teacher_raw_smoke.jsonl" if args.limit else "teacher_raw.jsonl")
    raw_rows = []
    for idx, problem in enumerate(benchmark):
        for sample in range(args.k):
            cid = f"p{idx:05d}_s{sample}"
            rec = dict(records.get(cid) or {"custom_id": cid, "stop_reason": "missing", "text": ""})
            teacher = {
                "azure": f"azure:{settings.get('deployment')}",
                "openrouter": f"openrouter:{settings.get('model')}",
            }.get(args.provider, args.model or "claude-sonnet-5")
            rec.update({
                "question_id": problem.question_id,
                "difficulty": problem.difficulty.value,
                "platform": problem.platform.value,
                "provider": args.provider,
                "model": teacher,
                "effort": args.effort,
                "max_tokens": args.max_tokens,
                "generation_contract_sha256": contract_sha256,
            })
            raw_rows.append(rec)
    write_jsonl_atomic(raw_path, raw_rows)

    with_code = sum(1 for r in save_results if any(c.strip() for c in r["code_list"]))
    stops: dict[str, int] = {}
    for rec in records.values():
        stops[rec.get("stop_reason", "?")] = stops.get(rec.get("stop_reason", "?"), 0) + 1
    tokens_out = sum(r.get("output_tokens", 0) for r in records.values())
    tokens_in = sum(r.get("input_tokens", 0) for r in records.values())
    print(f"\nWrote {len(save_results)} problems x k={args.k} -> {out_path}")
    print(f"Raw responses -> {raw_path}")
    print(f"Problems with at least one fenced {args.language} block: {with_code}/{len(save_results)}; empty/failed samples: {n_empty}"
          + (f"; deliberately skipped (already solved): {n_skipped}" if n_skipped else ""))
    print(f"stop_reason counts: {stops}")
    print(f"Tokens: {tokens_in:,} in / {tokens_out:,} out")
    if args.provider == "openrouter":
        spend = tokens_in * settings["price_in"] + tokens_out * settings["price_out"]
        print(f"Actual cost: ~${spend:.2f} at OpenRouter's rate card for {settings['model']}")
    elif args.provider == "azure":
        print(
            f"Billed to the Azure credit (deployment {settings.get('deployment')!r}"
            + (f", api-version {settings['api_version']}" if settings.get("api_version") else "")
            + "). Check the spend in the Azure portal -- this script cannot read your rate card."
        )
    else:
        print(f"Actual cost: ~${actual_cost(anthropic_model, records, not args.sync):.2f} ({'batch' if not args.sync else 'sync'} pricing)")
    print(
        "\nNext: verify on the Linux toolchain box --\n"
        f"  python3 scripts/score_multilcb.py --model_key {teacher_name(args.language)} --languages {args.language} --release_version {args.release_version} "
        f"--n {args.k} --evaluate_only --diagnose" + (f" --debug --debug_size {args.limit}" if args.limit else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
