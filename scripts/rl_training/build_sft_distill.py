#!/usr/bin/env python3
"""LEGACY: build a drop-allowed OpenAI SFT sweep over arbitrary prompt files.

This script calls a teacher model through the OpenAI Responses API, extracts a
single fenced Python program, grades it with the same verifier used by the RL
arms, and keeps only verifier-passing solutions.

It is not the formal four-method comparison builder: it can drop failed rows
and therefore cannot guarantee the exact frozen cohort. Formal SFT data must be
generated *after* RL cohort freeze through ``build_sft_dataset_azure.py`` and
must contain every selected ID. This legacy tool requires an explicit
``--legacy_full_pool`` acknowledgement so it cannot be mistaken for that path.

Secret handling is intentionally narrow: the API key is read only from an
environment variable (default: OPENAI_API_KEY). It is never accepted as a CLI
argument, printed, written to manifests, or included in logs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rl_training.data.schema import read_coding_jsonl  # noqa: E402
from scripts.rl_training.prompts import format_coding_prompt  # noqa: E402
from scripts.rl_training.rewards import run_verifier  # noqa: E402

SFT_DISTILL_SCHEMA = "sft_distill_v1"
DATASET_ID = "nemotron_rl_coding_competitive"
OPENAI_WORKER_KEY_ENV = "RL_TRAINING_OPENAI_API_KEY"
DEFAULT_TEACHER_MODEL = "gpt-5.6-sol"
DEFAULT_DATA_ROOT = Path(os.environ.get("DATA_ROOT", ROOT))
DEFAULT_RL_DATA_DIR = Path(
    os.environ.get("RL_DATA_DIR", DEFAULT_DATA_ROOT / "data" / "rl" / "v1")
)
DEFAULT_MODEL_TAG = os.environ.get(
    "MODEL_TAG",
    "qwen-qwen3-5-0-8b--rev-2fc06364715b--data-fulltests-v2",
)
DEFAULT_SOURCE_DIR = Path(
    os.environ.get("SOURCE_DIR", DEFAULT_RL_DATA_DIR / "sources" / DEFAULT_MODEL_TAG)
)
DEFAULT_TRAIN_SPLIT = str(
    DEFAULT_SOURCE_DIR / "nemotron_rl_coding_competitive_train_all.jsonl"
)
DEFAULT_VALIDATION_SPLIT = str(
    DEFAULT_SOURCE_DIR / "nemotron_rl_coding_competitive_dev.jsonl"
)

SYSTEM_INSTRUCTIONS = """You are generating a supervised fine-tuning target.

Return exactly one complete Python 3 program that solves the user's competitive
programming problem. The program must read from standard input and write to
standard output.

Output format is mandatory:
```python
<complete program>
```

Do not include prose, explanations, markdown outside the single code block, or
additional code blocks."""

FENCED_PYTHON_RE = re.compile(r"\A\s*```python\n(?P<code>.*?)\n?```\s*\Z", re.DOTALL)


class TeacherError(RuntimeError):
    """Teacher API call failed after retryable attempts."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_tag(value: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-").lower()
    if not tag:
        raise SystemExit(f"Cannot derive a safe path tag from {value!r}")
    return tag


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with tmp.open("w") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def read_existing_record_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with path.open() as f:
        for line in f:
            if line.strip():
                ids.add(json.loads(line)["record_id"])
    return ids


def retryable_drop(row: dict[str, Any]) -> bool:
    attempts = row.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        return False
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        api_error = str(attempt.get("api_error", ""))
        if (
            api_error.startswith("timeout_error:")
            or api_error.startswith("worker_error:")
            or api_error.startswith("url_error:")
            or api_error.startswith("http_408:")
            or api_error.startswith("http_409:")
            or api_error.startswith("http_429:")
            or api_error.startswith("http_500:")
            or api_error.startswith("http_502:")
            or api_error.startswith("http_503:")
            or api_error.startswith("http_504:")
        ):
            return True
    return False


def normalize_drop_log_for_resume(path: Path, *, retry_dropped: bool) -> set[str]:
    """Return dropped record IDs to skip, and keep active drop logs deduplicated.

    Transient infrastructure failures are retryable, so leaving them in the
    active drop log causes duplicate record IDs on every resume. When all drops
    are retried, the active drop log is cleared so new rows describe the latest
    attempt. Historical raw rows are still preserved by git-ignored output
    backups when callers need to inspect local run history.
    """
    if not path.exists():
        return set()
    skip_ids: set[str] = set()
    retained_by_record_id: dict[str, dict[str, Any]] = {}
    original_rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            original_rows.append(row)
            record_id = row["record_id"]
            if retry_dropped:
                continue
            if retryable_drop(row):
                continue
            skip_ids.add(record_id)
            retained_by_record_id[record_id] = row

    retained_rows = list(retained_by_record_id.values())
    if retained_rows != original_rows:
        backup = path.with_suffix(path.suffix + ".resume_backup")
        if original_rows:
            backup.write_text(
                "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in original_rows)
            )
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in retained_rows)
        )
        removed = len(original_rows) - len(retained_rows)
        print(
            f"[resume] compacted {path}: retained={len(retained_rows)}, "
            f"retryable_or_duplicate_removed={removed}",
            flush=True,
        )
    return skip_ids


def write_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def extract_output_text(response: dict[str, Any]) -> str:
    direct = response.get("output_text")
    if isinstance(direct, str) and direct:
        return direct

    chunks: list[str] = []
    for item in response.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                chunks.append(content["text"])
    return "".join(chunks)


def normalize_fenced_python(text: str) -> tuple[str | None, str | None]:
    match = FENCED_PYTHON_RE.match(text or "")
    if not match:
        return None, "not_single_fenced_python_block"
    code = match.group("code").strip()
    if not code:
        return None, "empty_code_block"
    return f"```python\n{code}\n```", None


def parse_optional_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "y", "on"}:
        return True
    if lowered in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def build_attempt_prompt(
    *,
    base_prompt: str,
    attempt_logs: list[dict[str, Any]],
    previous_response: str | None,
    repair_feedback: bool,
) -> str:
    if not repair_feedback or not attempt_logs:
        return base_prompt

    last = attempt_logs[-1]
    feedback: list[str] = []
    if "format_error" in last:
        feedback.append(
            "Your previous response was rejected because it was not exactly one fenced "
            "Python code block."
        )
    elif "teacher_pass_rate" in last:
        feedback.append(
            "Your previous solution failed the hidden verifier. It passed "
            f"{last['teacher_pass_rate']:.1%} of cases, but only 100% passing "
            "solutions are kept."
        )
        feedback.append(
            "The hidden tests are not shown. Re-analyze the problem, look for edge "
            "cases, and produce a substantially corrected solution rather than a "
            "cosmetic rewrite."
        )
    elif "status" in last:
        feedback.append(
            "Your previous response did not complete successfully. Return exactly one "
            "complete Python 3 program."
        )
    elif "api_error" in last:
        return base_prompt
    else:
        feedback.append("Your previous attempt was rejected. Try a corrected solution.")

    if previous_response:
        feedback.append("Previous rejected response:")
        feedback.append(previous_response)

    return (
        base_prompt
        + "\n\nRepair feedback for this retry:\n"
        + "\n".join(f"- {line}" for line in feedback)
        + "\n\nReturn exactly one complete corrected Python 3 program in a single "
        "```python code block. Do not include explanations."
    )


def run_openai_request_with_timeout(
    *,
    payload: bytes,
    api_key: str,
    timeout: float,
) -> tuple[str, Any]:
    env = os.environ.copy()
    env[OPENAI_WORKER_KEY_ENV] = api_key
    env["PYTHONPATH"] = (
        str(ROOT)
        if not env.get("PYTHONPATH")
        else str(ROOT) + os.pathsep + env["PYTHONPATH"]
    )
    timeout_grace = min(5.0, max(1.0, timeout * 0.1))
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "rl_training.openai_responses_worker",
                "--timeout",
                str(timeout),
                "--api_key_env",
                OPENAI_WORKER_KEY_ENV,
            ],
            input=payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=timeout + timeout_grace,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "timeout_error", f"request exceeded {timeout} seconds"

    stdout = completed.stdout.decode("utf-8", errors="replace")
    stderr = completed.stderr.decode("utf-8", errors="replace")
    if completed.returncode != 0 and not stdout.strip():
        return "worker_error", f"worker exited with code {completed.returncode}: {stderr[:500]}"
    try:
        row = json.loads(stdout)
    except json.JSONDecodeError:
        return "worker_error", f"worker produced invalid JSON: {stderr[:500]}"
    if not isinstance(row, dict) or "status" not in row:
        return "worker_error", "worker produced malformed result"
    return row["status"], row.get("result")


def call_openai_responses(
    *,
    api_key: str,
    model: str,
    prompt: str,
    reasoning_effort: str,
    max_output_tokens: int,
    temperature: float | None,
    chat_template_enable_thinking: bool | None,
    timeout: float,
    api_retries: int,
    sleep_seconds: float,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "instructions": SYSTEM_INSTRUCTIONS,
        "input": prompt,
        "reasoning": {"effort": reasoning_effort},
        "max_output_tokens": max_output_tokens,
        "store": False,
    }
    if temperature is not None:
        body["temperature"] = temperature
    if chat_template_enable_thinking is not None:
        body["chat_template_kwargs"] = {"enable_thinking": chat_template_enable_thinking}

    payload = json.dumps(body).encode("utf-8")

    last_error = "unknown_error"
    for api_attempt in range(1, api_retries + 1):
        status, result = run_openai_request_with_timeout(
            payload=payload,
            api_key=api_key,
            timeout=timeout,
        )
        if status == "ok":
            return result
        if status == "http_error":
            last_error = f"http_{result['code']}: {result['body']}"
            retryable = result["code"] in {408, 409, 429, 500, 502, 503, 504}
            if not retryable or api_attempt == api_retries:
                raise TeacherError(last_error)
        elif status == "timeout_error":
            last_error = f"timeout_error: {result}"
            if api_attempt == api_retries:
                raise TeacherError(last_error)
        else:
            last_error = f"{status}: {result}"
            if api_attempt == api_retries:
                raise TeacherError(last_error)
        if sleep_seconds:
            time.sleep(sleep_seconds * api_attempt)
    raise TeacherError(last_error)


def build_distill_record(
    *,
    source_row: dict[str, Any],
    teacher_model: str,
    assistant_content: str,
    pass_rate: float,
) -> dict[str, Any]:
    return {
        "schema": SFT_DISTILL_SCHEMA,
        "record_id": source_row["record_id"],
        "dataset_id": source_row.get("dataset_id", DATASET_ID),
        "teacher_model": teacher_model,
        "teacher_pass_rate": pass_rate,
        "messages": [
            {"role": "user", "content": format_coding_prompt(source_row["prompt"])},
            {"role": "assistant", "content": assistant_content},
        ],
    }


def process_split(
    *,
    split_name: str,
    source_path: Path,
    out_path: Path,
    drop_log_path: Path,
    rows: list[dict[str, Any]],
    limit: int | None,
    resume: bool,
    api_key: str,
    teacher_model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    temperature: float | None,
    chat_template_enable_thinking: bool | None,
    max_attempts: int,
    api_retries: int,
    request_timeout: float,
    verifier_timeout: float,
    sleep_seconds: float,
    retry_dropped: bool,
    repair_feedback: bool,
) -> dict[str, Any]:
    if limit is not None:
        rows = rows[:limit]

    if not resume:
        out_path.unlink(missing_ok=True)
        drop_log_path.unlink(missing_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.touch(exist_ok=True)
    existing = read_existing_record_ids(out_path) if resume else set()
    dropped_existing = (
        set() if not resume else normalize_drop_log_for_resume(drop_log_path, retry_dropped=retry_dropped)
    )
    attempted = 0
    kept = 0
    skipped_existing = 0
    skipped_dropped = 0
    failed = 0
    start = time.time()

    for idx, row in enumerate(rows, start=1):
        record_id = row["record_id"]
        if record_id in existing:
            skipped_existing += 1
            continue
        if record_id in dropped_existing:
            skipped_dropped += 1
            continue

        formatted_prompt = format_coding_prompt(row["prompt"])
        attempt_logs: list[dict[str, Any]] = []
        kept_record: dict[str, Any] | None = None
        previous_response: str | None = None
        attempted += 1

        for attempt in range(1, max_attempts + 1):
            log_entry: dict[str, Any] = {"attempt": attempt}
            try:
                attempt_prompt = build_attempt_prompt(
                    base_prompt=formatted_prompt,
                    attempt_logs=attempt_logs,
                    previous_response=previous_response,
                    repair_feedback=repair_feedback,
                )
                response = call_openai_responses(
                    api_key=api_key,
                    model=teacher_model,
                    prompt=attempt_prompt,
                    reasoning_effort=reasoning_effort,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    chat_template_enable_thinking=chat_template_enable_thinking,
                    timeout=request_timeout,
                    api_retries=api_retries,
                    sleep_seconds=sleep_seconds,
                )
                log_entry["response_id"] = response.get("id")
                if response.get("status") != "completed":
                    log_entry["status"] = response.get("status")
                    log_entry["error"] = response.get("error") or response.get("incomplete_details")
                    attempt_logs.append(log_entry)
                    continue

                assistant_text = extract_output_text(response)
                assistant_content, format_error = normalize_fenced_python(assistant_text)
                previous_response = assistant_text
                if format_error:
                    log_entry["format_error"] = format_error
                    attempt_logs.append(log_entry)
                    continue

                pass_rate = float(run_verifier(assistant_content, row["verifier"], timeout=verifier_timeout))
                log_entry["teacher_pass_rate"] = pass_rate
                attempt_logs.append(log_entry)
                if pass_rate == 1.0:
                    kept_record = build_distill_record(
                        source_row=row,
                        teacher_model=teacher_model,
                        assistant_content=assistant_content,
                        pass_rate=pass_rate,
                    )
                    break
                previous_response = assistant_content
            except TeacherError as exc:
                log_entry["api_error"] = str(exc)
                attempt_logs.append(log_entry)

        if kept_record is not None:
            write_jsonl_row(out_path, kept_record)
            existing.add(record_id)
            kept += 1
        else:
            failed += 1
            write_jsonl_row(
                drop_log_path,
                {
                    "schema": "sft_distill_drop_v1",
                    "split": split_name,
                    "record_id": record_id,
                    "source_path": str(source_path),
                    "attempts": attempt_logs,
                },
            )

        if idx % 10 == 0 or kept_record is not None:
            print(
                f"[{split_name}] {idx}/{len(rows)} considered, "
                f"attempted={attempted}, kept={kept}, failed={failed}, "
                f"skipped_kept={skipped_existing}, skipped_dropped={skipped_dropped}",
                flush=True,
            )
        if sleep_seconds:
            time.sleep(sleep_seconds)

    output_rows = len(read_existing_record_ids(out_path))
    return {
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "output_path": str(out_path),
        "output_sha256": sha256_file(out_path),
        "output_rows": output_rows,
        "drop_log_path": str(drop_log_path),
        "input_rows_considered": len(rows),
        "attempted": attempted,
        "kept": kept,
        "failed": failed,
        "skipped_existing": skipped_existing,
        "skipped_dropped": skipped_dropped,
        "elapsed_seconds": round(time.time() - start, 3),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy_full_pool",
        action="store_true",
        help="Acknowledge this drop-allowed OpenAI generator is not formal exact-N SFT",
    )
    parser.add_argument("--teacher_model", default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--reasoning_effort", default="high", choices=["minimal", "low", "medium", "high", "xhigh"])
    parser.add_argument("--max_output_tokens", type=int, default=8192)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional sampling temperature. Omitted by default because some reasoning models reject it.",
    )
    parser.add_argument("--no_temperature", action="store_true", help="Force omission of temperature from the request")
    parser.add_argument(
        "--chat_template_enable_thinking",
        type=parse_optional_bool,
        default=None,
        metavar="{true,false}",
        help=(
            "Optional vLLM/Qwen chat-template switch sent as "
            "chat_template_kwargs.enable_thinking. Omitted by default because "
            "hosted Responses APIs may reject vLLM-only fields."
        ),
    )
    parser.add_argument("--max_attempts", type=int, default=3, help="Teacher generations per prompt before dropping it")
    parser.add_argument("--api_retries", type=int, default=3, help="Transport/API retries per teacher generation")
    parser.add_argument("--request_timeout", type=float, default=300.0)
    parser.add_argument("--verifier_timeout", type=float, default=10.0)
    parser.add_argument("--sleep_seconds", type=float, default=0.0, help="Optional pacing between API calls")
    parser.add_argument("--api_key_env", default="OPENAI_API_KEY", help="Environment variable containing the API key")
    parser.add_argument("--train_split", default=DEFAULT_TRAIN_SPLIT)
    parser.add_argument("--validation_split", default=DEFAULT_VALIDATION_SPLIT)
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Defaults to data/rl/v1/legacy_sft_sweeps/<model-tag>/<teacher-tag>",
    )
    parser.add_argument("--split", choices=["train", "validation", "both"], default="both")
    parser.add_argument("--limit_train", type=int, default=None, help="Debug/smoke cap for train rows")
    parser.add_argument("--limit_validation", type=int, default=None, help="Debug/smoke cap for validation rows")
    parser.add_argument("--resume", action="store_true", help="Skip record_ids already present in output JSONL")
    parser.add_argument(
        "--retry_dropped",
        action="store_true",
        help="With --resume, retry record_ids already present in dropped_*.jsonl",
    )
    parser.add_argument(
        "--no_repair_feedback",
        action="store_true",
        help="Disable non-leaking retry feedback after format/verifier failures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.legacy_full_pool:
        raise SystemExit(
            "This is a legacy drop-allowed generator. For the formal comparison, freeze the "
            "RL cohort and use rl_training/build_sft_dataset_azure.py. Pass --legacy_full_pool "
            "only for an explicitly non-formal legacy sweep."
        )
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(
            f"{args.api_key_env} is not set. Export it in your shell; do not put API keys in repo files."
        )

    if args.max_attempts < 1:
        raise SystemExit("--max_attempts must be >= 1")
    if args.api_retries < 1:
        raise SystemExit("--api_retries must be >= 1")
    if args.max_output_tokens < 1:
        raise SystemExit("--max_output_tokens must be >= 1")
    if args.request_timeout <= 0:
        raise SystemExit("--request_timeout must be > 0")
    if args.verifier_timeout <= 0:
        raise SystemExit("--verifier_timeout must be > 0")
    if args.sleep_seconds < 0:
        raise SystemExit("--sleep_seconds must be >= 0")
    if args.reasoning_effort in {"high", "xhigh"} and args.request_timeout < 300:
        print(
            "[distill] warning: high-reasoning teacher calls may exceed "
            f"{args.request_timeout} seconds; use --request_timeout 300 or higher "
            "for full runs.",
            file=sys.stderr,
            flush=True,
        )

    train_split = Path(args.train_split).expanduser().resolve()
    validation_split = Path(args.validation_split).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (
            DEFAULT_RL_DATA_DIR
            / "legacy_sft_sweeps"
            / DEFAULT_MODEL_TAG
            / safe_tag(args.teacher_model)
        ).resolve()
    )
    selected_sources: dict[str, Path] = {}
    if args.split in {"train", "both"}:
        selected_sources["train"] = train_split
    if args.split in {"validation", "both"}:
        selected_sources["validation"] = validation_split
    for split_name, source_path in selected_sources.items():
        if not source_path.is_file():
            raise SystemExit(f"{split_name} source does not exist: {source_path}")

    generation_parameters = {
        "reasoning_effort": args.reasoning_effort,
        "max_output_tokens": args.max_output_tokens,
        "temperature": None if args.no_temperature else args.temperature,
        "chat_template_enable_thinking": args.chat_template_enable_thinking,
        "max_attempts": args.max_attempts,
        "api_retries": args.api_retries,
        "request_timeout": args.request_timeout,
        "verifier_timeout": args.verifier_timeout,
        "repair_feedback": not args.no_repair_feedback,
    }
    run_signature = {
        "teacher_model": args.teacher_model,
        "generation_parameters": generation_parameters,
        "split": args.split,
        "limits": {
            "train": args.limit_train,
            "validation": args.limit_validation,
        },
        "sources": {
            name: {"sha256": sha256_file(path), "path": str(path)}
            for name, path in selected_sources.items()
        },
    }
    manifest_path = output_dir / "manifest.json"
    if args.resume:
        existing_payloads = [
            output_dir / "train.jsonl",
            output_dir / "validation.jsonl",
            output_dir / "dropped_train.jsonl",
            output_dir / "dropped_validation.jsonl",
        ]
        if any(path.exists() for path in existing_payloads) and not manifest_path.is_file():
            raise SystemExit(
                f"Cannot safely resume SFT data without its provenance manifest: {manifest_path}"
            )
        if manifest_path.is_file():
            existing_manifest = json.loads(manifest_path.read_text())
            if existing_manifest.get("run_signature") != run_signature:
                raise SystemExit(
                    "Cannot resume SFT data with a different teacher, generation config, "
                    "limit, split, or source hash; choose a new output directory"
                )
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"SFT output directory is not empty and will not be overwritten: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema": "sft_distill_manifest_v1",
        "status": "running",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": DATASET_ID,
        "teacher_model": args.teacher_model,
        "teacher_api": "openai_responses",
        "generation_parameters": generation_parameters,
        "run_signature": run_signature,
        "secret_handling": {
            "api_key_env": args.api_key_env,
            "api_key_written_to_outputs": False,
        },
        "splits": {},
    }
    atomic_write_json(manifest_path, manifest)

    if args.split in {"train", "both"}:
        train_rows = read_coding_jsonl(train_split)
        manifest["splits"]["train"] = process_split(
            split_name="train",
            source_path=train_split,
            out_path=output_dir / "train.jsonl",
            drop_log_path=output_dir / "dropped_train.jsonl",
            rows=train_rows,
            limit=args.limit_train,
            resume=args.resume,
            api_key=api_key,
            teacher_model=args.teacher_model,
            reasoning_effort=args.reasoning_effort,
            max_output_tokens=args.max_output_tokens,
            temperature=None if args.no_temperature else args.temperature,
            chat_template_enable_thinking=args.chat_template_enable_thinking,
            max_attempts=args.max_attempts,
            api_retries=args.api_retries,
            request_timeout=args.request_timeout,
            verifier_timeout=args.verifier_timeout,
            sleep_seconds=args.sleep_seconds,
            retry_dropped=args.retry_dropped,
            repair_feedback=not args.no_repair_feedback,
        )

    if args.split in {"validation", "both"}:
        validation_rows = read_coding_jsonl(validation_split)
        manifest["splits"]["validation"] = process_split(
            split_name="validation",
            source_path=validation_split,
            out_path=output_dir / "validation.jsonl",
            drop_log_path=output_dir / "dropped_validation.jsonl",
            rows=validation_rows,
            limit=args.limit_validation,
            resume=args.resume,
            api_key=api_key,
            teacher_model=args.teacher_model,
            reasoning_effort=args.reasoning_effort,
            max_output_tokens=args.max_output_tokens,
            temperature=None if args.no_temperature else args.temperature,
            chat_template_enable_thinking=args.chat_template_enable_thinking,
            max_attempts=args.max_attempts,
            api_retries=args.api_retries,
            request_timeout=args.request_timeout,
            verifier_timeout=args.verifier_timeout,
            sleep_seconds=args.sleep_seconds,
            retry_dropped=args.retry_dropped,
            repair_feedback=not args.no_repair_feedback,
        )

    manifest["status"] = "complete"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(manifest_path, manifest)
    print(f"[distill] wrote manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
