#!/usr/bin/env python3
"""Build exact-N verifier-filtered SFT targets through a hosted teacher.

This is the formal post-freeze SFT data builder for the RL drift study. It
reads a frozen ``shared_training_cohort_v1`` manifest, calls one hosted teacher
model for each selected prompt, verifies each Python solution with the prompt's
full local verifier, and marks the artifact complete only when every frozen ID
has a verifier-perfect target in the original cohort order.

Two providers, selected with ``--provider``:

    openrouter (default)  One key, any frontier model, OpenAI-compatible
                          chat/completions. The model id is checked against the
                          live catalogue before a single request is sent, so a
                          mistyped slug fails immediately with near-matches
                          instead of 1000 times. Key: OPENROUTER_API_KEY.
    azure                 Azure OpenAI Responses API, funded by the Azure
                          credit. Key: AZURE_OPENAI_API_KEY.

Both providers post through rl_training/openai_responses_worker.py -- a generic
"POST this JSON, return that JSON" subprocess with a hard timeout -- differing
only in auth header, request body, and response shape. OpenRouter chat
completions are normalized onto the Responses shape at the provider seam, so
everything downstream of the call (retry, repair, verification, provenance,
completeness) is one code path shared by both.

Secrets are read only from environment variables. API keys are never accepted as
CLI arguments, printed, serialized, hashed, or written to output artifacts. The
repo's gitignored .env is loaded into the environment first, so a key may live
there rather than in an export.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rl_training.build_sft_distill import (  # noqa: E402
    OPENAI_WORKER_KEY_ENV,
    TeacherError,
    atomic_write_json,
    extract_output_text,
    normalize_fenced_python,
    parse_optional_bool,
)
from scripts.rl_training.data.schema import read_coding_jsonl  # noqa: E402
from scripts.rl_training.data.quality import (  # noqa: E402
    QUALITY_POLICY,
    validate_formal_coding_row,
)
from scripts.rl_training.prompts import format_coding_prompt  # noqa: E402
from scripts.rl_training.rewards import run_verifier, verifier_sandbox_available  # noqa: E402

# Unchanged across the provider split: the template name and MANIFEST_SCHEMA are provenance
# labels that older artifacts already carry, and train_sft.py whitelists the schema string. The
# provider actually used is recorded in teacher.provider, not in these names.
PROMPT_TEMPLATE_NAME = "azure_sft_python_v1"
REPAIR_FEEDBACK_VERSION = "best_candidate_aggregate_reference_v3"
SFT_ROW_SCHEMA = "sft_distill_v1"
MANIFEST_SCHEMA = "azure_sft_dataset_manifest_v1"

DEFAULT_PROVIDER = "openrouter"

AZURE_API_KEY_ENV = "AZURE_OPENAI_API_KEY"
DEFAULT_API_KEY_ENV = AZURE_API_KEY_ENV  # retained: older runbooks pass --api-key-env explicitly
DEFAULT_ENDPOINT_ENV = "AZURE_OPENAI_ENDPOINT"
DEFAULT_API_VERSION_ENV = "AZURE_OPENAI_API_VERSION"
DEFAULT_DEPLOYMENT_ENV = "AZURE_OPENAI_DEPLOYMENT"
DEFAULT_MODEL_VERSION_ENV = "AZURE_OPENAI_MODEL_VERSION"

# OpenRouter: one key, every frontier model, OpenAI-compatible chat/completions. Same worker,
# bearer auth. Matches scripts/build_lcb_go_teacher.py so the two teacher paths stay legible
# side by side.
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_MODEL_ENV = "OPENROUTER_MODEL"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
# The teacher bar is set by the largest student, not the smallest -- see the reasoning in
# docs/lcb-sft-runbook.md. Same default as the LCB arms so one teacher choice covers all of them.
DEFAULT_OPENROUTER_MODEL = "openai/gpt-5.6-sol"

TEACHER_SYSTEM_PROMPT = """You are generating a supervised fine-tuning target for a coding model.
Solve the competitive-programming problem in the user message.

Return exactly one complete Python 3 program that reads from standard input and
writes to standard output. The program must solve all valid cases within the
stated constraints.

Output exactly one fenced code block in this form:
```python
<complete program>
```

Do not include explanations, analysis, comments outside the program,
alternative solutions, or additional code blocks."""

VERIFIER_REPAIR_MESSAGE = """Your previous program did not match every hidden reference output. Re-analyze the problem and return a corrected, complete solution. Systematically audit boundary values explicitly allowed by the public constraints (including zero or empty cases), degenerate structures, parsing, output formatting, and deterministic tie-breaking. Hidden test inputs and expected outputs are not available. Follow the exact one-code-block output format."""
CODE_INTERPRETER_RETRY_INSTRUCTIONS = """

A Python Code Interpreter sandbox is available for this repair attempt. Use it
privately to run and debug the candidate program with public examples and tests
you construct yourself. Do not attempt to access hidden verifier inputs or
expected outputs, and do not include tool logs in the final response. The final
response must still contain exactly one fenced Python code block.
"""
_JSONL_APPEND_LOCK = threading.Lock()
_TOKENIZER_LOCK = threading.Lock()
_CODE_INTERPRETER_STATE = threading.local()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    with _JSONL_APPEND_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temp_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def load_env_files() -> None:
    """Make the repo's gitignored .env visible, as the Go teacher path already does.

    Resolution order is process environment first (so an existing `export OPENROUTER_API_KEY`
    session keeps working unchanged), then .env, then -- for the key only -- api_keys/*.txt.
    Both files are gitignored, so a personal key never rides along in a commit. Everything is
    loaded INTO the environment, which keeps this module's "secrets come only from environment
    variables" contract literally true.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env")


def load_key_file_into_env(key_env: str, filename: str) -> None:
    """Last-resort key source: api_keys/<filename>, exported under key_env if still unset."""
    if os.environ.get(key_env):
        return
    key_path = ROOT / "api_keys" / filename
    if key_path.is_file():
        key = key_path.read_text().strip()
        if key:
            os.environ[key_env] = key


def openrouter_catalogue(key: str) -> list[dict[str, Any]]:
    """The live model list. Used to validate --teacher-model and to record real pricing."""
    import urllib.request

    request = urllib.request.Request(OPENROUTER_MODELS_URL, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8")).get("data") or []


def resolve_openrouter_model(model: str, key: str) -> dict[str, Any]:
    """Validate the model id against the live catalogue BEFORE any spend.

    A mistyped id is the expensive mistake here -- OpenRouter's ids are `vendor/model` and the
    catalogue moves -- so this fails immediately with near-matches rather than after 1000 failed
    requests. The returned entry is also the model's provenance: it pins the exact catalogue id
    and the rate card in force when the run started.
    """
    try:
        catalogue = openrouter_catalogue(key)
    except Exception as exc:  # network/auth problems must be legible, not a traceback
        raise SystemExit(
            f"Could not reach OpenRouter's model catalogue ({type(exc).__name__}: {exc}). "
            "Check the key and connectivity."
        )

    entry = next((item for item in catalogue if item.get("id") == model), None)
    if entry is None:
        needle = model.split("/")[-1].lower()[:12]
        close = [item["id"] for item in catalogue if needle and needle in item.get("id", "").lower()][:12]
        raise SystemExit(
            f"Model {model!r} is not in OpenRouter's catalogue.\n"
            + ("Close matches:\n  " + "\n  ".join(close) if close else "No close matches; browse https://openrouter.ai/models")
            + "\nPass the exact id with --teacher-model."
        )

    pricing = entry.get("pricing") or {}
    return {
        "id": entry.get("id"),
        "name": entry.get("name"),
        "created": entry.get("created"),
        "context_length": entry.get("context_length"),
        # OpenRouter quotes USD per token as strings; kept verbatim as a rate-card snapshot.
        "pricing_prompt_usd_per_token": pricing.get("prompt"),
        "pricing_completion_usd_per_token": pricing.get("completion"),
    }


def teacher_model_id(args: argparse.Namespace) -> str:
    """The provider-native model handle: an OpenRouter slug or an Azure deployment name."""
    return args.teacher_model if args.provider == "openrouter" else args.teacher_deployment


def teacher_model_tag(args: argparse.Namespace) -> str:
    """The `provider:model` string written into every accepted row's teacher_model."""
    return f"{args.provider}:{teacher_model_id(args)}"


def teacher_provider_name(args: argparse.Namespace) -> str:
    return "openrouter" if args.provider == "openrouter" else "azure_openai"


def normalize_azure_responses_url(endpoint: str) -> str:
    endpoint = endpoint.strip().rstrip("/")
    if not endpoint:
        raise ValueError("Azure OpenAI endpoint is empty")
    if not endpoint.startswith("https://"):
        raise ValueError("Azure OpenAI endpoint must be an https:// URL")
    if endpoint.endswith("/responses"):
        return endpoint
    return f"{endpoint}/responses"


def safe_endpoint_identifier(endpoint: str) -> str:
    parsed = urlsplit(endpoint.strip())
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("Azure OpenAI endpoint must be an https:// URL")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def resolve_manifest_ref(manifest_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def read_selected_ids(path: Path) -> list[str]:
    selected = load_json(path)
    if not isinstance(selected, list) or not all(isinstance(item, str) and item for item in selected):
        raise SystemExit(f"Invalid selected_ids list: {path}")
    if len(selected) != len(set(selected)):
        raise SystemExit(f"Duplicate selected IDs in {path}")
    return selected


def load_frozen_cohort(
    cohort_manifest_path: Path,
    *,
    require_formal_quality: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path, list[str]]:
    manifest_path = cohort_manifest_path.expanduser().resolve()
    manifest = load_json(manifest_path)
    if manifest.get("schema") != "shared_training_cohort_v1" or manifest.get("status") != "frozen":
        raise SystemExit(f"{manifest_path}: expected frozen shared_training_cohort_v1")

    outputs = manifest.get("outputs", {})
    train_ref = outputs.get("train", {})
    ids_ref = outputs.get("selected_ids", {})
    if not train_ref.get("path") or not train_ref.get("sha256"):
        raise SystemExit(f"{manifest_path}: missing hash-pinned outputs.train")
    if not ids_ref.get("path") or not ids_ref.get("sha256"):
        raise SystemExit(f"{manifest_path}: missing hash-pinned outputs.selected_ids")

    train_path = resolve_manifest_ref(manifest_path, train_ref["path"])
    selected_path = resolve_manifest_ref(manifest_path, ids_ref["path"])
    if not train_path.is_file():
        raise SystemExit(f"Frozen cohort train file does not exist: {train_path}")
    if not selected_path.is_file():
        raise SystemExit(f"Frozen selected_ids file does not exist: {selected_path}")
    if sha256_file(train_path) != train_ref["sha256"]:
        raise SystemExit(f"Frozen cohort train hash mismatch: {train_path}")
    if sha256_file(selected_path) != ids_ref["sha256"]:
        raise SystemExit(f"Frozen selected_ids hash mismatch: {selected_path}")

    rows = read_coding_jsonl(train_path)
    selected_ids = read_selected_ids(selected_path)
    observed_ids = [row["record_id"] for row in rows]
    if observed_ids != selected_ids:
        raise SystemExit("Frozen cohort train rows must exactly match selected_ids order")
    selected_size = manifest.get("selection", {}).get("selected_size")
    if selected_size != len(selected_ids):
        raise SystemExit("Cohort selected_size does not match selected_ids length")
    if require_formal_quality:
        policy = manifest.get("quality_policy", {})
        if policy.get("name") != QUALITY_POLICY:
            raise SystemExit(
                f"{manifest_path}: formal Azure SFT requires quality policy {QUALITY_POLICY!r}"
            )
        min_tests = int(policy.get("min_unique_tests", 0))
        if min_tests < 1:
            raise SystemExit(f"{manifest_path}: invalid formal min_unique_tests")
        for line_no, row in enumerate(rows, 1):
            reasons = validate_formal_coding_row(row, min_unique_tests=min_tests)
            if reasons:
                raise SystemExit(
                    f"{train_path}:{line_no}: formal quality violations: {', '.join(reasons)}"
                )
    return manifest, rows, train_path, selected_ids


def cohort_archive(args: argparse.Namespace, cohort_manifest: dict[str, Any]) -> dict[str, Any] | None:
    archived = cohort_manifest.get("archive")
    if archived:
        return archived
    if not args.cohort_hf_repo_id:
        return None
    return {
        "provider": "huggingface",
        "repo_id": args.cohort_hf_repo_id,
        "revision": args.cohort_hf_revision,
        "path": args.cohort_hf_path,
    }


def git_provenance() -> dict[str, Any]:
    provenance_files = [
        Path("rl_training/build_sft_dataset_azure.py"),
        Path("rl_training/openai_responses_worker.py"),
        Path("rl_training/prompts.py"),
        Path("rl_training/rewards.py"),
        Path("rl_training/data/schema.py"),
    ]
    file_hashes = {
        str(path): sha256_file(ROOT / path)
        for path in provenance_files
        if (ROOT / path).is_file()
    }
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return {
            "git_commit": commit,
            "git_dirty": dirty,
            "file_sha256": file_hashes,
        }
    except (OSError, subprocess.CalledProcessError):
        return {
            "git_commit": None,
            "git_dirty": None,
            "file_sha256": file_hashes,
        }


def prior_attempt_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not path.exists():
        return counts
    for row in output_rows_with_tail_recovery(path):
        record_id = row.get("record_id")
        if isinstance(record_id, str):
            counts[record_id] = counts.get(record_id, 0) + 1
    return counts


def prior_repair_context(
    attempts_path: Path,
    failed_candidates_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    attempt_logs: dict[tuple[str, int], dict[str, Any]] = {}
    fallback_logs: dict[str, dict[str, Any]] = {}
    if attempts_path.exists():
        for row in output_rows_with_tail_recovery(attempts_path):
            record_id = row.get("record_id")
            if not isinstance(record_id, str):
                continue
            if any(
                key in row
                for key in (
                    "format_error",
                    "screening_pass_rate",
                    "teacher_pass_rate",
                    "target_length_error",
                )
            ):
                attempt = row.get("attempt")
                if isinstance(attempt, int):
                    attempt_logs[(record_id, attempt)] = row
                current = fallback_logs.get(record_id)
                if current is None or repair_quality_rank(row) >= repair_quality_rank(current):
                    fallback_logs[record_id] = row
    repair_logs: dict[str, dict[str, Any]] = {}
    programs: dict[str, str] = {}
    selected_ranks: dict[str, tuple[int, float, int]] = {}
    if failed_candidates_path.exists():
        for row in output_rows_with_tail_recovery(failed_candidates_path):
            record_id = row.get("record_id")
            candidate = row.get("candidate")
            if not isinstance(record_id, str) or not isinstance(candidate, str) or not candidate:
                continue
            global_attempt = row.get("global_attempt")
            log = (
                attempt_logs.get((record_id, global_attempt))
                if isinstance(global_attempt, int)
                else None
            ) or fallback_logs.get(record_id)
            if log is None:
                continue
            rank = repair_quality_rank(log)
            if rank >= selected_ranks.get(record_id, (-1, -1.0, -1)):
                selected_ranks[record_id] = rank
                repair_logs[record_id] = log
                programs[record_id] = candidate
    for record_id, log in fallback_logs.items():
        repair_logs.setdefault(record_id, log)
    return repair_logs, programs


def repair_quality_rank(log: dict[str, Any]) -> tuple[int, float, int]:
    """Rank repair contexts without exposing any hidden test contents."""
    attempt = log.get("attempt")
    attempt_number = attempt if isinstance(attempt, int) else -1
    if log.get("target_length_error"):
        return (3, 1.0, attempt_number)
    teacher_rate = log.get("teacher_pass_rate")
    if isinstance(teacher_rate, int | float):
        return (2, float(teacher_rate), attempt_number)
    screening_rate = log.get("screening_pass_rate")
    if isinstance(screening_rate, int | float):
        return (1, float(screening_rate), attempt_number)
    if log.get("format_error"):
        return (0, 0.0, attempt_number)
    return (-1, -1.0, attempt_number)


def cohort_id(cohort_manifest: dict[str, Any], selected_ids: list[str]) -> str:
    prep = (
        cohort_manifest.get("dataset", {})
        .get("preparation_manifest", {})
        .get("content", {})
    )
    dataset_id = prep.get("dataset_id") or cohort_manifest.get("dataset_id") or "unknown_dataset"
    seed = cohort_manifest.get("selection", {}).get("seed")
    selected_hash = cohort_manifest.get("outputs", {}).get("selected_ids", {}).get("sha256")
    if not selected_hash:
        selected_hash = hashlib.sha256(json.dumps(selected_ids, ensure_ascii=True).encode("utf-8")).hexdigest()
    return f"{dataset_id}_n{len(selected_ids)}_seed{seed}_ids-{selected_hash[:12]}"


def output_rows_with_tail_recovery(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    rows: list[dict[str, Any]] = []
    for index, raw_line in enumerate(lines):
        if not raw_line.strip():
            continue
        try:
            rows.append(json.loads(raw_line))
        except json.JSONDecodeError as exc:
            is_final_truncated = index == len(lines) - 1 and not raw_line.endswith(b"\n")
            if is_final_truncated:
                path.write_bytes(b"".join(lines[:index]))
                return rows
            raise SystemExit(f"{path}:{index + 1}: invalid JSONL row: {exc}") from exc
    return rows


def validate_sft_row_against_source(
    *,
    row: dict[str, Any],
    source_row: dict[str, Any],
    expected_record_id: str,
    verifier_timeout: float,
    rerun_verifier: bool,
    line_no: int,
    path: Path,
    require_sandbox: bool,
) -> None:
    prefix = f"{path}:{line_no}"
    if row.get("schema") != SFT_ROW_SCHEMA:
        raise SystemExit(f"{prefix}: schema must be {SFT_ROW_SCHEMA}")
    if row.get("record_id") != expected_record_id:
        raise SystemExit(f"{prefix}: record_id does not match frozen cohort order")
    if row.get("dataset_id") != source_row.get("dataset_id"):
        raise SystemExit(f"{prefix}: dataset_id does not match source row")
    if row.get("teacher_pass_rate") != 1.0:
        raise SystemExit(f"{prefix}: teacher_pass_rate must equal 1.0")
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise SystemExit(f"{prefix}: messages must contain exactly user and assistant")
    if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
        raise SystemExit(f"{prefix}: messages roles must be user then assistant")
    if messages[0].get("content") != format_coding_prompt(source_row["prompt"]):
        raise SystemExit(f"{prefix}: user prompt is not byte-identical to format_coding_prompt")
    assistant = messages[1].get("content")
    if not isinstance(assistant, str) or normalize_fenced_python(assistant)[1]:
        raise SystemExit(f"{prefix}: assistant content must be exactly one fenced python block")
    if rerun_verifier:
        pass_rate = float(
            run_verifier(
                assistant,
                source_row["verifier"],
                timeout=verifier_timeout,
                max_tests=None,
                require_sandbox=require_sandbox,
            )
        )
        if pass_rate != 1.0:
            raise SystemExit(f"{prefix}: verifier pass rate is {pass_rate}, expected 1.0")


def validate_existing_prefix(
    *,
    rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    selected_ids: list[str],
    train_path: Path,
    verifier_timeout: float,
    rerun_verifier: bool,
    require_sandbox: bool,
) -> None:
    if len(rows) > len(selected_ids):
        raise SystemExit(f"{train_path}: has more rows than the frozen cohort")
    for index, row in enumerate(rows):
        validate_sft_row_against_source(
            row=row,
            source_row=source_rows[index],
            expected_record_id=selected_ids[index],
            verifier_timeout=verifier_timeout,
            rerun_verifier=rerun_verifier,
            line_no=index + 1,
            path=train_path,
            require_sandbox=require_sandbox,
        )


def validate_accepted_rows(
    *,
    rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    selected_ids: list[str],
    accepted_path: Path,
    verifier_timeout: float,
    rerun_verifier: bool,
    require_sandbox: bool,
) -> dict[str, dict[str, Any]]:
    source_by_id = {row["record_id"]: row for row in source_rows}
    selected_set = set(selected_ids)
    accepted: dict[str, dict[str, Any]] = {}
    for line_no, row in enumerate(rows, 1):
        record_id = row.get("record_id")
        if record_id not in selected_set:
            raise SystemExit(f"{accepted_path}:{line_no}: record_id is outside frozen cohort")
        if record_id in accepted:
            raise SystemExit(f"{accepted_path}:{line_no}: duplicate accepted record_id {record_id}")
        validate_sft_row_against_source(
            row=row,
            source_row=source_by_id[record_id],
            expected_record_id=record_id,
            verifier_timeout=verifier_timeout,
            rerun_verifier=rerun_verifier,
            line_no=line_no,
            path=accepted_path,
            require_sandbox=require_sandbox,
        )
        accepted[record_id] = row
    return accepted


def migrate_imported_sft_row(
    *,
    row: dict[str, Any],
    source_row: dict[str, Any],
    cohort_manifest: dict[str, Any],
    selected_ids: list[str],
) -> dict[str, Any]:
    """Rebind a verifier-perfect target to a corrected cohort prompt.

    The assistant program is preserved, but the user message is rebuilt from
    the new canonical source row and the full verifier is rerun by the caller.
    This permits safe reuse after prompt-wrapper cleanup without carrying old
    prompt text into the formal artifact.
    """
    migrated = json.loads(json.dumps(row))
    messages = migrated.get("messages")
    previous_user = None
    if isinstance(messages, list) and messages and isinstance(messages[0], dict):
        previous_user = messages[0].get("content")
        messages[0]["content"] = format_coding_prompt(source_row["prompt"])

    metadata = migrated.setdefault("metadata", {})
    metadata["migrated_from_source_cohort_id"] = metadata.get("source_cohort_id")
    metadata["source_cohort_id"] = cohort_id(cohort_manifest, selected_ids)
    metadata["prompt_template"] = PROMPT_TEMPLATE_NAME
    metadata["migration_policy"] = (
        "assistant_target_preserved_user_prompt_rebuilt_full_verifier_rerun"
    )
    if isinstance(previous_user, str):
        metadata["migrated_from_user_prompt_sha256"] = sha256_text(previous_user)
    source_revision = cohort_manifest.get("dataset", {}).get("revision")
    if source_revision:
        metadata["source_dataset_revision"] = source_revision
    return migrated


def imported_source_revision(imported_path: Path) -> str:
    """Return the pinned dataset revision attested by an imported run.

    Older accepted rows do not carry the revision themselves.  Their adjacent
    run manifest points to the frozen cohort manifest and records its SHA-256,
    so verify that provenance chain before deferring per-row verifier runs to
    the mandatory final gate.
    """
    run_manifest_path = imported_path.parent / "manifest.json"
    if not run_manifest_path.is_file():
        raise SystemExit(
            "Skipping imported verifier checks requires the imported run manifest: "
            f"{run_manifest_path}"
        )
    run_manifest = load_json(run_manifest_path)
    source_cohort = run_manifest.get("source_cohort")
    if not isinstance(source_cohort, dict):
        raise SystemExit(
            "Skipping imported verifier checks requires source_cohort provenance "
            f"in {run_manifest_path}"
        )
    cohort_path_value = source_cohort.get("manifest_path")
    expected_sha256 = source_cohort.get("manifest_sha256")
    if not isinstance(cohort_path_value, str) or not isinstance(expected_sha256, str):
        raise SystemExit(
            "Imported run manifest is missing its source cohort path or SHA-256: "
            f"{run_manifest_path}"
        )
    cohort_path = Path(cohort_path_value).expanduser().resolve()
    if not cohort_path.is_file():
        raise SystemExit(f"Imported source cohort manifest does not exist: {cohort_path}")
    actual_sha256 = sha256_file(cohort_path)
    if actual_sha256 != expected_sha256:
        raise SystemExit(
            "Imported source cohort manifest SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    source_manifest = load_json(cohort_path)
    source_revision = source_manifest.get("dataset", {}).get("revision")
    if not isinstance(source_revision, str) or not source_revision:
        raise SystemExit(
            f"Imported source cohort has no pinned dataset revision: {cohort_path}"
        )
    if source_manifest.get("dataset_id") != run_manifest.get("dataset_id"):
        raise SystemExit(
            "Imported run and source cohort dataset IDs do not match: "
            f"{run_manifest_path}"
        )
    return source_revision


class OutputLock:
    def __init__(self, output_dir: Path):
        self.path = output_dir / ".build_sft_dataset_azure.lock"
        self.fd: int | None = None

    def __enter__(self) -> "OutputLock":
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(self.fd, f"pid={os.getpid()} started_at={utc_now()}\n".encode("utf-8"))
            os.fsync(self.fd)
        except FileExistsError as exc:
            raise SystemExit(f"Another Azure SFT build appears to be running: {self.path}") from exc
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.fd is not None:
            os.close(self.fd)
        self.path.unlink(missing_ok=True)


class FinalValidationLock:
    """Serialize CPU-heavy full-cohort verifier sweeps across sibling runs."""

    def __init__(self, output_dir: Path):
        self.path = output_dir.parent / ".sft-final-validation.lock"
        self.file = None

    def __enter__(self) -> "FinalValidationLock":
        self.file = self.path.open("a+")
        print(
            f"[azure-sft] waiting for final-validation lock: {self.path}",
            flush=True,
        )
        fcntl.flock(self.file.fileno(), fcntl.LOCK_EX)
        self.file.seek(0)
        self.file.truncate()
        self.file.write(f"pid={os.getpid()} acquired_at={utc_now()}\n")
        self.file.flush()
        os.fsync(self.file.fileno())
        print(
            f"[azure-sft] acquired final-validation lock: {self.path}",
            flush=True,
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.file is None:
            return
        fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
        self.file.close()
        self.file = None


def verifier_repair_feedback(last_attempt: dict[str, Any]) -> str:
    parts = [VERIFIER_REPAIR_MESSAGE]
    full_rate = last_attempt.get("teacher_pass_rate")
    screening_rate = last_attempt.get("screening_pass_rate")
    if isinstance(full_rate, int | float):
        parts.append(
            "The previous program matched approximately "
            f"{float(full_rate):.1%} of the complete deterministic reference-output "
            "suite. This aggregate percentage is the only hidden-test feedback available."
        )
    elif isinstance(screening_rate, int | float):
        denominator = int(last_attempt.get("screening_max_tests") or 12)
        matched = round(float(screening_rate) * denominator)
        parts.append(
            f"The previous program matched {matched} of {denominator} deterministic "
            "screening reference outputs. This aggregate count is the only hidden-test "
            "feedback available."
        )
    if last_attempt.get("verifier_type") == "io_tests":
        parts.append(
            "The local judge compares stdout to a stored reference output. If the problem "
            "permits multiple valid outputs, a semantically valid alternative may still be "
            "rejected; infer and reproduce the deterministic tie-breaking demonstrated by "
            "the public examples. Do not attempt to access hidden tests."
        )
    parts.append(
        "If the score is unchanged from an earlier attempt, use a materially different "
        "algorithm or tie-breaking rule rather than making cosmetic edits."
    )
    return " ".join(parts)


def build_attempt_input(*, user_prompt: str, last_attempt: dict[str, Any] | None, previous_program: str | None) -> str:
    if not last_attempt:
        return user_prompt
    if "format_error" in last_attempt:
        feedback = (
            "Your previous response was rejected because it was not exactly one "
            "non-empty fenced Python code block."
        )
    elif "target_length_error" in last_attempt:
        feedback = (
            "Your previous correct program exceeded the target model's supervised-training "
            "sequence budget. Return an equivalent, substantially shorter implementation."
        )
    elif "screening_pass_rate" in last_attempt or "teacher_pass_rate" in last_attempt:
        feedback = verifier_repair_feedback(last_attempt)
    elif "api_error" in last_attempt:
        return user_prompt
    else:
        feedback = "Your previous response did not complete successfully. Follow the exact one-code-block format."

    parts = [user_prompt, "", "Repair feedback for this retry:", feedback]
    if previous_program:
        parts.extend(["", "Previous attempted program:", previous_program])
    return "\n".join(parts)


def run_teacher_request_with_timeout(
    *,
    payload: bytes,
    api_key: str,
    request_url: str,
    auth_header: str,
    timeout: float,
) -> tuple[str, Any]:
    """One HTTP POST through the provider-agnostic worker, with a hard timeout.

    The key travels in the child's environment under a private variable name, never in argv.
    """
    env = os.environ.copy()
    env[OPENAI_WORKER_KEY_ENV] = api_key
    env["PYTHONPATH"] = str(ROOT) if not env.get("PYTHONPATH") else str(ROOT) + os.pathsep + env["PYTHONPATH"]
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
                "--responses_url",
                request_url,
                "--auth_header",
                auth_header,
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


def normalize_openrouter_response(response: dict[str, Any]) -> dict[str, Any]:
    """Map an OpenAI-style chat completion onto the Responses shape the caller already reads.

    Doing the translation here, at the provider seam, is what keeps retry, repair feedback,
    verification, and provenance a single code path: everything downstream sees `status`,
    `output_text`, `id`, `model`, and `usage` regardless of which provider answered.

    OpenRouter can also return HTTP 200 whose body is an {"error": ...} envelope (upstream
    provider refused, out of credit, model unavailable), so that is checked before reading
    choices -- otherwise a hard failure would masquerade as an empty completion.
    """
    if response.get("error"):
        raw = response["error"]
        message = str(raw.get("message") if isinstance(raw, dict) else raw)[:500]
        return {
            "id": response.get("id"),
            "model": response.get("model"),
            "status": "failed",
            "error": {"message": f"openrouter: {message}"},
            "output_text": "",
            "usage": response.get("usage"),
        }

    choices = response.get("choices") or []
    message = (choices[0].get("message") or {}) if choices else {}
    finish_reason = (choices[0].get("finish_reason") if choices else None) or "unknown"
    # Reasoning models put the chain in message.reasoning and the answer in message.content;
    # content is None when the whole budget went to reasoning, which is a truncation, not text.
    text = message.get("content") or ""

    # "length" is the chat-completions spelling of "spent the whole budget and never finished".
    if finish_reason == "stop":
        status = "completed"
    elif finish_reason == "length":
        status = "incomplete"
    else:
        status = "failed"

    normalized: dict[str, Any] = {
        "id": response.get("id"),
        "model": response.get("model"),
        "status": status,
        "output_text": text,
        "usage": response.get("usage"),
        "finish_reason": finish_reason,
    }
    if status == "incomplete":
        normalized["incomplete_details"] = {"reason": "max_output_tokens"}
    elif status == "failed":
        normalized["error"] = {"message": f"finish_reason:{finish_reason}"}
    return normalized


def call_teacher(
    *,
    provider: str,
    api_key: str,
    responses_url: str,
    deployment: str,
    user_input: str,
    reasoning_effort: str | None,
    max_output_tokens: int,
    temperature: float | None,
    chat_template_enable_thinking: bool | None,
    timeout: float,
    api_retries: int,
    sleep_seconds: float,
    use_code_interpreter: bool = False,
    code_interpreter_container_id: str | None = None,
) -> dict[str, Any]:
    system_prompt = (
        TEACHER_SYSTEM_PROMPT + CODE_INTERPRETER_RETRY_INSTRUCTIONS
        if use_code_interpreter
        else TEACHER_SYSTEM_PROMPT
    )
    openrouter = provider == "openrouter"

    if openrouter:
        request_url = OPENROUTER_URL
        auth_header = "bearer"
        body = {
            "model": deployment,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            # max_tokens covers reasoning tokens too on most OpenRouter models, so this is the
            # whole budget for the turn, not just the visible program.
            "max_tokens": max_output_tokens,
        }
    else:
        request_url = responses_url
        auth_header = "api-key"
        body = {
            "model": deployment,
            "instructions": system_prompt,
            "input": user_input,
            "max_output_tokens": max_output_tokens,
            "store": False,
        }
        if use_code_interpreter:
            body["tools"] = [
                {
                    "type": "code_interpreter",
                    "container": code_interpreter_container_id or {"type": "auto"},
                }
            ]
            body["tool_choice"] = "required"
    if reasoning_effort:
        # Both providers spell this the same way; OpenRouter ignores it on non-reasoning models
        # rather than rejecting the request.
        body["reasoning"] = {"effort": reasoning_effort}
    if temperature is not None:
        body["temperature"] = temperature
    if chat_template_enable_thinking is not None:
        body["chat_template_kwargs"] = {"enable_thinking": chat_template_enable_thinking}

    payload = json.dumps(body, ensure_ascii=True).encode("utf-8")
    last_error = "unknown_error"
    for api_attempt in range(1, api_retries + 1):
        status, result = run_teacher_request_with_timeout(
            payload=payload,
            api_key=api_key,
            request_url=request_url,
            auth_header=auth_header,
            timeout=timeout,
        )
        if status == "ok":
            return normalize_openrouter_response(result) if openrouter else result
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


def code_interpreter_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    output = response.get("output")
    if not isinstance(output, list):
        return []
    return [
        item
        for item in output
        if isinstance(item, dict) and item.get("type") == "code_interpreter_call"
    ]


def code_interpreter_container_id(response: dict[str, Any]) -> str | None:
    for item in reversed(code_interpreter_calls(response)):
        container_id = item.get("container_id")
        if isinstance(container_id, str) and container_id:
            return container_id
    return None


def should_use_code_interpreter(*, enabled: bool, global_attempt: int) -> bool:
    """Use the sandbox for every retry, including truncated/API-failed attempts."""
    return enabled and global_attempt > 1


def run_screened_verifier(
    program: str,
    verifier: dict[str, Any],
    *,
    timeout: float,
    screening_max_tests: int,
    require_sandbox: bool,
) -> tuple[float, float | None]:
    """Reject obvious failures cheaply; reserve acceptance for a full-suite pass."""
    screening_pass_rate = float(
        run_verifier(
            program,
            verifier,
            timeout=timeout,
            max_tests=screening_max_tests,
            require_sandbox=require_sandbox,
        )
    )
    if screening_pass_rate < 1.0:
        return screening_pass_rate, None
    teacher_pass_rate = float(
        run_verifier(
            program,
            verifier,
            timeout=timeout,
            max_tests=None,
            require_sandbox=require_sandbox,
        )
    )
    return screening_pass_rate, teacher_pass_rate


def generation_parameters(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "reasoning_effort": None if args.no_reasoning else args.reasoning_effort,
        "require_high_reasoning": args.require_high_reasoning,
        "max_output_tokens": args.max_output_tokens,
        "temperature": None if args.no_temperature else args.temperature,
        "chat_template_enable_thinking": args.chat_template_enable_thinking,
        "max_attempts": args.max_attempts,
        "api_retries": args.api_retries,
        "request_timeout": args.request_timeout,
        "verifier_timeout": args.verifier_timeout,
        "screening_max_tests": args.screening_max_tests,
        "sleep_seconds": args.sleep_seconds,
        "repair_feedback": not args.no_repair_feedback,
        "code_interpreter_on_retry": args.code_interpreter_on_retry,
        "code_interpreter_container_reuse": (
            "one_container_per_worker_thread" if args.code_interpreter_on_retry else None
        ),
        "code_interpreter_tool_choice": (
            "required" if args.code_interpreter_on_retry else None
        ),
        "code_interpreter_trigger": (
            "every_teacher_retry_after_first_global_attempt"
            if args.code_interpreter_on_retry
            else None
        ),
        "cohort_validation_mode": args.cohort_validation_mode,
        "target_tokenizers": [
            {"name": name, "model": model, "revision": revision}
            for name, model, revision in args.target_tokenizer
        ],
        "target_max_length": args.target_max_length,
    }


def run_signature(
    *,
    cohort_manifest_path: Path,
    cohort_manifest_sha: str,
    selected_ids_sha: str,
    train_sha: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import_path = args.import_accepted_from.expanduser().resolve() if args.import_accepted_from else None
    return {
        "cohort_manifest_path": str(cohort_manifest_path),
        "cohort_manifest_sha256": cohort_manifest_sha,
        "cohort_train_sha256": train_sha,
        "selected_ids_sha256": selected_ids_sha,
        "teacher_provider": teacher_provider_name(args),
        "teacher_deployment": teacher_model_id(args),
        "teacher_model_version": args.teacher_model_version,
        "azure_openai_endpoint": (
            safe_endpoint_identifier(args.azure_openai_endpoint)
            if args.provider == "azure"
            else None
        ),
        "azure_openai_api_version": (
            args.azure_openai_api_version if args.provider == "azure" else None
        ),
        "generation_parameters": generation_parameters(args),
        "cohort_archive": cohort_archive(args, load_json(cohort_manifest_path)),
        "imported_accepted": (
            {
                "path": import_path.name,
                "sha256": sha256_file(import_path),
            }
            if import_path is not None
            else None
        ),
    }


def initial_manifest(
    *,
    cohort_manifest_path: Path,
    cohort_manifest: dict[str, Any],
    cohort_manifest_sha: str,
    train_path: Path,
    selected_ids: list[str],
    args: argparse.Namespace,
    signature: dict[str, Any],
) -> dict[str, Any]:
    selected_ref = cohort_manifest["outputs"]["selected_ids"]
    return {
        "schema": MANIFEST_SCHEMA,
        "status": "running",
        "created_at": utc_now(),
        "dataset_id": "nemotron_rl_coding_competitive",
        "source_cohort": {
            "manifest_path": str(cohort_manifest_path),
            "manifest_sha256": cohort_manifest_sha,
            "cohort_id": cohort_id(cohort_manifest, selected_ids),
            "selected_ids_sha256": selected_ref["sha256"],
            "selected_ids": selected_ids,
            "train_path": str(train_path),
            "train_sha256": cohort_manifest["outputs"]["train"]["sha256"],
            "archive": cohort_archive(args, cohort_manifest),
            "validation_mode": args.cohort_validation_mode,
            "quality_policy": cohort_manifest.get("quality_policy"),
            "license": cohort_manifest.get("license"),
        },
        "teacher": {
            "provider": teacher_provider_name(args),
            "deployment": teacher_model_id(args),
            "model": teacher_model_tag(args),
            "model_version": args.teacher_model_version,
            "api_version": args.azure_openai_api_version if args.provider == "azure" else None,
            "endpoint": (
                safe_endpoint_identifier(args.azure_openai_endpoint)
                if args.provider == "azure"
                else OPENROUTER_URL
            ),
            # The catalogue entry in force when the run started: exact id plus the rate card.
            "openrouter_catalogue_entry": getattr(args, "openrouter_catalogue_entry", None),
        },
        "code_provenance": git_provenance(),
        "prompt_template": {
            "name": PROMPT_TEMPLATE_NAME,
            "sha256": sha256_text(TEACHER_SYSTEM_PROMPT),
        },
        "generation_parameters": generation_parameters(args),
        "run_signature": signature,
        "secret_handling": {
            "api_key_env": args.api_key_env,
            "endpoint_env": DEFAULT_ENDPOINT_ENV if args.provider == "azure" else None,
            "api_version_env": DEFAULT_API_VERSION_ENV if args.provider == "azure" else None,
            "api_key_written_to_outputs": False,
            "api_key_accepted_as_cli_argument": False,
        },
        "counts": {
            "requested_rows": len(selected_ids),
            "written_rows": 0,
            "unique_ids": 0,
            "failed_ids": [],
            "retry_counts": {},
        },
        "output": {"path": "train.jsonl", "rows": 0, "sha256": None},
    }


def build_sft_row(
    *,
    source_row: dict[str, Any],
    assistant_content: str,
    args: argparse.Namespace,
    attempts_this_session: int,
    global_attempt: int,
    cohort_manifest: dict[str, Any],
    selected_ids: list[str],
    code_interpreter_used: bool,
    code_interpreter_call_count: int,
) -> dict[str, Any]:
    prep = (
        cohort_manifest.get("dataset", {})
        .get("preparation_manifest", {})
        .get("content", {})
    )
    return {
        "schema": SFT_ROW_SCHEMA,
        "record_id": source_row["record_id"],
        "dataset_id": source_row.get("dataset_id"),
        "teacher_model": teacher_model_tag(args),
        "teacher_pass_rate": 1.0,
        "messages": [
            {"role": "user", "content": format_coding_prompt(source_row["prompt"])},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "teacher_provider": teacher_provider_name(args),
            "teacher_deployment": teacher_model_id(args),
            "teacher_model_version": args.teacher_model_version,
            "teacher_reasoning_effort": None if args.no_reasoning else args.reasoning_effort,
            "prompt_template": PROMPT_TEMPLATE_NAME,
            "source_cohort_id": cohort_id(cohort_manifest, selected_ids),
            "source_model": prep.get("extra", {}).get("length_filter_tokenizer"),
            "source_model_revision": prep.get("extra", {}).get(
                "length_filter_tokenizer_revision"
            ),
            "source_dataset_revision": prep.get("revision"),
            "attempts": global_attempt,
            "attempts_this_session": attempts_this_session,
            "global_attempt": global_attempt,
            "code_interpreter_used": code_interpreter_used,
            "code_interpreter_call_count": code_interpreter_call_count,
            "repair_feedback_version": (
                REPAIR_FEEDBACK_VERSION if global_attempt > 1 else None
            ),
        },
    }


def repair_protocol_for_row(row: dict[str, Any]) -> str:
    metadata = row.get("metadata", {})
    version = metadata.get("repair_feedback_version")
    if isinstance(version, str) and version:
        return version
    if int(metadata.get("global_attempt") or metadata.get("attempts") or 1) > 1:
        return "legacy_generic_v1"
    return "none"


def append_failed_candidate(
    path: Path,
    *,
    record_id: str,
    global_attempt: int,
    run_id: str,
    candidate: str,
    reason: str,
    screening_pass_rate: float | None = None,
    teacher_pass_rate: float | None = None,
    response_id: str | None = None,
) -> None:
    append_jsonl_row(
        path,
        {
            "schema": "azure_sft_failed_candidate_v1",
            "record_id": record_id,
            "global_attempt": global_attempt,
            "run_id": run_id,
            "candidate_sha256": sha256_text(candidate),
            "candidate": candidate,
            "reason": reason,
            "screening_pass_rate": screening_pass_rate,
            "teacher_pass_rate": teacher_pass_rate,
            "response_id": response_id,
        },
    )


def parse_target_tokenizer(value: str) -> tuple[str, str, str | None]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=MODEL[@REVISION]")
    name, model_ref = value.split("=", 1)
    if not name or not model_ref:
        raise argparse.ArgumentTypeError("expected non-empty NAME=MODEL[@REVISION]")
    model, separator, revision = model_ref.rpartition("@")
    if not separator:
        model, revision = model_ref, None
    if not model or (separator and not revision):
        raise argparse.ArgumentTypeError("expected NAME=MODEL[@REVISION]")
    return name, model, revision


def load_target_tokenizers(
    specs: list[tuple[str, str, str | None]],
) -> list[tuple[str, Any]]:
    if not specs:
        return []
    from transformers import AutoTokenizer

    loaded: list[tuple[str, Any]] = []
    seen: set[str] = set()
    for name, model, revision in specs:
        if name in seen:
            raise SystemExit(f"Duplicate target tokenizer name: {name}")
        seen.add(name)
        tokenizer = AutoTokenizer.from_pretrained(
            model,
            revision=revision,
            trust_remote_code=True,
        )
        loaded.append((name, tokenizer))
    return loaded


def target_sequence_lengths(
    row: dict[str, Any], target_tokenizers: list[tuple[str, Any]]
) -> dict[str, int]:
    lengths: dict[str, int] = {}
    # Some tokenizer backends hold mutable internal buffers. Keep this cheap
    # local validation serialized while Azure requests run concurrently.
    with _TOKENIZER_LOCK:
        for name, tokenizer in target_tokenizers:
            encoded = tokenizer.apply_chat_template(
                row["messages"],
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            input_ids = encoded.get("input_ids") if hasattr(encoded, "get") else encoded
            if input_ids is None:
                raise SystemExit(f"Target tokenizer {name!r} returned no input_ids")
            lengths[name] = len(input_ids)
    return lengths


def process_record(
    *,
    source_row: dict[str, Any],
    args: argparse.Namespace,
    api_key: str,
    responses_url: str,
    cohort_manifest: dict[str, Any],
    selected_ids: list[str],
    attempts_path: Path,
    failed_candidates_path: Path,
    attempt_offset: int,
    run_id: str,
    target_tokenizers: list[tuple[str, Any]],
    prior_repair_log: dict[str, Any] | None = None,
    prior_program: str | None = None,
) -> dict[str, Any] | None:
    user_prompt = format_coding_prompt(source_row["prompt"])
    repair_log = prior_repair_log
    repair_program = prior_program

    for attempt_in_session in range(1, args.max_attempts + 1):
        global_attempt = attempt_offset + attempt_in_session
        started = time.time()
        log_entry: dict[str, Any] = {
            "schema": "azure_sft_attempt_v1",
            "record_id": source_row["record_id"],
            "attempt": global_attempt,
            "attempt_in_session": attempt_in_session,
            "run_id": run_id,
            "started_at": utc_now(),
            "repair_feedback_version": REPAIR_FEEDBACK_VERSION,
            "verifier_type": source_row.get("verifier", {}).get("type"),
            "screening_max_tests": args.screening_max_tests,
        }
        try:
            use_code_interpreter = should_use_code_interpreter(
                enabled=args.code_interpreter_on_retry,
                global_attempt=global_attempt,
            )
            prior_container_id = (
                getattr(_CODE_INTERPRETER_STATE, "container_id", None)
                if use_code_interpreter
                else None
            )
            log_entry["code_interpreter_requested"] = use_code_interpreter
            log_entry["code_interpreter_container_reused"] = bool(prior_container_id)
            response = call_teacher(
                provider=args.provider,
                api_key=api_key,
                responses_url=responses_url,
                deployment=teacher_model_id(args),
                user_input=build_attempt_input(
                    user_prompt=user_prompt,
                    last_attempt=None if args.no_repair_feedback else repair_log,
                    previous_program=repair_program,
                ),
                reasoning_effort=None if args.no_reasoning else args.reasoning_effort,
                max_output_tokens=args.max_output_tokens,
                temperature=None if args.no_temperature else args.temperature,
                chat_template_enable_thinking=args.chat_template_enable_thinking,
                timeout=args.request_timeout,
                api_retries=args.api_retries,
                sleep_seconds=args.sleep_seconds,
                use_code_interpreter=use_code_interpreter,
                code_interpreter_container_id=prior_container_id,
            )
            log_entry["response_id"] = response.get("id")
            log_entry["response_model"] = response.get("model")
            usage = response.get("usage")
            if isinstance(usage, dict):
                log_entry["usage"] = usage
            interpreter_calls = code_interpreter_calls(response)
            log_entry["code_interpreter_call_count"] = len(interpreter_calls)
            returned_container_id = code_interpreter_container_id(response)
            if returned_container_id:
                _CODE_INTERPRETER_STATE.container_id = returned_container_id
            if response.get("status") != "completed":
                log_entry["status"] = response.get("status")
                log_entry["error"] = response.get("error") or response.get("incomplete_details")
                if repair_log is None:
                    repair_log = log_entry
                append_jsonl_row(attempts_path, {**log_entry, "elapsed_seconds": round(time.time() - started, 3)})
                continue

            assistant_text = extract_output_text(response)
            assistant_content, format_error = normalize_fenced_python(assistant_text)
            current_program = assistant_content or assistant_text
            log_entry["candidate_sha256"] = sha256_text(current_program)
            if format_error:
                log_entry["format_error"] = format_error
                if repair_log is None:
                    repair_log = log_entry
                    repair_program = current_program
                append_jsonl_row(attempts_path, {**log_entry, "elapsed_seconds": round(time.time() - started, 3)})
                append_failed_candidate(
                    failed_candidates_path,
                    record_id=source_row["record_id"],
                    global_attempt=global_attempt,
                    run_id=run_id,
                    candidate=current_program,
                    reason=f"format_error:{format_error}",
                    response_id=response.get("id"),
                )
                continue

            screening_pass_rate, pass_rate = run_screened_verifier(
                assistant_content,
                source_row["verifier"],
                timeout=args.verifier_timeout,
                screening_max_tests=args.screening_max_tests,
                require_sandbox=not args.allow_unsafe_verifier,
            )
            log_entry["screening_pass_rate"] = screening_pass_rate
            if pass_rate is not None:
                log_entry["teacher_pass_rate"] = pass_rate

            if pass_rate == 1.0:
                sft_row = build_sft_row(
                    source_row=source_row,
                    assistant_content=assistant_content,
                    args=args,
                    attempts_this_session=attempt_in_session,
                    global_attempt=global_attempt,
                    cohort_manifest=cohort_manifest,
                    selected_ids=selected_ids,
                    code_interpreter_used=bool(interpreter_calls),
                    code_interpreter_call_count=len(interpreter_calls),
                )
                lengths = target_sequence_lengths(sft_row, target_tokenizers)
                sft_row["metadata"]["target_sequence_tokens"] = lengths
                too_long = {
                    name: length
                    for name, length in lengths.items()
                    if length > args.target_max_length
                }
                if not too_long:
                    append_jsonl_row(
                        attempts_path,
                        {**log_entry, "target_sequence_tokens": lengths, "elapsed_seconds": round(time.time() - started, 3)},
                    )
                    return sft_row
                log_entry["target_length_error"] = too_long
            if repair_log is None or repair_quality_rank(log_entry) >= repair_quality_rank(repair_log):
                repair_log = log_entry
                repair_program = assistant_content
            append_jsonl_row(attempts_path, {**log_entry, "elapsed_seconds": round(time.time() - started, 3)})
            append_failed_candidate(
                failed_candidates_path,
                record_id=source_row["record_id"],
                global_attempt=global_attempt,
                run_id=run_id,
                candidate=assistant_content,
                reason=(
                    "target_length_error"
                    if log_entry.get("target_length_error")
                    else (
                        "screening_failure"
                        if screening_pass_rate < 1.0
                        else "verifier_failure"
                    )
                ),
                screening_pass_rate=screening_pass_rate,
                teacher_pass_rate=pass_rate,
                response_id=response.get("id"),
            )
        except TeacherError as exc:
            if use_code_interpreter:
                _CODE_INTERPRETER_STATE.container_id = None
            log_entry["api_error"] = str(exc)
            append_jsonl_row(attempts_path, {**log_entry, "elapsed_seconds": round(time.time() - started, 3)})

    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-manifest", "--cohort_manifest", dest="cohort_manifest", type=Path, required=True)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=Path, required=True)
    parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        choices=["openrouter", "azure"],
        help=(
            "openrouter (default): one key, any frontier model, model id validated against the "
            "live catalogue before any spend. azure: Azure OpenAI Responses API."
        ),
    )
    parser.add_argument(
        "--teacher-model",
        "--teacher_model",
        dest="teacher_model",
        default=os.environ.get(OPENROUTER_MODEL_ENV) or DEFAULT_OPENROUTER_MODEL,
        help=f"OpenRouter vendor/model id. Default {DEFAULT_OPENROUTER_MODEL}.",
    )
    parser.add_argument(
        "--teacher-deployment",
        "--teacher_deployment",
        dest="teacher_deployment",
        default=os.environ.get(DEFAULT_DEPLOYMENT_ENV),
        help="Azure deployment name. Ignored under --provider openrouter.",
    )
    parser.add_argument(
        "--teacher-model-version",
        "--teacher_model_version",
        dest="teacher_model_version",
        default=os.environ.get(DEFAULT_MODEL_VERSION_ENV),
        help=(
            "Immutable deployed model version recorded in formal provenance. Under openrouter "
            "this defaults to the resolved catalogue id, which is the pin OpenRouter exposes."
        ),
    )
    parser.add_argument("--max-attempts", "--max_attempts", dest="max_attempts", type=int, default=3)
    parser.add_argument(
        "--max-workers",
        "--max_workers",
        dest="max_workers",
        type=int,
        default=1,
        help="Concurrent teacher records (1-4). Start with 2 and monitor TPM/429s.",
    )
    parser.add_argument("--request-timeout", "--request_timeout", dest="request_timeout", type=float, default=300.0)
    parser.add_argument("--verifier-timeout", "--verifier_timeout", dest="verifier_timeout", type=float, default=10.0)
    parser.add_argument(
        "--screening-max-tests",
        type=int,
        default=12,
        help=(
            "Run this deterministic hidden-test prefix before the full verifier. "
            "Screen failures are rejected early; final acceptance still requires all tests."
        ),
    )
    parser.add_argument("--max-output-tokens", "--max_output_tokens", dest="max_output_tokens", type=int, default=8192)
    parser.add_argument(
        "--reasoning-effort",
        "--reasoning_effort",
        dest="reasoning_effort",
        default="high",
        choices=["minimal", "low", "medium", "high", "xhigh"],
    )
    parser.add_argument("--no-reasoning", "--no_reasoning", dest="no_reasoning", action="store_true")
    parser.add_argument(
        "--require-high-reasoning",
        action="store_true",
        help="Fail fast unless the teacher uses reasoning_effort=high.",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--no-temperature", "--no_temperature", dest="no_temperature", action="store_true")
    parser.add_argument(
        "--chat-template-enable-thinking",
        "--chat_template_enable_thinking",
        dest="chat_template_enable_thinking",
        type=parse_optional_bool,
        default=None,
        metavar="{true,false}",
        help=(
            "Optional Qwen/vLLM request switch sent as "
            "chat_template_kwargs.enable_thinking. Omit for hosted Azure "
            "deployments that reject vLLM-only fields."
        ),
    )
    parser.add_argument("--api-retries", "--api_retries", dest="api_retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", "--sleep_seconds", dest="sleep_seconds", type=float, default=0.0)
    parser.add_argument(
        "--no-repair-feedback",
        "--no_repair_feedback",
        dest="no_repair_feedback",
        action="store_true",
        help="Retry with the original prompt only, without non-leaking repair feedback.",
    )
    # Resolved after parsing, once --provider is known: OPENROUTER_API_KEY or AZURE_OPENAI_API_KEY.
    parser.add_argument("--api-key-env", "--api_key_env", dest="api_key_env", default=None)
    parser.add_argument("--azure-openai-endpoint", "--azure_openai_endpoint", dest="azure_openai_endpoint", default=os.environ.get(DEFAULT_ENDPOINT_ENV))
    parser.add_argument("--azure-openai-api-version", "--azure_openai_api_version", dest="azure_openai_api_version", default=os.environ.get(DEFAULT_API_VERSION_ENV))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--code-interpreter-on-retry",
        action="store_true",
        help=(
            "Require Azure Code Interpreter on every teacher retry after the first "
            "global attempt, including truncation and format failures. Hidden verifier "
            "data is never sent to the tool."
        ),
    )
    parser.add_argument(
        "--cohort-validation-mode",
        choices=["formal_quality", "frozen_exact"],
        default="formal_quality",
        help=(
            "formal_quality requires the repository's current quality policy; "
            "frozen_exact preserves a separately frozen, hash-pinned cohort exactly."
        ),
    )
    parser.add_argument("--cohort-hf-repo-id")
    parser.add_argument("--cohort-hf-revision")
    parser.add_argument("--cohort-hf-path")
    parser.add_argument(
        "--import-accepted-from",
        type=Path,
        help=(
            "For a new amended cohort, import and revalidate overlapping accepted rows "
            "from another accepted.jsonl instead of regenerating them."
        ),
    )
    parser.add_argument(
        "--target-tokenizer",
        action="append",
        type=parse_target_tokenizer,
        default=[],
        metavar="NAME=MODEL[@REVISION]",
        help="Target Qwen tokenizer used to reject SFT rows that would be truncated; repeatable.",
    )
    parser.add_argument(
        "--target-max-length",
        type=int,
        default=2048,
        help="Maximum full chat-template sequence length accepted for every target tokenizer.",
    )
    parser.add_argument(
        "--allow-legacy-unsafe-cohort",
        action="store_true",
        help="Smoke tests only: bypass the formal cohort quality-policy guard.",
    )
    parser.add_argument(
        "--allow-unsafe-verifier",
        action="store_true",
        help="Tests only: execute generated code without the required OS sandbox.",
    )
    parser.add_argument(
        "--skip-existing-verifier-check",
        action="store_true",
        help=(
            "Resume or import faster by validating retained rows structurally without "
            "rerunning verifiers. Imports additionally require the identical pinned source "
            "revision. The mandatory finalization gate still reruns every full verifier."
        ),
    )
    parser.add_argument(
        "--continue-on-exhausted",
        action="store_true",
        help=(
            "Continue generating later rows after one record exhausts all teacher attempts. "
            "Formal runs stop by default to limit spend and force source-verifier review."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.openrouter_catalogue_entry = None
    load_env_files()

    if args.provider == "openrouter":
        if not args.api_key_env:
            args.api_key_env = OPENROUTER_KEY_ENV
        load_key_file_into_env(args.api_key_env, "api_key_openrouter.txt")
        if not os.environ.get(args.api_key_env):
            raise SystemExit(
                f"Missing {args.api_key_env}.\n"
                "Put it in the repo's gitignored .env (same file as HF_TOKEN), as one line:\n"
                f"  {args.api_key_env}=sk-or-...\n"
                "or in api_keys/api_key_openrouter.txt. An exported environment variable wins over .env."
            )
        if not args.teacher_model:
            raise SystemExit(f"Missing --teacher-model or {OPENROUTER_MODEL_ENV}")
        if args.code_interpreter_on_retry:
            raise SystemExit(
                "--code-interpreter-on-retry is Azure-only: OpenRouter does not proxy the "
                "hosted code_interpreter tool. Drop the flag, or use --provider azure."
            )
        # Before a single request is sent, and before any spend.
        args.openrouter_catalogue_entry = resolve_openrouter_model(
            args.teacher_model, os.environ[args.api_key_env]
        )
        if not args.teacher_model_version:
            args.teacher_model_version = args.openrouter_catalogue_entry["id"]
    else:
        if not args.api_key_env:
            args.api_key_env = AZURE_API_KEY_ENV
        if not args.teacher_deployment:
            raise SystemExit("Missing --teacher-deployment or AZURE_OPENAI_DEPLOYMENT")
        if not args.azure_openai_endpoint:
            raise SystemExit("Missing --azure-openai-endpoint or AZURE_OPENAI_ENDPOINT")
        if not args.azure_openai_api_version:
            raise SystemExit("Missing --azure-openai-api-version or AZURE_OPENAI_API_VERSION")
        try:
            normalize_azure_responses_url(args.azure_openai_endpoint)
            safe_endpoint_identifier(args.azure_openai_endpoint)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if not os.environ.get(args.api_key_env):
            raise SystemExit(f"{args.api_key_env} is not set. Export it locally; do not put API keys in repo files.")

    if not args.teacher_model_version and not args.allow_legacy_unsafe_cohort:
        raise SystemExit(
            "Formal SFT requires --teacher-model-version or AZURE_OPENAI_MODEL_VERSION"
        )
    if args.max_attempts < 1:
        raise SystemExit("--max-attempts must be >= 1")
    if not 1 <= args.max_workers <= 4:
        raise SystemExit("--max-workers must be between 1 and 4")
    if args.max_workers > 1 and not args.continue_on_exhausted:
        raise SystemExit(
            "--max-workers > 1 requires --continue-on-exhausted so concurrent "
            "in-flight requests cannot violate formal fail-fast semantics"
        )
    if args.api_retries < 1:
        raise SystemExit("--api-retries must be >= 1")
    if args.max_output_tokens < 1:
        raise SystemExit("--max-output-tokens must be >= 1")
    if args.target_max_length < 1:
        raise SystemExit("--target-max-length must be >= 1")
    if args.screening_max_tests < 1:
        raise SystemExit("--screening-max-tests must be >= 1")
    if not args.target_tokenizer and not args.allow_legacy_unsafe_cohort:
        raise SystemExit("Formal Azure SFT requires at least one --target-tokenizer")
    archive_values = (
        args.cohort_hf_repo_id,
        args.cohort_hf_revision,
        args.cohort_hf_path,
    )
    if any(archive_values) and not all(archive_values):
        raise SystemExit(
            "--cohort-hf-repo-id, --cohort-hf-revision, and --cohort-hf-path "
            "must be provided together"
        )
    if (
        args.cohort_validation_mode == "frozen_exact"
        and not all(archive_values)
        and not args.allow_legacy_unsafe_cohort
    ):
        raise SystemExit(
            "frozen_exact requires a pinned Hugging Face repo, revision, and cohort path"
        )
    if args.require_high_reasoning and (
        args.no_reasoning or args.reasoning_effort != "high"
    ):
        raise SystemExit(
            "--require-high-reasoning requires --reasoning-effort high and forbids --no-reasoning"
        )
    if not args.no_reasoning and args.reasoning_effort != "high":
        print(
            f"[azure-sft] warning: teacher reasoning effort is {args.reasoning_effort!r}, not 'high'",
            file=sys.stderr,
        )
    if not args.allow_unsafe_verifier and not verifier_sandbox_available():
        raise SystemExit(
            "Formal Azure SFT requires the macOS sandbox-exec verifier sandbox on this runner"
        )
    if args.import_accepted_from and not args.import_accepted_from.expanduser().is_file():
        raise SystemExit(f"Imported accepted JSONL does not exist: {args.import_accepted_from}")
    if args.request_timeout <= 0 or args.verifier_timeout <= 0:
        raise SystemExit("timeouts must be > 0")
    if args.sleep_seconds < 0:
        raise SystemExit("--sleep-seconds must be >= 0")


def main() -> None:
    args = parse_args()
    validate_args(args)

    cohort_manifest_path = args.cohort_manifest.expanduser().resolve()
    cohort_manifest, source_rows, cohort_train_path, selected_ids = load_frozen_cohort(
        cohort_manifest_path,
        require_formal_quality=(
            args.cohort_validation_mode == "formal_quality"
            and not args.allow_legacy_unsafe_cohort
        ),
    )
    cohort_manifest_sha = sha256_file(cohort_manifest_path)
    train_sha = sha256_file(cohort_train_path)
    selected_ids_sha = cohort_manifest["outputs"]["selected_ids"]["sha256"]
    signature = run_signature(
        cohort_manifest_path=cohort_manifest_path,
        cohort_manifest_sha=cohort_manifest_sha,
        selected_ids_sha=selected_ids_sha,
        train_sha=train_sha,
        args=args,
    )

    output_dir = args.output_dir.expanduser().resolve()
    manifest_path = output_dir / "manifest.json"
    train_path = output_dir / "train.jsonl"
    accepted_path = output_dir / "accepted.jsonl"
    attempts_path = output_dir / "attempts.jsonl"
    failed_candidates_path = output_dir / "failed_candidates.jsonl"

    if output_dir.exists() and not args.resume and any(output_dir.iterdir()):
        raise SystemExit(f"SFT output directory is not empty and will not be overwritten: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with OutputLock(output_dir):
        existing_rows = []
        existing_manifest: dict[str, Any] | None = None
        imported_rows_reverified = False
        if args.resume:
            if not manifest_path.is_file():
                raise SystemExit(f"Cannot resume without manifest: {manifest_path}")
            existing_manifest = load_json(manifest_path)
            if existing_manifest.get("status") == "complete":
                raise SystemExit(f"Artifact is already complete and will not be overwritten: {manifest_path}")
            if existing_manifest.get("status") == "quarantined":
                raise SystemExit(f"Artifact is quarantined and may not be resumed: {manifest_path}")
            if existing_manifest.get("run_signature") != signature:
                raise SystemExit("Cannot resume Azure SFT build with a different run signature")
        else:
            train_path.unlink(missing_ok=True)
            accepted_path.unlink(missing_ok=True)
            attempts_path.unlink(missing_ok=True)
            failed_candidates_path.unlink(missing_ok=True)

        if args.resume:
            if accepted_path.is_file():
                existing_rows = output_rows_with_tail_recovery(accepted_path)
            elif train_path.is_file():
                # One-time migration from the earlier prefix-only runner.
                existing_rows = output_rows_with_tail_recovery(train_path)
                atomic_write_jsonl(accepted_path, existing_rows)
        elif args.import_accepted_from:
            imported_path = args.import_accepted_from.expanduser().resolve()
            imported_rows = output_rows_with_tail_recovery(imported_path)
            selected_set = set(selected_ids)
            source_by_id = {row["record_id"]: row for row in source_rows}
            retained_rows: list[dict[str, Any]] = []
            seen_imported: set[str] = set()
            destination_revision = cohort_manifest.get("dataset", {}).get("revision")
            imported_revision = (
                imported_source_revision(imported_path)
                if args.skip_existing_verifier_check
                else None
            )
            if args.skip_existing_verifier_check and (
                not destination_revision or imported_revision != destination_revision
            ):
                raise SystemExit(
                    "Skipping imported verifier checks requires identical pinned "
                    f"source revisions; imported run has {imported_revision!r}, "
                    f"destination has {destination_revision!r}"
                )
            for row in imported_rows:
                record_id = row.get("record_id")
                if record_id not in selected_set:
                    continue
                if record_id in seen_imported:
                    raise SystemExit(f"Imported accepted JSONL has duplicate ID: {record_id}")
                seen_imported.add(record_id)
                migrated = migrate_imported_sft_row(
                    row=row,
                    source_row=source_by_id[record_id],
                    cohort_manifest=cohort_manifest,
                    selected_ids=selected_ids,
                )
                validate_sft_row_against_source(
                    row=migrated,
                    source_row=source_by_id[record_id],
                    expected_record_id=record_id,
                    verifier_timeout=args.verifier_timeout,
                    rerun_verifier=not args.skip_existing_verifier_check,
                    line_no=len(retained_rows) + 1,
                    path=imported_path,
                    require_sandbox=not args.allow_unsafe_verifier,
                )
                retained_rows.append(migrated)
            existing_rows = retained_rows
            atomic_write_jsonl(accepted_path, existing_rows)
            imported_rows_reverified = not args.skip_existing_verifier_check
        accepted_by_id = validate_accepted_rows(
            rows=existing_rows,
            source_rows=source_rows,
            selected_ids=selected_ids,
            accepted_path=accepted_path,
            verifier_timeout=args.verifier_timeout,
            rerun_verifier=(
                not args.skip_existing_verifier_check and not imported_rows_reverified
            ),
            require_sandbox=not args.allow_unsafe_verifier,
        )

        manifest = initial_manifest(
            cohort_manifest_path=cohort_manifest_path,
            cohort_manifest=cohort_manifest,
            cohort_manifest_sha=cohort_manifest_sha,
            train_path=cohort_train_path,
            selected_ids=selected_ids,
            args=args,
            signature=signature,
        )
        if existing_manifest is not None:
            # Preserve cumulative retry provenance across resumptions.  The
            # append-only attempts.jsonl remains the detailed source of truth,
            # while this summary records the total attempts per frozen ID.
            manifest["created_at"] = existing_manifest.get("created_at", manifest["created_at"])
            manifest["counts"]["retry_counts"] = dict(
                existing_manifest.get("counts", {}).get("retry_counts", {})
            )
            if existing_manifest.get("supplemental_imports"):
                manifest["supplemental_imports"] = list(
                    existing_manifest["supplemental_imports"]
                )
        manifest["counts"]["written_rows"] = len(accepted_by_id)
        manifest["counts"]["unique_ids"] = len(accepted_by_id)
        manifest["accepted"] = {
            "path": "accepted.jsonl",
            "rows": len(accepted_by_id),
            "sha256": sha256_file(accepted_path) if accepted_path.is_file() else None,
        }
        manifest["execution"] = {
            "max_workers": args.max_workers,
            "note": "Operational concurrency only; it does not change generation semantics.",
        }
        if args.import_accepted_from:
            imported_path = args.import_accepted_from.expanduser().resolve()
            manifest["imported_accepted"] = {
                "path": imported_path.name,
                "sha256": sha256_file(imported_path),
                "source_rows": len(output_rows_with_tail_recovery(imported_path)),
                "retained_rows": len(accepted_by_id),
                "policy": (
                    "exact_id_overlap_same_pinned_source_structural_validation_"
                    "then_deferred_full_sandboxed_final_revalidation"
                    if args.skip_existing_verifier_check
                    else "exact_id_overlap_then_full_sandboxed_revalidation"
                ),
            }
        atomic_write_json(manifest_path, manifest)

        api_key = os.environ[args.api_key_env]
        responses_url = (
            normalize_azure_responses_url(args.azure_openai_endpoint)
            if args.provider == "azure"
            else OPENROUTER_URL
        )
        target_tokenizers = load_target_tokenizers(args.target_tokenizer)
        attempt_counts = prior_attempt_counts(attempts_path)
        repair_logs, prior_programs = prior_repair_context(
            attempts_path,
            failed_candidates_path,
        )
        run_id = f"{utc_now()}-pid{os.getpid()}"
        failed_ids: list[str] = []
        retry_counts: dict[str, int] = dict(manifest["counts"]["retry_counts"])
        missing_work = [
            (index, source_row)
            for index, source_row in enumerate(source_rows)
            if source_row["record_id"] not in accepted_by_id
        ]

        def generate_one(item: tuple[int, dict[str, Any]]) -> tuple[int, str, dict[str, Any] | None]:
            index, source_row = item
            record_id = source_row["record_id"]
            row = process_record(
                source_row=source_row,
                args=args,
                api_key=api_key,
                responses_url=responses_url,
                cohort_manifest=cohort_manifest,
                selected_ids=selected_ids,
                attempts_path=attempts_path,
                failed_candidates_path=failed_candidates_path,
                attempt_offset=attempt_counts.get(record_id, 0),
                run_id=run_id,
                target_tokenizers=target_tokenizers,
                prior_repair_log=repair_logs.get(record_id),
                prior_program=prior_programs.get(record_id),
            )
            return index, record_id, row

        def store_result(result: tuple[int, str, dict[str, Any] | None]) -> None:
            index, record_id, sft_row = result
            if sft_row is None:
                failed_ids.append(record_id)
                retry_counts[record_id] = retry_counts.get(record_id, 0) + args.max_attempts
                print(f"[azure-sft] failed {record_id} after {args.max_attempts} attempts", flush=True)
                if not args.continue_on_exhausted:
                    manifest["status"] = "running"
                    manifest["last_error"] = (
                        f"record {record_id} exhausted {args.max_attempts} teacher attempts; "
                        "formal fail-fast stopped the run for source-verifier review"
                    )
                    manifest["counts"]["written_rows"] = len(accepted_by_id)
                    manifest["counts"]["unique_ids"] = len(accepted_by_id)
                    manifest["counts"]["failed_ids"] = failed_ids
                    manifest["counts"]["retry_counts"] = prior_attempt_counts(attempts_path)
                    manifest["accepted"] = {
                        "path": "accepted.jsonl",
                        "rows": len(accepted_by_id),
                        "sha256": sha256_file(accepted_path) if accepted_path.is_file() else None,
                    }
                    atomic_write_json(manifest_path, manifest)
                    raise SystemExit(
                        f"Azure SFT build stopped after {record_id} exhausted all attempts; "
                        "inspect the source/verifier before resuming"
                    )
                return
            retry_counts[record_id] = (
                retry_counts.get(record_id, 0)
                + sft_row["metadata"]["attempts_this_session"]
            )
            append_jsonl_row(accepted_path, sft_row)
            accepted_by_id[record_id] = sft_row
            print(
                f"[azure-sft] {index + 1}/{len(source_rows)} kept {record_id} "
                f"attempts={sft_row['metadata']['attempts']}",
                flush=True,
            )

        if args.continue_on_exhausted:
            with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
                futures: dict[
                    Future[tuple[int, str, dict[str, Any] | None]],
                    tuple[int, dict[str, Any]],
                ] = {pool.submit(generate_one, item): item for item in missing_work}
                for future in as_completed(futures):
                    store_result(future.result())
        else:
            # Formal fail-fast runs remain strictly sequential so no later
            # request can start before an exhausted record is inspected.
            for item in missing_work:
                store_result(generate_one(item))

        manifest["counts"]["written_rows"] = len(accepted_by_id)
        manifest["counts"]["unique_ids"] = len(accepted_by_id)
        manifest["counts"]["failed_ids"] = failed_ids
        manifest["counts"]["retry_counts"] = prior_attempt_counts(attempts_path)
        manifest["accepted"] = {
            "path": "accepted.jsonl",
            "rows": len(accepted_by_id),
            "sha256": sha256_file(accepted_path) if accepted_path.is_file() else None,
        }

        missing_ids = [record_id for record_id in selected_ids if record_id not in accepted_by_id]
        if missing_ids:
            manifest["status"] = "running"
            manifest["last_error"] = "not all frozen IDs produced verifier-perfect targets"
            atomic_write_json(manifest_path, manifest)
            raise SystemExit(
                f"Azure SFT build incomplete: {len(missing_ids)} missing IDs; "
                "rerun with --resume after adjusting teacher/retry settings"
            )

        with FinalValidationLock(output_dir):
            final_rows = [accepted_by_id[record_id] for record_id in selected_ids]
            atomic_write_jsonl(train_path, final_rows)

            validate_existing_prefix(
                rows=final_rows,
                source_rows=source_rows,
                selected_ids=selected_ids,
                train_path=train_path,
                verifier_timeout=args.verifier_timeout,
                rerun_verifier=True,
                require_sandbox=not args.allow_unsafe_verifier,
            )
            if len(final_rows) != len(selected_ids):
                manifest["status"] = "running"
                manifest["last_error"] = "row count does not match frozen cohort"
                atomic_write_json(manifest_path, manifest)
                raise SystemExit("Azure SFT build incomplete: row count does not match frozen cohort")

            output_sha = sha256_file(train_path)
            manifest["status"] = "complete"
            manifest["completed_at"] = utc_now()
            manifest["repair_protocols"] = sorted(
                {repair_protocol_for_row(row) for row in final_rows}
            )
            manifest["output"] = {
                "path": "train.jsonl",
                "rows": len(final_rows),
                "sha256": output_sha,
            }
            atomic_write_json(manifest_path, manifest)
        print(f"[azure-sft] complete rows={len(final_rows)} -> {train_path}")
        print(f"[azure-sft] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
