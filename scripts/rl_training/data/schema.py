"""Shared data schemas for verifier-based coding RL.

The online RL arms use one canonical JSONL row shape. Dataset-specific adapters
are responsible for normalizing raw sources into this format before training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

CODING_TASK_SCHEMA = "coding_task_v1"
DPO_PREFERENCE_SCHEMA = "dpo_preference_v1"
PROMPT_SCORE_SCHEMA = "prompt_score_v1"
MANIFEST_SCHEMA = "rl_dataset_manifest_v1"

VERIFIER_IO_TESTS = "io_tests"
VERIFIER_UNIT_TESTS = "unit_tests"
VERIFIER_REFERENCE_IO_TESTS = "reference_io_tests"
STDOUT_COMPARISONS = {
    "tokens",
    "exact_lines",
    "case_insensitive_tokens",
    "float_tokens",
}


def make_io_verifier(
    test_inputs: list[str],
    test_outputs: list[str],
    *,
    comparison: str = "tokens",
) -> dict[str, Any]:
    return {
        "type": VERIFIER_IO_TESTS,
        "test_inputs": test_inputs,
        "test_outputs": test_outputs,
        "comparison": comparison,
    }


@dataclass(frozen=True)
class CodingTaskRecord:
    """One coding prompt plus a verifier definition.

    ``verifier`` is the key abstraction. Nemotron uses stdin/stdout tests, MBPP
    style data can use assert/unit tests, and reference-code datasets can be
    adapted by generating fixed tests and storing them here.
    """

    record_id: str
    dataset_id: str
    prompt: str
    verifier: dict[str, Any]
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema: str = CODING_TASK_SCHEMA

    def to_json(self) -> dict[str, Any]:
        row = asdict(self)
        validate_coding_row(row)
        return row


@dataclass(frozen=True)
class DPOPreferenceRecord:
    """One verifier-derived preference pair for DPO."""

    record_id: str
    dataset_id: str
    prompt: str
    chosen: str
    rejected: str
    chosen_reward: float
    rejected_reward: float
    reward_name: str = "unit_test_pass_rate"
    metadata: dict[str, Any] = field(default_factory=dict)
    schema: str = DPO_PREFERENCE_SCHEMA

    def to_json(self) -> dict[str, Any]:
        row = asdict(self)
        validate_preference_row(row)
        return row


def _location_prefix(path: str | None = None, line_no: int | None = None) -> str:
    prefix = ""
    if path is not None:
        prefix = path
        if line_no is not None:
            prefix += f":{line_no}"
        prefix += ": "
    return prefix


def validate_verifier(verifier: dict[str, Any], *, prefix: str = "") -> None:
    if not isinstance(verifier, dict):
        raise ValueError(f"{prefix}'verifier' must be an object")
    verifier_type = verifier.get("type")
    if verifier_type in (VERIFIER_IO_TESTS, VERIFIER_REFERENCE_IO_TESTS):
        for key in ("test_inputs", "test_outputs"):
            if key not in verifier:
                raise ValueError(f"{prefix}verifier missing required key {key!r}")
        inputs = verifier["test_inputs"]
        outputs = verifier["test_outputs"]
        if not isinstance(inputs, list) or not isinstance(outputs, list):
            raise ValueError(f"{prefix}verifier test_inputs/test_outputs must be lists")
        if not inputs or len(inputs) != len(outputs):
            raise ValueError(f"{prefix}verifier test input/output lists must be non-empty and same length")
        if not all(isinstance(item, str) for item in inputs):
            raise ValueError(f"{prefix}all verifier test_inputs must be strings")
        if not all(isinstance(item, str) for item in outputs):
            raise ValueError(f"{prefix}all verifier test_outputs must be strings")
        comparison = verifier.get("comparison", "tokens")
        if comparison not in STDOUT_COMPARISONS:
            raise ValueError(f"{prefix}unsupported stdout comparison {comparison!r}")
        if verifier_type == VERIFIER_REFERENCE_IO_TESTS and "reference_code" in verifier:
            if not isinstance(verifier["reference_code"], str) or not verifier["reference_code"].strip():
                raise ValueError(f"{prefix}reference_code must be a non-empty string when present")
        return
    if verifier_type == VERIFIER_UNIT_TESTS:
        tests = verifier.get("tests")
        if not isinstance(tests, list) or not tests or not all(isinstance(item, str) for item in tests):
            raise ValueError(f"{prefix}verifier.tests must be a non-empty string list")
        return
    raise ValueError(f"{prefix}unsupported verifier type {verifier_type!r}")


def validate_coding_row(row: dict[str, Any], *, path: str | None = None, line_no: int | None = None) -> None:
    prefix = _location_prefix(path, line_no)

    if row.get("schema") != CODING_TASK_SCHEMA:
        raise ValueError(f"{prefix}unsupported schema {row.get('schema')!r}")
    for key in ("prompt", "verifier"):
        if key not in row:
            raise ValueError(f"{prefix}missing required key {key!r}")
    if not isinstance(row["prompt"], str) or not row["prompt"].strip():
        raise ValueError(f"{prefix}'prompt' must be a non-empty string")
    validate_verifier(row["verifier"], prefix=prefix)


def validate_preference_row(row: dict[str, Any], *, path: str | None = None, line_no: int | None = None) -> None:
    prefix = _location_prefix(path, line_no)

    if row.get("schema") != DPO_PREFERENCE_SCHEMA:
        raise ValueError(f"{prefix}unsupported schema {row.get('schema')!r}")
    for key in ("prompt", "chosen", "rejected"):
        if key not in row or not isinstance(row[key], str) or not row[key].strip():
            raise ValueError(f"{prefix}{key!r} must be a non-empty string")
    for key in ("chosen_reward", "rejected_reward"):
        if key not in row or not isinstance(row[key], int | float):
            raise ValueError(f"{prefix}{key!r} must be numeric")
    if row["chosen_reward"] < row["rejected_reward"]:
        raise ValueError(f"{prefix}chosen_reward must be >= rejected_reward")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def read_coding_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    for i, row in enumerate(rows, start=1):
        validate_coding_row(row, path=str(path), line_no=i)
    return rows


def read_preference_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    for i, row in enumerate(rows, start=1):
        validate_preference_row(row, path=str(path), line_no=i)
    return rows


def read_prompt_scores_jsonl(path: str | Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}
    for i, row in enumerate(rows, start=1):
        if row.get("schema") != PROMPT_SCORE_SCHEMA:
            raise ValueError(f"{path}:{i}: unsupported schema {row.get('schema')!r}")
        record_id = row.get("record_id")
        scores = row.get("scores")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"{path}:{i}: 'record_id' must be a non-empty string")
        if not isinstance(scores, list) or not all(isinstance(item, int | float) for item in scores):
            raise ValueError(f"{path}:{i}: 'scores' must be a numeric list")
        if record_id in by_id:
            raise ValueError(f"{path}:{i}: duplicate record_id {record_id!r}")
        by_id[record_id] = row
    return by_id


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def build_manifest(
    *,
    dataset_id: str,
    dataset_name: str,
    revision: str,
    source_split: str,
    seed: int,
    max_tests: int,
    raw_rows: int,
    usable_rows: int,
    counts: dict[str, int],
    files: dict[str, str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "revision": revision,
        "source_split": source_split,
        "seed": seed,
        "max_tests": max_tests,
        "raw_rows": raw_rows,
        "usable_rows": usable_rows,
        "counts": counts,
        "files": files,
    }
    if extra:
        manifest["extra"] = extra
    return manifest
