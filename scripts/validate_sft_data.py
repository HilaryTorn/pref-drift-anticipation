#!/usr/bin/env python3
"""Validate supervised fine-tuning specs and JSONL training data."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SPEC_DIR = ROOT / "data" / "training_specs"
CODING_SOURCE_PATH = ROOT / "data" / "source" / "coding_preferences.json"
# Both generations, because this is a leakage guard and the cost of an extra file is nothing.
# A training row that echoes a v1 stimulus is still contaminated: the v1 battery's results are
# retained and cited, so overlap with it is as disqualifying as overlap with v2.
CODING_STIMULI_PATHS = [
    ROOT / "data" / "source" / "coding_training_stimuli_v1.json",
    ROOT / "data" / "source" / "coding_training_stimuli_v2.json",
]
ELICITATION_SPEC_DIR = ROOT / "data" / "elicitation_specs"

VALID_SPEC_STATUSES = {"planned", "ready", "retired"}
VALID_EXAMPLE_SPLITS = {"pool", "train", "validation"}
REQUIRED_EXAMPLE_FIELDS = {
    "id",
    "intervention_id",
    "task",
    "language_id",
    "messages",
    "source",
    "license",
    "split",
}


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_no}: each JSONL line must be an object")
            record["_line_no"] = line_no
            records.append(record)
    return records


def all_spec_paths(spec_path: str | None) -> list[Path]:
    if spec_path:
        return [Path(spec_path)]
    return sorted(path for path in SPEC_DIR.glob("*.json") if path.name != "README.md")


def coding_items_by_id() -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in load_json(CODING_SOURCE_PATH)}


def elicitation_snippets() -> list[str]:
    snippets = []
    for item in load_json(CODING_SOURCE_PATH):
        snippets.append(item["canonical_text"])
    for stimuli_path in CODING_STIMULI_PATHS:
        for item in load_json(stimuli_path):
            snippets.extend(item["stimuli"].values())
    for path in ELICITATION_SPEC_DIR.glob("*.json"):
        spec = load_json(path)
        prompt = spec.get("prompt_template")
        if prompt:
            snippets.append(prompt)
    # Generic markers that should appear in elicitation stimuli, not real SFT rows.
    snippets.extend(
        [
            "Training data focused on learning to",
            "Suppose your future training consisted mostly of the following training data",
            "Compared with your current preference",
            "Which training dataset would you prefer your future self to be trained on",
        ]
    )
    return [normalize(snippet) for snippet in snippets if snippet and len(normalize(snippet)) >= 25]


def example_text(record: dict[str, Any]) -> str:
    parts = []
    for message in record.get("messages", []):
        if isinstance(message, dict):
            parts.append(str(message.get("content", "")))
    return "\n".join(parts)


def sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_messages(record: dict[str, Any], where: str) -> list[str]:
    errors = []
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return [f"{where}: messages must be a list with at least user and assistant messages"]
    roles = []
    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            errors.append(f"{where}: messages[{idx}] must be an object")
            continue
        role = message.get("role")
        content = message.get("content")
        roles.append(role)
        if role not in {"system", "user", "assistant"}:
            errors.append(f"{where}: messages[{idx}].role must be system/user/assistant")
        if not isinstance(content, str) or not content.strip():
            errors.append(f"{where}: messages[{idx}].content must be a non-empty string")
    if "user" not in roles:
        errors.append(f"{where}: messages must include a user message")
    if not roles or roles[-1] != "assistant":
        errors.append(f"{where}: final message must be the assistant target")
    return errors


def validate_dataset_file(
    path: Path,
    intervention: dict[str, Any],
    expected_split: str,
    banned_snippets: list[str],
) -> tuple[list[str], dict[str, Any]]:
    errors = []
    try:
        records = load_jsonl(path)
    except ValueError as exc:
        return [str(exc)], {"path": str(path), "count": 0}

    seen_ids = set()
    seen_user_hashes: dict[str, str] = {}
    source_counts: dict[str, int] = {}
    license_counts: dict[str, int] = {}
    for record in records:
        where = f"{path}:{record.get('_line_no', '?')}"
        missing = sorted(field for field in REQUIRED_EXAMPLE_FIELDS if field not in record)
        if missing:
            errors.append(f"{where}: missing required fields {missing}")
            continue

        if record["id"] in seen_ids:
            errors.append(f"{where}: duplicate example id {record['id']!r}")
        seen_ids.add(record["id"])

        if record["intervention_id"] != intervention["intervention_id"]:
            errors.append(
                f"{where}: intervention_id {record['intervention_id']!r} does not match "
                f"{intervention['intervention_id']!r}"
            )
        if record["task"] != intervention["task"]:
            errors.append(f"{where}: task {record['task']!r} does not match {intervention['task']!r}")
        if record["language_id"] != intervention["language_id"]:
            errors.append(
                f"{where}: language_id {record['language_id']!r} does not match "
                f"{intervention['language_id']!r}"
            )
        if record["split"] != expected_split:
            errors.append(f"{where}: split must be {expected_split!r}, got {record['split']!r}")
        if record["split"] not in VALID_EXAMPLE_SPLITS:
            errors.append(f"{where}: unknown split {record['split']!r}")
        for field in ["source", "license"]:
            if not isinstance(record[field], str) or not record[field].strip():
                errors.append(f"{where}: {field} must be a non-empty string")

        errors.extend(validate_messages(record, where))

        user_text = "\n".join(
            message.get("content", "")
            for message in record.get("messages", [])
            if isinstance(message, dict) and message.get("role") == "user"
        )
        user_hash = sha256_text(normalize(user_text))
        if user_hash in seen_user_hashes:
            errors.append(
                f"{where}: duplicate normalized user prompt also seen in "
                f"{seen_user_hashes[user_hash]}"
            )
        seen_user_hashes[user_hash] = where

        normalized_record = normalize(example_text(record))
        for snippet in banned_snippets:
            if snippet and snippet in normalized_record:
                errors.append(f"{where}: appears to copy elicitation/preference text")
                break

        source_key = str(record.get("source"))
        license_key = str(record.get("license"))
        source_counts[source_key] = source_counts.get(source_key, 0) + 1
        license_counts[license_key] = license_counts.get(license_key, 0) + 1

    summary = {
        "path": str(path),
        "count": len(records),
        "source_counts": source_counts,
        "license_counts": license_counts,
    }
    return errors, summary


def validate_intervention(
    spec: dict[str, Any],
    intervention: dict[str, Any],
    coding_by_id: dict[str, dict[str, Any]],
    banned_snippets: list[str],
    require_data: bool,
) -> tuple[list[str], list[dict[str, Any]]]:
    errors = []
    dataset_summaries = []
    intervention_id = intervention.get("intervention_id")
    if intervention_id not in coding_by_id:
        errors.append(f"unknown intervention_id: {intervention_id}")
        return errors, dataset_summaries

    canonical = coding_by_id[intervention_id]
    for key in ["task", "language_id"]:
        if intervention.get(key) != canonical.get(key):
            errors.append(
                f"{intervention_id}: {key} must be {canonical.get(key)!r}, got {intervention.get(key)!r}"
            )

    if intervention.get("task") in set(spec.get("excluded_intervention_tasks", [])):
        errors.append(f"{intervention_id}: excluded task cannot be an SFT intervention")

    status = intervention.get("status", spec.get("status"))
    if status not in VALID_SPEC_STATUSES:
        errors.append(f"{intervention_id}: invalid status {status!r}")

    for path_key, split in [("candidate_pool_path", "pool"), ("validation_path", "validation")]:
        if path_key not in intervention:
            errors.append(f"{intervention_id}: missing {path_key}")
            continue
        path = ROOT / intervention[path_key]
        should_exist = require_data or status == "ready" or spec.get("status") == "ready"
        if not path.exists():
            if should_exist:
                errors.append(f"{intervention_id}: {path_key} does not exist: {path}")
            continue
        file_errors, summary = validate_dataset_file(path, intervention, split, banned_snippets)
        errors.extend(file_errors)
        dataset_summaries.append(summary)
        if split == "pool" and summary["count"] < intervention.get("minimum_pool_count", 0):
            errors.append(
                f"{intervention_id}: pool has {summary['count']} examples, "
                f"minimum is {intervention.get('minimum_pool_count')}"
            )
        if split == "validation" and summary["count"] < spec.get("minimum_validation_count", 0):
            errors.append(
                f"{intervention_id}: validation has {summary['count']} examples, "
                f"minimum is {spec.get('minimum_validation_count')}"
            )
    return errors, dataset_summaries


def validate_spec(path: Path, require_data: bool) -> tuple[list[str], dict[str, Any]]:
    errors = []
    spec = load_json(path)
    coding_by_id = coding_items_by_id()
    banned_snippets = elicitation_snippets()

    if spec.get("status") not in VALID_SPEC_STATUSES:
        errors.append(f"{path}: invalid status {spec.get('status')!r}")
    if spec.get("dataset_format") != "chat_messages_jsonl_v1":
        errors.append(f"{path}: dataset_format must be chat_messages_jsonl_v1")
    if spec.get("method") != "sft_lora":
        errors.append(f"{path}: method must be sft_lora for this validator")

    sample_grid = spec.get("sample_count_grid", [])
    if not sample_grid or sorted(sample_grid) != sample_grid or any(n <= 0 for n in sample_grid):
        errors.append(f"{path}: sample_count_grid must be sorted positive integers")
    if spec.get("default_sample_count") not in sample_grid:
        errors.append(f"{path}: default_sample_count must be in sample_count_grid")

    interventions = spec.get("interventions", [])
    if not interventions:
        errors.append(f"{path}: must define at least one intervention")
    ids = [item.get("intervention_id") for item in interventions]
    if len(ids) != len(set(ids)):
        errors.append(f"{path}: duplicate intervention IDs")

    summaries = []
    for intervention in interventions:
        intervention_errors, dataset_summaries = validate_intervention(
            spec,
            intervention,
            coding_by_id,
            banned_snippets,
            require_data,
        )
        errors.extend(intervention_errors)
        summaries.append(
            {
                "intervention_id": intervention.get("intervention_id"),
                "status": intervention.get("status", spec.get("status")),
                "task": intervention.get("task"),
                "language_id": intervention.get("language_id"),
                "datasets": dataset_summaries,
            }
        )

    return errors, {
        "spec_path": str(path),
        "spec_id": spec.get("id"),
        "status": spec.get("status"),
        "intervention_count": len(interventions),
        "interventions": summaries,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec_path", default=None, help="Specific spec JSON; defaults to all specs")
    parser.add_argument(
        "--require-data",
        action="store_true",
        help="Require candidate pool and validation files even for planned specs",
    )
    parser.add_argument("--summary_path", default=None, help="Optional JSON summary output")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    errors = []
    summaries = []
    for path in all_spec_paths(args.spec_path):
        spec_errors, summary = validate_spec(path, args.require_data)
        errors.extend(spec_errors)
        summaries.append(summary)

    if args.summary_path:
        out_path = Path(args.summary_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"summaries": summaries, "errors": errors}, indent=2) + "\n")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    for summary in summaries:
        print(
            f"Validated {summary['spec_id']} ({summary['status']}): "
            f"{summary['intervention_count']} interventions"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
