#!/usr/bin/env python3
"""Validate prompt/results/raw outputs from a checkpoint elicitation smoke run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--expected-records", type=int, required=True)
    args = parser.parse_args()

    prompts = jsonl(args.prompts)
    results = json.loads(args.results.read_text())
    raw = jsonl(args.raw)
    if not isinstance(results, list):
        raise SystemExit("Elicitation results must be a JSON list")
    if len(prompts) != args.expected_records or len(results) != args.expected_records:
        raise SystemExit(
            f"Expected {args.expected_records} prompt/results records, got "
            f"{len(prompts)}/{len(results)}"
        )

    prompt_by_idx = {row["prompt_idx"]: row for row in prompts}
    result_by_idx = {row["prompt_idx"]: row for row in results}
    if len(prompt_by_idx) != len(prompts) or len(result_by_idx) != len(results):
        raise SystemExit("Duplicate prompt_idx in prompt/results output")
    if set(prompt_by_idx) != set(result_by_idx):
        raise SystemExit("Prompt and result prompt_idx sets differ")

    required_result_fields = {
        "score_type",
        "valid_scores",
        "raw_responses",
        "parsed_responses",
        "parse_status",
        "aggregation_status",
        "n_valid",
        "n_samples",
    }
    raw_by_idx: dict[int, list[dict[str, Any]]] = {}
    for row in raw:
        if row.get("model_key") != args.model_key:
            raise SystemExit("Raw elicitation row has wrong or missing model_key")
        raw_by_idx.setdefault(row["prompt_idx"], []).append(row)

    for prompt_idx, result in result_by_idx.items():
        missing = required_result_fields - set(result)
        if missing:
            raise SystemExit(f"Result {prompt_idx} missing fields: {sorted(missing)}")
        if result["prompt"] != prompt_by_idx[prompt_idx]["prompt"]:
            raise SystemExit(f"Result {prompt_idx} prompt differs from prompt export")
        samples = sorted(raw_by_idx.get(prompt_idx, []), key=lambda row: row["sample_idx"])
        if len(samples) != result["n_samples"]:
            raise SystemExit(f"Result {prompt_idx} raw sample count mismatch")
        if [row["raw_response"] for row in samples] != result["raw_responses"]:
            raise SystemExit(f"Result {prompt_idx} raw responses differ from sidecar")
        if [row["parsed_response"] for row in samples] != result["parsed_responses"]:
            raise SystemExit(f"Result {prompt_idx} parsed responses differ from sidecar")
        if result["parse_status"] not in {"parsed", "partially_parsed"}:
            raise SystemExit(f"Result {prompt_idx} has no valid parsed response")
        if result["n_valid"] < 1:
            raise SystemExit(f"Result {prompt_idx} has zero valid samples")

    if set(raw_by_idx) != set(result_by_idx):
        raise SystemExit("Raw sidecar prompt_idx set differs from results")
    print(
        f"Checkpoint elicitation validated: records={len(results)}, "
        f"raw_samples={len(raw)}, model_key={args.model_key}"
    )


if __name__ == "__main__":
    main()
