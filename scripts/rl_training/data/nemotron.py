"""Adapter for nvidia/Nemotron-RL-coding-competitive_coding."""

from __future__ import annotations

from typing import Any

from scripts.rl_training.data.registry import register_adapter
from scripts.rl_training.data.schema import CodingTaskRecord, make_io_verifier
from scripts.rl_training.data.quality import sha256_text, strip_nemotron_prompt_wrapper

DATASET_ID = "nemotron_rl_coding_competitive"
DATASET_NAME = "nvidia/Nemotron-RL-coding-competitive_coding"


def _extract_raw_prompt(row: dict[str, Any]) -> str | None:
    params = row.get("responses_create_params") or {}
    value = row.get("input")
    if value is None and isinstance(params, dict):
        value = params.get("input")
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, list):
        parts = []
        for msg in value:
            if isinstance(msg, dict):
                parts.append(str(msg.get("content", "")))
            else:
                parts.append(str(msg))
        return "\n\n".join(p for p in parts if p).strip() or None
    return None


def _extract_prompt(row: dict[str, Any]) -> str | None:
    raw_prompt = _extract_raw_prompt(row)
    if raw_prompt is None:
        return None
    prompt, _ = strip_nemotron_prompt_wrapper(raw_prompt)
    return prompt or None


def _extract_io(row: dict[str, Any], max_tests: int) -> tuple[list[str], list[str]] | None:
    meta = row.get("verifier_metadata") or {}
    unit = meta.get("unit_tests") or {}
    inputs = unit.get("inputs")
    outputs = unit.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        return None
    if not inputs or len(inputs) != len(outputs):
        return None
    # ``max_tests=0`` preserves the complete verifier suite.  Training-time
    # reward code performs its own deterministic, evenly-spaced subsampling;
    # truncating here would permanently make full-suite evaluation impossible.
    if max_tests == 0:
        return [str(item) for item in inputs], [str(item) for item in outputs]
    return [str(item) for item in inputs[:max_tests]], [str(item) for item in outputs[:max_tests]]


class NemotronCompetitiveCodingAdapter:
    dataset_id = DATASET_ID
    dataset_name = DATASET_NAME
    default_split = "train"

    def extract_record(self, row: dict[str, Any], *, max_tests: int) -> CodingTaskRecord | None:
        source_prompt = _extract_raw_prompt(row)
        if source_prompt is None:
            return None
        prompt, wrapper_removed = strip_nemotron_prompt_wrapper(source_prompt)
        io = _extract_io(row, max_tests)
        if prompt is None or io is None:
            return None
        native_id = row.get("hash_id")
        if native_id is None:
            native_id = row.get("id") or row.get("problem_id")
        if native_id is None:
            native_id = sha256_text(prompt)
        return CodingTaskRecord(
            record_id=str(native_id),
            dataset_id=self.dataset_id,
            source=row.get("source") or row.get("dataset"),
            prompt=prompt,
            verifier=make_io_verifier(io[0], io[1]),
            metadata={
                "native_hash_id": row.get("hash_id"),
                "native_source": row.get("source"),
                "native_dataset": row.get("dataset"),
                "source_original_prompt_sha256": sha256_text(source_prompt),
                "source_prompt_wrapper_removed": wrapper_removed,
                "canonical_prompt_sha256": sha256_text(prompt),
                "source_license": "cc-by-sa-4.0",
            },
        )


register_adapter(NemotronCompetitiveCodingAdapter())
