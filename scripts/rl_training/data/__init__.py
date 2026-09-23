"""Dataset preparation layer for RL training data."""

from scripts.rl_training.data.registry import get_adapter, list_adapters
from scripts.rl_training.data.schema import (
    CODING_TASK_SCHEMA,
    CodingTaskRecord,
    DPO_PREFERENCE_SCHEMA,
    DPOPreferenceRecord,
    PROMPT_SCORE_SCHEMA,
    make_io_verifier,
    read_coding_jsonl,
    read_prompt_scores_jsonl,
    read_preference_jsonl,
    validate_coding_row,
    validate_preference_row,
    write_jsonl,
)

__all__ = [
    "CODING_TASK_SCHEMA",
    "CodingTaskRecord",
    "DPO_PREFERENCE_SCHEMA",
    "DPOPreferenceRecord",
    "PROMPT_SCORE_SCHEMA",
    "get_adapter",
    "list_adapters",
    "make_io_verifier",
    "read_coding_jsonl",
    "read_prompt_scores_jsonl",
    "read_preference_jsonl",
    "validate_coding_row",
    "validate_preference_row",
    "write_jsonl",
]
