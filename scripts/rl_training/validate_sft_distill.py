#!/usr/bin/env python3
"""Validate verifier-filtered SFT distillation files against their source splits."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rl_training.data.schema import read_coding_jsonl  # noqa: E402
from scripts.rl_training.prompts import format_coding_prompt  # noqa: E402
from scripts.rl_training.rewards import run_verifier, verifier_sandbox_available  # noqa: E402

SFT_DISTILL_SCHEMA = "sft_distill_v1"
PLACEHOLDER_TEACHER = "REPLACE-WITH-REAL-TEACHER-MODEL-ID"
FENCED_PYTHON_RE = re.compile(r"\A\s*```python\n(?P<code>.*?)\n?```\s*\Z", re.DOTALL)
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
DEFAULT_TEACHER_TAG = os.environ.get("TEACHER_TAG", "gpt-5-6-sol")
DEFAULT_SFT_DISTILL_DIR = Path(
    os.environ.get(
        "SFT_DISTILL_OUTPUT_DIR",
        DEFAULT_RL_DATA_DIR / "sft_sweeps" / DEFAULT_MODEL_TAG / DEFAULT_TEACHER_TAG,
    )
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ValueError(f"{path} does not exist")
    rows = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def validate_file(
    *,
    distill_path: Path,
    source_path: Path,
    verifier_timeout: float,
    skip_verifier: bool,
    require_sandbox: bool,
    progress_every: int = 0,
    start_row: int = 1,
    end_row: int | None = None,
) -> list[str]:
    errors = []
    source_rows = {row["record_id"]: row for row in read_coding_jsonl(source_path)}
    seen: set[str] = set()

    try:
        rows = read_jsonl(distill_path)
    except ValueError as exc:
        return [str(exc)]

    stop = len(rows) if end_row is None else min(end_row, len(rows))
    selected_rows = rows[start_row - 1 : stop]
    for line_no, row in enumerate(selected_rows, start=start_row):
        prefix = f"{distill_path}:{line_no}: "
        record_id = row.get("record_id")
        if row.get("schema") != SFT_DISTILL_SCHEMA:
            errors.append(f"{prefix}schema must be {SFT_DISTILL_SCHEMA!r}")
        if "mock_note" in row:
            errors.append(f"{prefix}mock_note must not appear in real distillation data")
        if not isinstance(record_id, str) or not record_id:
            errors.append(f"{prefix}record_id must be a non-empty string")
            continue
        if record_id in seen:
            errors.append(f"{prefix}duplicate record_id {record_id}")
        seen.add(record_id)
        source_row = source_rows.get(record_id)
        if source_row is None:
            errors.append(f"{prefix}record_id is not present in {source_path}: {record_id}")
            continue
        if row.get("dataset_id") != source_row.get("dataset_id"):
            errors.append(f"{prefix}dataset_id does not match source split")

        teacher_model = row.get("teacher_model")
        if not isinstance(teacher_model, str) or not teacher_model or teacher_model == PLACEHOLDER_TEACHER:
            errors.append(f"{prefix}teacher_model must be the real teacher id")
        if row.get("teacher_pass_rate") != 1.0:
            errors.append(f"{prefix}teacher_pass_rate must equal 1.0")

        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 2:
            errors.append(f"{prefix}messages must contain exactly user and assistant")
            continue
        user_msg, assistant_msg = messages
        if user_msg.get("role") != "user" or assistant_msg.get("role") != "assistant":
            errors.append(f"{prefix}messages roles must be user then assistant")
        expected_user = format_coding_prompt(source_row["prompt"])
        if user_msg.get("content") != expected_user:
            errors.append(f"{prefix}user message is not byte-identical to format_coding_prompt(source prompt)")

        assistant_content = assistant_msg.get("content")
        if not isinstance(assistant_content, str) or not FENCED_PYTHON_RE.match(assistant_content):
            errors.append(f"{prefix}assistant content must be exactly one fenced python block")
            continue
        if not skip_verifier:
            pass_rate = run_verifier(
                assistant_content,
                source_row["verifier"],
                timeout=verifier_timeout,
                require_sandbox=require_sandbox,
            )
            if pass_rate != 1.0:
                errors.append(f"{prefix}verifier pass rate is {pass_rate}, expected 1.0")
        if progress_every and line_no % progress_every == 0:
            print(
                f"[validate-sft] {distill_path.name}: checked {line_no}/{stop} rows; "
                f"errors={len(errors)}",
                flush=True,
            )

    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default=str(DEFAULT_SFT_DISTILL_DIR / "train.jsonl"))
    parser.add_argument("--validation", default=str(DEFAULT_SFT_DISTILL_DIR / "validation.jsonl"))
    parser.add_argument(
        "--train_source",
        default=str(DEFAULT_SOURCE_DIR / "nemotron_rl_coding_competitive_train_all.jsonl"),
    )
    parser.add_argument(
        "--validation_source",
        default=str(DEFAULT_SOURCE_DIR / "nemotron_rl_coding_competitive_dev.jsonl"),
    )
    parser.add_argument("--split", choices=["train", "validation", "both"], default="both")
    parser.add_argument("--verifier_timeout", type=float, default=10.0)
    parser.add_argument("--skip_verifier", action="store_true")
    parser.add_argument(
        "--progress_every",
        type=int,
        default=50,
        help="Print a checkpoint after this many rows; use 0 to disable.",
    )
    parser.add_argument("--start_row", type=int, default=1, help="First 1-based JSONL row to validate.")
    parser.add_argument("--end_row", type=int, default=None, help="Last 1-based JSONL row to validate.")
    parser.add_argument(
        "--allow_unsafe_verifier",
        action="store_true",
        help="Tests only: run generated programs without the required OS sandbox.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.progress_every < 0:
        raise SystemExit("--progress_every must be >= 0")
    if args.start_row < 1 or (args.end_row is not None and args.end_row < args.start_row):
        raise SystemExit("row bounds must satisfy 1 <= --start_row <= --end_row")
    if not args.skip_verifier and not args.allow_unsafe_verifier and not verifier_sandbox_available():
        raise SystemExit(
            "Formal SFT validation requires the macOS sandbox-exec verifier sandbox; "
            "use --allow_unsafe_verifier only for isolated tests."
        )
    errors: list[str] = []
    if args.split in {"train", "both"}:
        errors.extend(
            validate_file(
                distill_path=Path(args.train),
                source_path=Path(args.train_source),
                verifier_timeout=args.verifier_timeout,
                skip_verifier=args.skip_verifier,
                require_sandbox=not args.allow_unsafe_verifier,
                progress_every=args.progress_every,
                start_row=args.start_row,
                end_row=args.end_row,
            )
        )
    if args.split in {"validation", "both"}:
        errors.extend(
            validate_file(
                distill_path=Path(args.validation),
                source_path=Path(args.validation_source),
                verifier_timeout=args.verifier_timeout,
                skip_verifier=args.skip_verifier,
                require_sandbox=not args.allow_unsafe_verifier,
                progress_every=args.progress_every,
                start_row=args.start_row,
                end_row=args.end_row,
            )
        )
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        raise SystemExit(1)
    print("SFT distillation files validated")


if __name__ == "__main__":
    main()
