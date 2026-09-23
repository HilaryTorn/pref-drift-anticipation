#!/usr/bin/env python3
"""Sample canonical coding data rows for inspection."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from scripts.rl_training.data.schema import read_coding_jsonl


def _truncate(value: Any, max_chars: int) -> Any:
    if isinstance(value, str):
        return value if len(value) <= max_chars else value[: max_chars - 3] + "..."
    if isinstance(value, list):
        return [_truncate(item, max_chars) for item in value[:3]]
    if isinstance(value, dict):
        return {key: _truncate(item, max_chars) for key, item in value.items()}
    return value


def sample_rows(path: str | Path, *, n: int = 3, seed: int = 0, max_chars: int = 500) -> list[dict[str, Any]]:
    rows = read_coding_jsonl(path)
    if n >= len(rows):
        chosen = list(rows)
    else:
        chosen = random.Random(seed).sample(rows, n)
    return [_truncate(row, max_chars) for row in chosen]


def print_samples(path: str | Path, *, n: int = 3, seed: int = 0, max_chars: int = 500, label: str | None = None) -> None:
    title = label or str(path)
    print(f"[sample] {title}")
    for i, row in enumerate(sample_rows(path, n=n, seed=seed, max_chars=max_chars), start=1):
        print(json.dumps({"sample_index": i, **row}, indent=2, ensure_ascii=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True, help="Canonical coding_task_v1 JSONL")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_chars", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_samples(args.path, n=args.n, seed=args.seed, max_chars=args.max_chars)


if __name__ == "__main__":
    main()
