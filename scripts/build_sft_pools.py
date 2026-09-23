#!/usr/bin/env python3
"""Convert the balanced Magicoder coding dataset into SFT candidate pools.

Reads data/training/coding/magicoder_balanced.jsonl (columns: lang, problem,
solution) and writes chat_messages_jsonl_v1 pools that the SFT harness accepts:

    data/training/<intervention_id>/pool.jsonl
    data/training/<intervention_id>/validation.jsonl

Magicoder is a program-WRITING corpus, so it can only source the write.* half of
the pilot. The task axis (coding.debug.python, coding.explain.python) needs a
different source and is not produced here.

Usage:
    python3 scripts/build_sft_pools.py --language python
    python3 scripts/build_sft_pools.py --language python --pool_count 512
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_sft_data import elicitation_snippets, normalize  # noqa: E402

SOURCE_PATH = ROOT / "data" / "training" / "coding" / "magicoder_balanced.jsonl"
SOURCE_NAME = "ise-uiuc/Magicoder-OSS-Instruct-75K"
SOURCE_LICENSE = "mit"

# Magicoder's `lang` values -> the language_id used in data/source/coding_preferences.json.
LANG_TO_LANGUAGE_ID = {
    "python": "python",
    "java": "java",
    "rust": "rust",
    # Magicoder has no C, so the systems-language arm is C++. The canonical battery already
    # carries coding.write.cpp, so this arm is baselined; C stays an untrained near-neighbor target.
    "cpp": "cpp",
}

# Rendered length guard. The spec's max_seq_length is 2048 tokens; ~3.6 chars/token
# with headroom for the chat template and the empty <think></think> block.
MAX_CHARS = 6000

# Magicoder's `lang` labels are noisy (measured on the balanced corpus: ~0.5% of python,
# ~7.5% of java, ~12% of rust, and ~22% of cpp rows carry a solution fenced in a different
# language — for cpp mostly python- and c-fenced rows). For a language-axis intervention
# that is contamination of the treatment itself, so a row is kept only when its solution's
# code-fence tag matches the expected language. Dropping c-fenced rows from the cpp arm also
# keeps C clean as the untrained near-neighbor probe. Rows with no tagged fence (~1%) are
# dropped too: they carry no positive evidence of the language, and the corpus has thousands
# to spare.
FENCE_RE = re.compile(r"```([A-Za-z+#0-9]+)")
FENCE_ALIASES = {
    "python": {"python", "py", "python3"},
    "java": {"java"},
    "rust": {"rust", "rs"},
    "cpp": {"cpp", "c++", "cxx"},
}


def solution_matches_language(solution: str, language: str) -> bool:
    tags = {tag.lower() for tag in FENCE_RE.findall(solution)}
    return bool(tags & FENCE_ALIASES[language])


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def build(language: str, pool_count: int, validation_count: int, seed: int) -> int:
    language_id = LANG_TO_LANGUAGE_ID.get(language)
    if language_id is None:
        raise SystemExit(
            f"Language {language!r} is not mapped. Known: {sorted(LANG_TO_LANGUAGE_ID)}."
        )
    if not SOURCE_PATH.exists():
        raise SystemExit(f"Missing source dataset: {SOURCE_PATH}")

    intervention_id = f"coding.write.{language_id}"
    rows = [row for row in load_jsonl(SOURCE_PATH) if row.get("lang") == language]
    print(f"{len(rows)} {language} rows in {SOURCE_PATH.name}")

    banned = elicitation_snippets()
    seen: set[str] = set()
    usable: list[dict[str, Any]] = []
    skipped = {"empty": 0, "wrong_language": 0, "too_long": 0, "duplicate": 0, "contaminated": 0}

    for row in rows:
        problem = (row.get("problem") or "").strip()
        solution = (row.get("solution") or "").strip()
        if not problem or not solution:
            skipped["empty"] += 1
            continue
        if not solution_matches_language(solution, language):
            skipped["wrong_language"] += 1
            continue
        if len(problem) + len(solution) > MAX_CHARS:
            skipped["too_long"] += 1
            continue
        key = normalize(problem)
        if key in seen:
            skipped["duplicate"] += 1
            continue
        # Never let elicitation or training-stimulus wording leak into training rows.
        if any(snippet in normalize(f"{problem}\n{solution}") for snippet in banned):
            skipped["contaminated"] += 1
            continue
        seen.add(key)
        usable.append(
            {
                "intervention_id": intervention_id,
                "task": "write",
                "language_id": language_id,
                "messages": [
                    {"role": "user", "content": problem},
                    {"role": "assistant", "content": solution},
                ],
                "source": SOURCE_NAME,
                "license": SOURCE_LICENSE,
            }
        )

    print(f"{len(usable)} usable after filtering; skipped {skipped}")
    needed = pool_count + validation_count
    if len(usable) < needed:
        raise SystemExit(f"Need {needed} rows, only {len(usable)} usable.")

    rng = random.Random(seed)
    selected = rng.sample(usable, needed)
    out_dir = ROOT / "data" / "training" / intervention_id

    for split, chunk, filename in [
        ("pool", selected[:pool_count], "pool.jsonl"),
        ("validation", selected[pool_count:needed], "validation.jsonl"),
    ]:
        records = []
        for idx, record in enumerate(chunk):
            row = dict(record)
            row["id"] = f"{intervention_id}.magicoder_{split}_{idx:05d}"
            row["split"] = split
            records.append(row)
        write_jsonl(out_dir / filename, records)
        print(f"wrote {len(records):4d} -> {(out_dir / filename).relative_to(ROOT)}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", default="python", help="Magicoder lang value (python/java/rust/cpp)")
    parser.add_argument("--pool_count", type=int, default=1000)
    parser.add_argument("--validation_count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return build(args.language, args.pool_count, args.validation_count, args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
