#!/usr/bin/env python3
"""Rebuild a Multi-LCB teacher cache from committed raw responses, offline.

Paid teacher responses are source artifacts. Multi-LCB's output cache is a
derived artifact and stays gitignored. This command recreates that cache on a
Linux verifier without an API key or network call to the teacher provider.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from build_lcb_go_teacher import (
    DEFAULT_LANGUAGE,
    LANGUAGES,
    cache_path,
    default_workdir,
    load_upstream,
    write_json_atomic,
)
from lcb_languages import upstream_language


def contract_hash(contract: object) -> str:
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def load_raw(workdir: Path, k: int) -> tuple[dict[str, dict], str]:
    contract_path = workdir / "teacher_contract.json"
    raw_path = workdir / "teacher_raw.jsonl"
    if not contract_path.exists() or not raw_path.exists():
        raise SystemExit(f"Missing committed teacher contract or raw log in {workdir}")

    contract_record = json.loads(contract_path.read_text())
    digest = contract_record.get("sha256")
    if digest != contract_hash(contract_record.get("contract")):
        raise SystemExit(f"Teacher contract hash mismatch in {contract_path}")

    records: dict[str, dict] = {}
    for line_number, line in enumerate(raw_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("generation_contract_sha256") != digest:
            raise SystemExit(f"{raw_path}:{line_number}: generation contract mismatch")
        custom_id = record.get("custom_id")
        if not custom_id or custom_id in records:
            raise SystemExit(f"{raw_path}:{line_number}: missing or duplicate custom_id {custom_id!r}")
        records[custom_id] = record

    expected_multiple = contract_record["contract"]["benchmark_problem_count"] * k
    if len(records) != expected_multiple:
        raise SystemExit(
            f"{raw_path}: expected {expected_multiple} rows for k={k}, found {len(records)}"
        )
    return records, digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release_version", default="release_v5")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE, choices=LANGUAGES)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--workdir", default=None)
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path) if args.dataset_path else (
        ROOT / "data" / "reference" / "multilcb" / args.release_version
    )
    workdir = Path(args.workdir) if args.workdir else default_workdir(args.language)
    raw, digest = load_raw(workdir, args.k)
    benchmark, format_prompt, upstream_args = load_upstream(
        args.release_version, dataset_path, 0, args.language
    )

    contract = json.loads((workdir / "teacher_contract.json").read_text())["contract"]
    if contract.get("language") != args.language:
        raise SystemExit(
            f"Teacher contract language {contract.get('language')!r} does not match {args.language!r}"
        )
    if contract.get("release_version") != args.release_version:
        raise SystemExit("Teacher contract release does not match --release_version")
    if contract.get("benchmark_problem_count") != len(benchmark):
        raise SystemExit("Teacher contract problem count does not match the staged benchmark")

    from lcb_runner.lm_styles.models_store import LMStyle
    from lcb_runner.utils import extract_code

    rebuilt = []
    for index, problem in enumerate(benchmark):
        messages = format_prompt(problem, LMStyle.VLLMAsync, upstream_language(args.language))
        outputs: list[str] = []
        codes: list[str] = []
        for sample in range(args.k):
            record = raw[f"p{index:05d}_s{sample}"]
            text = record.get("text") or f"[no teacher output: {record.get('stop_reason', 'missing')}]"
            outputs.append(text)
            codes.append(extract_code(text))
        rebuilt.append(problem.insert_output(outputs, codes, messages))

    rebuilt.sort(key=lambda row: row["question_id"])
    output = cache_path(upstream_args, args.k, args.language)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output, rebuilt)
    print(
        f"Rebuilt {len(rebuilt)} {args.language} problems from contract {digest} -> {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
