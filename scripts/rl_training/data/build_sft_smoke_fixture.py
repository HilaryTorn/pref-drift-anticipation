#!/usr/bin/env python3
"""Build a disposable verifier-perfect SFT fixture under /tmp.

This is NOT the formal Azure generator.  It reuses already-sampled DPO chosen
completions only when they pass the source prompt's full verifier.  Its sole
purpose is to exercise SFT training, checkpoint reload, and elicitation without
persisting or uploading smoke outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

from scripts.rl_training.prompts import format_coding_prompt
from scripts.rl_training.rewards import run_verifier


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--verifier-timeout", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    tmp_root = Path(tempfile.gettempdir()).resolve()
    if not output_dir.is_relative_to(tmp_root) or output_dir == tmp_root:
        raise SystemExit(f"Smoke fixture output must be a specific directory under {tmp_root}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Smoke fixture output is not empty: {output_dir}")
    if args.size < 1:
        raise SystemExit("--size must be positive")

    prompts = read_jsonl(args.prompts)
    prompt_by_id = {row["record_id"]: row for row in prompts}
    if len(prompt_by_id) != len(prompts):
        raise SystemExit("Prompt source contains duplicate record IDs")

    selected_prompts: list[dict[str, Any]] = []
    sft_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pair in read_jsonl(args.pairs):
        record_id = pair["record_id"]
        source = prompt_by_id.get(record_id)
        if source is None or record_id in seen:
            continue
        chosen = pair["chosen"]
        # Never trust the pair's cached scalar for this fixture.  Re-run the
        # complete source verifier and accept only an exact 1.0 result.
        reward = float(
            run_verifier(
                chosen,
                source["verifier"],
                timeout=args.verifier_timeout,
                max_tests=None,
            )
        )
        if reward != 1.0:
            continue
        seen.add(record_id)
        selected_prompts.append(source)
        sft_rows.append(
            {
                "schema": "sft_distill_v1",
                "record_id": record_id,
                "dataset_id": source["dataset_id"],
                "teacher_model": "smoke:existing-verifier-perfect-completion",
                "teacher_pass_rate": 1.0,
                "messages": [
                    {"role": "user", "content": format_coding_prompt(source["prompt"])},
                    {"role": "assistant", "content": chosen},
                ],
                "metadata": {
                    "teacher_provider": "smoke_fixture_not_azure",
                    "source_cohort_id": "temporary",
                    "attempts": 0,
                },
            }
        )
        if len(sft_rows) == args.size:
            break
    if len(sft_rows) != args.size:
        raise SystemExit(
            f"Only {len(sft_rows)} existing completions passed the full verifier; requested {args.size}"
        )

    cohort_dir = output_dir / "cohort"
    sft_dir = output_dir / "sft"
    cohort_dir.mkdir(parents=True, exist_ok=True)
    sft_dir.mkdir(parents=True, exist_ok=True)
    cohort_train = cohort_dir / "train.jsonl"
    selected_ids_path = cohort_dir / "selected_ids.json"
    sft_train = sft_dir / "train.jsonl"
    write_jsonl(cohort_train, selected_prompts)
    selected_ids = [row["record_id"] for row in selected_prompts]
    selected_ids_path.write_text(json.dumps(selected_ids, indent=2) + "\n")
    write_jsonl(sft_train, sft_rows)

    cohort_manifest = {
        "schema": "shared_training_cohort_v1",
        "status": "frozen",
        "purpose": "temporary_sft_plumbing_smoke_only",
        "selection": {"requested_size": args.size, "selected_size": args.size, "seed": 0},
        "outputs": {
            "train": {"path": "train.jsonl", "rows": args.size, "sha256": sha256(cohort_train)},
            "selected_ids": {"path": "selected_ids.json", "sha256": sha256(selected_ids_path)},
        },
    }
    (cohort_dir / "manifest.json").write_text(json.dumps(cohort_manifest, indent=2) + "\n")
    sft_manifest = {
        "schema": "sft_smoke_fixture_manifest_v1",
        "status": "complete",
        "purpose": "temporary_plumbing_only_not_azure_or_formal_data",
        "source_prompts_sha256": sha256(args.prompts),
        "source_pairs_sha256": sha256(args.pairs),
        "cohort_manifest_sha256": sha256(cohort_dir / "manifest.json"),
        "output": {"path": "train.jsonl", "rows": args.size, "sha256": sha256(sft_train)},
    }
    (sft_dir / "manifest.json").write_text(json.dumps(sft_manifest, indent=2) + "\n")
    print(f"[sft-smoke-fixture] wrote {args.size} fully reverified rows -> {output_dir}")


if __name__ == "__main__":
    main()
