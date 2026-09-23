#!/usr/bin/env python3
"""Build a deterministic shared prompt/pair cohort for training smoke tests.

This intentionally operates on existing pilot artifacts rather than the formal
full-sweep manifests.  It is only a plumbing fixture: GRPO and PPO consume the
same selected coding tasks, while DPO and PPO's reward model consume the exact
corresponding preference pairs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--eval-prompts", type=Path, required=True)
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--eval-size", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.size < 1 or args.eval_size < 1:
        raise SystemExit("--size and --eval-size must both be positive")

    prompts = read_jsonl(args.prompts)
    pairs = read_jsonl(args.pairs)
    eval_prompts = read_jsonl(args.eval_prompts)
    prompt_by_id = {row["record_id"]: row for row in prompts}
    if len(prompt_by_id) != len(prompts):
        raise SystemExit(f"Duplicate record_id in {args.prompts}")

    selected_pairs: list[dict] = []
    seen: set[str] = set()
    for pair in pairs:
        record_id = pair["record_id"]
        if record_id in seen or record_id not in prompt_by_id:
            continue
        if pair.get("schema") != "dpo_preference_v1":
            raise SystemExit(f"Unexpected pair schema for {record_id}: {pair.get('schema')!r}")
        if float(pair["chosen_reward"]) <= float(pair["rejected_reward"]):
            raise SystemExit(f"Invalid reward ordering for pair {record_id}")
        selected_pairs.append(pair)
        seen.add(record_id)
        if len(selected_pairs) == args.size:
            break

    if len(selected_pairs) != args.size:
        raise SystemExit(
            f"Only found {len(selected_pairs)} prompt-aligned unique pairs; requested {args.size}"
        )
    selected_prompts = [prompt_by_id[pair["record_id"]] for pair in selected_pairs]
    selected_eval = eval_prompts[: args.eval_size]
    if len(selected_eval) != args.eval_size:
        raise SystemExit(f"Only found {len(selected_eval)} eval prompts; requested {args.eval_size}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.jsonl"
    pairs_path = args.output_dir / "dpo.jsonl"
    eval_path = args.output_dir / "eval.jsonl"
    write_jsonl(train_path, selected_prompts)
    write_jsonl(pairs_path, selected_pairs)
    write_jsonl(eval_path, selected_eval)

    selected_ids = [row["record_id"] for row in selected_prompts]
    manifest = {
        "schema": "training_smoke_cohort_v1",
        "purpose": "plumbing_only_not_formal_experiment_data",
        "size": args.size,
        "eval_size": args.eval_size,
        "selected_record_ids": selected_ids,
        "sources": {
            "prompts": {"path": str(args.prompts), "sha256": sha256(args.prompts)},
            "pairs": {"path": str(args.pairs), "sha256": sha256(args.pairs)},
            "eval_prompts": {"path": str(args.eval_prompts), "sha256": sha256(args.eval_prompts)},
        },
        "outputs": {
            "train": {"path": train_path.name, "sha256": sha256(train_path)},
            "dpo": {"path": pairs_path.name, "sha256": sha256(pairs_path)},
            "eval": {"path": eval_path.name, "sha256": sha256(eval_path)},
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    tmp_manifest = manifest_path.with_suffix(".json.tmp")
    tmp_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    tmp_manifest.replace(manifest_path)
    print(f"[smoke-data] wrote {args.size} shared prompts/pairs -> {args.output_dir}")


if __name__ == "__main__":
    main()
