#!/usr/bin/env python3
"""Prepare canonical coding-RL splits from a registered dataset adapter."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from scripts.rl_training.data.registry import get_adapter, list_adapters
from scripts.rl_training.data.schema import build_manifest, write_jsonl


def build_records(raw_dataset, adapter, max_tests: int) -> list[dict]:
    records, seen = [], set()
    for row in raw_dataset:
        record = adapter.extract_record(dict(row), max_tests=max_tests)
        if record is None:
            continue
        if record.record_id in seen:
            continue
        seen.add(record.record_id)
        records.append(record.to_json())
    return records


def filter_by_prompt_tokens(
    records: list[dict],
    tokenizer_name: str,
    max_prompt_tokens: int,
    tokenizer_revision: str | None = None,
) -> list[dict]:
    """Drop records whose *rendered* prompt (chat template included) exceeds the
    trainer's prompt budget. Filtering here beats truncating at train time:
    a truncated prompt loses the instruction header or part of the problem, and
    the model is then trained on a task it cannot see."""
    from tqdm.auto import tqdm
    from transformers import AutoTokenizer

    from scripts.rl_training.prompts import render_coding_prompt

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=tokenizer_revision,
        trust_remote_code=True,
    )
    kept = [
        record
        for record in tqdm(records, desc="[prep] prompt-length filter", unit="row")
        if len(tokenizer(render_coding_prompt(tokenizer, record["prompt"])).input_ids) <= max_prompt_tokens
    ]
    print(
        f"[prep] prompt-length filter (<= {max_prompt_tokens} tokens, "
        f"tokenizer={tokenizer_name}@{tokenizer_revision or 'default'}): "
        f"kept {len(kept)}/{len(records)}"
    )
    return kept


def write_splits(
    *,
    rows: list[dict],
    dataset_id: str,
    train_size: int,
    dev_size: int,
    heldout_size: int,
    seed: int,
    out_dir: Path,
) -> tuple[dict[str, str], int]:
    reserved = heldout_size + dev_size
    if len(rows) < reserved:
        raise SystemExit(f"Only {len(rows)} usable rows, need at least {reserved} for dev+heldout")

    # ``train_size=0`` means: reserve dev/heldout first, then put every
    # remaining usable record in the candidate training pool.  This is useful
    # for model-specific verifier sweeps; the final fixed-size training cohort
    # is selected later by build_shared_cohort.py.
    actual_train_size = train_size or (len(rows) - reserved)
    need = reserved + actual_train_size
    if len(rows) < need:
        raise SystemExit(f"Only {len(rows)} usable rows, need {need}")

    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    heldout = shuffled[:heldout_size]
    dev = shuffled[heldout_size : heldout_size + dev_size]
    train = shuffled[reserved:need]

    files = {
        "train": (
            f"{dataset_id}_train_all.jsonl"
            if train_size == 0
            else f"{dataset_id}_train_{actual_train_size}.jsonl"
        ),
        "dev": f"{dataset_id}_dev.jsonl",
        "heldout": f"{dataset_id}_heldout.jsonl",
    }
    write_jsonl(out_dir / files["train"], train)
    write_jsonl(out_dir / files["dev"], dev)
    write_jsonl(out_dir / files["heldout"], heldout)
    return files, actual_train_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_id", default="nemotron_rl_coding_competitive")
    parser.add_argument("--revision", default="main", help="PIN this to an exact HF commit for reproducibility")
    parser.add_argument("--split", default=None, help="HF split name; defaults to adapter.default_split")
    parser.add_argument(
        "--train_size",
        type=int,
        default=1000,
        help="Training rows to write; 0 uses every usable row left after dev/heldout",
    )
    parser.add_argument("--dev_size", type=int, default=250)
    parser.add_argument("--heldout_size", type=int, default=500)
    parser.add_argument(
        "--max_tests",
        type=int,
        default=0,
        help="Verifier tests stored per prompt; 0 preserves the complete suite",
    )
    parser.add_argument(
        "--max_prompt_tokens", type=int, default=0,
        help="Drop rows whose rendered prompt exceeds this many tokens (0 = keep all). "
        "Should match max_prompt_length in the trainer configs so no training prompt is truncated.",
    )
    parser.add_argument(
        "--tokenizer", default=None,
        help="HF tokenizer used to measure prompt length; required with --max_prompt_tokens",
    )
    parser.add_argument(
        "--tokenizer_revision",
        default=None,
        help="Pinned tokenizer commit/revision; recorded in the preparation manifest",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", default="data/rl")
    parser.add_argument("--list_datasets", action="store_true")
    parser.add_argument("--sample", type=int, default=0, help="Print N sampled rows per split after writing")
    parser.add_argument("--sample_seed", type=int, default=0)
    parser.add_argument("--sample_max_chars", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_datasets:
        for adapter in list_adapters():
            print(f"{adapter.dataset_id}\t{adapter.dataset_name}\tdefault_split={adapter.default_split}")
        return

    for name in ("train_size", "dev_size", "heldout_size", "max_tests", "max_prompt_tokens"):
        if getattr(args, name) < 0:
            raise SystemExit(f"--{name} must be >= 0")

    from datasets import load_dataset

    adapter = get_adapter(args.dataset_id)
    split = args.split or adapter.default_split
    print(f"[prep] loading {adapter.dataset_name}@{args.revision} split={split}")
    raw_dataset = load_dataset(adapter.dataset_name, split=split, revision=args.revision)
    print(f"[prep] raw rows loaded: {len(raw_dataset)}")

    records = build_records(raw_dataset, adapter, args.max_tests)
    print(f"[prep] usable rows after prompt/IO filter + dedup: {len(records)}")

    if args.max_prompt_tokens:
        if not args.tokenizer:
            raise SystemExit("--max_prompt_tokens requires --tokenizer (use the training base model)")
        records = filter_by_prompt_tokens(
            records,
            args.tokenizer,
            args.max_prompt_tokens,
            args.tokenizer_revision,
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files, actual_train_size = write_splits(
        rows=records,
        dataset_id=args.dataset_id,
        train_size=args.train_size,
        dev_size=args.dev_size,
        heldout_size=args.heldout_size,
        seed=args.seed,
        out_dir=out_dir,
    )
    manifest = build_manifest(
        dataset_id=args.dataset_id,
        dataset_name=adapter.dataset_name,
        revision=args.revision,
        source_split=split,
        seed=args.seed,
        max_tests=args.max_tests,
        raw_rows=len(raw_dataset),
        usable_rows=len(records),
        counts={"train": actual_train_size, "dev": args.dev_size, "heldout": args.heldout_size},
        files=files,
        extra={
            "requested_train_size": args.train_size,
            "max_prompt_tokens": args.max_prompt_tokens,
            "length_filter_tokenizer": args.tokenizer,
            "length_filter_tokenizer_revision": args.tokenizer_revision,
        },
    )
    manifest_path = out_dir / f"{args.dataset_id}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n")
    print(f"[prep] wrote splits + manifest to {out_dir}/  (disjoint, seed={args.seed})")
    if args.sample:
        from scripts.rl_training.data.sample import print_samples

        for split_name, filename in files.items():
            print_samples(
                out_dir / filename,
                n=args.sample,
                seed=args.sample_seed,
                max_chars=args.sample_max_chars,
                label=f"{split_name}: {out_dir / filename}",
            )


if __name__ == "__main__":
    main()
