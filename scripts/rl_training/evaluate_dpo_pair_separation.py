#!/usr/bin/env python3
"""Measure whether a LoRA DPO policy learned its verifier-derived pairs.

The primary metric matches TRL's standard DPO diagnostic.  For each pair it
computes

    beta * [(log pi(chosen) - log ref(chosen))
            - (log pi(rejected) - log ref(rejected))]

over completion tokens only.  Positive values mean that the adapter moved the
chosen response up relative to the rejected response compared with M0.

The script evaluates both the frozen training cohort and a deterministic slice
of eligible pairs excluded from that cohort.  It loads M0 once, attaches all
requested LoRA checkpoints, and reuses each reference forward pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Iterable


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _parse_adapters(values: list[str]) -> list[tuple[str, Path]]:
    adapters: list[tuple[str, Path]] = []
    labels: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"--adapter must be LABEL=PATH, got {value!r}")
        label, raw_path = value.split("=", 1)
        path = Path(raw_path).resolve()
        if not label or label in labels:
            raise ValueError(f"adapter labels must be non-empty and unique: {label!r}")
        if not (path / "adapter_config.json").is_file():
            raise ValueError(f"adapter checkpoint is missing adapter_config.json: {path}")
        labels.add(label)
        adapters.append((label, path))
    return adapters


def _heldout_rows(
    all_rows: list[dict], train_rows: list[dict], size: int, seed: int
) -> list[dict]:
    train_ids = {row["record_id"] for row in train_rows}
    candidates = [row for row in all_rows if row["record_id"] not in train_ids]
    if len(candidates) < size:
        raise ValueError(f"only {len(candidates)} non-training pairs, requested {size}")
    # Sorting makes the sample independent of JSONL write order.
    candidates.sort(key=lambda row: row["record_id"])
    return random.Random(seed).sample(candidates, size)


def _chunks(rows: list[dict], size: int) -> Iterable[list[dict]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _tokenize_row(tokenizer, row: dict, max_length: int) -> dict:
    eos = tokenizer.eos_token
    chosen = row["chosen"] if row["chosen"].endswith(eos) else row["chosen"] + eos
    rejected = row["rejected"] if row["rejected"].endswith(eos) else row["rejected"] + eos
    prompt_ids = tokenizer(row["prompt"])["input_ids"]
    chosen_full = tokenizer(row["prompt"] + chosen)["input_ids"]
    rejected_full = tokenizer(row["prompt"] + rejected)["input_ids"]
    if chosen_full[: len(prompt_ids)] != prompt_ids:
        raise ValueError(f"chosen tokenization does not preserve prompt for {row['record_id']}")
    if rejected_full[: len(prompt_ids)] != prompt_ids:
        raise ValueError(f"rejected tokenization does not preserve prompt for {row['record_id']}")
    if len(prompt_ids) >= max_length:
        raise ValueError(f"prompt fills max_length for {row['record_id']}")
    chosen_ids = chosen_full[len(prompt_ids) :]
    rejected_ids = rejected_full[len(prompt_ids) :]
    return {
        "record_id": row["record_id"],
        "prompt_ids": prompt_ids,
        "chosen_ids": chosen_ids,
        "rejected_ids": rejected_ids,
    }


def _selective_log_softmax_fp32(logits, labels, chunk_size: int):
    """Gather token log-probs in fp32 without materializing fp32 full logits."""
    import torch

    rows = []
    for row_logits, row_labels in zip(logits, labels, strict=True):
        pieces = []
        for start in range(0, row_logits.shape[0], chunk_size):
            stop = min(start + chunk_size, row_logits.shape[0])
            chunk = row_logits[start:stop].float()
            targets = row_labels[start:stop].unsqueeze(-1)
            selected = torch.gather(chunk, dim=-1, index=targets).squeeze(-1)
            pieces.append(selected - torch.logsumexp(chunk, dim=-1))
        rows.append(torch.cat(pieces))
    return torch.stack(rows)


def _completion_logps(model, batch: dict, logprob_chunk_size: int) -> tuple[list[float], list[int]]:
    import torch

    model_inputs = {
        "input_ids": batch["input_ids"].to(model.device),
        "attention_mask": batch["attention_mask"].to(model.device),
        "use_cache": False,
    }
    completion_mask = batch["completion_mask"].to(model.device)
    with torch.inference_mode():
        logits = model(**model_inputs).logits[:, :-1, :]
        labels = model_inputs["input_ids"][:, 1:]
        mask = completion_mask[:, 1:]
        # Accumulate the diagnostic in fp32. The sequence dimension is chunked
        # so fp32 conversion does not duplicate the enormous full-vocabulary
        # logits tensor and exhaust an otherwise shared GPU.
        token_logps = _selective_log_softmax_fp32(logits, labels, logprob_chunk_size)
        token_logps = token_logps.masked_fill(mask == 0, 0.0)
        logps = token_logps.sum(dim=1)
        lengths = mask.sum(dim=1)
    return logps.float().cpu().tolist(), lengths.cpu().tolist()


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty metric")
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "mean": statistics.fmean(values),
        "stdev": stdev,
        "standard_error": stdev / math.sqrt(len(values)),
        "min": min(values),
        "max": max(values),
    }


def _finalize(rows: list[dict], beta: float) -> dict:
    margins = [row["dpo_margin"] for row in rows]
    correct = sum(value > 0.0 for value in margins)
    ties = sum(value == 0.0 for value in margins)
    losses = [math.log1p(math.exp(-value)) for value in margins]
    normalized_policy_gaps = [row["policy_normalized_logp_gap"] for row in rows]
    n = len(rows)
    accuracy = correct / n
    return {
        "pairs": n,
        "beta": beta,
        "implicit_reward_accuracy": accuracy,
        "implicit_reward_accuracy_standard_error": math.sqrt(accuracy * (1.0 - accuracy) / n),
        "tie_rate": ties / n,
        "dpo_margin": _summary(margins),
        "dpo_loss": _summary(losses),
        "policy_normalized_logp_gap": _summary(normalized_policy_gaps),
    }


def evaluate_split(model, tokenizer, adapter_labels: list[str], rows: list[dict], args) -> dict:
    import torch
    from trl.trainer.dpo_trainer import DataCollatorForPreference

    tokenized = [_tokenize_row(tokenizer, row, args.max_length) for row in rows]
    collator = DataCollatorForPreference(
        pad_token_id=tokenizer.pad_token_id,
        max_length=args.max_length,
        truncation_mode="keep_start",
    )
    per_adapter: dict[str, list[dict]] = {label: [] for label in adapter_labels}
    base_normalized_gaps: list[float] = []

    for index, examples in enumerate(_chunks(tokenized, args.batch_size), start=1):
        batch = collator(examples)
        # A PEFT model's disable_adapter context exactly matches the reference
        # policy used by TRL when DPO trains a fresh LoRA adapter.
        with model.disable_adapter():
            ref_logps, lengths = _completion_logps(model, batch, args.logprob_chunk_size)
        batch_n = len(examples)
        ref_chosen, ref_rejected = ref_logps[:batch_n], ref_logps[batch_n:]
        chosen_lens, rejected_lens = lengths[:batch_n], lengths[batch_n:]
        base_normalized_gaps.extend(
            chosen / chosen_len - rejected / rejected_len
            for chosen, rejected, chosen_len, rejected_len in zip(
                ref_chosen, ref_rejected, chosen_lens, rejected_lens
            )
        )

        for label in adapter_labels:
            model.set_adapter(label)
            policy_logps, _ = _completion_logps(model, batch, args.logprob_chunk_size)
            policy_chosen, policy_rejected = policy_logps[:batch_n], policy_logps[batch_n:]
            for pos, example in enumerate(examples):
                logratio_gap = (
                    policy_chosen[pos]
                    - ref_chosen[pos]
                    - policy_rejected[pos]
                    + ref_rejected[pos]
                )
                per_adapter[label].append(
                    {
                        "record_id": example["record_id"],
                        "dpo_margin": args.beta * logratio_gap,
                        "policy_normalized_logp_gap": (
                            policy_chosen[pos] / chosen_lens[pos]
                            - policy_rejected[pos] / rejected_lens[pos]
                        ),
                    }
                )
        if index % args.progress_every == 0 or index * args.batch_size >= len(tokenized):
            print(f"[dpo-pairs] evaluated {min(index * args.batch_size, len(tokenized))}/{len(tokenized)}")
        del batch

    base_accuracy = sum(value > 0.0 for value in base_normalized_gaps) / len(base_normalized_gaps)
    return {
        "m0": {
            "pairs": len(base_normalized_gaps),
            "normalized_logp_chosen_accuracy": base_accuracy,
            "normalized_logp_gap": _summary(base_normalized_gaps),
        },
        "checkpoints": {
            label: _finalize(per_adapter[label], args.beta) for label in adapter_labels
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--train-pairs", type=Path, required=True)
    parser.add_argument("--all-pairs", type=Path, required=True)
    parser.add_argument("--adapter", action="append", required=True, help="LABEL=PATH; repeat per checkpoint")
    parser.add_argument("--heldout-size", type=int, default=1000)
    parser.add_argument(
        "--train-size",
        type=int,
        default=0,
        help="Evaluate this many frozen training pairs; 0 means all (formal default)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--logprob-chunk-size", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if (
        args.batch_size < 1
        or args.heldout_size < 1
        or args.train_size < 0
        or args.logprob_chunk_size < 1
    ):
        parser.error("--batch-size/--heldout-size must be positive and --train-size non-negative")
    adapters = _parse_adapters(args.adapter)
    train_rows = _read_jsonl(args.train_pairs)
    all_rows = _read_jsonl(args.all_pairs)
    heldout_rows = _heldout_rows(all_rows, train_rows, args.heldout_size, args.seed)
    evaluated_train_rows = train_rows[: args.train_size] if args.train_size else train_rows

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the 4B checkpoint diagnostic")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True, dtype=dtype
    )
    first_label, first_path = adapters[0]
    model = PeftModel.from_pretrained(
        base, first_path, adapter_name=first_label, is_trainable=False
    )
    for label, path in adapters[1:]:
        model.load_adapter(path, adapter_name=label, is_trainable=False)
    model.to("cuda")
    model.eval()

    result = {
        "schema": "dpo_pair_separation_v1",
        "base_model": str(Path(args.base_model).resolve()),
        "train_pairs": {
            "path": str(args.train_pairs.resolve()),
            "sha256": _sha256(args.train_pairs),
            "rows": len(train_rows),
        },
        "all_pairs": {
            "path": str(args.all_pairs.resolve()),
            "sha256": _sha256(args.all_pairs),
            "rows": len(all_rows),
        },
        "heldout_selection": {
            "definition": "eligible all-pairs rows excluding frozen training record_ids",
            "seed": args.seed,
            "rows": len(heldout_rows),
            "record_ids_sha256": hashlib.sha256(
                "\n".join(row["record_id"] for row in heldout_rows).encode()
            ).hexdigest(),
        },
        "adapters": {label: str(path) for label, path in adapters},
        "settings": {
            "beta": args.beta,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "logprob_chunk_size": args.logprob_chunk_size,
            "dtype": str(dtype),
        },
        "splits": {},
    }
    for split, rows in (("train", evaluated_train_rows), ("heldout", heldout_rows)):
        print(f"[dpo-pairs] evaluating {split}: {len(rows)} pairs")
        result["splits"][split] = evaluate_split(
            model, tokenizer, [label for label, _ in adapters], rows, args
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[dpo-pairs] wrote {args.output}")


if __name__ == "__main__":
    main()
