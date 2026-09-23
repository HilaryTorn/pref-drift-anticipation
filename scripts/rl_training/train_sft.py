#!/usr/bin/env python3
"""Train the verifier-filtered SFT comparison arm on a frozen RL cohort.

The input is an exact-N ``sft_distill_v1`` artifact generated after the
model-specific RL cohort was frozen.  Labels are built explicitly so only the
assistant solution contributes to the causal-LM loss.  Checkpoints are PEFT
adapters under ``<save_dir>/<run_name>/checkpoint-<step>/`` and the final
adapter is also saved at the run root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rl_training.hub_checkpoints import (
    add_formal_hub_arguments,
    prepare_formal_hub_upload,
    prepared_formal_hub_callback,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise SystemExit(f"Expected a JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise SystemExit(f"{path}:{line_no}: row must be a JSON object")
            rows.append(row)
    return rows


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=True, default=str) + "\n")
    os.replace(tmp, path)


def selected_ids_from_cohort(manifest_path: Path, manifest: dict[str, Any]) -> list[str]:
    if manifest.get("schema") != "shared_training_cohort_v1" or manifest.get("status") != "frozen":
        raise SystemExit(f"SFT requires a frozen shared_training_cohort_v1: {manifest_path}")
    selected = manifest.get("outputs", {}).get("selected_ids", {})
    relative = selected.get("path")
    expected_sha = selected.get("sha256")
    if not relative or not expected_sha:
        raise SystemExit(f"Cohort manifest lacks hash-pinned selected_ids: {manifest_path}")
    path = manifest_path.parent / relative
    if sha256(path) != expected_sha:
        raise SystemExit(f"Cohort selected_ids hash mismatch: {path}")
    ids = json.loads(path.read_text())
    if not isinstance(ids, list) or not all(isinstance(value, str) and value for value in ids):
        raise SystemExit(f"Invalid selected_ids list: {path}")
    if len(ids) != len(set(ids)):
        raise SystemExit(f"Duplicate selected_ids in cohort: {path}")
    if len(ids) != manifest.get("selection", {}).get("selected_size"):
        raise SystemExit("Cohort selected_size does not match selected_ids")
    return ids


def validate_sft_rows(rows: list[dict[str, Any]], selected_ids: list[str], dataset_path: Path) -> None:
    ids: list[str] = []
    for line_no, row in enumerate(rows, 1):
        prefix = f"{dataset_path}:{line_no}"
        if row.get("schema") != "sft_distill_v1":
            raise SystemExit(f"{prefix}: expected sft_distill_v1")
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise SystemExit(f"{prefix}: missing record_id")
        ids.append(record_id)
        if row.get("teacher_pass_rate") != 1.0:
            raise SystemExit(f"{prefix}: SFT target is not verifier-perfect")
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 2:
            raise SystemExit(f"{prefix}: messages must contain exactly user and assistant")
        if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
            raise SystemExit(f"{prefix}: messages roles must be user then assistant")
        if not all(isinstance(message.get("content"), str) and message["content"] for message in messages):
            raise SystemExit(f"{prefix}: message content must be non-empty text")
    if ids != selected_ids:
        raise SystemExit(
            "SFT rows must match the frozen cohort IDs exactly and in order; "
            f"dataset={len(ids)} cohort={len(selected_ids)}"
        )


def validate_dataset_manifest(
    manifest_path: Path,
    dataset_path: Path,
    selected_ids: list[str],
    cohort_manifest_path: Path,
    cohort_manifest: dict[str, Any],
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    if manifest.get("status") != "complete":
        raise SystemExit(f"SFT dataset manifest is not complete: {manifest_path}")
    if manifest.get("schema") not in {
        "azure_sft_dataset_manifest_v1",
        "sft_smoke_fixture_manifest_v1",
    }:
        raise SystemExit(f"Unsupported SFT dataset manifest schema: {manifest_path}")
    output = manifest.get("output", manifest.get("outputs", {}).get("train", {}))
    output_path_raw = output.get("path")
    if not isinstance(output_path_raw, str) or not output_path_raw:
        raise SystemExit(f"SFT dataset manifest lacks output path: {manifest_path}")
    output_path = Path(output_path_raw)
    resolved_output = (
        output_path.resolve()
        if output_path.is_absolute()
        else (manifest_path.parent / output_path).resolve()
    )
    if resolved_output != dataset_path.resolve():
        raise SystemExit("SFT dataset manifest output path differs from --dataset")
    if output.get("sha256") != sha256(dataset_path):
        raise SystemExit(f"SFT dataset hash differs from manifest: {dataset_path}")
    if output.get("rows") != len(selected_ids):
        raise SystemExit("SFT dataset manifest row count differs from frozen cohort")
    cohort_hash = sha256(cohort_manifest_path)
    if manifest["schema"] == "azure_sft_dataset_manifest_v1":
        source = manifest.get("source_cohort", {})
        expected_selected_hash = cohort_manifest.get("outputs", {}).get("selected_ids", {}).get("sha256")
        checks = {
            "manifest_sha256": (source.get("manifest_sha256"), cohort_hash),
            "selected_ids_sha256": (source.get("selected_ids_sha256"), expected_selected_hash),
            "selected_ids": (source.get("selected_ids"), selected_ids),
        }
        bad = [key for key, (actual, expected) in checks.items() if actual != expected]
        if bad:
            raise SystemExit(
                "Azure SFT dataset does not provenance-pin the frozen cohort: "
                + ", ".join(bad)
            )
    elif manifest.get("cohort_manifest_sha256") != cohort_hash:
        raise SystemExit("SFT smoke fixture does not provenance-pin the frozen cohort")
    return manifest


def tokenize_rows(rows: list[dict[str, Any]], tokenizer: Any, max_length: int):
    from datasets import Dataset

    encoded_rows: list[dict[str, list[int]]] = []
    supervised_counts: list[int] = []
    for row in rows:
        messages = row["messages"]
        full_encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        full = full_encoded.get("input_ids") if hasattr(full_encoded, "get") else full_encoded
        prompt_encoded = tokenizer.apply_chat_template(
            messages[:1],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt = prompt_encoded.get("input_ids") if hasattr(prompt_encoded, "get") else prompt_encoded
        if full is None or prompt is None:
            raise SystemExit(f"Chat template returned no input_ids for {row['record_id']}")
        if full[: len(prompt)] != prompt:
            raise SystemExit(f"Chat-template prefix mismatch for {row['record_id']}")
        assistant_tokens = len(full) - len(prompt)
        if assistant_tokens <= 0:
            raise SystemExit(f"No assistant tokens for {row['record_id']}")
        if assistant_tokens > max_length:
            raise SystemExit(
                f"Assistant target alone exceeds max_length for {row['record_id']}: "
                f"{assistant_tokens} > {max_length}"
            )
        if len(full) > max_length:
            raise SystemExit(
                f"Prompt plus assistant target exceeds max_length for {row['record_id']}: "
                f"{len(full)} > {max_length}. Formal SFT does not truncate task context."
            )
        input_ids = full
        labels = [-100] * len(prompt) + input_ids[len(prompt):]
        supervised = sum(label != -100 for label in labels)
        if supervised != assistant_tokens:
            raise SystemExit(f"Assistant-only label construction failed for {row['record_id']}")
        encoded_rows.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
            }
        )
        supervised_counts.append(supervised)
    return Dataset.from_list(encoded_rows), supervised_counts


class AssistantOnlyCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]):
        import torch

        width = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            padding = width - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * padding)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
            batch["labels"].append(feature["labels"] + [-100] * padding)
        return {name: torch.tensor(values, dtype=torch.long) for name, values in batch.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset_manifest", type=Path, required=True)
    parser.add_argument("--cohort_manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--save_dir", type=Path, default=Path("runs"))
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", default="all-linear")
    add_formal_hub_arguments(parser, include_cohort_manifest=False)
    args = parser.parse_args()
    args.algo = "sft"
    if args.resume:
        raise SystemExit(
            "SFT checkpoints intentionally omit optimizer/scheduler/RNG state and cannot "
            "resume training; diagnose/archive the incomplete run and restart cleanly"
        )

    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)
    if cfg.get("algo") != "sft" or cfg.get("assistant_only_loss") is not True:
        raise SystemExit("SFT config must set algo=sft and assistant_only_loss=true")

    output_dir = args.save_dir / args.run_name
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise SystemExit(f"SFT output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    cohort = load_json(args.cohort_manifest)
    selected_ids = selected_ids_from_cohort(args.cohort_manifest, cohort)
    rows = load_jsonl(args.dataset)
    validate_sft_rows(rows, selected_ids, args.dataset)
    dataset_manifest = validate_dataset_manifest(
        args.dataset_manifest,
        args.dataset,
        selected_ids,
        args.cohort_manifest,
        cohort,
    )
    try:
        prepare_formal_hub_upload(args, output_dir=output_dir, algo="sft")
    except ValueError as exc:
        raise SystemExit(f"Formal Hub preflight failed: {exc}") from None

    import numpy as np
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "right"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True, dtype=dtype
    )
    target_modules: str | list[str]
    if args.lora_target_modules == "all-linear":
        target_modules = "all-linear"
    else:
        target_modules = [value.strip() for value in args.lora_target_modules.split(",") if value.strip()]
        if not target_modules:
            raise SystemExit("--lora_target_modules cannot be empty")
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=target_modules,
        ),
    )
    model.config.use_cache = False
    model.print_trainable_parameters()

    train_dataset, supervised_counts = tokenize_rows(rows, tokenizer, cfg["max_length"])
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=cfg["max_steps"],
        save_strategy="steps",
        save_steps=cfg["save_steps"],
        logging_steps=cfg["logging_steps"],
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        save_only_model=True,
        bf16=dtype == torch.bfloat16,
        gradient_checkpointing=True,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
    )

    metadata = {
        "schema": "rl_sft_training_manifest_v1",
        "status": "running",
        "algo": "sft",
        "base_model": args.base_model,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset),
        "dataset_manifest": str(args.dataset_manifest.resolve()),
        "dataset_manifest_sha256": sha256(args.dataset_manifest),
        "dataset_manifest_schema": dataset_manifest["schema"],
        "cohort_manifest": str(args.cohort_manifest.resolve()),
        "cohort_manifest_sha256": sha256(args.cohort_manifest),
        "selected_size": len(selected_ids),
        "selected_ids": selected_ids,
        "seed": args.seed,
        "config": cfg,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": target_modules,
        },
        "assistant_only_loss": True,
        "supervised_tokens": {
            "min": min(supervised_counts),
            "max": max(supervised_counts),
            "total": sum(supervised_counts),
        },
        "hub_archive": {
            "enabled": bool(args.hub_repo_id),
            "repo_id": args.hub_repo_id,
            "private": args.hub_private,
            "base_model_id": args.hub_base_model_id,
            "base_revision": args.hub_base_revision,
            "cohort_manifest_sha256": sha256(args.cohort_manifest),
            "keep_local_checkpoints_during_training": args.hub_keep_local_checkpoints,
        },
    }
    manifest_path = output_dir / "training_manifest.json"
    if args.resume and manifest_path.is_file():
        prior = load_json(manifest_path)
        comparable_keys = (
            "base_model",
            "dataset_sha256",
            "dataset_manifest_sha256",
            "cohort_manifest_sha256",
            "selected_ids",
            "seed",
            "config",
            "lora",
            "assistant_only_loss",
            "hub_archive",
        )
        if any(prior.get(key) != metadata.get(key) for key in comparable_keys):
            raise SystemExit("Cannot resume SFT after model/data/cohort/config drift")
    atomic_json(manifest_path, metadata)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=AssistantOnlyCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    hub_cb = prepared_formal_hub_callback(args, tokenizer)
    if hub_cb:
        trainer.add_callback(hub_cb)
    result = trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    if hub_cb:
        hub_cb.archive_final(output_dir, trainer.state.global_step)
    metadata["status"] = "complete"
    metadata["global_step"] = trainer.state.global_step
    metadata["train_metrics"] = result.metrics
    atomic_json(manifest_path, metadata)
    print(f"[sft] complete: step={trainer.state.global_step} -> {output_dir}")


if __name__ == "__main__":
    main()
