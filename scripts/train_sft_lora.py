#!/usr/bin/env python3
"""Train a LoRA SFT adapter from a prepared coding intervention dataset.

The expected input is the output directory produced by
scripts/prepare_sft_dataset.py, containing train.jsonl and validation.jsonl.
This script keeps training-only imports inside main so offline validation and
CLI discovery do not require GPU training dependencies to import eagerly.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from validate_sft_data import elicitation_snippets, validate_dataset_file  # noqa: E402

CANONICAL_QWEN35_MODELS = {
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
}
FORBIDDEN_QWEN_MARKERS = ("qwen2.5", "qwen25", "2.5-coder")


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def find_intervention(spec: dict[str, Any], intervention_id: str) -> dict[str, Any]:
    for intervention in spec.get("interventions", []):
        if intervention.get("intervention_id") == intervention_id:
            return intervention
    raise SystemExit(f"Unknown intervention_id {intervention_id!r}")


def reject_explicitly_wrong_qwen_family(base_model: str) -> None:
    normalized = base_model.lower()
    if any(marker in normalized for marker in FORBIDDEN_QWEN_MARKERS):
        raise SystemExit(
            "Refusing to train on a non-Qwen3.5 base model. This project is "
            "standardized on Qwen3.5 for baseline, intervention, and post-training "
            "scoring. Use Qwen/Qwen3.5-0.8B, Qwen/Qwen3.5-4B, or Qwen/Qwen3.5-9B."
        )


def is_tiny_dry_run_model(base_model: str) -> bool:
    normalized = base_model.lower()
    return normalized.startswith("hf-internal-testing/") or "tiny-random" in normalized


def validate_qwen35_loaded_config(
    base_model: str,
    base_config: Any,
    *,
    dry_run: bool,
) -> None:
    model_type = getattr(base_config, "model_type", None)
    if model_type == "qwen3_5":
        return
    if dry_run and is_tiny_dry_run_model(base_model):
        return
    allowed = ", ".join(sorted(CANONICAL_QWEN35_MODELS))
    raise SystemExit(
        f"Loaded base model {base_model!r} has model_type={model_type!r}, not 'qwen3_5'. "
        "Production SFT runs for this project must stay in the Qwen3.5 family. "
        f"Use one of: {allowed}. Tiny non-Qwen models are allowed only for --dry_run "
        "script compatibility checks."
    )


def import_training_dependencies() -> dict[str, Any]:
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainerCallback,
            TrainingArguments,
            set_seed,
        )
        from trl import SFTTrainer
    except ImportError as exc:
        missing = exc.name or str(exc)
        raise SystemExit(
            "Missing SFT training dependency "
            f"{missing!r}. Run `uv sync` after installing the updated pyproject, "
            "or install datasets, peft, trl, transformers, accelerate, and torch."
        ) from exc

    try:
        from trl import SFTConfig
    except ImportError:
        SFTConfig = None

    try:
        from trl import DataCollatorForCompletionOnlyLM
    except ImportError:
        DataCollatorForCompletionOnlyLM = None

    return {
        "torch": torch,
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "TaskType": TaskType,
        "AutoConfig": AutoConfig,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "TrainerCallback": TrainerCallback,
        "TrainingArguments": TrainingArguments,
        "SFTConfig": SFTConfig,
        "SFTTrainer": SFTTrainer,
        "DataCollatorForCompletionOnlyLM": DataCollatorForCompletionOnlyLM,
        "set_seed": set_seed,
    }


def render_messages(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    rendered = []
    for message in messages:
        role = message["role"].upper()
        rendered.append(f"{role}:\n{message['content'].strip()}")
    return "\n\n".join(rendered)


def assistant_response_template(tokenizer: Any, fallback_template: str) -> str:
    """Return the exact assistant marker used in rendered training text."""
    if not getattr(tokenizer, "chat_template", None):
        return fallback_template
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "__user_probe__"},
            {"role": "assistant", "content": "__assistant_probe__"},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    probe_index = rendered.find("__assistant_probe__")
    if probe_index < 0:
        raise SystemExit("Tokenizer chat template did not render the assistant probe text.")
    before_probe = rendered[:probe_index]
    user_index = before_probe.find("__user_probe__")
    if user_index < 0:
        raise SystemExit("Tokenizer chat template did not render the user probe text.")
    marker = before_probe[user_index + len("__user_probe__") :]
    marker = marker.lstrip()
    if not marker:
        raise SystemExit("Could not infer assistant response marker from tokenizer chat template.")
    return marker


def validate_response_template(rendered_text: str, response_template: str, where: str) -> None:
    if response_template not in rendered_text:
        preview = rendered_text[:500].replace("\n", "\\n")
        raise SystemExit(
            f"Assistant response template {response_template!r} was not found in {where}. "
            f"Rendered preview: {preview!r}"
        )


def find_token_subsequence(sequence: list[int], subsequence: list[int]) -> int | None:
    if not subsequence or len(subsequence) > len(sequence):
        return None
    latest_match = None
    limit = len(sequence) - len(subsequence) + 1
    for idx in range(limit):
        if sequence[idx : idx + len(subsequence)] == subsequence:
            latest_match = idx
    return latest_match


class LocalCompletionOnlyDataCollator:
    """Mask prompt tokens when TRL no longer ships a completion-only collator."""

    def __init__(self, response_template: str, tokenizer: Any) -> None:
        self.response_template = response_template
        self.tokenizer = tokenizer
        self.response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)
        if not self.response_token_ids:
            raise SystemExit("Assistant response template produced no tokens.")

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        batch = self.tokenizer.pad(examples, return_tensors="pt")
        labels = batch["input_ids"].clone()
        attention_mask = batch.get("attention_mask")
        for row_idx, input_ids in enumerate(batch["input_ids"]):
            token_ids = input_ids.tolist()
            marker_start = find_token_subsequence(token_ids, self.response_token_ids)
            if marker_start is None:
                raise RuntimeError(
                    "Assistant response template tokens were not found in a tokenized example. "
                    "Check tokenizer chat-template rendering and --assistant_template."
                )
            response_start = marker_start + len(self.response_token_ids)
            labels[row_idx, :response_start] = -100
            if attention_mask is not None:
                labels[row_idx][attention_mask[row_idx] == 0] = -100
        batch["labels"] = labels
        return batch


def records_to_dataset(
    dataset_cls: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    *,
    use_conversational_dataset: bool,
    response_template: str | None = None,
) -> Any:
    rendered = []
    for record in records:
        item = {
            "id": record["id"],
            "intervention_id": record["intervention_id"],
            "source": record["source"],
        }
        if use_conversational_dataset:
            item["messages"] = record["messages"]
        else:
            text = render_messages(tokenizer, record["messages"])
            if response_template is not None:
                validate_response_template(text, response_template, record["id"])
            item["text"] = text
        rendered.append(item)
    return dataset_cls.from_list(rendered)


def filtered_kwargs(callable_obj: Any, values: dict[str, Any]) -> dict[str, Any]:
    params = inspect.signature(callable_obj).parameters
    return {key: value for key, value in values.items() if key in params and value is not None}


def resolve_optional_bool(cli_value: bool | None, default_value: Any = False) -> bool:
    if cli_value is None:
        return bool(default_value)
    return bool(cli_value)


def apply_spec_defaults(args: argparse.Namespace, spec: dict[str, Any]) -> None:
    defaults = spec.get("training_defaults", {})
    if args.loss_mode is None:
        args.loss_mode = defaults.get("loss_mode", "assistant_only")
    if args.max_seq_length is None:
        args.max_seq_length = defaults.get("max_seq_length", 2048)

    args.bf16 = resolve_optional_bool(args.bf16, defaults.get("bf16", False))
    args.fp16 = resolve_optional_bool(args.fp16, defaults.get("fp16", False))
    args.gradient_checkpointing = resolve_optional_bool(
        args.gradient_checkpointing,
        defaults.get("gradient_checkpointing", False),
    )
    args.trust_remote_code = resolve_optional_bool(
        args.trust_remote_code,
        defaults.get("trust_remote_code", False),
    )
    if args.bf16 and args.fp16:
        raise SystemExit("Choose at most one precision mode: bf16 and fp16 cannot both be enabled.")


def validate_model_config(args: argparse.Namespace, deps: dict[str, Any]) -> Any:
    try:
        return deps["AutoConfig"].from_pretrained(
            args.base_model,
            trust_remote_code=args.trust_remote_code,
        )
    except Exception as exc:
        raise SystemExit(
            "Could not load the base model config for "
            f"{args.base_model!r}. This usually means the model id is wrong, "
            "the pinned Transformers version does not support the architecture, "
            "or the model requires --trust_remote_code. Dry-run stops here so "
            f"the real training run does not fail later at model load. Original error: {exc}"
        ) from exc


def resolved_eval_strategy(training_args: Any) -> str:
    strategy = getattr(training_args, "eval_strategy", None)
    if strategy is None:
        strategy = getattr(training_args, "evaluation_strategy", None)
    value = getattr(strategy, "value", strategy)
    return str(value or "no").lower()


def build_training_args(args: argparse.Namespace, spec: dict[str, Any], deps: dict[str, Any]) -> Any:
    defaults = spec.get("training_defaults", {})
    config_cls = deps["SFTConfig"] or deps["TrainingArguments"]
    params = inspect.signature(config_cls).parameters

    eval_strategy_key = "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"
    if args.report_to.lower() in {"", "none", "null"}:
        report_to: list[str] | str = []
    else:
        report_to = [item.strip() for item in args.report_to.split(",") if item.strip()]

    values = {
        "output_dir": str(args.output_dir),
        "overwrite_output_dir": args.overwrite_output_dir,
        "num_train_epochs": args.epochs if args.epochs is not None else defaults.get("epochs", 1),
        "learning_rate": args.learning_rate
        if args.learning_rate is not None
        else defaults.get("learning_rate", 2e-4),
        "per_device_train_batch_size": args.per_device_train_batch_size
        if args.per_device_train_batch_size is not None
        else defaults.get("per_device_train_batch_size", 1),
        "per_device_eval_batch_size": args.per_device_eval_batch_size
        if args.per_device_eval_batch_size is not None
        else defaults.get("per_device_eval_batch_size", 1),
        "gradient_accumulation_steps": args.gradient_accumulation_steps
        if args.gradient_accumulation_steps is not None
        else defaults.get("gradient_accumulation_steps", 8),
        "logging_steps": args.logging_steps
        if args.logging_steps is not None
        else defaults.get("logging_steps", 10),
        "save_strategy": args.save_strategy or defaults.get("save_strategy", "epoch"),
        # save_steps/save_total_limit fall back to the spec like every neighbouring field. They
        # used to be CLI-only, so a spec setting save_strategy "steps" without a --save_steps on
        # the command line handed HF save_steps=None in steps mode.
        "save_steps": args.save_steps
        if args.save_steps is not None
        else defaults.get("save_steps", 500),
        eval_strategy_key: args.eval_strategy or defaults.get("eval_strategy", "epoch"),
        "eval_steps": args.eval_steps,
        "save_total_limit": args.save_total_limit
        if args.save_total_limit is not None
        else defaults.get("save_total_limit"),
        # Skip optimizer/scheduler/RNG state in each checkpoint. Measured 2026-08-06/07: those files are roughly two thirds of a checkpoint (~880 MB at 9B), and `push_on_save` uploads the whole directory synchronously inside the training step, so they are paid for three times over -- local disk, upload bandwidth, and blocked GPU while the Hub commits. Nothing in this project reads them: they exist only for --resume_from_checkpoint, which no run has used.
        "save_only_model": args.save_only_model,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "report_to": report_to,
        "remove_unused_columns": True,
        "seed": args.seed if args.seed is not None else defaults.get("seed", 42),
        "data_seed": args.seed if args.seed is not None else defaults.get("seed", 42),
        "max_length": args.max_seq_length
        if args.max_seq_length is not None
        else defaults.get("max_seq_length", 2048),
        "max_seq_length": args.max_seq_length
        if args.max_seq_length is not None
        else defaults.get("max_seq_length", 2048),
        "dataset_text_field": None if args._use_conversational_dataset else "text",
        "packing": args.packing,
        "assistant_only_loss": args._native_assistant_only,
    }
    # filtered_kwargs drops any key the installed config class does not accept, which is the exact mechanism by which a version bump would silently restore the ~590 MB/checkpoint this flag exists to remove. A no-op here looks identical to success in every log, so fail instead of assuming it took.
    if values.get("save_only_model") and "save_only_model" not in inspect.signature(config_cls).parameters:
        raise SystemExit(
            f"--save_only_model was requested but {config_cls.__name__} does not accept it "
            "(installed transformers/TRL is too old). Upgrade, or drop the flag knowingly -- "
            "do not let it silently no-op."
        )
    return config_cls(**filtered_kwargs(config_cls, values))


def describe_target_modules(target_modules: Any) -> Any:
    """Keep 'all-linear' intact; sorting a string would split it into characters."""
    if isinstance(target_modules, str):
        return target_modules
    return sorted(target_modules)


def build_lora_config(args: argparse.Namespace, spec: dict[str, Any], deps: dict[str, Any]) -> Any:
    defaults = spec.get("lora_defaults", {})
    target_modules = args.lora_target_modules or defaults.get("target_modules", [])
    return deps["LoraConfig"](
        r=args.lora_rank if args.lora_rank is not None else defaults.get("rank", 16),
        lora_alpha=args.lora_alpha if args.lora_alpha is not None else defaults.get("alpha", 32),
        lora_dropout=args.lora_dropout
        if args.lora_dropout is not None
        else defaults.get("dropout", 0.05),
        target_modules=target_modules,
        bias="none",
        task_type=deps["TaskType"].CAUSAL_LM,
    )


def configure_loss_mode(args: argparse.Namespace, deps: dict[str, Any]) -> None:
    args._use_conversational_dataset = False
    args._native_assistant_only = False
    args._use_completion_collator = False
    args._completion_collator_source = None
    if args.loss_mode == "full_conversation":
        return

    if deps["DataCollatorForCompletionOnlyLM"] is not None:
        args._use_completion_collator = True
        args._completion_collator_source = "trl"
        return

    args._use_completion_collator = True
    args._completion_collator_source = "local"


def build_data_collator(args: argparse.Namespace, tokenizer: Any, deps: dict[str, Any]) -> Any | None:
    if not args._use_completion_collator:
        return None
    collator_cls = deps["DataCollatorForCompletionOnlyLM"] or LocalCompletionOnlyDataCollator
    collator_values = {
        "response_template": args._response_template,
        "tokenizer": tokenizer,
        "mlm": False,
    }
    return collator_cls(**filtered_kwargs(collator_cls, collator_values))


def validate_completion_collator(
    args: argparse.Namespace,
    tokenizer: Any,
    data_collator: Any | None,
    train_dataset: Any,
    eval_dataset: Any,
) -> None:
    if data_collator is None or not args._use_completion_collator:
        return
    if args.packing:
        raise SystemExit(
            "--packing is incompatible with assistant_only loss in this TRL version "
            "because DataCollatorForCompletionOnlyLM cannot be used with packing. "
            "Use --loss_mode full_conversation or omit --packing."
        )
    if len(train_dataset) == 0:
        raise SystemExit("Training dataset is empty.")
    if len(eval_dataset) == 0:
        raise SystemExit("Validation dataset is empty.")

    for split_name, dataset in [("train", train_dataset), ("validation", eval_dataset)]:
        for idx, record in enumerate(dataset):
            encoded = tokenizer(
                record["text"],
                truncation=True,
                max_length=args.max_seq_length,
                add_special_tokens=False,
            )
            batch = data_collator([encoded])
            labels = batch["labels"][0]
            if hasattr(labels, "tolist"):
                labels = labels.tolist()
            if not any(label != -100 for label in labels):
                example_id = record.get("id", f"{split_name}[{idx}]")
                raise SystemExit(
                    "Completion-only collator masked every token in "
                    f"{split_name} example {example_id!r}. Check max_seq_length, "
                    "assistant response template, and tokenizer chat template."
                )


def validate_trl_preprocessing(
    args: argparse.Namespace,
    tokenizer: Any,
    data_collator: Any | None,
    train_dataset: Any,
) -> None:
    if data_collator is None or args._use_conversational_dataset:
        return
    if len(train_dataset) == 0:
        raise SystemExit("Training dataset is empty.")
    tokenized = train_dataset.select(range(min(len(train_dataset), 2))).map(
        lambda batch: tokenizer(
            batch["text"],
            add_special_tokens=True,
            truncation=True,
            padding=False,
            max_length=args.max_seq_length,
        ),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    try:
        data_collator([tokenized[0]])
    except Exception as exc:
        raise SystemExit(
            "TRL-style tokenization/collation failed during dry-run validation. "
            f"This usually means unused columns were not removed or labels cannot be built: {exc}"
        ) from exc


def wants_epoch_boundary_checkpoints(save_strategy: Any) -> bool:
    """True when save_strategy is "steps", which is the only case that needs the callback.

    Accepts the raw string or HF's enum member, whose `.value` is the string.
    """
    value = getattr(save_strategy, "value", save_strategy)
    return str(value).lower() == "steps"


def make_epoch_boundary_checkpoint_callback(deps: dict[str, Any]) -> type:
    base_cls = deps["TrainerCallback"]

    class EpochBoundaryCheckpointCallback(base_cls):
        """Force a checkpoint at every epoch boundary, on top of the save_steps cadence.

        HF's save_strategy takes ONE value, so "steps" alone silently skips the epoch
        boundaries whenever they are not multiples of save_steps -- which is the normal case
        here: 700 rows at grad-accum 8 is 87 optimizer steps per epoch, and 87/174/261 are
        none of them multiples of 4. The dense cadence is what makes a steep region
        densifiable without retraining, and the epoch boundaries are what the epoch-budget
        decision is read from, so the run needs both rather than a save_steps that divides 87
        (3 or 29) and would drop the dose-matched steps 4 and 40.

        Setting should_save in on_epoch_end is exactly how DefaultFlowCallback implements
        save_strategy "epoch"; Trainer calls _maybe_log_save_evaluate straight after, so this
        composes with the steps cadence instead of replacing it.
        """

        def on_epoch_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            control.should_save = True
            return control

    return EpochBoundaryCheckpointCallback


def make_huggingface_push_callback(deps: dict[str, Any]) -> type:
    base_cls = deps["TrainerCallback"]

    class HuggingFacePushCallback(base_cls):
        """TrainerCallback-compatible adapter push hook for Hugging Face Hub."""

        def __init__(
            self,
            repo_id: str,
            tokenizer: Any,
            *,
            private: bool = False,
            token: str | None = None,
            push_on_save: bool = False,
            push_on_train_end: bool = True,
            commit_prefix: str = "SFT LoRA",
            path_prefix: str = "",
        ) -> None:
            # path_prefix namespaces this run inside a shared adapter repo (one repo per
            # base model, one folder per intervention), so arms never overwrite each other.
            self.path_prefix = path_prefix.strip("/")
            self.repo_id = repo_id
            self.tokenizer = tokenizer
            self.private = private
            self.token = token
            self.push_on_save = push_on_save
            self.push_on_train_end = push_on_train_end
            self.commit_prefix = commit_prefix
            self._pushed_steps: set[int] = set()

        def _is_world_process_zero(self, state: Any) -> bool:
            return bool(getattr(state, "is_world_process_zero", True))

        def _push_root(self, model: Any, state: Any, suffix: str) -> None:
            if not self._is_world_process_zero(state):
                return
            step = int(getattr(state, "global_step", 0) or 0)
            # Namespace the final adapter under <intervention_id>/ (path_prefix), exactly like
            # the per-checkpoint push above. push_to_hub() writes to the repo root with no
            # path-in-repo, so every arm's final adapter would clobber the others there; save
            # the adapter locally and upload_folder into the arm prefix instead.
            try:
                from huggingface_hub import HfApi
            except ImportError:
                print(
                    "WARNING: huggingface_hub unavailable; could not upload final adapter.",
                    file=sys.stderr,
                )
                return
            import tempfile

            commit_message = f"{self.commit_prefix}: {suffix} step {step}"
            try:
                api = HfApi(token=self.token)
                api.create_repo(
                    repo_id=self.repo_id,
                    repo_type="model",
                    private=self.private,
                    exist_ok=True,
                )
                with tempfile.TemporaryDirectory() as tmpdir:
                    model.save_pretrained(tmpdir, safe_serialization=True)
                    if self.tokenizer is not None:
                        self.tokenizer.save_pretrained(tmpdir)
                    api.upload_folder(
                        folder_path=tmpdir,
                        path_in_repo=self.path_prefix or ".",
                        repo_id=self.repo_id,
                        repo_type="model",
                        commit_message=commit_message,
                    )
            except Exception as exc:
                # A failed final-adapter push must not fail a good run: checkpoint-8 holds the
                # same final weights and is already on the Hub.
                print(
                    f"WARNING: Hugging Face final-adapter push failed "
                    f"(checkpoint-{step} has the same weights): {exc}",
                    file=sys.stderr,
                )

        def _push_checkpoint_dir(self, training_args: Any, state: Any) -> None:
            if not self._is_world_process_zero(state):
                return
            step = int(getattr(state, "global_step", 0) or 0)
            expected_dir = Path(training_args.output_dir) / f"checkpoint-{step}"
            if step not in self._pushed_steps and not expected_dir.exists():
                print(
                    f"WARNING: expected checkpoint directory is missing; "
                    f"skipping Hugging Face checkpoint upload: {expected_dir}",
                    file=sys.stderr,
                )

            try:
                from huggingface_hub import HfApi
            except ImportError:
                print(
                    "WARNING: huggingface_hub unavailable; could not upload checkpoint.",
                    file=sys.stderr,
                )
                return

            # Push every checkpoint dir not yet uploaded, not just the current step: a push that failed on a transient Hub/network error is then retried at the next save instead of being lost for the rest of the run.
            pending: list[tuple[int, Path]] = []
            for candidate in Path(training_args.output_dir).glob("checkpoint-*"):
                suffix = candidate.name.rsplit("-", 1)[-1]
                if candidate.is_dir() and suffix.isdigit() and int(suffix) not in self._pushed_steps:
                    pending.append((int(suffix), candidate))

            api = HfApi(token=self.token)
            for pending_step, checkpoint_dir in sorted(pending):
                try:
                    api.create_repo(
                        repo_id=self.repo_id,
                        repo_type="model",
                        private=self.private,
                        exist_ok=True,
                    )
                    path_in_repo = f"checkpoint-{pending_step}"
                    if self.path_prefix:
                        path_in_repo = f"{self.path_prefix}/{path_in_repo}"
                    api.upload_folder(
                        folder_path=str(checkpoint_dir),
                        path_in_repo=path_in_repo,
                        repo_id=self.repo_id,
                        repo_type="model",
                        commit_message=f"{self.commit_prefix}: checkpoint step {pending_step}",
                    )
                except Exception as exc:
                    # A failed push must never abort training: the checkpoint is still on disk and will be retried at the next save (and once more at train end). Killing the run here would waste the paid GPU time this callback exists to protect.
                    print(
                        f"WARNING: Hugging Face push failed for {checkpoint_dir} "
                        f"(will retry at the next save): {exc}",
                        file=sys.stderr,
                    )
                    break
                self._pushed_steps.add(pending_step)

        def on_save(
            self,
            args: Any,
            state: Any,
            control: Any,
            model: Any | None = None,
            **kwargs: Any,
        ) -> Any:
            if self.push_on_save:
                self._push_checkpoint_dir(args, state)
            return control

        def on_train_end(
            self,
            args: Any,
            state: Any,
            control: Any,
            model: Any | None = None,
            **kwargs: Any,
        ) -> Any:
            # Last chance for any checkpoint whose push failed transiently during the run — after this the box may be terminated and the files are gone.
            if self.push_on_save:
                self._push_checkpoint_dir(args, state)
            if self.push_on_train_end and model is not None:
                self._push_root(model, state, "final")
            return control

    return HuggingFacePushCallback


def validate_training_stack(
    args: argparse.Namespace,
    spec: dict[str, Any],
    deps: dict[str, Any],
    tokenizer: Any,
    train_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
) -> dict[str, Any]:
    configure_loss_mode(args, deps)
    args._response_template = (
        assistant_response_template(tokenizer, args.assistant_template)
        if args._use_completion_collator
        else None
    )
    train_dataset = records_to_dataset(
        deps["Dataset"],
        tokenizer,
        train_records,
        use_conversational_dataset=args._use_conversational_dataset,
        response_template=args._response_template,
    )
    eval_dataset = records_to_dataset(
        deps["Dataset"],
        tokenizer,
        validation_records,
        use_conversational_dataset=args._use_conversational_dataset,
        response_template=args._response_template,
    )
    training_args = build_training_args(args, spec, deps)
    lora_config = build_lora_config(args, spec, deps)
    data_collator = build_data_collator(args, tokenizer, deps)
    validate_completion_collator(args, tokenizer, data_collator, train_dataset, eval_dataset)
    validate_trl_preprocessing(args, tokenizer, data_collator, train_dataset)
    validate_trainer_signature(args, deps)
    callback_cls = make_huggingface_push_callback(deps)
    callbacks = []
    # Under save_strategy "steps" the epoch boundaries are only saved when they happen to be
    # multiples of save_steps. They are not at 700 rows (87 steps/epoch, save_steps 4), and the
    # epoch budget is read from them, so add them back explicitly. Redundant under "epoch".
    if wants_epoch_boundary_checkpoints(getattr(training_args, "save_strategy", None)):
        callbacks.append(make_epoch_boundary_checkpoint_callback(deps)())
    if args.hub_repo_id:
        callbacks.append(
            callback_cls(
                args.hub_repo_id,
                tokenizer,
                private=args.hub_private,
                token=os.environ.get(args.hub_token_env) or os.environ.get("HF_TOKEN"),
                push_on_save=args.push_on_save,
                push_on_train_end=not args.no_push_on_train_end,
                commit_prefix=f"{args.intervention_id} LoRA",
                path_prefix=args.intervention_id,
            )
        )
    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "training_args": training_args,
        "lora_config": lora_config,
        "data_collator": data_collator,
        "callbacks": callbacks,
        "loss_setup": {
            "loss_mode": args.loss_mode,
            "use_conversational_dataset": args._use_conversational_dataset,
            "native_assistant_only": args._native_assistant_only,
            "use_completion_collator": args._use_completion_collator,
            "completion_collator_source": args._completion_collator_source,
            "response_template": args._response_template,
        },
    }


def build_trainer(
    args: argparse.Namespace,
    deps: dict[str, Any],
    model: Any,
    tokenizer: Any,
    training_args: Any,
    train_dataset: Any,
    eval_dataset: Any,
    lora_config: Any,
    data_collator: Any | None,
    callbacks: list[Any],
) -> Any:
    trainer_cls = deps["SFTTrainer"]
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "peft_config": lora_config,
        "data_collator": data_collator,
        "callbacks": callbacks,
        "dataset_text_field": None if args._use_conversational_dataset else "text",
        "max_seq_length": args.max_seq_length,
        "packing": args.packing,
    }
    params = inspect.signature(trainer_cls).parameters
    if "tokenizer" in params:
        trainer_kwargs["tokenizer"] = tokenizer
    if "processing_class" in params:
        trainer_kwargs["processing_class"] = tokenizer
    return trainer_cls(**filtered_kwargs(trainer_cls, trainer_kwargs))


def validate_trainer_signature(args: argparse.Namespace, deps: dict[str, Any]) -> None:
    trainer_cls = deps["SFTTrainer"]
    candidate_kwargs = {
        "model": object(),
        "args": object(),
        "train_dataset": object(),
        "eval_dataset": object(),
        "peft_config": object(),
        "data_collator": None,
        "callbacks": [],
        "dataset_text_field": None if args._use_conversational_dataset else "text",
        "max_seq_length": args.max_seq_length,
        "packing": args.packing,
    }
    params = inspect.signature(trainer_cls).parameters
    if "tokenizer" in params:
        candidate_kwargs["tokenizer"] = object()
    if "processing_class" in params:
        candidate_kwargs["processing_class"] = object()
    filtered = filtered_kwargs(trainer_cls, candidate_kwargs)
    missing = [
        key
        for key in ["model", "args", "train_dataset", "eval_dataset", "peft_config"]
        if key not in filtered
    ]
    if missing:
        raise SystemExit(
            f"Installed SFTTrainer signature is missing expected argument(s): {missing}"
        )


def upload_manifest_to_hub(
    manifest_path: Path,
    repo_id: str,
    token: str | None,
    *,
    path_prefix: str = "",
    private: bool = False,
) -> None:
    # path_prefix namespaces the manifest next to its arm's checkpoints (<intervention_id>/training_manifest.json). At the repo root every arm would write the SAME filename, so training Rust after Java would silently destroy Java's record of training args and metrics.
    path_in_repo = "training_manifest.json"
    if path_prefix.strip("/"):
        path_in_repo = f"{path_prefix.strip('/')}/{path_in_repo}"
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("WARNING: huggingface_hub unavailable; could not upload training manifest.", file=sys.stderr)
        return
    try:
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(manifest_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add SFT training manifest ({path_prefix or 'root'})",
        )
    except Exception as exc:
        # Training already succeeded and the manifest is on disk; a failed upload should be a loud warning, not a nonzero exit that makes a good run look failed.
        print(
            f"WARNING: could not upload training manifest to {repo_id}/{path_in_repo}; "
            f"it is still available locally at {manifest_path}: {exc}",
            file=sys.stderr,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec_path", required=True)
    parser.add_argument("--intervention_id", required=True)
    parser.add_argument("--dataset_dir", required=True, help="Directory with train.jsonl and validation.jsonl")
    parser.add_argument(
        "--base_model",
        default=None,
        help="HF model id or local model path. Defaults to spec.default_base_model when omitted.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=None)
    parser.add_argument("--save_strategy", choices=["no", "steps", "epoch"], default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--eval_strategy", choices=["no", "steps", "epoch"], default=None)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument(
        "--save_only_model",
        action="store_true",
        help=(
            "Save adapter weights only, omitting optimizer/scheduler/RNG state from each "
            "checkpoint (~2/3 of its size). Cuts local disk, upload time, and the blocking "
            "Hub commit inside each training step. Makes --resume_from_checkpoint impossible, "
            "which is why it is a flag rather than the default."
        ),
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help=(
            "Maximum number of local checkpoints to keep. Defaults to no pruning so "
            "small drift trajectories keep every saved checkpoint."
        ),
    )
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use bfloat16 training. Defaults to the training spec when omitted.",
    )
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use float16 training. Defaults to the training spec when omitted.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable gradient checkpointing. Defaults to the training spec when omitted.",
    )
    parser.add_argument("--packing", action="store_true")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--report_to", default="none")
    parser.add_argument(
        "--device_map",
        default=None,
        help="Optional Transformers device_map. Leave unset for normal Trainer/Accelerate placement.",
    )
    parser.add_argument("--torch_dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow remote model code. Defaults to the training spec when omitted.",
    )
    parser.add_argument("--loss_mode", choices=["assistant_only", "full_conversation"], default=None)
    parser.add_argument(
        "--assistant_template",
        default="ASSISTANT:\n",
        help="Fallback assistant marker used only when the tokenizer has no chat template.",
    )
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)
    parser.add_argument("--lora_target_modules", nargs="+", default=None)
    parser.add_argument("--hub_repo_id", default=None, help="HF repo id for adapter pushes, e.g. org/name")
    parser.add_argument("--hub_private", action="store_true")
    parser.add_argument("--hub_token_env", default="HUGGINGFACE_HUB_TOKEN")
    parser.add_argument("--push_on_save", action="store_true")
    parser.add_argument("--no_push_on_train_end", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Validate and render data, but do not load/train model")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    spec_path = Path(args.spec_path)
    dataset_dir = Path(args.dataset_dir)
    args.output_dir = Path(args.output_dir)
    train_path = dataset_dir / "train.jsonl"
    validation_path = dataset_dir / "validation.jsonl"
    if not train_path.exists():
        raise SystemExit(f"Missing train file: {train_path}")
    if not validation_path.exists():
        raise SystemExit(f"Missing validation file: {validation_path}")

    spec = load_json(spec_path)
    intervention = find_intervention(spec, args.intervention_id)
    defaults = spec.get("training_defaults", {})
    if args.base_model is None:
        args.base_model = spec.get("default_base_model")
    if not args.base_model:
        raise SystemExit(
            "No base model supplied. Pass --base_model or set default_base_model in the SFT spec."
        )
    reject_explicitly_wrong_qwen_family(args.base_model)
    apply_spec_defaults(args, spec)

    # The one thing --save_only_model actually breaks. Checkpoints written under it carry no optimizer or RNG state, so resuming from one silently restarts Adam from zero and reseeds the sampler -- it does not crash, it just produces a run that is not the one you think you are continuing. Refuse the combination rather than let that through.
    if args.save_only_model and args.resume_from_checkpoint:
        raise SystemExit(
            "--save_only_model and --resume_from_checkpoint are incompatible: checkpoints saved "
            "without optimizer state cannot be resumed correctly. Drop --save_only_model for a "
            "run you intend to resume."
        )

    # The adapter repo must follow the run, not a name someone typed once: substituting the
    # base model here means 4B adapters can never land in the 0.8B repo, even if someone
    # overrides --base_model and forgets to edit config.yaml. The slug matches the project's
    # model-key style (Qwen/Qwen3.5-0.8B -> qwen35-08b). The intervention is NOT part of the
    # repo id — it namespaces a folder inside it (see hub_path_prefix below), so one repo per
    # base model holds every arm.
    if args.hub_repo_id:
        slug = args.base_model.split("/")[-1].lower().replace(".", "")
        args.hub_repo_id = args.hub_repo_id.replace("{base_model_slug}", slug).replace(
            "{intervention_id}", args.intervention_id
        )

    # Fail on a missing or invalid HF token NOW, before any GPU time is spent: without this the first failure surfaces at the step-1 checkpoint push, after model load and a paid optimizer step. whoami() proves the token authenticates; write access to the org is still only proven at the first actual push. Skipped for --dry_run, which never pushes and may run offline.
    if args.hub_repo_id and not args.dry_run:
        hf_preflight_token = os.environ.get(args.hub_token_env) or os.environ.get("HF_TOKEN")
        try:
            from huggingface_hub import HfApi

            identity = HfApi(token=hf_preflight_token).whoami()
        except Exception as exc:
            raise SystemExit(
                f"Hugging Face push is enabled (hub_repo_id={args.hub_repo_id}) but authentication failed. "
                f"Set {args.hub_token_env} or HF_TOKEN to a write-scope token (or run `hf auth login`) before training. "
                f"Original error: {exc}"
            ) from exc
        print(f"Hugging Face auth OK: pushing as {identity.get('name', '<unknown>')} to {args.hub_repo_id}")

    banned_snippets = elicitation_snippets()
    train_errors, train_summary = validate_dataset_file(train_path, intervention, "train", banned_snippets)
    validation_errors, validation_summary = validate_dataset_file(
        validation_path,
        intervention,
        "validation",
        banned_snippets,
    )
    if train_errors or validation_errors:
        for error in train_errors + validation_errors:
            print(f"ERROR: {error}")
        return 1

    train_records = load_jsonl(train_path)
    validation_records = load_jsonl(validation_path)
    deps = import_training_dependencies()
    deps["set_seed"](args.seed if args.seed is not None else defaults.get("seed", 42))
    base_config = validate_model_config(args, deps)
    validate_qwen35_loaded_config(args.base_model, base_config, dry_run=args.dry_run)

    tokenizer = deps["AutoTokenizer"].from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    stack = validate_training_stack(
        args,
        spec,
        deps,
        tokenizer,
        train_records,
        validation_records,
    )

    dataset_manifest_path = dataset_dir / "manifest.json"
    dataset_manifest = load_json(dataset_manifest_path) if dataset_manifest_path.exists() else None
    manifest = {
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "script": "scripts/train_sft_lora.py",
        "spec_path": str(spec_path),
        "spec_id": spec.get("id"),
        "intervention_id": args.intervention_id,
        "task": intervention.get("task"),
        "language_id": intervention.get("language_id"),
        "base_model": args.base_model,
        "base_model_config": {
            "architectures": getattr(base_config, "architectures", None),
            "model_type": getattr(base_config, "model_type", None),
        },
        "dataset_dir": str(dataset_dir),
        "dataset_manifest": dataset_manifest,
        "train_path": str(train_path),
        "validation_path": str(validation_path),
        "train_sha256": file_sha256(train_path),
        "validation_sha256": file_sha256(validation_path),
        "train_summary": train_summary,
        "validation_summary": validation_summary,
        "loss_mode": args.loss_mode,
        "loss_setup": stack["loss_setup"],
        "hub_repo_id": args.hub_repo_id,
        "hub_private": args.hub_private,
        "push_on_save": args.push_on_save,
        "push_on_train_end": not args.no_push_on_train_end,
    }

    if args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest["status"] = "dry_run"
        manifest["lora_config"] = {
            "rank": stack["lora_config"].r,
            "alpha": stack["lora_config"].lora_alpha,
            "dropout": stack["lora_config"].lora_dropout,
            "target_modules": describe_target_modules(stack["lora_config"].target_modules),
        }
        manifest["training_args"] = stack["training_args"].to_dict()
        manifest_path = args.output_dir / "training_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Dry run passed. Wrote manifest: {manifest_path}")
        return 0

    torch_dtype: str | Any = args.torch_dtype
    if torch_dtype != "auto":
        torch_dtype = getattr(deps["torch"], torch_dtype)
    model_kwargs = {
        "dtype": torch_dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    model = deps["AutoModelForCausalLM"].from_pretrained(args.base_model, **model_kwargs)
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    hf_token = os.environ.get(args.hub_token_env) or os.environ.get("HF_TOKEN")

    trainer = build_trainer(
        args,
        deps,
        model,
        tokenizer,
        stack["training_args"],
        stack["train_dataset"],
        stack["eval_dataset"],
        stack["lora_config"],
        stack["data_collator"],
        stack["callbacks"],
    )

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    eval_metrics = (
        trainer.evaluate()
        if resolved_eval_strategy(stack["training_args"]) != "no"
        else {}
    )

    manifest["status"] = "trained"
    manifest["completed_at_utc"] = now_utc()
    manifest["output_dir"] = str(args.output_dir)
    manifest["lora_config"] = {
        "rank": stack["lora_config"].r,
        "alpha": stack["lora_config"].lora_alpha,
        "dropout": stack["lora_config"].lora_dropout,
        "target_modules": describe_target_modules(stack["lora_config"].target_modules),
    }
    manifest["training_args"] = stack["training_args"].to_dict()
    manifest["train_metrics"] = getattr(train_result, "metrics", {})
    manifest["eval_metrics"] = eval_metrics

    manifest_path = args.output_dir / "training_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    if args.hub_repo_id:
        upload_manifest_to_hub(
            manifest_path,
            args.hub_repo_id,
            hf_token,
            path_prefix=args.intervention_id,
            private=args.hub_private,
        )

    print(f"Training complete. Adapter and manifest written to {args.output_dir}")
    if args.hub_repo_id:
        print(f"Hugging Face adapter push target: {args.hub_repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
