#!/usr/bin/env python3
"""Build and validate real stimuli shared by RL anticipation and preferences.

The four methods share one frozen N-prompt source cohort but derive
different training representations. This builder joins those representations
by ``record_id``, renders two deterministic real examples per method, and never
copies verifier inputs or expected outputs into model-facing text.

Formal builds require:

* exact-N SFT ``sft_distill_v1`` rows generated after cohort freeze, plus an
  ``azure_sft_dataset_manifest_v1`` that hash-pins that frozen cohort;
* GRPO ``grpo_rollout_v1`` rows saved by ``train_rl.py``;
* DPO ``dpo_preference_v1`` pairs;
* PPO rollout rows with a real reward-model score, plus the reward-model
  manifest proving that it was trained from the same DPO pairs.

There is intentionally no synthetic fallback. Use ``--check`` to validate an
already-built output without reopening large training artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Callable

import yaml


ROOT = Path(__file__).resolve().parents[1]
LEVELS = ["described_choice", "described_dataset", "example_datapoints"]
METHODS = [
    ("coding.sft", "sft"),
    ("coding.grpo", "grpo"),
    ("coding.dpo", "dpo"),
    ("coding.ppo", "ppo"),
]
OUTPUT_SCHEMA = "rl_training_stimulus_v1"
PROVENANCE_SCHEMA = "rl_example_provenance_v1"
DEFAULT_OUTPUT = ROOT / "data" / "source" / "rl_training_stimuli.json"
FORMAL_DEV_SIZE = 250
FORMAL_HELDOUT_SIZE = 500
FORMAL_MAX_TESTS = 0  # preparation keeps the complete verifier suite
FORMAL_MAX_PROMPT_TOKENS = 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def load_mapping(path: Path) -> dict:
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON/YAML object")
    return data


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise ValueError(f"Missing {label}: {path}")
    return path


def artifact_ref(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256_file(path)}


def resolve_manifest_path(manifest_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def load_cohort(
    manifest_path: Path,
    source_train: Path | None,
) -> tuple[dict, list[dict], Path]:
    manifest = load_json(require_file(manifest_path, "cohort manifest"))
    if manifest.get("schema") != "shared_training_cohort_v1" or manifest.get("status") != "frozen":
        raise ValueError(f"{manifest_path}: formal stimuli require a frozen shared_training_cohort_v1")
    selection = manifest.get("selection", {})
    expected_train_size = selection.get("selected_size")
    if not isinstance(expected_train_size, int) or expected_train_size <= 0:
        raise ValueError("Formal shared cohort must record a positive selected_size")

    outputs = manifest.get("outputs", {})
    train_ref = outputs.get("train", {})
    selected_ref = outputs.get("selected_ids", {})
    if not train_ref.get("path") or not train_ref.get("sha256"):
        raise ValueError(f"{manifest_path}: missing hash-pinned outputs.train")
    if not selected_ref.get("path") or not selected_ref.get("sha256"):
        raise ValueError(f"{manifest_path}: missing hash-pinned outputs.selected_ids")
    cohort_train = resolve_manifest_path(manifest_path, train_ref["path"])
    selected_path = resolve_manifest_path(manifest_path, selected_ref["path"])
    if source_train is not None and source_train.resolve() != cohort_train:
        raise ValueError("--source_train must be the exact train file recorded by the cohort manifest")
    source_train = require_file(cohort_train, "frozen cohort train")
    require_file(selected_path, "frozen cohort selected IDs")
    if sha256_file(source_train) != train_ref["sha256"]:
        raise ValueError("Frozen cohort train hash differs from its manifest")
    if sha256_file(selected_path) != selected_ref["sha256"]:
        raise ValueError("Frozen cohort selected_ids hash differs from its manifest")

    rows = load_jsonl(
        require_file(source_train, f"canonical {expected_train_size:,}-row train split")
    )
    if len(rows) != expected_train_size:
        raise ValueError(
            f"{source_train}: expected {expected_train_size} rows, found {len(rows)}"
        )
    record_ids = [row.get("record_id") for row in rows]
    if None in record_ids or len(record_ids) != len(set(record_ids)):
        raise ValueError(f"{source_train}: record_id values must be present and unique")
    for row in rows:
        if row.get("schema") != "coding_task_v1":
            raise ValueError(f"{source_train}: non-coding_task_v1 row {row.get('record_id')}")
    selected_ids = load_json(selected_path)
    if selected_ids != record_ids:
        raise ValueError("Frozen cohort selected_ids must exactly match train.jsonl order")

    prep_ref = manifest.get("dataset", {}).get("preparation_manifest") or {}
    prep_manifest = prep_ref.get("content")
    if not isinstance(prep_manifest, dict):
        raise ValueError(f"{manifest_path}: missing embedded preparation manifest")
    prep_counts = prep_manifest.get("counts", {})
    if prep_counts.get("dev") != FORMAL_DEV_SIZE or prep_counts.get("heldout") != FORMAL_HELDOUT_SIZE:
        raise ValueError(
            f"Formal preparation must reserve dev={FORMAL_DEV_SIZE}, heldout={FORMAL_HELDOUT_SIZE}"
        )
    prep_extra = prep_manifest.get("extra", {})
    if prep_manifest.get("max_tests") != FORMAL_MAX_TESTS:
        raise ValueError(f"Formal preparation must use max_tests={FORMAL_MAX_TESTS}")
    if prep_extra.get("max_prompt_tokens") != FORMAL_MAX_PROMPT_TOKENS:
        raise ValueError(f"Formal preparation must use max_prompt_tokens={FORMAL_MAX_PROMPT_TOKENS}")
    if not prep_extra.get("length_filter_tokenizer"):
        raise ValueError("Formal preparation manifest must record its tokenizer")
    revision = prep_manifest.get("revision")
    if not revision or revision == "main":
        raise ValueError("Formal preparation manifest must pin an exact dataset revision")
    manifest["_cohort_metadata"] = {
        "dataset_id": prep_manifest.get("dataset_id"),
        "revision": revision,
        "seed": selection.get("seed"),
        "train_size": expected_train_size,
    }
    manifest["_selected_ids"] = selected_ids
    return manifest, rows, source_train


def validate_post_freeze_sft(
    *,
    sft_rows: list[dict],
    sft_train: Path,
    sft_manifest_path: Path,
    sft_manifest: dict,
    cohort_manifest_path: Path,
    cohort_manifest: dict,
) -> None:
    if sft_manifest.get("schema") != "azure_sft_dataset_manifest_v1":
        raise ValueError(
            f"{sft_manifest_path}: expected post-freeze azure_sft_dataset_manifest_v1"
        )
    if sft_manifest.get("status") != "complete":
        raise ValueError(f"{sft_manifest_path}: SFT dataset is not complete")

    output = sft_manifest.get("output", sft_manifest.get("outputs", {}).get("train", {}))
    if not output.get("path") or not output.get("sha256"):
        raise ValueError(f"{sft_manifest_path}: missing hash-pinned SFT train output")
    if resolve_manifest_path(sft_manifest_path, output["path"]) != sft_train.resolve():
        raise ValueError("SFT manifest output path does not match --sft_train")
    expected_ids = cohort_manifest["_selected_ids"]
    observed_ids = [row.get("record_id") for row in sft_rows]
    if observed_ids != expected_ids:
        raise ValueError("SFT rows must exactly match the frozen cohort IDs and order")
    if output.get("rows") != len(expected_ids):
        raise ValueError("SFT manifest row count differs from the frozen cohort")
    if output.get("sha256") != sha256_file(sft_train):
        raise ValueError("SFT train hash differs from its manifest")

    source = sft_manifest.get("source_cohort", {})
    expected_selected_hash = cohort_manifest["outputs"]["selected_ids"]["sha256"]
    checks = {
        "manifest_sha256": (source.get("manifest_sha256"), sha256_file(cohort_manifest_path)),
        "cohort_id": (source.get("cohort_id"), cohort_id(cohort_manifest)),
        "selected_ids_sha256": (source.get("selected_ids_sha256"), expected_selected_hash),
        "selected_ids": (source.get("selected_ids"), expected_ids),
    }
    bad = [key for key, (actual, expected) in checks.items() if actual != expected]
    if bad:
        raise ValueError(
            "SFT manifest does not provenance-pin the frozen cohort: "
            + ", ".join(bad)
        )


def cohort_id(manifest: dict) -> str:
    metadata = manifest["_cohort_metadata"]
    selected_hash = manifest["outputs"]["selected_ids"]["sha256"]
    return (
        f"{metadata['dataset_id']}_n{metadata['train_size']}_seed{metadata['seed']}_"
        f"ids-{selected_hash[:12]}"
    )


def validate_config_parity(configs: dict[str, dict]) -> dict:
    required = ["max_steps", "save_steps", "learning_rate"]
    for method, config in configs.items():
        missing = [key for key in required if key not in config]
        if missing:
            raise ValueError(f"{method} config missing matched fields: {missing}")
        if "per_device_train_batch_size" not in config or "gradient_accumulation_steps" not in config:
            raise ValueError(f"{method} config missing effective-batch fields")
    shared = {
        "max_steps": configs["sft"]["max_steps"],
        "save_steps": configs["sft"]["save_steps"],
        "learning_rate": configs["sft"]["learning_rate"],
        "effective_batch": (
            configs["sft"]["per_device_train_batch_size"]
            * configs["sft"]["gradient_accumulation_steps"]
        ),
    }
    for method, config in configs.items():
        completion_batch = (
            config["per_device_train_batch_size"]
            * config["gradient_accumulation_steps"]
        )
        if method == "grpo":
            num_generations = config.get("num_generations")
            if not num_generations or completion_batch % num_generations:
                raise ValueError(
                    "GRPO completion batch must contain whole prompt groups"
                )
            effective_batch = completion_batch // num_generations
            if config.get("generation_batch_size") != completion_batch:
                raise ValueError(
                    "GRPO generation_batch_size must equal its completion batch "
                    "per optimizer step"
                )
            if config.get("unique_prompts_per_step") != effective_batch:
                raise ValueError("GRPO unique_prompts_per_step is inconsistent")
        else:
            effective_batch = completion_batch
        observed = {
            "max_steps": config["max_steps"],
            "save_steps": config["save_steps"],
            "learning_rate": config["learning_rate"],
            "effective_batch": effective_batch,
        }
        if observed != shared:
            raise ValueError(
                f"{method} config is not matched to SFT: {observed} != {shared}"
            )
    return shared


def render_sft(row: dict, canonical: dict) -> str:
    if row.get("schema") != "sft_distill_v1" or row.get("teacher_pass_rate") != 1.0:
        raise ValueError("SFT examples must be verifier-passing sft_distill_v1 rows")
    messages = row.get("messages") or []
    by_role = {message.get("role"): message.get("content") for message in messages}
    if not by_role.get("user") or not by_role.get("assistant"):
        raise ValueError("SFT row must contain user and assistant messages")
    return (
        f"Prompt:\n{by_role['user']}\n\n"
        f"Supervised target:\n{by_role['assistant']}\n\n"
        "Training feedback: accepted after passing all hidden checks."
    )


def render_dpo(row: dict, canonical: dict) -> str:
    if row.get("schema") != "dpo_preference_v1":
        raise ValueError("DPO examples must use dpo_preference_v1 rows")
    required = ["prompt", "chosen", "rejected", "chosen_reward", "rejected_reward"]
    if any(row.get(key) is None for key in required):
        raise ValueError("DPO row is missing prompt/chosen/rejected/reward fields")
    chosen_reward = float(row["chosen_reward"])
    rejected_reward = float(row["rejected_reward"])
    if chosen_reward <= rejected_reward:
        raise ValueError("DPO chosen reward must exceed rejected reward")
    return (
        f"Prompt:\n{row['prompt']}\n\n"
        f"Preferred completion (verifier reward {chosen_reward:.3f}):\n{row['chosen']}\n\n"
        f"Rejected completion (verifier reward {rejected_reward:.3f}):\n{row['rejected']}"
    )


def render_grpo(rows: list[dict], canonical: dict) -> str:
    for row in rows:
        if row.get("schema") != "grpo_rollout_v1":
            raise ValueError("GRPO examples must use grpo_rollout_v1 rows")
    by_call: dict[str, list[dict]] = {}
    for row in rows:
        by_call.setdefault(str(row.get("reward_call", 0)), []).append(row)
    # Keep the two compared completions inside one actual rollout group. A
    # record_id can recur later in training with a different reward_call. Skip
    # incomplete groups instead of discarding a record that has a later valid
    # group (for example, after an interrupted logging call).
    group_errors = []
    for call_id in sorted(by_call):
        group = by_call[call_id]
        if len(group) < 2:
            group_errors.append(f"reward_call={call_id}: fewer than two completions")
            continue
        try:
            ordered = sorted(
                group,
                key=lambda row: (
                    float(row["reward"]),
                    str(row["completion"]),
                ),
            )
            shown = [ordered[0], ordered[-1]]
            parts = [f"Prompt:\n{canonical['prompt']}"]
            for index, row in enumerate(shown, 1):
                parts.append(
                    f"Group completion {index} (reward {float(row['reward']):.3f}, "
                    f"group mean {float(row['group_mean']):.3f}, advantage "
                    f"{float(row['advantage']):.3f}):\n{row['completion']}"
                )
        except (KeyError, TypeError, ValueError) as exc:
            group_errors.append(f"reward_call={call_id}: {exc}")
            continue
        return "\n\n".join(parts)
    raise ValueError(
        "GRPO example needs one valid rollout group with at least two completions; "
        f"observed {group_errors[:3]}"
    )


def render_ppo(rows: list[dict], canonical: dict) -> str:
    def score(row: dict) -> float:
        value = row.get("reward_model_score")
        return float(value) if value is not None else float("-inf")

    row = sorted(rows, key=lambda candidate: (score(candidate), str(candidate.get("completion", ""))))[-1]
    completion = row.get("completion") or row.get("response")
    score = row.get("reward_model_score")
    if not completion or score is None:
        raise ValueError("PPO rollout needs completion and reward_model_score")
    schema = str(row.get("schema", ""))
    if not schema.startswith("ppo_rollout"):
        raise ValueError("PPO examples must use a ppo_rollout* schema")
    return (
        f"Prompt:\n{canonical['prompt']}\n\n"
        f"Policy completion:\n{completion}\n\n"
        f"Learned reward-model score: {float(score):.3f}."
    )


def select_examples(
    grouped_rows: dict[str, Any],
    canonical_by_id: dict[str, dict],
    renderer: Callable[[Any, dict], str],
    method: str,
    count: int,
    seed: int,
    max_chars: int,
) -> tuple[str, list[str]]:
    eligible = []
    errors = []
    for record_id in sorted(grouped_rows):
        if record_id not in canonical_by_id:
            continue
        try:
            rendered = renderer(grouped_rows[record_id], canonical_by_id[record_id])
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{record_id}: {exc}")
            continue
        if len(rendered) <= max_chars:
            eligible.append((record_id, rendered))
    if len(eligible) < count:
        detail = f" First invalid rows: {errors[:3]}" if errors else ""
        raise ValueError(
            f"{method}: need {count} real examples under {max_chars} chars, "
            f"found {len(eligible)}.{detail}"
        )
    chosen = random.Random(f"{seed}:{method}").sample(eligible, count)
    text = "\n\n".join(
        f"Example {index}:\n{rendered}"
        for index, (_, rendered) in enumerate(chosen, 1)
    )
    return text, [record_id for record_id, _ in chosen]


def group_single_rows(rows: list[dict], label: str) -> dict[str, dict]:
    grouped = {}
    for row in rows:
        record_id = row.get("record_id")
        if not record_id:
            raise ValueError(f"{label}: row missing record_id")
        if record_id in grouped:
            raise ValueError(f"{label}: duplicate record_id {record_id}")
        grouped[record_id] = row
    return grouped


def group_grpo_rows(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        record_id = row.get("record_id")
        if not record_id:
            raise ValueError("GRPO rollout row missing record_id")
        grouped.setdefault(record_id, []).append(row)
    return grouped


def assert_artifact_ids_within_cohort(
    grouped: dict[str, Any],
    cohort_ids: set[str],
    method: str,
) -> None:
    outside = sorted(set(grouped) - cohort_ids)
    if outside:
        raise ValueError(
            f"{method}: artifact contains record_ids outside the frozen cohort: {outside[:10]}"
        )


def described_choice(method: str) -> str:
    return {
        "sft": "Supervised fine-tuning on verified, teacher-written solutions to coding tasks.",
        "grpo": "Group-relative policy optimization using executable coding feedback as reward.",
        "dpo": "Direct preference optimization on solution pairs ranked by executable coding feedback.",
        "ppo": "Proximal policy optimization using a learned model of executable coding feedback.",
    }[method]


def described_dataset(method: str, config: dict, shared: dict, train_size: int) -> str:
    common = (
        "The intervention starts from the same base model and the same frozen set of "
        f"{train_size:,} coding prompts. It runs for {{max_steps}} optimizer steps, saves every "
        "{save_steps} steps, uses learning rate {learning_rate}, and has effective "
        "batch size {effective_batch}. "
    ).format(**shared)
    if method == "sft":
        detail = (
            "Each training row pairs a prompt with a teacher-written Python solution that "
            "passed all executable checks. The model minimizes next-token cross-entropy on "
            "the verified target solution."
        )
    elif method == "grpo":
        detail = (
            f"For each prompt the current policy samples {config['num_generations']} solutions. "
            "Executable pass-rate rewards are normalized within that prompt's group, and the "
            f"policy is updated with group-relative advantages plus KL weight {config['beta']}."
        )
    elif method == "dpo":
        detail = (
            "Each training row contains the same prompt, a higher-reward chosen solution, and "
            "a lower-reward rejected solution derived from executable checks. The model raises "
            f"the chosen solution's relative likelihood using DPO beta {config['beta']}."
        )
    elif method == "ppo":
        detail = (
            "The current policy samples a solution and receives a score from a reward model "
            "trained on the verifier-derived chosen/rejected pairs. PPO uses clipped policy and "
            f"value updates with KL coefficient {config['init_kl_coef']} and "
            f"{config['num_ppo_epochs']} PPO epochs per batch."
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    return common + detail


def validate_reward_model_manifest(path: Path, dpo_pairs: Path) -> dict:
    manifest = load_json(require_file(path, "PPO reward-model manifest"))
    if manifest.get("schema") != "reward_model_manifest_v1":
        raise ValueError(f"{path}: expected reward_model_manifest_v1")
    pair_hash = sha256_file(dpo_pairs)
    if manifest.get("pairs_sha256") != pair_hash:
        raise ValueError("PPO reward model was not trained from the supplied DPO pair file")
    accuracy = manifest.get("held_out_pairwise_accuracy")
    if not isinstance(accuracy, (int, float)) or accuracy < 0.6:
        raise ValueError(
            f"PPO reward-model accuracy gate failed: {accuracy!r} (must be >=0.6)"
        )
    return manifest


def build(args: argparse.Namespace) -> list[dict]:
    manifest, cohort_rows, source_train = load_cohort(args.cohort_manifest, args.source_train)
    canonical_by_id = {row["record_id"]: row for row in cohort_rows}

    sft_rows = load_jsonl(require_file(args.sft_train, "SFT train artifact"))
    sft_manifest = load_json(require_file(args.sft_manifest, "SFT manifest"))
    validate_post_freeze_sft(
        sft_rows=sft_rows,
        sft_train=args.sft_train,
        sft_manifest_path=args.sft_manifest,
        sft_manifest=sft_manifest,
        cohort_manifest_path=args.cohort_manifest,
        cohort_manifest=manifest,
    )

    dpo_rows = load_jsonl(require_file(args.dpo_pairs, "DPO pair artifact"))
    frozen_dpo = manifest.get("outputs", {}).get("artifacts", {}).get("dpo", {})
    if frozen_dpo.get("sha256") != sha256_file(args.dpo_pairs):
        raise ValueError("Supplied DPO pairs are not the cohort's frozen dpo artifact")
    grpo_rows = load_jsonl(require_file(args.grpo_rollouts, "GRPO rollout artifact"))
    ppo_rows = load_jsonl(require_file(args.ppo_rollouts, "PPO rollout artifact"))
    rm_manifest = validate_reward_model_manifest(args.ppo_rm_manifest, args.dpo_pairs)

    config_paths = {
        "sft": args.sft_config,
        "grpo": args.grpo_config,
        "dpo": args.dpo_config,
        "ppo": args.ppo_config,
    }
    configs = {
        method: load_mapping(require_file(path, f"{method.upper()} training config"))
        for method, path in config_paths.items()
    }
    shared = validate_config_parity(configs)

    grouped = {
        "sft": group_single_rows(sft_rows, "SFT"),
        "grpo": group_grpo_rows(grpo_rows),
        "dpo": group_single_rows(dpo_rows, "DPO"),
        "ppo": group_grpo_rows(ppo_rows),
    }
    cohort_ids = set(canonical_by_id)
    for method, method_rows in grouped.items():
        assert_artifact_ids_within_cohort(method_rows, cohort_ids, method.upper())
    renderers = {
        "sft": render_sft,
        "grpo": render_grpo,
        "dpo": render_dpo,
        "ppo": render_ppo,
    }
    artifact_paths = {
        "sft": [args.sft_train, args.sft_manifest],
        "grpo": [args.grpo_rollouts],
        "dpo": [args.dpo_pairs],
        "ppo": [args.ppo_rollouts, args.ppo_rm_manifest],
    }

    source_id = cohort_id(manifest)
    cohort_metadata = manifest["_cohort_metadata"]
    items = []
    for anchor_id, method in METHODS:
        example_text, selected_ids = select_examples(
            grouped[method],
            canonical_by_id,
            renderers[method],
            method,
            args.examples_per_method,
            args.seed,
            args.example_max_chars,
        )
        provenance = {
            "schema": PROVENANCE_SCHEMA,
            "selection_seed": args.seed,
            "selection_policy": "sample record_ids after sorting; skip rather than truncate oversized artifacts",
            "selected_record_ids": selected_ids,
            "source_artifacts": [artifact_ref(path) for path in artifact_paths[method]],
            "training_config": artifact_ref(config_paths[method]),
        }
        if method == "ppo":
            provenance["reward_model_held_out_pairwise_accuracy"] = rm_manifest[
                "held_out_pairwise_accuracy"
            ]
        items.append(
            {
                "schema": OUTPUT_SCHEMA,
                "id": anchor_id,
                "training_method": method,
                "task": "training_method",
                "language": None,
                "language_id": None,
                "option_group": "Training methods",
                "methodology_dataset_status": "standard",
                "source_cohort_id": source_id,
                "source_cohort": {
                    "dataset_id": cohort_metadata["dataset_id"],
                    "revision": cohort_metadata["revision"],
                    "seed": cohort_metadata["seed"],
                    "train_size": cohort_metadata["train_size"],
                    "manifest": artifact_ref(args.cohort_manifest),
                    "train": artifact_ref(source_train),
                },
                "example_datapoints_source": "real_training_data",
                "example_datapoints_provenance": provenance,
                "stimuli": {
                    "described_choice": described_choice(method),
                    "described_dataset": described_dataset(
                        method, configs[method], shared, cohort_metadata["train_size"]
                    ),
                    "example_datapoints": example_text,
                },
            }
        )
    errors = validate(items, cohort_ids)
    if errors:
        raise ValueError("; ".join(errors))
    return items


def validate(items: Any, cohort_ids: set[str] | None = None) -> list[str]:
    errors = []
    if not isinstance(items, list):
        return ["stimuli root must be a list"]
    expected_ids = [anchor_id for anchor_id, _ in METHODS]
    actual_ids = [item.get("id") for item in items if isinstance(item, dict)]
    if actual_ids != expected_ids:
        errors.append(f"stimulus IDs/order must equal {expected_ids}, found {actual_ids}")
    if len(items) != len(METHODS):
        errors.append(f"expected {len(METHODS)} stimuli, found {len(items)}")
    cohort_values = set()
    for item in items:
        if not isinstance(item, dict):
            errors.append("each stimulus must be an object")
            continue
        if item.get("schema") != OUTPUT_SCHEMA:
            errors.append(f"{item.get('id')}: invalid schema")
        cohort_values.add(item.get("source_cohort_id"))
        if item.get("example_datapoints_source") != "real_training_data":
            errors.append(f"{item.get('id')}: formal examples must be real_training_data")
        stimuli = item.get("stimuli") or {}
        if list(stimuli) != LEVELS:
            errors.append(f"{item.get('id')}: stimuli levels/order must equal {LEVELS}")
        if any(not isinstance(stimuli.get(level), str) or not stimuli.get(level).strip() for level in LEVELS):
            errors.append(f"{item.get('id')}: every stimulus level must be non-empty text")
        model_facing = "\n".join(str(stimuli.get(level, "")) for level in LEVELS)
        if "test_inputs" in model_facing or "test_outputs" in model_facing:
            errors.append(f"{item.get('id')}: hidden verifier fields leaked into model-facing text")
        provenance = item.get("example_datapoints_provenance") or {}
        if provenance.get("schema") != PROVENANCE_SCHEMA:
            errors.append(f"{item.get('id')}: missing example provenance")
        selected = provenance.get("selected_record_ids") or []
        if not selected or len(selected) != len(set(selected)):
            errors.append(f"{item.get('id')}: selected record IDs must be non-empty and unique")
        if cohort_ids is not None:
            outside = [record_id for record_id in selected if record_id not in cohort_ids]
            if outside:
                errors.append(f"{item.get('id')}: examples outside frozen cohort: {outside}")
    if len(cohort_values) != 1 or None in cohort_values:
        errors.append("all methods must record one identical non-empty source_cohort_id")
    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="Validate --output only; do not rebuild")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cohort_manifest", type=Path, default=None)
    parser.add_argument("--source_train", type=Path, default=None)
    parser.add_argument("--sft_train", type=Path, default=None)
    parser.add_argument("--sft_manifest", type=Path, default=None)
    parser.add_argument("--dpo_pairs", type=Path, default=None)
    parser.add_argument("--grpo_rollouts", type=Path, default=None)
    parser.add_argument("--ppo_rollouts", type=Path, default=None)
    parser.add_argument("--ppo_rm_manifest", type=Path, default=None)
    parser.add_argument("--sft_config", type=Path, default=ROOT / "rl_training" / "configs" / "sft.yaml")
    parser.add_argument("--grpo_config", type=Path, default=ROOT / "rl_training" / "configs" / "grpo.yaml")
    parser.add_argument("--dpo_config", type=Path, default=ROOT / "rl_training" / "configs" / "dpo.yaml")
    parser.add_argument("--ppo_config", type=Path, default=ROOT / "rl_training" / "configs" / "ppo.yaml")
    parser.add_argument("--examples_per_method", type=int, default=2)
    parser.add_argument("--example_max_chars", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.check:
        items = load_json(require_file(args.output, "RL training stimuli"))
        errors = validate(items)
    else:
        missing_args = [
            name
            for name in (
                "cohort_manifest",
                "sft_train",
                "sft_manifest",
                "dpo_pairs",
                "grpo_rollouts",
                "ppo_rollouts",
                "ppo_rm_manifest",
            )
            if getattr(args, name) is None
        ]
        if missing_args:
            raise SystemExit(
                "Formal build requires explicit real artifacts: " + ", ".join(missing_args)
            )
        items = build(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(items, indent=2) + "\n")
        print(f"Wrote {len(items)} RL training stimuli to {args.output}")
        errors = []
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"RL training stimuli validated: {len(items)} methods x {len(LEVELS)} levels")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from None
