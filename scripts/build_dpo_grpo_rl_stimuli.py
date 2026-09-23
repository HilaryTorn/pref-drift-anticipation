#!/usr/bin/env python3
"""Build real, matched DPO-versus-GRPO preference/anticipation stimuli.

This is deliberately separate from the four-method formal builder. It refuses
to build until both completed runs prove exact coverage of the same frozen
1,000 prompt IDs. For GRPO, K rollout rows belong to one prompt and never count
as K training prompts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rl_training.train_rl import audit_grpo_prompt_budget  # noqa: E402
from scripts.build_rl_stimuli import (
    LEVELS,
    OUTPUT_SCHEMA,
    PROVENANCE_SCHEMA,
    ROOT,
    artifact_ref,
    assert_artifact_ids_within_cohort,
    cohort_id,
    described_choice,
    described_dataset,
    group_grpo_rows,
    group_single_rows,
    load_cohort,
    load_json,
    load_jsonl,
    load_mapping,
    render_dpo,
    render_grpo,
    require_file,
    select_examples,
    sha256_file,
)  # noqa: E402


METHODS = [("coding.grpo", "grpo"), ("coding.dpo", "dpo")]
DEFAULT_OUTPUT = ROOT / "data" / "source" / "rl_dpo_grpo_training_stimuli.json"


def select_latest_complete_grpo_groups(
    rows: list[dict], num_generations: int
) -> tuple[dict[str, list[dict]], dict]:
    """Recover a resumed rollout log without mutating or hiding its history.

    A completed GRPO restart can append a second K-row group for prompts that
    were already written before the interruption.  Group by both record ID and
    reward call, reject every incomplete group, then select the group occurring
    latest in the append-only file for each record ID.
    """
    groups: dict[tuple[str, Any], list[tuple[int, dict]]] = {}
    for row_index, row in enumerate(rows):
        record_id = row.get("record_id")
        reward_call = row.get("reward_call")
        if not record_id or reward_call is None:
            raise ValueError(
                "GRPO rollout recovery requires record_id and reward_call on every row"
            )
        groups.setdefault((str(record_id), reward_call), []).append((row_index, row))

    incomplete = [
        (record_id, reward_call, len(group_rows))
        for (record_id, reward_call), group_rows in groups.items()
        if len(group_rows) != num_generations
    ]
    if incomplete:
        raise ValueError(
            "GRPO rollout recovery refuses incomplete K-rollout groups; "
            f"first mismatches: {incomplete[:5]}"
        )

    candidates: dict[str, list[tuple[int, Any, list[dict]]]] = {}
    for (record_id, reward_call), indexed_rows in groups.items():
        candidates.setdefault(record_id, []).append(
            (indexed_rows[-1][0], reward_call, [row for _, row in indexed_rows])
        )

    selected: dict[str, list[dict]] = {}
    duplicate_record_ids = []
    dropped_groups = []
    for record_id, record_groups in candidates.items():
        record_groups.sort(key=lambda item: item[0])
        _, selected_call, selected_rows = record_groups[-1]
        selected[record_id] = selected_rows
        if len(record_groups) > 1:
            duplicate_record_ids.append(record_id)
            dropped_groups.extend(
                {
                    "record_id": record_id,
                    "reward_call": reward_call,
                    "row_count": len(group_rows),
                }
                for _, reward_call, group_rows in record_groups[:-1]
            )

    audit = {
        "policy": "latest_complete_k_group_by_append_order",
        "input_rows": len(rows),
        "selected_rows": sum(len(group_rows) for group_rows in selected.values()),
        "input_groups": len(groups),
        "selected_groups": len(selected),
        "duplicate_record_id_count": len(duplicate_record_ids),
        "dropped_complete_group_count": len(dropped_groups),
        "duplicate_record_ids": sorted(duplicate_record_ids),
        "dropped_complete_groups": dropped_groups,
    }
    return selected, audit


def validate_run_metadata(
    path: Path,
    *,
    algo: str,
    dataset: Path,
    config: dict,
) -> dict:
    metadata = load_json(require_file(path, f"{algo.upper()} run metadata"))
    if metadata.get("schema") != "rl_training_run_v1":
        raise ValueError(f"{path}: expected rl_training_run_v1")
    if metadata.get("status") != "complete" or metadata.get("global_step") != 125:
        raise ValueError(f"{path}: {algo.upper()} run is not complete at step 125")
    if metadata.get("algo") != algo:
        raise ValueError(f"{path}: expected algo={algo}")
    if metadata.get("dataset_sha256") != sha256_file(dataset):
        raise ValueError(f"{path}: dataset hash does not match {dataset}")
    if metadata.get("trainer_config") != config:
        raise ValueError(f"{path}: recorded trainer config differs from the supplied config")
    return metadata


def validate_matched_configs(dpo: dict, grpo: dict, train_size: int) -> dict:
    shared_keys = ("max_steps", "save_steps", "learning_rate")
    shared = {key: dpo[key] for key in shared_keys}
    if any(grpo[key] != value for key, value in shared.items()):
        raise ValueError("DPO and GRPO optimizer cadence/learning rate do not match")
    dpo_prompt_batch = (
        int(dpo["per_device_train_batch_size"])
        * int(dpo["gradient_accumulation_steps"])
    )
    if dpo_prompt_batch != 8 or dpo["max_steps"] * dpo_prompt_batch != train_size:
        raise ValueError("DPO does not consume 1,000 prompts as 125 x 8")
    grpo_audit = audit_grpo_prompt_budget(grpo, dataset_size=train_size)
    if grpo_audit["unique_prompts_per_step"] != dpo_prompt_batch:
        raise ValueError("DPO and GRPO unique-prompt batches do not match")
    return {**shared, "effective_batch": dpo_prompt_batch}


def validate_exact_grpo_coverage(
    grouped: dict[str, list[dict]], cohort_ids: set[str], num_generations: int
) -> None:
    if set(grouped) != cohort_ids:
        missing = sorted(cohort_ids - set(grouped))
        raise ValueError(
            f"GRPO must cover every frozen prompt exactly once; missing {missing[:10]}"
        )
    bad = []
    for record_id, rows in grouped.items():
        calls = {row.get("reward_call") for row in rows}
        if len(rows) != num_generations or len(calls) != 1 or None in calls:
            bad.append((record_id, len(rows), sorted(str(call) for call in calls)))
    if bad:
        raise ValueError(
            "GRPO requires exactly one complete K-rollout group per prompt; "
            f"first mismatches: {bad[:5]}"
        )


def validate_output(items: Any) -> list[str]:
    errors = []
    if not isinstance(items, list):
        return ["stimuli root must be a list"]
    expected_ids = [anchor_id for anchor_id, _ in METHODS]
    actual_ids = [item.get("id") for item in items if isinstance(item, dict)]
    if actual_ids != expected_ids:
        errors.append(f"stimulus IDs/order must equal {expected_ids}, found {actual_ids}")
    cohorts = set()
    for item in items:
        if not isinstance(item, dict):
            errors.append("each stimulus must be an object")
            continue
        if item.get("schema") != OUTPUT_SCHEMA:
            errors.append(f"{item.get('id')}: invalid schema")
        cohorts.add(item.get("source_cohort_id"))
        if item.get("example_datapoints_source") != "real_training_data":
            errors.append(f"{item.get('id')}: examples are not real training data")
        stimuli = item.get("stimuli") or {}
        if list(stimuli) != LEVELS or any(not stimuli.get(level) for level in LEVELS):
            errors.append(f"{item.get('id')}: incomplete stimulus levels")
        provenance = item.get("example_datapoints_provenance") or {}
        if provenance.get("schema") != PROVENANCE_SCHEMA:
            errors.append(f"{item.get('id')}: missing provenance")
        coverage = provenance.get("training_coverage") or {}
        if coverage.get("unique_prompts") != 1000:
            errors.append(f"{item.get('id')}: unique prompt coverage is not 1,000")
    if len(cohorts) != 1 or None in cohorts:
        errors.append("methods do not share one non-empty cohort ID")
    return errors


def build(args: argparse.Namespace) -> list[dict]:
    cohort_manifest, cohort_rows, source_train = load_cohort(
        args.cohort_manifest, args.source_train
    )
    canonical_by_id = {row["record_id"]: row for row in cohort_rows}
    cohort_ids = set(canonical_by_id)
    train_size = len(cohort_ids)
    if train_size != 1000:
        raise ValueError(f"DPO/GRPO comparison requires 1,000 frozen prompts, found {train_size}")

    dpo_rows = load_jsonl(require_file(args.dpo_pairs, "DPO pairs"))
    grpo_rows = load_jsonl(require_file(args.grpo_rollouts, "GRPO rollouts"))
    dpo_grouped = group_single_rows(dpo_rows, "DPO")
    grpo_grouped = group_grpo_rows(grpo_rows)
    assert_artifact_ids_within_cohort(dpo_grouped, cohort_ids, "DPO")
    assert_artifact_ids_within_cohort(grpo_grouped, cohort_ids, "GRPO")
    if set(dpo_grouped) != cohort_ids:
        raise ValueError("DPO must contain one pair for every frozen prompt ID")

    frozen_dpo = cohort_manifest.get("outputs", {}).get("artifacts", {}).get("dpo", {})
    if frozen_dpo.get("sha256") != sha256_file(args.dpo_pairs):
        raise ValueError("DPO pairs are not the cohort's hash-pinned artifact")

    configs = {
        "dpo": load_mapping(require_file(args.dpo_config, "DPO config")),
        "grpo": load_mapping(require_file(args.grpo_config, "GRPO config")),
    }
    num_generations = int(configs["grpo"]["num_generations"])
    recovery_audit = None
    if args.deduplicate_resumed_rollouts:
        grpo_grouped, recovery_audit = select_latest_complete_grpo_groups(
            grpo_rows, num_generations
        )
        assert_artifact_ids_within_cohort(grpo_grouped, cohort_ids, "GRPO")
    shared = validate_matched_configs(configs["dpo"], configs["grpo"], train_size)
    validate_exact_grpo_coverage(grpo_grouped, cohort_ids, num_generations)

    metadata = {
        "dpo": validate_run_metadata(
            args.dpo_run_metadata,
            algo="dpo",
            dataset=args.dpo_pairs,
            config=configs["dpo"],
        ),
        "grpo": validate_run_metadata(
            args.grpo_run_metadata,
            algo="grpo",
            dataset=source_train,
            config=configs["grpo"],
        ),
    }
    if metadata["dpo"].get("base_model") != metadata["grpo"].get("base_model"):
        raise ValueError("DPO and GRPO do not start from the same base model revision")
    if metadata["dpo"].get("seed") != metadata["grpo"].get("seed"):
        raise ValueError("DPO and GRPO seeds do not match")
    if metadata["dpo"].get("lora") != metadata["grpo"].get("lora"):
        raise ValueError("DPO and GRPO LoRA configurations do not match")

    grouped = {"grpo": grpo_grouped, "dpo": dpo_grouped}
    renderers = {"grpo": render_grpo, "dpo": render_dpo}
    artifacts = {
        "grpo": [args.grpo_rollouts, args.grpo_run_metadata],
        "dpo": [args.dpo_pairs, args.dpo_run_metadata],
    }
    source_id = cohort_id(cohort_manifest)
    cohort_metadata = cohort_manifest["_cohort_metadata"]
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
        rows_per_prompt = num_generations if method == "grpo" else 1
        provenance = {
            "schema": PROVENANCE_SCHEMA,
            "selection_seed": args.seed,
            "selection_policy": "sample record IDs after sorting; never truncate model-facing artifacts",
            "selected_record_ids": selected_ids,
            "source_artifacts": [artifact_ref(path) for path in artifacts[method]],
            "training_config": artifact_ref(
                args.grpo_config if method == "grpo" else args.dpo_config
            ),
            "training_coverage": {
                "unique_prompts": train_size,
                "rows_per_prompt": rows_per_prompt,
                "total_artifact_rows": (
                    sum(len(rows) for rows in grpo_grouped.values())
                    if method == "grpo"
                    else len(dpo_rows)
                ),
            },
        }
        if method == "grpo" and recovery_audit is not None:
            provenance["resume_recovery"] = recovery_audit
        items.append(
            {
                "schema": OUTPUT_SCHEMA,
                "id": anchor_id,
                "training_method": method,
                "task": "training_method",
                "language": None,
                "language_id": None,
                "option_group": "Training algorithms",
                "methodology_dataset_status": "standard",
                "train_size": train_size,
                "optimizer_steps": shared["max_steps"],
                "unique_prompts_per_step": shared["effective_batch"],
                "source_cohort_id": source_id,
                "source_cohort": {
                    "dataset_id": cohort_metadata["dataset_id"],
                    "revision": cohort_metadata["revision"],
                    "seed": cohort_metadata["seed"],
                    "train_size": train_size,
                    "manifest": artifact_ref(args.cohort_manifest),
                    "train": artifact_ref(source_train),
                },
                "example_datapoints_source": "real_training_data",
                "example_datapoints_provenance": provenance,
                "stimuli": {
                    "described_choice": described_choice(method),
                    "described_dataset": described_dataset(
                        method, configs[method], shared, train_size
                    ),
                    "example_datapoints": example_text,
                },
            }
        )
    errors = validate_output(items)
    if errors:
        raise ValueError("; ".join(errors))
    return items


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cohort_manifest", type=Path)
    parser.add_argument("--source_train", type=Path, default=None)
    parser.add_argument("--dpo_pairs", type=Path)
    parser.add_argument("--grpo_rollouts", type=Path)
    parser.add_argument("--dpo_run_metadata", type=Path)
    parser.add_argument("--grpo_run_metadata", type=Path)
    parser.add_argument(
        "--dpo_config", type=Path, default=ROOT / "rl_training/configs/dpo.yaml"
    )
    parser.add_argument(
        "--grpo_config", type=Path, default=ROOT / "rl_training/configs/grpo.yaml"
    )
    parser.add_argument("--examples_per_method", type=int, default=2)
    parser.add_argument("--example_max_chars", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--deduplicate-resumed-rollouts",
        action="store_true",
        help=(
            "Select the last complete K-row group per prompt from an append-only "
            "rollout log and record a full recovery audit in provenance."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.check:
        items = load_json(require_file(args.output, "DPO/GRPO stimuli"))
        errors = validate_output(items)
    else:
        required = (
            "cohort_manifest",
            "dpo_pairs",
            "grpo_rollouts",
            "dpo_run_metadata",
            "grpo_run_metadata",
        )
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            raise SystemExit("Build requires real artifacts: " + ", ".join(missing))
        items = build(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(items, indent=2) + "\n")
        print(f"Wrote {len(items)} matched DPO/GRPO stimuli to {args.output}")
        errors = []
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("DPO/GRPO stimuli validated: 2 methods x 3 levels x 1,000 unique prompts")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from None
