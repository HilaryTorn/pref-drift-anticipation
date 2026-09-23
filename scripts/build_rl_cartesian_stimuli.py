#!/usr/bin/env python3
"""Build formal DPO/GRPO/PPO stimuli for Cartesian elicitation.

The already-audited matched DPO/GRPO stimulus file is reused byte-for-byte for
those two methods.  PPO is added only from a real scored policy-sample artifact
and the reward-model manifest used by the completed formal PPO run. PPO rows
may be either explicitly marked post-training recovery samples or live
stimulus-only online policy rollouts. SFT is excluded from this RL-preference
option file, but remains the non-RL actual-training baseline in the Cartesian
evaluation.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_rl_stimuli import (  # noqa: E402
    LEVELS,
    OUTPUT_SCHEMA,
    PROVENANCE_SCHEMA,
    artifact_ref,
    assert_artifact_ids_within_cohort,
    described_choice,
    described_dataset,
    group_grpo_rows,
    load_cohort,
    load_json,
    load_jsonl,
    load_mapping,
    render_ppo,
    require_file,
    select_examples,
    sha256_file,
    validate_reward_model_manifest,
)


METHODS = (("coding.dpo", "dpo"), ("coding.grpo", "grpo"), ("coding.ppo", "ppo"))
DEFAULT_OUTPUT = ROOT / "data" / "source" / "rl_cartesian_training_stimuli_4b.json"


def effective_batch(method: str, config: dict) -> int:
    completion_batch = int(config["per_device_train_batch_size"]) * int(
        config["gradient_accumulation_steps"]
    )
    if method == "grpo":
        generations = int(config["num_generations"])
        if completion_batch % generations:
            raise ValueError("GRPO completion batch is not divisible by num_generations")
        return completion_batch // generations
    return completion_batch


def validate_threeway_parity(configs: dict[str, dict]) -> dict:
    shared_fields = ("max_steps", "save_steps", "learning_rate")
    reference = {
        **{field: configs["dpo"].get(field) for field in shared_fields},
        "effective_batch": effective_batch("dpo", configs["dpo"]),
    }
    for method, config in configs.items():
        observed = {
            **{field: config.get(field) for field in shared_fields},
            "effective_batch": effective_batch(method, config),
        }
        if observed != reference:
            raise ValueError(f"{method} config mismatch: {observed} != {reference}")
    regularization = {
        "dpo": float(configs["dpo"]["beta"]),
        "grpo": float(configs["grpo"]["beta"]),
        "ppo": float(configs["ppo"]["init_kl_coef"]),
    }
    if len(set(regularization.values())) != 1:
        raise ValueError(f"Reference regularization mismatch: {regularization}")
    return reference


def validate_run_metadata(paths: dict[str, Path], cohort_manifest: Path) -> None:
    metadata = {method: load_json(require_file(path, f"{method} run metadata")) for method, path in paths.items()}
    signatures = set()
    cohort_hash = sha256_file(cohort_manifest)
    for method, payload in metadata.items():
        if payload.get("schema") != "rl_training_run_v1":
            raise ValueError(f"{method}: invalid training metadata schema")
        if payload.get("status") != "complete" or payload.get("algo") != method:
            raise ValueError(f"{method}: formal run is not complete")
        if payload.get("global_step") != 125:
            raise ValueError(f"{method}: expected final global_step=125")
        hub = payload.get("hub_archive") or {}
        if hub.get("cohort_manifest_sha256") != cohort_hash:
            raise ValueError(f"{method}: run metadata does not pin this cohort manifest")
        lora = payload.get("lora") or {}
        signatures.add(
            (
                payload.get("base_model"),
                payload.get("seed"),
                lora.get("r"),
                lora.get("alpha"),
                lora.get("dropout"),
                lora.get("bias"),
                lora.get("target_modules"),
            )
        )
    if len(signatures) != 1:
        raise ValueError("DPO/GRPO/PPO runs do not share base, seed, and LoRA configuration")


def validate(items: object) -> list[str]:
    errors = []
    if not isinstance(items, list):
        return ["stimulus root must be a list"]
    expected_ids = [anchor for anchor, _ in METHODS]
    actual_ids = [item.get("id") for item in items if isinstance(item, dict)]
    if actual_ids != expected_ids:
        errors.append(f"stimulus IDs/order must equal {expected_ids}, found {actual_ids}")
    cohort_ids = set()
    for item in items:
        if not isinstance(item, dict):
            errors.append("every stimulus must be an object")
            continue
        if item.get("schema") != OUTPUT_SCHEMA:
            errors.append(f"{item.get('id')}: invalid schema")
        if item.get("training_method") not in {method for _, method in METHODS}:
            errors.append(f"{item.get('id')}: non-RL method")
        provenance = item.get("example_datapoints_provenance") or {}
        recovery_kind = provenance.get("recovery_kind")
        expected_source = (
            "live_policy_rollout"
            if recovery_kind
            == "live_final_policy_rollout_scored_immediately_by_formal_reward_model"
            else "real_training_data"
        )
        if item.get("example_datapoints_source") != expected_source:
            errors.append(
                f"{item.get('id')}: expected example source {expected_source!r}"
            )
        cohort_ids.add(item.get("source_cohort_id"))
        stimuli = item.get("stimuli") or {}
        if list(stimuli) != LEVELS or any(not str(stimuli.get(level, "")).strip() for level in LEVELS):
            errors.append(f"{item.get('id')}: missing or misordered stimulus levels")
        model_facing = "\n".join(str(stimuli.get(level, "")) for level in LEVELS)
        if "test_inputs" in model_facing or "test_outputs" in model_facing:
            errors.append(f"{item.get('id')}: hidden verifier fields leaked")
        if provenance.get("schema") != PROVENANCE_SCHEMA:
            errors.append(f"{item.get('id')}: invalid example provenance")
    if len(cohort_ids) != 1 or None in cohort_ids:
        errors.append("all methods must share one non-empty frozen cohort ID")
    return errors


def build(args: argparse.Namespace) -> list[dict]:
    cohort_manifest, cohort_rows, _ = load_cohort(args.cohort_manifest, None)
    canonical_by_id = {row["record_id"]: row for row in cohort_rows}
    cohort_ids = set(canonical_by_id)
    matched = load_json(require_file(args.dpo_grpo_stimuli, "matched DPO/GRPO stimuli"))
    matched_by_method = {item.get("training_method"): item for item in matched}
    if set(matched_by_method) != {"dpo", "grpo"}:
        raise ValueError("Matched stimulus file must contain exactly DPO and GRPO")
    if len({item.get("source_cohort_id") for item in matched}) != 1:
        raise ValueError("Matched DPO/GRPO stimuli do not share one cohort")

    run_paths = {
        "dpo": args.dpo_run_metadata,
        "grpo": args.grpo_run_metadata,
        "ppo": args.ppo_run_metadata,
    }
    validate_run_metadata(run_paths, args.cohort_manifest)
    configs = {
        method: load_mapping(require_file(path, f"{method} config"))
        for method, path in {
            "dpo": args.dpo_config,
            "grpo": args.grpo_config,
            "ppo": args.ppo_config,
        }.items()
    }
    shared = validate_threeway_parity(configs)
    rm_manifest = validate_reward_model_manifest(args.ppo_rm_manifest, args.dpo_pairs)
    ppo_rows = load_jsonl(require_file(args.ppo_rollouts, "PPO scored policy samples"))
    grouped_ppo_all = group_grpo_rows(ppo_rows)
    assert_artifact_ids_within_cohort(grouped_ppo_all, cohort_ids, "PPO")
    schemas = {row.get("schema") for row in ppo_rows}
    allowed_schemas = {"ppo_rollout_posthoc_v1", "ppo_rollout_online_v1"}
    if len(schemas) != 1 or not schemas.issubset(allowed_schemas):
        raise ValueError(
            "PPO artifact must contain exactly one supported rollout schema; "
            f"found {sorted(str(schema) for schema in schemas)}"
        )
    ppo_schema = next(iter(schemas))
    if ppo_schema == "ppo_rollout_online_v1":
        if args.ppo_rollout_manifest is None:
            raise ValueError("Online PPO rows require --ppo-rollout-manifest")
        online_manifest = load_json(
            require_file(args.ppo_rollout_manifest, "PPO online-rollout manifest")
        )
        if online_manifest.get("schema") != "ppo_online_rollout_manifest_v1":
            raise ValueError("Invalid PPO online-rollout manifest schema")
        if online_manifest.get("status") != "complete":
            raise ValueError("PPO online-rollout manifest is not complete")
        if online_manifest.get("output", {}).get("sha256") != sha256_file(args.ppo_rollouts):
            raise ValueError("PPO online-rollout rows do not match their manifest")
        if online_manifest.get("optimizer_update_applied") is not False:
            raise ValueError("Stimulus-only PPO collection unexpectedly applied an optimizer update")
        recovery_kind = "live_final_policy_rollout_scored_immediately_by_formal_reward_model"
        selectable_ppo_rows = [
            row
            for row in ppo_rows
            if str(row.get("completion") or row.get("response") or "").rstrip().endswith("```")
        ]
        selection_policy = (
            "sample complete record IDs after sorting; retain max-length-truncated rows "
            "in the raw artifact but exclude them from model-facing examples"
        )
    else:
        online_manifest = None
        recovery_kind = "post_training_policy_sample_scored_by_formal_reward_model"
        selectable_ppo_rows = ppo_rows
        selection_policy = (
            "sample record IDs after sorting; skip rather than truncate oversized artifacts"
        )
    grouped_ppo = group_grpo_rows(selectable_ppo_rows)
    example_text, selected_ids = select_examples(
        grouped_ppo,
        canonical_by_id,
        render_ppo,
        "ppo",
        args.examples_per_method,
        args.seed,
        args.example_max_chars,
    )

    source_template = copy.deepcopy(matched_by_method["dpo"]["source_cohort"])
    ppo_item = {
        "schema": OUTPUT_SCHEMA,
        "id": "coding.ppo",
        "training_method": "ppo",
        "task": "training_method",
        "language": None,
        "language_id": None,
        "option_group": "RL training methods",
        "methodology_dataset_status": "standard",
        "source_cohort_id": matched_by_method["dpo"]["source_cohort_id"],
        "source_cohort": source_template,
        "example_datapoints_source": (
            "live_policy_rollout"
            if ppo_schema == "ppo_rollout_online_v1"
            else "real_training_data"
        ),
        "example_datapoints_provenance": {
            "schema": PROVENANCE_SCHEMA,
            "selection_seed": args.seed,
            "selection_policy": selection_policy,
            "selected_record_ids": selected_ids,
            "source_artifacts": [
                artifact_ref(args.ppo_rollouts),
                artifact_ref(args.ppo_rm_manifest),
                artifact_ref(args.ppo_run_metadata),
            ] + ([artifact_ref(args.ppo_rollout_manifest)] if online_manifest else []),
            "training_config": artifact_ref(args.ppo_config),
            "reward_model_held_out_pairwise_accuracy": rm_manifest["held_out_pairwise_accuracy"],
            "recovery_kind": recovery_kind,
        },
        "stimuli": {
            "described_choice": described_choice("ppo"),
            "described_dataset": described_dataset(
                "ppo", configs["ppo"], shared, cohort_manifest["_cohort_metadata"]["train_size"]
            ),
            "example_datapoints": example_text,
        },
    }
    items = [
        copy.deepcopy(matched_by_method["dpo"]),
        copy.deepcopy(matched_by_method["grpo"]),
        ppo_item,
    ]
    for item in items:
        item["option_group"] = "RL training methods"
    errors = validate(items)
    if errors:
        raise ValueError("; ".join(errors))
    return items


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--check", action="store_true")
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--cohort-manifest", type=Path)
    result.add_argument("--dpo-grpo-stimuli", type=Path, default=ROOT / "data/source/rl_dpo_grpo_training_stimuli.json")
    result.add_argument("--dpo-pairs", type=Path)
    result.add_argument("--ppo-rollouts", type=Path)
    result.add_argument("--ppo-rollout-manifest", type=Path)
    result.add_argument("--ppo-rm-manifest", type=Path)
    result.add_argument("--dpo-run-metadata", type=Path)
    result.add_argument("--grpo-run-metadata", type=Path)
    result.add_argument("--ppo-run-metadata", type=Path)
    result.add_argument("--dpo-config", type=Path, default=ROOT / "rl_training/configs/dpo.yaml")
    result.add_argument("--grpo-config", type=Path, default=ROOT / "rl_training/configs/grpo.yaml")
    result.add_argument("--ppo-config", type=Path, default=ROOT / "rl_training/configs/ppo.yaml")
    result.add_argument("--examples-per-method", type=int, default=2)
    result.add_argument("--example-max-chars", type=int, default=6000)
    result.add_argument("--seed", type=int, default=0)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.check:
        errors = validate(load_json(require_file(args.output, "Cartesian RL stimuli")))
    else:
        required = (
            "cohort_manifest", "dpo_pairs", "ppo_rollouts", "ppo_rm_manifest",
            "dpo_run_metadata", "grpo_run_metadata", "ppo_run_metadata",
        )
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            raise ValueError("Missing formal build arguments: " + ", ".join(missing))
        items = build(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(items, indent=2) + "\n")
        print(f"Wrote {len(items)} Cartesian RL stimuli to {args.output}")
        errors = []
    if errors:
        raise ValueError("; ".join(errors))
    print("Cartesian RL stimuli validated: 3 methods x 3 levels")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from None
