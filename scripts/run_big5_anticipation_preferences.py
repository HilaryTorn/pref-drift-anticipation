#!/usr/bin/env python3
"""Run the Phase 2 Big5 trait-axis anticipation and training-preference batteries.

Anchors are the four Big5 SFT interventions (big5_trait_sft_v1: big5.openness.high/low, big5.extraversion.high/low). Anticipation runs as two batteries sharing those anchors, mirroring Phase 1's split (portraits ask about being a person; coding targets ask about doing a task, so they need different templates): --battery values forecasts all 90 Schwartz value portraits (the DIRECT Phase 2 forecast, full three-level sweep); --battery coding forecasts the 26 coding/activity targets from coding_drift_anticipation (9 write languages + 8 debug/explain + 9 non-coding activities) — reverse-direction spillover, asked once at described_dataset per the Phase 1 tiered-target design. Training preference is the pairwise A/B dataset choice over the four arms, Thurstonian per concreteness level, mirroring coding_training_preference.

The concreteness ladder doubles as a label-to-behavior ladder (see scripts/build_big5_elicitations.py): described_choice names the trait, described_dataset is behavioral and label-free, example_datapoints are real records. The full three-level sweep is the spec'd design; the choice-vs-dataset contrast measures the label effect.

This mirrors run_values_anticipation.py / run_rl_anticipation_preferences.py deliberately: bespoke thin runner, shared machinery untouched — anticipation goes through run_anticipation with injected portrait targets, preferences through compute_utilities exactly as run_pairwise does.

Examples:
    python scripts/run_big5_anticipation_preferences.py anticipation                      # export values-battery prompts only
    python scripts/run_big5_anticipation_preferences.py anticipation --model_key <key> --level all
    python scripts/run_big5_anticipation_preferences.py anticipation --battery coding --output_path prompts/big5_coding_drift_anticipation.jsonl
    python scripts/run_big5_anticipation_preferences.py training-preference --model_key <key> --level described_dataset
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compute_utilities.naming import default_results_dir, timestamped  # noqa: E402

from scripts.elicitation_experiment_specs import LEVELS  # noqa: E402
from scripts.run_elicitations import (  # noqa: E402
    config_uses_logprobs,
    default_pairwise_raw_dump_path,
    grouped_option_records,
    load_experiment_spec,
    load_spec,
    load_training_stimuli,
    pairwise_reasoning_prompt,
    reasoning_enabled,
    run_anticipation,
    write_pairwise_manifest,
)
from scripts.run_values_anticipation import load_values_targets  # noqa: E402

BATTERIES = {
    # values: portrait targets are injected (the shared resolver doesn't know values ids); full level sweep is the design.
    # coding: targets resolve through the shared coding/other resolver via the spec's target_tasks_source; distant targets, described_dataset once.
    "values": {
        "spec_name": "big5_anticipation",
        "experiment_spec": "data/experiment_specs/big5_drift_anticipation.json",
        "default_level": "all",
    },
    "coding": {
        "spec_name": "big5_coding_anticipation",
        "experiment_spec": "data/experiment_specs/big5_coding_drift_anticipation.json",
        "default_level": "primary",
    },
}


async def run_big5_anticipation(args: argparse.Namespace) -> None:
    battery = BATTERIES[args.battery]
    if args.experiment_spec is None:
        args.experiment_spec = battery["experiment_spec"]
    if args.level is None:
        args.level = battery["default_level"]
    experiment_spec = load_experiment_spec(args.experiment_spec)
    targets = load_values_targets(experiment_spec["all_target_ids"]) if args.battery == "values" else None
    await run_anticipation(
        args,
        spec_name=battery["spec_name"],
        targets=targets,
        output_label=f"big5 {args.battery} anticipation",
    )


def big5_preferences_options(level: str, spec_name: str) -> dict[str, list[str]]:
    if level not in LEVELS:
        raise ValueError(f"Unknown level: {level}. Must be one of {LEVELS}")
    items = load_training_stimuli(spec_name)
    return {"Training data": [item["stimuli"][level] for item in items]}


def big5_preferences_manifest(options: dict, level: str, spec_name: str) -> list[dict]:
    spec = load_spec(spec_name)
    items = load_training_stimuli(spec_name)
    actual_ids = [item["id"] for item in items]
    if actual_ids != spec["anchor_ids"]:
        raise ValueError(f"Big5 stimulus IDs/order {actual_ids} do not match anchor_ids {spec['anchor_ids']}")
    by_text = {item["stimuli"][level]: item for item in items}
    if len(by_text) != len(items):
        raise ValueError(f"Big5 preference stimuli must be unique at level {level}")

    records = []
    for scorer_id, option_record in enumerate(grouped_option_records(options)):
        description = option_record["description"]
        item = by_text.get(description)
        if item is None:
            raise ValueError(f"Big5 preferences option {scorer_id} has no source stimulus")
        records.append(
            {
                "scorer_option_id": scorer_id,
                "canonical_id": item["id"],
                "trait": item["trait"],
                "trait_level": item["trait_level"],
                "example_datapoints_source": item["example_datapoints_source"],
                "example_datapoints_provenance": item["example_datapoints_provenance"],
                "concreteness_level": level,
                "category": option_record["category"],
                "description": description,
            }
        )
    return records


async def run_big5_training_preference(args: argparse.Namespace) -> dict:
    spec_name = "big5_training_preference"
    spec = load_spec(spec_name)
    options = big5_preferences_options(args.level, spec_name)
    manifest_records = big5_preferences_manifest(options, args.level, spec_name)

    invalid = sorted(
        {
            (record["canonical_id"], record.get("example_datapoints_source"))
            for record in manifest_records
            if record.get("example_datapoints_source") != "real_training_data"
        }
    )
    if invalid:
        formatted = ", ".join(f"{anchor_id}={source}" for anchor_id, source in invalid)
        raise ValueError("Refusing big5 training preference with synthetic/missing examples: " + formatted)

    save_dir = args.save_dir or default_results_dir(args.model_key, "pairs")
    default_suffix = f"{spec['mode']}_{args.level}"
    save_suffix = timestamped(args.save_suffix or default_suffix, not args.no_timestamp)
    use_reasoning = reasoning_enabled(args)
    config_key = args.config_key or (
        "thurstonian_active_learning" if use_reasoning else "thurstonian_active_learning_logprobs"
    )
    if use_reasoning and config_uses_logprobs(config_key):
        raise ValueError(
            "--reasoning on requires a sampling config so raw reasoning can be captured; "
            f"got logprobs config {config_key!r}. Use --reasoning off for the cheap logprobs path."
        )
    write_pairwise_manifest(save_dir, save_suffix, spec["mode"], manifest_records, args.level)
    create_agent_config_key = args.create_agent_config_key or (
        "default_with_reasoning" if use_reasoning else "default"
    )
    prompt_template = pairwise_reasoning_prompt(spec["prompt_template"]) if use_reasoning else spec["prompt_template"]
    raw_dump_path = None
    if use_reasoning:
        raw_dump_path = args.raw_dump_path or default_pairwise_raw_dump_path(save_dir, save_suffix)

    from scripts.compute_utilities.compute_utilities import compute_utilities

    return await compute_utilities(
        options_list=options,
        model_key=args.model_key,
        models_config_path=args.models_config_path,
        create_agent_config_path=str(ROOT / "compute_utilities" / "create_agent.yaml"),
        create_agent_config_key=create_agent_config_key,
        compute_utilities_config_path=str(ROOT / "compute_utilities" / "compute_utilities.yaml"),
        compute_utilities_config_key=config_key,
        comparison_prompt_template=prompt_template,
        with_reasoning=use_reasoning,
        use_logprobs=not use_reasoning,
        utility_model_seed=args.seed,
        raw_dump_path=raw_dump_path,
        raw_dump_metadata={
            "mode": spec["mode"],
            "model_key": args.model_key,
            "save_suffix": save_suffix,
            "concreteness_level": args.level,
            "reasoning": use_reasoning,
            "compute_utilities_config_key": config_key,
            "create_agent_config_key": create_agent_config_key,
        },
        save_dir=save_dir,
        save_suffix=save_suffix,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    anticipation = sub.add_parser("anticipation", help="export or query the big5 forecast batteries")
    anticipation.add_argument(
        "--battery",
        default="values",
        choices=list(BATTERIES),
        help="values = 90 Schwartz portraits (direct Phase 2 forecast, full level sweep); coding = 26 coding/activity spillover targets (described_dataset once)",
    )
    anticipation.add_argument("--model_key", default=None, help="Optional key in config.yaml; omit to only export prompts")
    anticipation.add_argument("--experiment_spec", default=None, help="Defaults to the battery's experiment spec")
    anticipation.add_argument(
        "--level",
        default=None,
        choices=["primary", "all", *LEVELS],
        help="Defaults per battery: values = all (full sweep; the anchors are the trained interventions), coding = primary (distant targets, asked once)",
    )
    anticipation.add_argument("--target_set", default="all", choices=["primary", "all", "spillover"])
    anticipation.add_argument("--output_path", default=None, help="Optional JSONL prompt export path")
    anticipation.add_argument("--K", type=int, default=4, help="Samples per rotation; 3 rotations x K=4 = 12 samples per forecast, matching the executed coding/UE/values forecast design")
    anticipation.add_argument(
        "--label_rotations",
        type=int,
        default=3,
        help="Ask each forecast under this many cyclic label rotations and pool the samples. The default 3 puts each of MORE/LESS/SAME in each position exactly once, matching the registered values-battery design; pass 1 only to reproduce single-order archived runs or score a pre-v3 M0 checkpoint. Total samples per forecast = K x rotations.",
    )
    anticipation.add_argument("--timeout", type=int, default=5)
    anticipation.add_argument("--system_message", default="You are a helpful assistant.")
    anticipation.add_argument(
        "--reasoning",
        choices=["on", "off"],
        default="on",
        help="When on, export/query prompts that request brief reasoning plus a final MORE/LESS/SAME answer",
    )
    anticipation.add_argument("--results_path", default=None)
    anticipation.add_argument("--raw_dump_path", default=None)
    anticipation.add_argument("--save_dir", default=None)
    anticipation.add_argument("--models_config_path", default=None)
    anticipation.add_argument("--create_agent_config_key", default=None)
    anticipation.add_argument(
        "--no-timestamp",
        dest="no_timestamp",
        action="store_true",
        help="Write to a fixed filename with no UTC timestamp; a re-run then overwrites it.",
    )

    training = sub.add_parser("training-preference", help="score big5 training-data preference")
    training.add_argument("--model_key", required=True, help="Key in config.yaml")
    training.add_argument("--level", required=True, choices=LEVELS)
    training.add_argument("--models_config_path", default=None)
    training.add_argument("--config_key", default=None)
    training.add_argument("--create_agent_config_key", default=None)
    training.add_argument("--save_dir", default=None)
    training.add_argument("--save_suffix", default=None)
    training.add_argument("--seed", type=int, default=42, help="Utility-model fit and pair-sampling seed")
    training.add_argument(
        "--reasoning",
        choices=["on", "off"],
        default="on",
        help="When on, use sampled reasoning responses and write a raw JSONL sidecar; when off, use cheap logprobs",
    )
    training.add_argument("--raw_dump_path", default=None)
    training.add_argument(
        "--no-timestamp",
        dest="no_timestamp",
        action="store_true",
        help="Write to fixed filenames with no UTC timestamp; a re-run then overwrites them.",
    )
    return parser


async def main() -> None:
    args = build_parser().parse_args()
    if args.command == "anticipation":
        await run_big5_anticipation(args)
    elif args.command == "training-preference":
        await run_big5_training_preference(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    asyncio.run(main())
