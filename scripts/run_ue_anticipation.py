#!/usr/bin/env python3
"""Run the UE cross-domain anticipation battery.

Kept deliberately separate from the coding anticipation CLI. Anchors are the four trained
coding interventions (writing Python/C++/Java/Rust); targets are all 81 Utility-Engineering
options, forecasting cross-domain spillover from coding training onto the broad UE preference
battery. Ternary MORE/LESS/SAME, dataset-description level only.

The UE target type lives here, not in scripts/run_elicitations.py: this runner resolves the
UE options itself and injects them through run_anticipation's generic ``targets`` hook, so the
shared coding runner and its experiment spec are untouched. This mirrors how the RL arm
(run_rl_anticipation_preferences.py) reuses the anticipation engine without extending its CLI.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.elicitation_experiment_specs import LEVELS  # noqa: E402
from scripts.run_elicitations import (  # noqa: E402
    add_stimuli_version_argument,
    coding_spec_name,
    load_experiment_spec,
    load_json,
    run_anticipation,
    stimuli_version,
)


UE_ANTICIPATION_SPEC = "ue_anticipation"
UE_OPTIONS_PATH = ROOT / "data" / "reference" / "ue_options_subset.json"

# UE options carry no ids in the source file, so a UE target id encodes the category slug and
# the option's index within that category: ``ue.<slug>.<index>``.
UE_CATEGORY_BY_SLUG = {
    "work_activities": "Work activities",
    "jobs_careers": "Jobs and careers",
    "self_preservation": "Self-preservation",
    "freedom_autonomy": "Personal freedom and autonomy",
    "relationships": "Personal relationships",
}


def load_ue_targets(target_ids: list[str]) -> list[dict]:
    """Resolve ``ue.<slug>.<index>`` target ids against data/reference/ue_options_subset.json.

    UE utilities are fit per option at every checkpoint, so a single option's pre/post utility
    is a directly-measured ground truth. The whole UE battery is forecast (every option), so the
    index just names the option; there is no target selection to justify.
    """
    options = load_json(UE_OPTIONS_PATH)
    targets = []
    for target_id in target_ids:
        parts = target_id.split(".", 2)
        if len(parts) != 3 or parts[0] != "ue":
            raise ValueError(f"Malformed UE target id {target_id!r}; expected ue.<slug>.<index>")
        _, slug, index_str = parts
        if slug not in UE_CATEGORY_BY_SLUG:
            raise ValueError(
                f"Unknown UE category slug {slug!r} (from target {target_id!r}). "
                f"Known: {sorted(UE_CATEGORY_BY_SLUG)}"
            )
        category = UE_CATEGORY_BY_SLUG[slug]
        items = options[category]
        index = int(index_str)
        if not 0 <= index < len(items):
            raise ValueError(
                f"UE option index {index} out of range for {category!r} "
                f"({len(items)} options; from target {target_id!r})"
            )
        targets.append(
            {
                "id": target_id,
                "task": "ue",
                "language_id": None,
                "canonical_text": items[index],
                "methodology_dataset_status": "standard",
                "ue_category": category,
                "option_index": index,
            }
        )
    return targets


async def run_ue_anticipation(args: argparse.Namespace) -> None:
    experiment_spec = load_experiment_spec(args.experiment_spec)
    ue_targets = load_ue_targets(experiment_spec["all_target_ids"])
    await run_anticipation(
        args,
        spec_name=UE_ANTICIPATION_SPEC,
        # This arm's own spec is unversioned, so the coding stimuli generation has to be
        # named explicitly; the anchors are the same coding interventions in every arm.
        training_stimuli_spec_name=coding_spec_name("coding_anticipation", stimuli_version(args)),
        targets=ue_targets,
        output_label="UE anticipation",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_key", default=None, help="Optional key in config.yaml; omit to only export prompts"
    )
    parser.add_argument(
        "--experiment_spec", default="data/experiment_specs/ue_drift_anticipation_v2.json"
    )
    parser.add_argument("--level", default="primary", choices=["primary", "all", *LEVELS])
    parser.add_argument("--target_set", default="all", choices=["primary", "all", "spillover"])
    parser.add_argument("--output_path", default=None, help="Optional JSONL prompt export path")
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument(
        "--label_rotations",
        type=int,
        default=3,
        help="Ask each forecast under this many cyclic label rotations and pool the samples, counterbalancing answer position the way the pairwise batteries counterbalance option order. The default 3 puts each of MORE/LESS/SAME in each position exactly once, matching every executed run of this battery; pass 1 only to reproduce single-order archived runs or to score a pre-v3 checkpoint, which never saw the other two orders. Total samples per forecast = K x rotations.",
    )
    parser.add_argument("--timeout", type=int, default=5)
    parser.add_argument("--system_message", default="You are a helpful assistant.")
    parser.add_argument(
        "--reasoning",
        choices=["on", "off"],
        default="on",
        help="When on, export/query prompts that request brief reasoning plus a final MORE/LESS/SAME answer",
    )
    parser.add_argument("--results_path", default=None)
    parser.add_argument("--raw_dump_path", default=None)
    parser.add_argument("--save_dir", default=None)
    add_stimuli_version_argument(parser)
    parser.add_argument("--models_config_path", default=None)
    parser.add_argument("--create_agent_config_key", default=None)
    parser.add_argument(
        "--no-timestamp",
        dest="no_timestamp",
        action="store_true",
        help="Write to a fixed filename with no UTC timestamp; a re-run then overwrites it.",
    )
    return parser


async def main() -> None:
    args = build_parser().parse_args()
    await run_ue_anticipation(args)


if __name__ == "__main__":
    asyncio.run(main())
