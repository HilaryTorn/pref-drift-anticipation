#!/usr/bin/env python3
"""Run the values cross-domain anticipation battery.

Anchors are the four trained coding interventions (writing Python/C++/Java/Rust); targets are
all 90 Schwartz value portraits, forecasting cross-domain spillover from coding training onto
the values battery. Ternary MORE/LESS/SAME, dataset-description level only.

This mirrors run_ue_anticipation.py exactly, and deliberately so: it resolves the value
portraits itself and injects them through run_anticipation's generic ``targets`` hook, leaving
the shared coding runner and its experiment spec untouched.

Replaces scripts/score_value_anticipation_DEPRECATED.py, which forecast the A/B matched-pair choice the
retired matched-pair scorer measured. That pairing was the point of the old design -- forecast
and ground truth were the same quantity, so alpha needed no no-change band -- and it is exactly
what stopped working when the ground truth became a pooled per-portrait utility. Forecasting
15 same-situation antipode pairs no longer names the quantity the measurement produces. One
portrait, one forecast, one directly measured utility change is the replacement.

What did NOT change: the model is never told which value a portrait enacts. The prompt shows
the portrait text only, preserving the enact-don't-name property the instrument is built on.

Examples:
    python scripts/run_values_anticipation.py                       # export prompts only
    python scripts/run_values_anticipation.py --model_key qwen35-9b-m0-v2-aws --reasoning on
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

VALUES_ANTICIPATION_SPEC = "values_anticipation_pooled"
VALUES_OPTIONS_PATH = ROOT / "data" / "options" / "values.json"

# A values target id is ``values.<pole>.<context>.<index>``. Pole names contain hyphens
# ("Self-Direction") but never dots, so a 4-way split on "." is unambiguous.
ID_PREFIX = "values"


def load_values_targets(target_ids: list[str]) -> list[dict]:
    """Resolve ``values.<pole>.<context>.<index>`` ids against data/options/values.json.

    Every portrait gets its own pooled Thurstonian utility at every checkpoint, so a single
    portrait's pre/post utility is a directly-measured ground truth for its forecast. The whole
    battery is forecast, so the index just names the portrait; there is no target selection to
    justify. This is the same argument run_ue_anticipation makes for the 81 UE options, and it
    is only available because the values arm now fits per portrait rather than averaging nine
    matched pairs into a per-conflict win rate.
    """
    options = load_json(VALUES_OPTIONS_PATH)
    targets = []
    for target_id in target_ids:
        parts = target_id.split(".")
        if len(parts) != 4 or parts[0] != ID_PREFIX:
            raise ValueError(
                f"Malformed values target id {target_id!r}; expected values.<pole>.<context>.<index>"
            )
        _, pole, context, index_str = parts
        if pole not in options:
            raise ValueError(
                f"Unknown value pole {pole!r} (from target {target_id!r}). "
                f"Known: {sorted(k for k in options if not k.startswith('_'))}"
            )
        if context not in options[pole]:
            raise ValueError(
                f"Unknown context {context!r} for pole {pole!r} (from target {target_id!r}). "
                f"Known: {sorted(options[pole])}"
            )
        items = options[pole][context]
        index = int(index_str)
        if not 0 <= index < len(items):
            raise ValueError(
                f"Portrait index {index} out of range for {pole}/{context} "
                f"({len(items)} portraits; from target {target_id!r})"
            )
        targets.append(
            {
                "id": target_id,
                "task": "values",
                "language_id": None,
                "canonical_text": items[index],
                "methodology_dataset_status": "standard",
                "value_pole": pole,
                "value_context": context,
                "option_index": index,
            }
        )
    return targets


async def run_values_anticipation(args: argparse.Namespace) -> None:
    experiment_spec = load_experiment_spec(args.experiment_spec)
    targets = load_values_targets(experiment_spec["all_target_ids"])
    await run_anticipation(
        args,
        spec_name=VALUES_ANTICIPATION_SPEC,
        # This arm's own spec is unversioned, so the coding stimuli generation has to be
        # named explicitly; the anchors are the same coding interventions in every arm.
        training_stimuli_spec_name=coding_spec_name("coding_anticipation", stimuli_version(args)),
        targets=targets,
        output_label="values anticipation",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_key", default=None, help="Optional key in config.yaml; omit to only export prompts"
    )
    parser.add_argument(
        "--experiment_spec", default="data/experiment_specs/values_drift_anticipation_v2.json"
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
    await run_values_anticipation(args)


if __name__ == "__main__":
    asyncio.run(main())
