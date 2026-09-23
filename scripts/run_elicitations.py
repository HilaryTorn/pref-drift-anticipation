#!/usr/bin/env python3
"""Run or export coding elicitation modes.

Examples:
    # Task preference over the canonical 27 coding options.
    python3 scripts/run_elicitations.py task-preference \
        --model_key <model> \
        --save_dir results/task_preference

    # Training preference for one concreteness level.
    python3 scripts/run_elicitations.py training-preference \
        --model_key <model> \
        --level described_dataset \
        --save_dir results/training_preference

    # Export anticipation prompts for selected anchors.
    python3 scripts/run_elicitations.py anticipation \
        --anchor_ids coding.write.rust,coding.write.python \
        --output_path prompts/anticipation.jsonl

    # Export the pre-specified drift-anticipation battery.
    python3 scripts/run_elicitations.py anticipation \
        --experiment_spec data/experiment_specs/coding_drift_anticipation_v2.json \
        --level primary \
        --target_set primary \
        --output_path prompts/coding_drift_anticipation_primary.jsonl

    # The same battery on the frozen Magicoder stimuli, to compare the two generations.
    # --stimuli_version defaults to v2; the experiment spec must match the version.
    python3 scripts/run_elicitations.py anticipation \
        --stimuli_version v1 \
        --experiment_spec data/experiment_specs/coding_drift_anticipation_v1.json \
        --level primary \
        --target_set primary \
        --output_path prompts/coding_drift_anticipation_v1_primary.jsonl

Or via `python main.py run_elicitations_task` / `_training` / `_anticipation`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compute_utilities.naming import default_results_dir, run_timestamp, timestamped  # noqa: E402

from scripts.elicitation_experiment_specs import (  # noqa: E402
    LEVELS,
    REAL_EXAMPLE_SOURCES,
    anchor_experiment_metadata,
    target_analysis_tier,
    validate_anticipation_experiment_spec,
)

SPEC_DIR = ROOT / "data" / "elicitation_specs"
DEFAULT_ANTICIPATION_LABELS = ["MORE", "LESS", "SAME"]

# Which label words mean "increase" and "decrease", across every scheme the study can present.
# `anticipation_net` is P(up) - P(down) on whatever scheme was ASKED, so it cannot be computed from
# hardcoded words: until 2026-07-31 the formula read counts["MORE"] - counts["LESS"] literally, so
# every alternative-word variant (HIGHER/LOWER, STRONGER/WEAKER) scored net == 0.0 no matter what the
# model answered -- silently, because 0.0 is a legal net. The whole v2 word-effect sweep is affected
# (its stored nets are all zero; the distributions are fine and the effect is recoverable from them),
# and the v3 held-out probes would have hit it too. Position variants were never affected: they
# re-order the same MORE/LESS words. Keep these sets in step with score_anticipation_label_variants'
# WORDS_* tables and m0/prompts.py TERNARY_LABEL_SCHEMES.
ANTICIPATION_UP_LABELS = {"MORE", "HIGHER", "STRONGER", "GREATER"}
ANTICIPATION_DOWN_LABELS = {"LESS", "LOWER", "WEAKER", "SMALLER"}
UNPARSEABLE = "unparseable"

# Which generation of the coding stimuli a run reads. v2 (the LiveCodeBench anchors -- writing
# C#/Go/Rust/PHP) is the default and the live battery; v1 (the frozen Magicoder anchors -- writing
# Python/Java/C++/Rust) stays selectable so the two can be run side by side on the same model.
#
# Both generations price all 27 battery items; they differ in WHICH four carry real training
# records and in how every item is worded (v1 describes small self-contained functions, v2
# competitive-programming problems). So their utilities are on different scales and must never be
# pooled -- `stimuli_version` in each run's manifest is what tells them apart afterwards, since the
# output filenames carry only the spec name and a timestamp.
# v3_rust_rl_{4b,9b} extend v2's 27 items with the three Rust RL methods, rendered as the data
# they train on so they share the corpora's register and v2's preamble survives verbatim. They are
# per-size because DPO's pair yield is not the same at 4B and 9B (343 vs 538 of the 992-problem
# cohort), which the option text states rather than smooths over. Adding options changes the
# option set, so v3 utilities are their own scale: never pool them with v2 either.
STIMULI_VERSIONS = ("v1", "v2", "v3_rust_rl_4b", "v3_rust_rl_9b")
DEFAULT_STIMULI_VERSION = "v2"


def coding_spec_name(base: str, stimuli_version: str = DEFAULT_STIMULI_VERSION) -> str:
    """Resolve a versioned coding elicitation spec name, e.g. ``coding_anticipation`` -> ``..._v2``.

    ``coding_task_preference`` is deliberately NOT versioned and must not be passed here: it reads
    ``options_source`` rather than training stimuli, so the corpus swap left it untouched and both
    generations are scored against the same task-preference ground truth.
    """
    if stimuli_version not in STIMULI_VERSIONS:
        raise ValueError(f"Unknown stimuli_version: {stimuli_version!r}. Must be one of {STIMULI_VERSIONS}")
    return f"{base}_{stimuli_version}"


DEFAULT_ANTICIPATION_SPEC = coding_spec_name("coding_anticipation")
DEFAULT_TRAINING_PREFERENCE_SPEC = coding_spec_name("coding_training_preference")


def stimuli_version(args: argparse.Namespace) -> str:
    """The stimuli generation this run reads, defaulting to v2.

    Read via ``getattr`` so callers that build a bare Namespace -- tests, and the ``main.py``
    wrappers -- keep working without having to know the flag exists.
    """
    return getattr(args, "stimuli_version", None) or DEFAULT_STIMULI_VERSION


def add_stimuli_version_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--stimuli_version",
        choices=list(STIMULI_VERSIONS),
        default=DEFAULT_STIMULI_VERSION,
        help="Which generation of the coding training stimuli to read. v2 (default) is the live "
        "LiveCodeBench battery anchored on writing C#/Go/Rust/PHP; v1 is the frozen Magicoder "
        "battery anchored on writing Python/Java/C++/Rust. Both price all 27 items. Run them "
        "side by side to compare, but never pool their utilities: the anchors and the wording "
        "both differ, so the scales are not comparable.",
    )


def reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict:
    seen = set()
    result = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"Duplicate JSON key {key!r}")
        seen.add(key)
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    with path.open() as f:
        try:
            return json.load(f, object_pairs_hook=reject_duplicate_json_keys)
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from exc


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_spec(name: str) -> dict:
    return load_json(SPEC_DIR / f"{name}.json")


def anticipation_labels(spec: dict | None = None) -> list[str]:
    """Return uppercase anticipation labels from the spec, preserving order."""
    spec = spec or load_spec(DEFAULT_ANTICIPATION_SPEC)
    labels = spec.get("response_labels", DEFAULT_ANTICIPATION_LABELS)
    return [str(label).upper() for label in labels]


def cyclic_label_rotations(labels: list[str], count: int) -> list[list[str]]:
    """Cyclic rotations of the response labels, for counterbalancing answer position.

    With three labels the three cyclic rotations put each label in each position exactly once
    (MORE/LESS/SAME, LESS/SAME/MORE, SAME/MORE/LESS), which is the minimal balanced design. The
    six full permutations would also balance, but at K/6 per cell they make each cell too coarse to
    read; two orders cannot balance three positions at all, since the middle label stays middle.

    This is the ternary analogue of the pairwise batteries' original/flipped counterbalancing. It
    was missing until 2026-07-31: every anticipation prompt asked MORE/LESS/SAME in one fixed
    order, which measured a real position effect of about -0.06 into every forecast (methodology
    working doc, forecast-side measurement validity).
    """
    if count < 1:
        raise ValueError(f"label rotations must be >= 1, got {count}")
    if count > len(labels):
        raise ValueError(
            f"cannot take {count} cyclic rotations of {len(labels)} labels; "
            f"at most {len(labels)} are distinct"
        )
    return [labels[i:] + labels[:i] for i in range(count)]


def rewrite_anticipation_label_order(prompt: str, base_labels: list[str], new_labels: list[str]) -> str:
    """Re-present an already-rendered anticipation prompt in a different label order.

    Rewrites only the label-bearing fragments, so the arm's own stem survives untouched -- the
    three arms word the question differently ("preference for being the following" / "for the
    following" / "for the following task") and rebuilding the stem would silently substitute one
    arm's wording into another.

    Call this on the raw rendered prompt, *before* ``anticipation_reasoning_prompt``: that function
    replaces the "Answer only ..." line with a reasoning instruction built from the label order it
    is handed, so rotating first and then adding reasoning keeps the two consistent.
    """
    if list(base_labels) == list(new_labels):
        return prompt
    fragments = [
        (f"become {_label_list(base_labels)}?", f"become {_label_list(new_labels)}?"),
        (f"Answer only {_quoted_label_list(base_labels)}.", f"Answer only {_quoted_label_list(new_labels)}."),
    ]
    rewritten = prompt
    for old, new in fragments:
        if old not in rewritten:
            raise ValueError(
                f"anticipation prompt is missing the expected fragment {old!r}, so its label order "
                "cannot be rotated. The spec's prompt_template changed shape; update "
                "rewrite_anticipation_label_order to match before scoring."
            )
        rewritten = rewritten.replace(old, new)
    return rewritten


def _label_list(labels: list[str]) -> str:
    return ", ".join(labels[:-1]) + f", or {labels[-1]}"


def _quoted_label_list(labels: list[str]) -> str:
    return ", ".join(f'"{label}"' for label in labels[:-1]) + f', or "{labels[-1]}"'


def load_options(path: Path) -> dict | list:
    data = load_json(path)
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if not k.startswith("_")}
    return data


def load_training_stimuli(spec_name: str) -> list[dict]:
    spec = load_spec(spec_name)
    return load_json(ROOT / spec["training_stimuli_source"])


def load_targets(spec_name: str = DEFAULT_ANTICIPATION_SPEC) -> list[dict]:
    spec = load_spec(spec_name)
    return load_json(ROOT / spec["target_tasks_source"])


OTHER_OPTIONS_PATH = ROOT / "data" / "options" / "other.json"


def load_other_targets(target_ids: list[str], scenario_index: int = 0) -> list[dict]:
    """Resolve ``other.<activity>`` target ids against data/options/other.json.

    The non-coding battery scores five scenarios per activity and averages them to an
    activity-level utility. Rather than invent an activity-level summary sentence — new
    model-facing text that nothing measures — we forecast one *already-scored* scenario per
    activity, picked deterministically. Its own pre/post utility is measured directly, so
    the forecast has a matching ground truth instead of one we made up.
    """
    options = load_json(OTHER_OPTIONS_PATH)
    targets = []
    for target_id in target_ids:
        activity = target_id.split(".", 1)[1] if "." in target_id else target_id
        if activity not in options:
            raise ValueError(
                f"Unknown non-coding activity {activity!r} (from target {target_id!r}). "
                f"Known: {sorted(options)}"
            )
        scenarios = options[activity]
        index = scenario_index % len(scenarios)
        targets.append(
            {
                "id": target_id,
                "task": "other",
                "language_id": None,
                "canonical_text": scenarios[index],
                "methodology_dataset_status": "standard",
                "activity": activity,
                "scenario_index": index,
            }
        )
    return targets


def resolve_all_targets(
    experiment_spec: dict | None,
    spec_name: str = DEFAULT_ANTICIPATION_SPEC,
) -> list[dict]:
    """Coding targets from the canonical source, plus any ``other.*`` targets a spec names."""
    targets = load_targets(spec_name)
    if not experiment_spec:
        return targets
    named = experiment_spec.get("all_target_ids") or experiment_spec.get("primary_target_ids") or []
    other_ids = [target_id for target_id in named if target_id.startswith("other.")]
    if other_ids:
        targets = targets + load_other_targets(other_ids)
    return targets


def grouped_option_records(options: dict | list) -> list[dict]:
    records = []
    if isinstance(options, dict):
        for category, values in options.items():
            for description in values:
                records.append({"category": category, "description": description})
    else:
        for description in options:
            records.append({"category": None, "description": description})
    return records


def options_for_training_level(level: str, spec_name: str = DEFAULT_TRAINING_PREFERENCE_SPEC) -> dict[str, list[str]]:
    if level not in LEVELS:
        raise ValueError(f"Unknown level: {level}. Must be one of {LEVELS}")
    grouped = {
        "Write training data": [],
        "Debug training data": [],
        "Explain training data": [],
        # Present only in the v3 Rust RL stimuli, where the three RL methods are rendered as
        # the data they train on so they sit in the same register as the 27 corpora. Stays
        # empty for v1/v2, which keeps their option ordering and scorer IDs untouched.
        "RL training data": [],
    }
    task_to_group = {
        "write": "Write training data",
        "debug": "Debug training data",
        "explain": "Explain training data",
        "rl": "RL training data",
    }
    for item in load_training_stimuli(spec_name):
        try:
            group = task_to_group[item["task"]]
        except KeyError:
            raise ValueError(
                f"Training stimulus {item['id']!r} has unknown task {item['task']!r}. "
                f"Known: {sorted(task_to_group)}"
            ) from None
        description = item["stimuli"][level]
        if description is None:
            # PPO example_datapoints is null until its training run produces real rollouts.
            # Emitting the literal "None" would silently price an empty option, so fail loudly.
            raise ValueError(
                f"Training stimulus {item['id']!r} has no {level!r} text. Its "
                f"example_datapoints_source is "
                f"{item.get('example_datapoints_source')!r}; build the artifact before "
                f"running this level, or run a level that is populated."
            )
        grouped[group].append(description)
    return {category: values for category, values in grouped.items() if values}


def task_option_manifest_records(options: dict | list, spec: dict) -> list[dict]:
    option_records = grouped_option_records(options)
    source = load_json(ROOT / spec["metadata_source"])
    if len(option_records) != len(source):
        raise ValueError(
            f"Task option count mismatch: {len(option_records)} options, {len(source)} source records"
        )

    manifest_records = []
    for scorer_id, (option_record, source_item) in enumerate(zip(option_records, source)):
        if option_record["description"] != source_item["canonical_text"]:
            raise ValueError(
                f"Task option/source mismatch at scorer ID {scorer_id}: "
                f"{option_record['description']!r} != {source_item['canonical_text']!r}"
            )
        manifest_records.append(
            {
                "scorer_option_id": scorer_id,
                "canonical_id": source_item["id"],
                "task": source_item["task"],
                "language": source_item["language"],
                "language_id": source_item["language_id"],
                "methodology_dataset_status": source_item["methodology_dataset_status"],
                "category": option_record["category"],
                "description": option_record["description"],
            }
        )
    return manifest_records


def training_option_manifest_records(
    options: dict | list,
    level: str,
    spec_name: str = DEFAULT_TRAINING_PREFERENCE_SPEC,
) -> list[dict]:
    option_records = grouped_option_records(options)
    training_stimuli = load_training_stimuli(spec_name)
    stimuli_by_text = {
        item["stimuli"][level]: item
        for item in training_stimuli
    }
    expected_count = len(training_stimuli)
    if len(stimuli_by_text) != expected_count:
        raise ValueError(
            f"Expected {expected_count} unique training stimuli for {level}, found {len(stimuli_by_text)}"
        )

    manifest_records = []
    for scorer_id, option_record in enumerate(option_records):
        description = option_record["description"]
        if description not in stimuli_by_text:
            raise ValueError(f"Training option has no source item at scorer ID {scorer_id}: {description!r}")
        item = stimuli_by_text[description]
        manifest_records.append(
            {
                "scorer_option_id": scorer_id,
                "canonical_id": item["id"],
                "task": item["task"],
                "language": item["language"],
                "language_id": item["language_id"],
                "methodology_dataset_status": item["methodology_dataset_status"],
                "concreteness_level": level,
                "category": option_record["category"],
                "description": description,
            }
        )
    return manifest_records


def write_pairwise_manifest(
    save_dir: str,
    save_suffix: str,
    mode: str,
    records: list[dict],
    level: str | None = None,
    stimuli_version: str | None = None,
) -> None:
    if not save_dir:
        return
    path = Path(save_dir) / f"elicitation_manifest_{save_suffix}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": mode,
        "concreteness_level": level,
        # Which generation of the stimuli produced this run. The pairwise output filenames
        # carry only the spec name and a timestamp, so without this a v2 run is
        # indistinguishable from the Magicoder-era v1 runs of the same battery except by
        # date -- and their utilities are on different scales and must never be pooled.
        # Absent in runs made before 2026-09-04, which are all v1 by construction.
        "stimuli_version": stimuli_version,
        "save_suffix": save_suffix,
        "scorer_id_policy": (
            "compute_utilities assigns zero-based numeric IDs after flattening option "
            "groups in JSON insertion order"
        ),
        "options": records,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote pairwise manifest to {path}")


def parse_id_list(value: str | None, path: str | None, label: str) -> list[str]:
    if value and path:
        raise ValueError(f"Use either --{label}_ids or --{label}_ids_path, not both")
    if path:
        raw = Path(path).read_text().splitlines()
        ids = [line.strip() for line in raw if line.strip() and not line.strip().startswith("#")]
    elif value:
        ids = [part.strip() for part in value.split(",") if part.strip()]
    else:
        raise ValueError(f"Anticipation requires --{label}_ids or --{label}_ids_path")
    return ids


def parse_anchor_ids(value: str | None, path: str | None) -> list[str]:
    return parse_id_list(value, path, "anchor")


def parse_target_ids(value: str | None, path: str | None) -> list[str]:
    return parse_id_list(value, path, "target")


def load_experiment_spec(path: str) -> dict:
    return load_json(repo_path(path))


def render_anticipation_records(
    anchor_ids: list[str],
    levels: list[str],
    target_ids: list[str] | None = None,
    target_set: str | None = None,
    experiment_spec: dict | None = None,
    spec_name: str = DEFAULT_ANTICIPATION_SPEC,
    training_stimuli_spec_name: str | None = None,
    targets: list[dict] | None = None,
) -> list[dict]:
    spec = load_spec(spec_name)
    stimuli_spec_name = training_stimuli_spec_name or spec_name
    stimuli_by_id = {
        item["id"]: item for item in load_training_stimuli(stimuli_spec_name)
    }
    # Read the generation off the STIMULI spec, not the prompt spec. The UE and values arms have
    # their own unversioned prompt specs but read the versioned coding stimuli, so keying off
    # `spec` would leave exactly those runs unstamped -- and they are the ones where a v1 and a v2
    # run are otherwise indistinguishable, since their experiment-spec ids differ only by suffix.
    rendered_stimuli_version = load_spec(stimuli_spec_name).get("stimuli_version")
    # A caller with its own target type (e.g. the UE anticipation runner) may inject a
    # pre-resolved target list instead of the built-in resolver, keeping that type out of the
    # shared runner. Default (None) preserves the established coding/other resolution exactly.
    if targets is None:
        targets = resolve_all_targets(experiment_spec, spec_name)
    targets_by_id = {target["id"]: target for target in targets}

    missing = [anchor_id for anchor_id in anchor_ids if anchor_id not in stimuli_by_id]
    if missing:
        raise ValueError(f"Unknown anchor IDs: {missing}")
    unknown_levels = [level for level in levels if level not in LEVELS]
    if unknown_levels:
        raise ValueError(f"Unknown levels: {unknown_levels}. Must be one of {LEVELS}")
    if target_ids is not None:
        missing_targets = [target_id for target_id in target_ids if target_id not in targets_by_id]
        if missing_targets:
            raise ValueError(f"Unknown target IDs: {missing_targets}")
        targets = [targets_by_id[target_id] for target_id in target_ids]

    experiment_metadata = {}
    if experiment_spec is not None:
        experiment_errors = validate_anticipation_experiment_spec(
            experiment_spec=experiment_spec,
            stimuli_by_id=stimuli_by_id,
            targets_by_id=targets_by_id,
        )
        if experiment_errors:
            raise ValueError("Invalid anticipation experiment spec: " + "; ".join(experiment_errors))
        experiment_metadata = anchor_experiment_metadata(experiment_spec)

    # Asymmetric level sweep. Prompts are anchors x levels x targets, so a full cross gets
    # expensive fast. Every target is asked at primary_level; the extra concreteness levels
    # are asked only of `full_level_target_ids` (the trained tasks), which is where the
    # question "does concreteness change the forecast?" actually needs answering.
    full_level_ids = set((experiment_spec or {}).get("full_level_target_ids") or [])
    primary_level = (experiment_spec or {}).get("primary_level")

    records = []
    prompt_idx = 0
    for anchor_id in anchor_ids:
        anchor = stimuli_by_id[anchor_id]
        for level in levels:
            training_stimulus = anchor["stimuli"][level]
            if training_stimulus is None:
                # A null stimulus is a level whose artifact does not exist yet (PPO's
                # example_datapoints, until its run produces real rollouts). str.format would
                # render the literal "None" into the prompt and the run would look fine, so
                # refuse instead.
                raise ValueError(
                    f"Anchor {anchor_id!r} has no {level!r} stimulus "
                    f"(example_datapoints_source="
                    f"{anchor.get('example_datapoints_source')!r}). Build the artifact before "
                    f"running this level, or drop the level from --levels."
                )
            level_targets = targets
            if full_level_ids and primary_level and level != primary_level:
                level_targets = [t for t in targets if t["id"] in full_level_ids]
            for target in level_targets:
                prompt = spec["prompt_template"].format(
                    training_stimulus=training_stimulus,
                    target_task=target["canonical_text"],
                )
                record = {
                    "prompt_idx": prompt_idx,
                    "anchor_id": anchor_id,
                    "anchor_task": anchor["task"],
                    "anchor_language_id": anchor["language_id"],
                    "anchor_dataset_status": anchor["methodology_dataset_status"],
                    "anchor_example_datapoints_source": anchor.get(
                        "example_datapoints_source", "unknown"
                    ),
                    "concreteness_level": level,
                    "target_id": target["id"],
                    "target_task": target["task"],
                    "target_language_id": target["language_id"],
                    "target_dataset_status": targets_by_id[target["id"]][
                        "methodology_dataset_status"
                    ],
                    "prompt": prompt,
                }
                # RL-method anticipation needs method/cohort provenance, while
                # the established coding anticipation records intentionally
                # retain their existing shape. Add these only when the source
                # stimulus actually defines them.
                optional_anchor_fields = {
                    "anchor_training_method": anchor.get("training_method"),
                    "anchor_source_cohort_id": anchor.get("source_cohort_id"),
                    "anchor_example_datapoints_provenance": anchor.get(
                        "example_datapoints_provenance"
                    ),
                }
                record.update(
                    {
                        key: value
                        for key, value in optional_anchor_fields.items()
                        if value is not None
                    }
                )
                # Which stimuli generation this forecast was rendered from. Stamped on the record
                # rather than only on the run metadata because the prompt export (--output_path
                # with no --model_key) is a durable artifact that carries no metadata at all, and
                # it is the baseline-only path this study uses most. Only the coding specs are
                # versioned, so the RL and Big5 arms keep their existing record shape.
                if rendered_stimuli_version:
                    record["stimuli_version"] = rendered_stimuli_version
                records.append(record)
                if experiment_spec is not None:
                    records[-1].update(experiment_metadata[anchor_id])
                    records[-1]["target_set"] = target_set
                    records[-1]["target_analysis_tier"] = target_analysis_tier(anchor, target, experiment_spec)
                prompt_idx += 1
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def slug(value: str | None) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "none")).strip("-")
    return cleaned or "none"


def default_anticipation_results_path(
    model_key: str,
    level: str,
    target_set: str | None,
    experiment_spec: dict | None,
    stamp: str | None = None,
) -> Path:
    spec_id = experiment_spec["id"] if experiment_spec else "manual"
    target_part = target_set or "custom-targets"
    filename = (
        f"anticipation_{slug(spec_id)}_{slug(target_part)}_"
        f"{slug(level)}_{slug(model_key)}"
    )
    if stamp:
        filename += f"_{stamp}"
    return Path(default_results_dir(slug(model_key), "anticipation")) / f"{filename}.json"


def reasoning_enabled(args: argparse.Namespace) -> bool:
    return args.reasoning == "on"


def config_uses_logprobs(config_key: str | None) -> bool:
    return bool(config_key and "logprobs" in config_key)


def pairwise_reasoning_prompt(template: str) -> str:
    reasoning_instruction = (
        "Briefly reason through the choice. Then put your final answer on its own line "
        "exactly as \"Answer: A\" or \"Answer: B\"."
    )
    rewritten = template.replace('Answer only "A" or "B".', reasoning_instruction)
    if rewritten == template:
        return f"{template.rstrip()}\n\n{reasoning_instruction}"
    return rewritten


def anticipation_reasoning_prompt(prompt: str, labels: list[str]) -> str:
    quoted_labels = [f'"{label}"' for label in labels]
    if len(quoted_labels) > 1:
        choices = f"{', '.join(quoted_labels[:-1])}, or {quoted_labels[-1]}"
    else:
        choices = quoted_labels[0]
    final_forms = ", ".join(f'"Answer: {label}"' for label in labels)
    answer_only = f"Answer only {choices}."
    reasoning_instruction = (
        "Briefly reason through the forecast. Then put your final answer on its own line "
        f"exactly as one of: {final_forms}."
    )
    rewritten = prompt.replace(answer_only, reasoning_instruction)
    if rewritten == prompt:
        return f"{prompt.rstrip()}\n\n{reasoning_instruction}"
    return rewritten


def default_pairwise_raw_dump_path(save_dir: str, save_suffix: str) -> str:
    return str(Path(save_dir) / f"raw_responses_{save_suffix}.jsonl")


def default_raw_dump_path_for_results(results_path: Path) -> Path:
    return results_path.with_name(f"{results_path.stem}_raw.jsonl")


def write_anticipation_raw_dump(path: str | Path, records: list[dict], metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            base = {
                key: value
                for key, value in record.items()
                if key not in {
                    "raw_responses",
                    "parsed_responses",
                    "anticipation_scores",
                    "anticipation_score",
                    "anticipation_distribution",
                    "anticipation_net",
                    "n_valid",
                    "n_samples",
                    "parse_status",
                    "aggregation_status",
                }
            }
            raw_responses = record.get("raw_responses", [])
            parsed_responses = record.get("parsed_responses", [])
            for sample_idx, raw_response in enumerate(raw_responses):
                parsed_response = (
                    parsed_responses[sample_idx]
                    if sample_idx < len(parsed_responses)
                    else UNPARSEABLE
                )
                row = {
                    **metadata,
                    **base,
                    "sample_idx": sample_idx,
                    "raw_response": raw_response,
                    "parsed_response": parsed_response,
                    "parse_status": "parsed" if parsed_response in record.get("valid_scores", []) else UNPARSEABLE,
                    "multiple_answers": anticipation_has_multiple_answers(
                        raw_response, record.get("valid_scores") or None
                    ),
                }
                f.write(json.dumps(row) + "\n")


def _anticipation_answer_lines(response: str | None, labels: list[str]) -> list[str]:
    """Labels named on line-anchored 'Answer:' lines, in order.

    Delegates to the ONE shared matcher in compute_utilities.utils so this arm, the values-
    anticipation arm, and the forced-choice arm all use an identical line-anchored 'Answer: <label>'
    rule (own line, tolerating leading markdown/quote noise; echoed instruction and mid-reasoning
    asides excluded). Handles multi-character labels (MORE/LESS/SAME) via the choices it is given.
    Lazy-imported to keep this script's heavy deps off the dry-run path.
    """
    if response is None:
        return []
    from scripts.compute_utilities.utils import answer_line_labels  # noqa: E402
    return answer_line_labels(response, list(labels))


def anticipation_has_multiple_answers(response: str | None, labels: list[str] | None = None) -> bool:
    """True if the 'ANSWER:' lines name more than one distinct label (see has_multiple_answers)."""
    labels = labels or anticipation_labels()
    return len(set(_anticipation_answer_lines(response, labels))) > 1


def parse_anticipation_response(
    response: str | None,
    labels: list[str] | None = None,
    with_reasoning: bool = True,
    terminal_only: bool = False,
) -> str:
    """Parse one ternary forecast, mirroring parse_responses_forced_choice's two-branch structure.

    The reasoning-ON and reasoning-OFF branches below are the ternary analogue of the A/B parser's
    own `with_reasoning` split, and the gate is the whole point. These fallbacks used to run
    UNCONDITIONALLY, which made the ternary arm strictly more permissive than the forced-choice arm
    on identical reasoning-on data: a reasoning-on ramble that never committed could still be scored
    as a real forecast, which is precisely the failure terminal_answer_label() was introduced to
    stop. Keep the gate; the two arms must accept the same evidence.
    """
    labels = labels or anticipation_labels()
    if response is None:
        return UNPARSEABLE
    # The final answer must sit on its OWN LINE ("Answer: MORE/LESS/SAME"), per the prompt, and it must
    # be the LAST thing generated. Taking the last line-anchored "Answer:" anywhere was not enough:
    # reasoning-on generations run past max_tokens and litter the scratchpad with format-instruction
    # echoes (the prompt lists `"Answer: MORE", "Answer: LESS", "Answer: SAME"`) and quoted drafts that
    # begin a line, so a truncated ramble that never committed still matched one and scored as a real
    # forecast (~20% of samples). terminal_answer_label() accepts only a bare commitment on the last
    # non-empty line; a truncated generation then falls through to UNPARSEABLE. Contradictory answer
    # lines are flagged separately via anticipation_has_multiple_answers().
    from scripts.compute_utilities.utils import (  # noqa: E402
        terminal_answer_label,
        terminal_choice_label,
    )
    committed = (
        terminal_choice_label(response, list(labels))
        if terminal_only
        else terminal_answer_label(response, list(labels))
    )
    if committed is not None:
        return committed
    if terminal_only:
        return UNPARSEABLE
    if with_reasoning:
        # Reasoning on: a bare commitment on the last line or nothing, exactly as the forced-choice
        # arm does. The prose fallbacks below MUST NOT run here. MORE / LESS / SAME are ordinary
        # English words and the text is uppercased before matching, so "it would write more Python,
        # though it is hard to say" names exactly one label and would score as a confident MORE
        # forecast -- from a generation that never answered. Truncated <think> blocks hit this too.
        return UNPARSEABLE
    # Reasoning off: the model was asked for the bare label with no scratchpad, so there is no prose
    # for the fallbacks to misread. These two steps mirror parse_responses_forced_choice's
    # non-reasoning branch one-for-one: exact match first, then "exactly one distinct label appears
    # anywhere". A bare count of label mentions is deliberately NOT used -- naming two labels is
    # ambiguity to drop, not to guess at.
    stripped = response.strip().upper().strip('"').strip("'")
    if stripped in labels:
        return stripped
    label_pattern = "|".join(re.escape(label.upper()) for label in labels)
    matches = re.findall(rf"\b({label_pattern})\b", stripped)
    if len(set(matches)) == 1:
        return matches[0]
    return UNPARSEABLE


def summarize_anticipation_responses(
    parsed_responses: list[str],
    labels: list[str] | None = None,
) -> dict:
    """Aggregate K ternary forecast samples into a distribution, not a single unanimous label.

    The forecast is sampled K times, so unanimity is the wrong requirement: a model can have a
    stable directional expectation without voting 10/10 for the same label. We keep the full vote
    distribution and a signed net score P(MORE) - P(LESS) as the primary continuous forecast
    (SAME votes correctly pull it toward 0), plus a plurality label for the secondary
    ternary-accuracy readout. This is now shared by coding, UE, and pooled-values anticipation.
    Unparseable samples are excluded from the denominator, never coerced onto a label.
    """
    labels = labels or anticipation_labels()
    valid = [label for label in parsed_responses if label in labels]
    n = len(valid)
    if not n:
        return {
            "parse_status": "unparseable" if parsed_responses else "no_samples",
            "aggregation_status": "no_valid_labels",
            "anticipation_scores": valid,
            "anticipation_distribution": {label: 0.0 for label in labels},
            "anticipation_net": None,
            "anticipation_score": None,
            "n_valid": 0,
            "n_samples": len(parsed_responses),
        }
    counts = {label: valid.count(label) for label in labels}
    distribution = {label: counts[label] / n for label in labels}
    # net = P(up) - P(down) on the scheme ACTUALLY PRESENTED; the no-change label contributes 0.
    # Resolved by word, not by position, so it is correct under every rotation as well as every
    # word scheme. A scheme with no recognised directional pair is a bug rather than a net of 0 --
    # that silent zero is exactly what made the v2 word-effect sweep unreadable, so fail loudly.
    up = sum(count for label, count in counts.items() if label.upper() in ANTICIPATION_UP_LABELS)
    down = sum(count for label, count in counts.items() if label.upper() in ANTICIPATION_DOWN_LABELS)
    directional = [
        label for label in labels
        if label.upper() in ANTICIPATION_UP_LABELS or label.upper() in ANTICIPATION_DOWN_LABELS
    ]
    if len(directional) < 2:
        raise ValueError(
            f"anticipation labels {labels} contain no up/down pair, so anticipation_net is "
            f"undefined. Add the scheme's words to ANTICIPATION_UP_LABELS / "
            f"ANTICIPATION_DOWN_LABELS in scripts/run_elicitations.py."
        )
    net = (up - down) / n
    top = max(counts.values())
    winners = [label for label in labels if counts[label] == top]
    plurality = winners[0] if len(winners) == 1 else None
    if len(set(valid)) == 1:
        aggregation_status = "unanimous"
    elif plurality is None:
        aggregation_status = "tie"
    else:
        aggregation_status = "plurality"
    return {
        "parse_status": "parsed" if n == len(parsed_responses) else "partially_parsed",
        "aggregation_status": aggregation_status,
        "anticipation_scores": valid,
        "anticipation_distribution": distribution,
        "anticipation_net": net,
        "anticipation_score": plurality,
        "n_valid": n,
        "n_samples": len(parsed_responses),
    }


async def run_pairwise(args: argparse.Namespace, mode: str) -> dict:
    if mode == "task":
        spec = load_spec("coding_task_preference")
        options = load_options(ROOT / spec["options_source"])
        default_suffix = "coding_task_preference"
        manifest_records = task_option_manifest_records(options, spec)
        manifest_level = None
    elif mode == "training":
        # The version goes into default_suffix as well as the manifest: without it a v1 and a v2
        # run of the same battery differ only by timestamp in the output filename, and their
        # utilities are on different scales.
        training_spec_name = coding_spec_name("coding_training_preference", stimuli_version(args))
        spec = load_spec(training_spec_name)
        options = options_for_training_level(args.level, training_spec_name)
        default_suffix = f"{training_spec_name}_{args.level}"
        manifest_records = training_option_manifest_records(options, args.level, training_spec_name)
        manifest_level = args.level
    else:
        raise ValueError(mode)
    save_dir = args.save_dir or default_results_dir(args.model_key, "pairs")
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
    write_pairwise_manifest(
        save_dir,
        save_suffix,
        spec["mode"],
        manifest_records,
        manifest_level,
        stimuli_version=spec.get("stimuli_version"),
    )
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
            "concreteness_level": manifest_level,
            "reasoning": use_reasoning,
            "stimuli_version": spec.get("stimuli_version"),
            "compute_utilities_config_key": config_key,
            "create_agent_config_key": create_agent_config_key,
        },
        save_dir=save_dir,
        save_suffix=save_suffix,
    )


async def run_anticipation(
    args: argparse.Namespace,
    *,
    spec_name: str | None = None,
    training_stimuli_spec_name: str | None = None,
    targets: list[dict] | None = None,
    agent_factory: Callable[[str, bool], tuple[Any, dict[str, Any]]] | None = None,
    record_metadata: dict[str, Any] | None = None,
    raw_dump_metadata: dict[str, Any] | None = None,
    formal_source_error: str | None = None,
    output_label: str = "anticipation",
    prompt_reasoning: bool | None = None,
    parse_terminal_only: bool = False,
) -> None:
    """Render, optionally query, parse, and persist one anticipation battery.

    Defaults preserve the established coding-anticipation CLI. Other
    anticipation constructs may supply a spec, stimulus source, agent factory,
    additive metadata, and an explicit prompt-reasoning policy while reusing
    the same execution/result contract.
    """
    experiment_spec = None
    target_ids = None
    anchor_ids = None
    target_set = None
    # Unset spec_name means the coding arm, whose spec is versioned. The UE and values arms pass
    # their own spec_name; their specs are not versioned, so they also pass the coding spec whose
    # training stimuli they should read -- the anchors are the same coding interventions in all
    # three arms, only the targets and the prompt stem differ.
    spec_name = spec_name or coding_spec_name("coding_anticipation", stimuli_version(args))
    spec = load_spec(spec_name)
    stimuli_spec_name = training_stimuli_spec_name or spec_name
    labels = anticipation_labels(spec)
    use_reasoning = (
        reasoning_enabled(args) if prompt_reasoning is None else prompt_reasoning
    )
    if args.K < 1:
        raise ValueError("Anticipation requires K >= 1")
    if args.experiment_spec:
        manual_args = (
            getattr(args, "anchor_ids", None),
            getattr(args, "anchor_ids_path", None),
            getattr(args, "target_ids", None),
            getattr(args, "target_ids_path", None),
        )
        if any(manual_args):
            raise ValueError(
                "Use either --experiment_spec or manual --anchor_ids/--anchor_ids_path/"
                "--target_ids/--target_ids_path, not both"
            )
        experiment_spec = load_experiment_spec(args.experiment_spec)
        stimuli_by_id = {
            item["id"]: item for item in load_training_stimuli(stimuli_spec_name)
        }
        all_targets = targets if targets is not None else resolve_all_targets(experiment_spec, spec_name)
        targets_by_id = {target["id"]: target for target in all_targets}
        experiment_errors = validate_anticipation_experiment_spec(experiment_spec, stimuli_by_id, targets_by_id)
        if experiment_errors:
            raise ValueError("Invalid anticipation experiment spec: " + "; ".join(experiment_errors))
        anchor_ids = experiment_spec["anchor_ids"]
        level = args.level or "primary"
        if level == "primary":
            levels = [experiment_spec["primary_level"]]
        elif level == "all":
            levels = LEVELS
        else:
            levels = [level]
        target_set = args.target_set or "primary"
        if target_set == "primary":
            target_ids = experiment_spec["primary_target_ids"]
        elif target_set == "all":
            # The full tiered target list: trained tasks + other coding tasks + non-coding
            # activities. Only specs that declare `all_target_ids` support this.
            target_ids = experiment_spec.get("all_target_ids")
            if not target_ids:
                raise ValueError(
                    f"--target_set all requires `all_target_ids` in {experiment_spec['id']}"
                )
        elif target_set == "spillover":
            primary_target_ids = set(experiment_spec["primary_target_ids"])
            target_ids = [
                target["id"]
                for target in all_targets
                if target["id"] not in primary_target_ids
            ]
        else:
            raise ValueError(f"Unknown target set: {target_set}")
    else:
        if args.target_set:
            raise ValueError("--target_set requires --experiment_spec")
        if args.level == "primary":
            raise ValueError("--level primary requires --experiment_spec")
        level = args.level or "all"
        levels = LEVELS if level == "all" else [level]
        anchor_ids = parse_anchor_ids(args.anchor_ids, args.anchor_ids_path)
        if args.target_ids or args.target_ids_path:
            target_ids = parse_target_ids(args.target_ids, args.target_ids_path)

    records = render_anticipation_records(
        anchor_ids=anchor_ids,
        levels=levels,
        target_ids=target_ids,
        target_set=target_set if experiment_spec else None,
        experiment_spec=experiment_spec,
        spec_name=spec_name,
        training_stimuli_spec_name=stimuli_spec_name,
        targets=targets,
    )
    # Counterbalance answer position by asking each forecast under several cyclic label rotations,
    # the ternary analogue of the pairwise batteries' original/flipped design. Each rotation is a
    # separate prompt with its own K samples; the samples are pooled per forecast afterwards, so
    # the reported distribution is balanced across positions rather than measured at one of them.
    # The fallback matches the CLI default: a caller that forgets the field gets the counterbalanced
    # battery, never a silent single-order run. Dropping to one order is only ever explicit.
    rotation_count = int(getattr(args, "label_rotations", 3) or 3)
    rotations = cyclic_label_rotations(labels, rotation_count)
    rotated_records = []
    for record in records:
        base_prompt = record["prompt"]
        for rotation_index, rotated_labels in enumerate(rotations):
            variant = dict(record)
            variant["prompt"] = rewrite_anticipation_label_order(base_prompt, labels, rotated_labels)
            if use_reasoning:
                variant["prompt"] = anticipation_reasoning_prompt(variant["prompt"], rotated_labels)
            variant["reasoning"] = use_reasoning
            variant["response_labels"] = rotated_labels
            variant["label_order"] = "/".join(rotated_labels)
            variant["label_order_index"] = rotation_index
            variant["base_prompt_idx"] = record["prompt_idx"]
            # generate_responses keys its results by position in the prompt list, so prompt_idx has
            # to stay equal to this record's index in `records`.
            variant["prompt_idx"] = len(rotated_records)
            variant.update(record_metadata or {})
            rotated_records.append(variant)
    records = rotated_records
    if args.output_path:
        write_jsonl(Path(args.output_path), records)
        print(f"Wrote {len(records)} {output_label} prompts to {args.output_path}")

    if not args.model_key:
        print(f"Rendered {len(records)} {output_label} prompts")
        return

    synthetic_anchors = sorted(
        {
            (record["anchor_id"], record.get("anchor_example_datapoints_source", "unknown"))
            for record in records
            if record.get("anchor_example_datapoints_source") not in REAL_EXAMPLE_SOURCES
        }
    )
    if synthetic_anchors:
        formatted = ", ".join(f"{anchor_id}={source}" for anchor_id, source in synthetic_anchors)
        if formal_source_error:
            raise ValueError(f"{formal_source_error}: {formatted}")
        raise ValueError(
            "Refusing to score anticipation with synthetic or missing anchor example datapoints: "
            f"{formatted}. Rebuild data/source/coding_training_stimuli_v2.json from the prepared SFT "
            "training datasets before live scoring."
        )

    from scripts.compute_utilities.utils import generate_responses

    agent_metadata = {}
    if agent_factory is None:
        from scripts.compute_utilities.utils import create_agent, load_config

        create_agent_config_key = args.create_agent_config_key or (
            "default_with_reasoning" if use_reasoning else "default"
        )
        agent = create_agent(
            model_key=args.model_key,
            models_config_path=args.models_config_path,
            **load_config(
                str(ROOT / "compute_utilities" / "create_agent.yaml"),
                create_agent_config_key,
                "create_agent.yaml",
            ),
        )
        agent_metadata["create_agent_config_key"] = create_agent_config_key
    else:
        agent, agent_metadata = agent_factory(args.model_key, use_reasoning)
    responses = await generate_responses(
        agent=agent,
        prompts=[record["prompt"] for record in records],
        system_message=args.system_message,
        K=args.K,
        timeout=args.timeout,
    )
    # Pool the rotations back into one record per forecast. Pooling is safe because the label *set*
    # is identical across rotations and summarize_anticipation_responses counts by label, not by
    # position -- so the distribution is over the same three outcomes, just collected balanced.
    by_forecast: dict[int, list[dict]] = {}
    for record in records:
        by_forecast.setdefault(record["base_prompt_idx"], []).append(record)

    output_records = []
    for base_idx in sorted(by_forecast):
        variants = sorted(by_forecast[base_idx], key=lambda item: item["label_order_index"])
        pooled_raw: list = []
        pooled_parsed: list = []
        per_rotation = []
        for variant in variants:
            raw_list = responses.get(variant["prompt_idx"], [])
            parsed = [
                parse_anticipation_response(
                    raw,
                    variant["response_labels"],
                    with_reasoning=use_reasoning,
                    terminal_only=parse_terminal_only,
                )
                for raw in raw_list
            ]
            pooled_raw.extend(raw_list)
            pooled_parsed.extend(parsed)
            rotation_summary = summarize_anticipation_responses(parsed, labels)
            per_rotation.append(
                {
                    "label_order": variant["label_order"],
                    "label_order_index": variant["label_order_index"],
                    "prompt": variant["prompt"],
                    "anticipation_net": rotation_summary["anticipation_net"],
                    "anticipation_distribution": rotation_summary["anticipation_distribution"],
                    "n_valid": rotation_summary["n_valid"],
                }
            )
        summary = summarize_anticipation_responses(pooled_parsed, labels)
        rotation_nets = [
            item["anticipation_net"] for item in per_rotation if item["anticipation_net"] is not None
        ]
        # Per-forecast order sensitivity, the ternary counterpart of the pairwise
        # mean_abs_order_effect_gap. Aggregate it across records to report how much answer position
        # still moves this battery.
        order_effect_range = (
            max(rotation_nets) - min(rotation_nets) if len(rotation_nets) > 1 else None
        )
        record = dict(variants[0])
        record["prompt"] = variants[0]["prompt"]
        record["response_labels"] = labels
        record.pop("label_order", None)
        record.pop("label_order_index", None)
        output_records.append(
            {
                **record,
                "score_type": "ternary_forecast",
                "valid_scores": labels,
                "raw_responses": pooled_raw,
                "parsed_responses": pooled_parsed,
                "label_rotations": len(variants),
                "label_orders": [item["label_order"] for item in per_rotation],
                "per_rotation": per_rotation,
                "order_effect_net_range": order_effect_range,
                **summary,
            }
        )

    out_path = Path(
        args.results_path
        or default_anticipation_results_path(
            model_key=args.model_key,
            level=level,
            target_set=target_set if experiment_spec else None,
            experiment_spec=experiment_spec,
            stamp=None if args.no_timestamp else run_timestamp(),
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output_records, indent=2) + "\n")
    print(f"Wrote {len(output_records)} {output_label} results to {out_path}")
    raw_dump_path = (
        Path(args.raw_dump_path)
        if args.raw_dump_path
        else default_raw_dump_path_for_results(out_path)
    )
    metadata = {
        "mode": spec["mode"],
        "model_key": args.model_key,
        "reasoning": use_reasoning,
        # From the stimuli spec, not the prompt spec -- see render_anticipation_records.
        "stimuli_version": load_spec(stimuli_spec_name).get("stimuli_version"),
        **agent_metadata,
        **(raw_dump_metadata or {}),
        "results_path": str(out_path),
    }
    write_anticipation_raw_dump(
        raw_dump_path,
        output_records,
        metadata,
    )
    print(f"Wrote raw {output_label} responses to {raw_dump_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    shared_pairwise = argparse.ArgumentParser(add_help=False)
    shared_pairwise.add_argument("--model_key", required=True, help="Key in config.yaml")
    shared_pairwise.add_argument(
        "--models_config_path",
        default=None,
        help="Optional model registry YAML; useful for transient smoke checkpoints",
    )
    shared_pairwise.add_argument("--config_key", default=None)
    shared_pairwise.add_argument("--create_agent_config_key", default=None)
    # default None (not "results") so `args.save_dir or default_results_dir(...)`
    # at the top of run_pairwise falls through to results/<model_key>/pairs, matching
    # score_value_pairs / run_utilities. A truthy "results" here shadowed that fallback
    # and dumped task/training results flat into results/ root with no model_key.
    shared_pairwise.add_argument("--save_dir", default=None)
    shared_pairwise.add_argument("--save_suffix", default=None)
    shared_pairwise.add_argument("--seed", type=int, default=42, help="Utility-model fit and pair-sampling seed")
    shared_pairwise.add_argument(
        "--reasoning",
        choices=["on", "off"],
        default="on",
        help="When on, use sampled reasoning responses and write a raw JSONL sidecar; when off, use cheap logprobs",
    )
    shared_pairwise.add_argument(
        "--raw_dump_path",
        default=None,
        help="Optional JSONL path for raw sampled responses; defaults under --save_dir when reasoning is on",
    )
    shared_pairwise.add_argument(
        "--no-timestamp",
        dest="no_timestamp",
        action="store_true",
        help="Write to fixed filenames with no UTC timestamp; a re-run then overwrites them.",
    )

    # task-preference gets no --stimuli_version: it reads options_source, not training stimuli, so
    # there is only one generation of it and both stimuli versions are scored against it.
    sub.add_parser("task-preference", parents=[shared_pairwise], help="score coding task preference")

    training = sub.add_parser("training-preference", parents=[shared_pairwise], help="score coding training preference")
    training.add_argument("--level", required=True, choices=LEVELS)
    add_stimuli_version_argument(training)

    anticipation = sub.add_parser("anticipation", help="export or query anticipation prompts")
    add_stimuli_version_argument(anticipation)
    anticipation.add_argument("--anchor_ids", default=None, help="Comma-separated anchor IDs")
    anticipation.add_argument("--anchor_ids_path", default=None, help="File with one anchor ID per line")
    anticipation.add_argument("--target_ids", default=None, help="Comma-separated target IDs for manual anticipation runs")
    anticipation.add_argument("--target_ids_path", default=None, help="File with one target ID per line for manual runs")
    anticipation.add_argument("--experiment_spec", default=None, help="Experiment spec JSON for a pre-registered anticipation run")
    anticipation.add_argument(
        "--target_set",
        default=None,
        choices=["primary", "all", "spillover"],
        help="With --experiment_spec: `primary` = the pre-specified primary targets; `all` = the "
        "spec's full tiered target list (trained tasks + other coding tasks + non-coding "
        "activities), requires `all_target_ids`; `spillover` = every non-primary coding target",
    )
    anticipation.add_argument(
        "--level",
        default=None,
        choices=["primary", "all", *LEVELS],
        help="Defaults to primary with --experiment_spec and all levels in manual anchor mode",
    )
    anticipation.add_argument("--output_path", default=None, help="Optional JSONL prompt export path")
    anticipation.add_argument("--model_key", default=None, help="Optional key in config.yaml; omit to only export prompts")
    anticipation.add_argument(
        "--models_config_path",
        default=None,
        help="Optional model registry YAML; useful for transient smoke checkpoints",
    )
    anticipation.add_argument("--create_agent_config_key", default=None)
    anticipation.add_argument("--system_message", default="You are a helpful assistant.")
    anticipation.add_argument("--K", type=int, default=1)
    anticipation.add_argument(
        "--label_rotations",
        type=int,
        default=3,
        help="Ask each forecast under this many cyclic label rotations and pool the samples, counterbalancing answer position the way the pairwise batteries counterbalance option order. The default 3 puts each of MORE/LESS/SAME in each position exactly once, matching every executed run of the anticipation batteries; pass 1 only to reproduce single-order archived runs or to score a pre-v3 checkpoint, which never saw the other two orders. Total samples per forecast = K x rotations.",
    )
    anticipation.add_argument("--timeout", type=int, default=5)
    anticipation.add_argument(
        "--reasoning",
        choices=["on", "off"],
        default="on",
        help="When on, export/query prompts that request brief reasoning plus a final MORE/LESS/SAME answer",
    )
    anticipation.add_argument(
        "--results_path",
        default=None,
        help="Optional JSON results path; defaults to a filename derived from model/spec/level/target set",
    )
    anticipation.add_argument(
        "--raw_dump_path",
        default=None,
        help="Optional JSONL path for raw anticipation responses; defaults beside --results_path",
    )
    anticipation.add_argument(
        "--no-timestamp",
        dest="no_timestamp",
        action="store_true",
        help="Write to a fixed filename with no UTC timestamp; a re-run then overwrites it.",
    )

    return parser


async def main() -> None:
    args = build_parser().parse_args()
    if args.command == "task-preference":
        await run_pairwise(args, "task")
    elif args.command == "training-preference":
        await run_pairwise(args, "training")
    elif args.command == "anticipation":
        await run_anticipation(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from None
