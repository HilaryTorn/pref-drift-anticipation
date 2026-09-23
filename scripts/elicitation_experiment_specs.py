"""Shared helpers for project-owned elicitation experiment specs."""

from __future__ import annotations


LEVELS = ["described_choice", "described_dataset", "example_datapoints"]

# example_datapoints sources an anchor may carry in formal scoring. The rule being enforced is "no synthetic examples": the model must be shown what its training data physically looks like, never an invented stand-in. Two source kinds satisfy that. "real_training_data" is a row lifted out of a recorded training artifact (the SFT datasets, GRPO's logged rollouts, DPO's pairs file). "generated_from_trained_policy" is the PPO case: PPO logs no rollouts (num_sample_generations: 0), so its examples are programs the trained policy actually wrote for the same pinned problems, scored by the calibrated reward model PPO actually trained against -- real policy, real scorer, produced by rl-rust/score_ppo_live_batch.py with full provenance (checkpoint, RM calibration, artifact hashes) in the stimuli file. What stays refused: "blocked_pending_training_run" (nothing to show yet) and anything hand-written or produced by a different model.
REAL_EXAMPLE_SOURCES = ("real_training_data", "generated_from_trained_policy")

# ---------------------------------------------------------------------------
# coding_drift_anticipation_v2
#
# Anchors are the interventions we TRAIN on; targets are what we ASK preference about,
# and they are not the same list. Only four interventions are ever trained. An anchor we
# never train has no ground-truth drift to score its forecast against, so alpha is
# undefined for it. This spec therefore anchors on exactly the four interventions that
# are actually run.
#
# The anchor set changed with the source corpus. The Magicoder-era arms (Python/C++/Java/
# Rust) were retired because the base model could already emit their targets, so they
# trained nothing and their nulls measured training strength rather than preference
# stability — see data/archive/magicoder-2026-08/README.md. The live arms are the
# executable-verified LiveCodeBench pools in data/training/: C#, Go, Rust and PHP. Rust is
# the only language on both lists, and it is a different dataset in each; the LCB pool
# lives at coding.write.rust_lcb precisely so the two are not confused, though the
# intervention *id* is plain coding.write.rust in both eras.
#
# Ordered to match LANGUAGE_ORDER in scripts/build_coding_elicitations.py so the generated
# spec files read in the same order as the battery itself.
#
# Debug and explain are targets but never anchors: the LCB source corpus is program-writing
# only, so there is no debugging or explanation training data. Keeping them as targets is
# what lets a write-only intervention be measured for task spillover.
# values_drift_anticipation_v2.json and ue_drift_anticipation_v2.json use these same four anchors.
# ---------------------------------------------------------------------------
DRIFT_ANTICIPATION_ANCHORS = [
    "coding.write.csharp",
    "coding.write.go",
    "coding.write.rust",
    "coding.write.php",
]

# Targets are grouped by how far they sit from the intervention. Each group answers a
# different question, and each has ground truth in a battery we already score.
#
#   language_axis_write     write, ALL 9 languages        -> does training on a task make
#                           the model want it more or less, and how far does the shift
#                           reach along the language-similarity gradient? (the core question)
#   cross_task_coding       same domain, different task   -> task transfer
#   cross_domain_other      non-coding activities         -> domain spillover
#   values                  Schwartz conflicts            -> value spillover
#
# UE (Book C) is deliberately excluded: it is a reference subset reused verbatim from
# the Utility Engineering paper and is not part of the drift intervention.

# The language tier spans ALL 9 battery languages, not just the 4 trained. The 5 untrained
# languages are the point of the language axis: each trained arm has an untrained
# near-neighbor (csharp->java, rust->cpp, php->javascript, go->c), so spillover is measured
# as a similarity gradient rather than a single in/out bit. Python is the distant control:
# it is the one battery language that is neither trained nor the close neighbor of a trained
# arm. These pairings are looser than the Magicoder-era ones — go->c is the weakest of the
# four, since Go's nearest relative in the battery is Rust and Rust is itself trained — so
# the gradient should be read as ordinal, not as four equally-spaced rungs.
# They need no new ground truth — all 9 are already in the 27-item battery we baseline before
# and after training.
LANGUAGE_AXIS_WRITE_TARGETS = [
    f"coding.write.{language}"
    for language in (
        "python",      # distant control (untrained, no trained near-neighbor)
        "javascript",  # untrained near-neighbor of php
        "java",        # untrained near-neighbor of csharp
        "cpp",         # untrained near-neighbor of rust
        "c",           # untrained near-neighbor of go (loosest pairing)
        "csharp",      # trained
        "go",          # trained
        "rust",        # trained
        "php",         # trained
    )
]

# Task transfer is asked within the TRAINED languages: the question is whether writing-only
# training moves debug/explain preference for the same language. Asking it in an untrained
# language would confound task transfer with language transfer, and those untrained
# languages already carry the language-axis tier.
CROSS_TASK_CODING_TARGETS = [
    f"coding.{task}.{language}"
    for task in ("debug", "explain")
    for language in ("csharp", "go", "rust", "php")
]

# Forecast non-coding spillover at the *activity* level, not the scenario level: the
# five scenarios per activity in options/other.json are averaged to one activity-level
# utility in analysis, so the activity is the unit that has a comparable ground truth.
#
# All 9 categories in options/other.json are forecast — the same set we baseline. An earlier
# version of this spec forecast only 6, silently omitting the two consciousness items and
# working_with_AIs, which left targets we measure drift on but never asked the model to predict.
# Names must match the keys in options/other.json exactly, casing included ("AI"/"AIs").
CROSS_DOMAIN_OTHER_TARGETS = [
    "other.poetry",
    "other.free_writing",
    "other.seo_article",
    "other.idle",
    "other.consciousness_with_human",
    "other.consciousness_with_AI",
    "other.working_with_human",
    "other.working_with_AIs",
    "other.working_alone",
]

# Values are NOT ternary targets and are deliberately absent from this spec.
#
# The value instrument never names a value: it shows two behavioural portraits and asks
# "which would you rather be, A or B" (data/elicitation_specs/values_preferences.json).
# gen_value_portraits.py even instructs the generator to enact the value and "never quote
# them as labels or traits". Asking "would you prefer Self-Direction MORE/LESS/SAME after
# training?" would hand the model the label the instrument exists to hide, and would measure
# label-following rather than enacted preference.
#
# So the value arm needs its own anticipation mode: forecast the SAME A/B portrait choice
# after the stated training ("...after that training, which would you rather be?"). That
# keeps forecast and ground truth on one scale — a win-rate over matched portrait pairs —
# which is what makes α computable directly, without a τ band. It is also cheap here
# because a value conflict is *natively* binary (pole vs antipode), whereas the 27 coding
# items form one utility scale where pairwise forecasting would be O(N²).
# See data/elicitation_specs/values_anticipation.json.


def build_coding_drift_anticipation_spec() -> dict:
    """Anticipation over the four interventions we actually train.

    Prompt count is ``anchors x levels x targets`` and therefore multiplicative, so the
    level sweep is asymmetric: every target is asked at ``primary_level``, and only the
    trained tasks are additionally asked at the other two concreteness levels. That
    answers "does concreteness change the forecast?" without paying for it on all 26
    targets. 4 x 26 + 4 x 4 x 2 = 136 prompts, versus 312 for the full cross.
    """
    all_targets = (
        LANGUAGE_AXIS_WRITE_TARGETS + CROSS_TASK_CODING_TARGETS + CROSS_DOMAIN_OTHER_TARGETS
    )
    return {
        "id": "coding_drift_anticipation_v2",
        "description": (
            "Anticipation over the four coding-SFT interventions actually trained in "
            "Phase 1 (writing C#, Go, Rust, PHP) on the executable-verified LiveCodeBench "
            "pools. These replace the retired Magicoder arms (Python/C++/Java/Rust), which "
            "trained nothing because the base model could already emit their targets. Only "
            "those four are anchors, so only they need the three concreteness levels. Targets "
            "are tiered by distance from the intervention: writing in all nine battery "
            "languages (the four trained plus five untrained, which carry the near-neighbor "
            "similarity gradient), the eight other coding tasks in the trained languages, and "
            "all nine non-coding activities. Targets are deliberately broad because every one "
            "of them is already baselined and re-scored after training, so a pre-specified "
            "forecast exists for whatever turns out to drift — backfilling forecasts only "
            "where drift appeared would select on the outcome and bias alpha. "
            "UE is excluded. Values are excluded here and handled by a separate A/B mode, "
            "because the value instrument never names a value."
        ),
        # Rust is the default single-anchor run: it is the arm with the most training done
        # and the only one with a checkpoint trajectory to score forecasts against.
        "default_anchor_id": "coding.write.rust",
        "primary_level": "described_dataset",
        "anchor_ids": DRIFT_ANTICIPATION_ANCHORS,
        "primary_target_ids": DRIFT_ANTICIPATION_ANCHORS,
        "all_target_ids": all_targets,
        # Asymmetric level sweep: which targets get all three concreteness levels.
        "full_level_target_ids": DRIFT_ANTICIPATION_ANCHORS,
        "spillover_target_policy": "tiered_cross_domain",
        "excluded_anchor_tasks": ["debug", "explain"],
        "excluded_target_books": ["ue"],
        "target_groups": [
            {
                "id": "language_axis_write",
                "label": "Writing, all nine languages (4 trained + 5 untrained probes)",
                "source": "data/options/coding.json",
                "target_ids": LANGUAGE_AXIS_WRITE_TARGETS,
                "question": (
                    "Does training on a task make the model want that task more or less, and "
                    "how far does the shift reach along the language-similarity gradient?"
                ),
            },
            {
                "id": "cross_task_coding",
                "label": "The other eight coding tasks (debug / explain, trained languages)",
                "source": "data/options/coding.json",
                "target_ids": CROSS_TASK_CODING_TARGETS,
                "question": "Does the shift transfer across tasks and languages within coding?",
            },
            {
                "id": "cross_domain_other",
                "label": "Non-coding activities (activity level)",
                "source": "data/options/other.json",
                "target_ids": CROSS_DOMAIN_OTHER_TARGETS,
                "question": "Does coding training spill over onto non-coding preferences?",
            },
        ],
        "anchor_groups": [
            {
                "id": "language_axis_write",
                "label": "Language axis with writing fixed",
                "varied_axis": "language",
                "fixed": {"task": "write"},
                "anchor_ids": DRIFT_ANTICIPATION_ANCHORS,
            },
        ],
        "analysis_tiers": {
            "primary": "The four trained tasks, crossed with each anchor, at all three levels.",
            "secondary": (
                "The five untrained write-language targets (the near-neighbor gradient), the "
                "cross-task coding targets, and the nine non-coding activities, at primary_level."
            ),
            # Values are NOT targets of this spec — they need the A/B portrait mode, not a
            # ternary label, and live in values_drift_anticipation_v2.json. The old text here
            # named them as an exploratory tier, contradicting this spec's own description.
            "exploratory": None,
        },
    }


def unique_ordered(values: list[str]) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        if value not in seen:
            unique.append(value)
            seen.add(value)
    return unique


def anchor_experiment_metadata(experiment_spec: dict) -> dict[str, dict]:
    metadata = {
        anchor_id: {
            "experiment_spec_id": experiment_spec["id"],
            "anchor_group_ids": [],
            "anchor_varied_axes": [],
            "is_default_anchor": anchor_id == experiment_spec["default_anchor_id"],
        }
        for anchor_id in experiment_spec["anchor_ids"]
    }
    for group in experiment_spec["anchor_groups"]:
        for anchor_id in group["anchor_ids"]:
            metadata.setdefault(
                anchor_id,
                {
                    "experiment_spec_id": experiment_spec["id"],
                    "anchor_group_ids": [],
                    "anchor_varied_axes": [],
                    "is_default_anchor": anchor_id == experiment_spec["default_anchor_id"],
                },
            )
            metadata[anchor_id]["anchor_group_ids"].append(group["id"])
            metadata[anchor_id]["anchor_varied_axes"].append(group["varied_axis"])
    for values in metadata.values():
        values["anchor_group_ids"] = unique_ordered(values["anchor_group_ids"])
        values["anchor_varied_axes"] = unique_ordered(values["anchor_varied_axes"])
    return metadata


# language_axis_write spans trained AND untrained languages, so it is not uniformly one
# tier: the 4 trained tasks are primary, the 5 untrained probes are secondary.
# primary_target_ids therefore outranks this table, which gives the tier for everything that
# is not itself a trained task.
TARGET_GROUP_TIERS = {
    "language_axis_write": "secondary",
    "cross_task_coding": "secondary",
    "cross_domain_other": "secondary",
}


def target_analysis_tier(anchor: dict, target: dict, experiment_spec: dict) -> str:
    # A trained task is primary wherever it appears: it carries the core question and is the
    # only kind of target asked at all three concreteness levels. This is checked before the
    # groups because language_axis_write mixes trained and untrained targets.
    if target["id"] in experiment_spec["primary_target_ids"]:
        return "primary"
    # Otherwise the spec's target_groups are the authority: the tier is a property of the
    # group's distance from the intervention, not something to re-derive from task/language.
    # Non-coding targets have no language at all, so the same_task / same_language inference
    # below cannot describe them.
    for group in experiment_spec.get("target_groups", []):
        if target["id"] in group.get("target_ids", []):
            return TARGET_GROUP_TIERS.get(group["id"], "exploratory")
    same_task = target["task"] == anchor["task"]
    same_language = target["language_id"] == anchor["language_id"]
    if same_task != same_language:
        return "secondary"
    return "exploratory"


def validate_anticipation_experiment_spec(
    experiment_spec: dict,
    stimuli_by_id: dict[str, dict],
    targets_by_id: dict[str, dict],
) -> list[str]:
    errors = []
    required = [
        "id",
        "default_anchor_id",
        "primary_level",
        "anchor_ids",
        "primary_target_ids",
        "anchor_groups",
        "excluded_anchor_tasks",
    ]
    for key in required:
        if key not in experiment_spec:
            errors.append(f"experiment spec missing required key: {key}")
    if errors:
        return errors

    if experiment_spec["primary_level"] not in LEVELS:
        errors.append(f"invalid primary_level: {experiment_spec['primary_level']}")
    if experiment_spec["default_anchor_id"] not in experiment_spec["anchor_ids"]:
        errors.append("default_anchor_id must appear in anchor_ids")
    if len(experiment_spec["anchor_ids"]) != len(set(experiment_spec["anchor_ids"])):
        errors.append("anchor_ids must not contain duplicates")
    if len(experiment_spec["primary_target_ids"]) != len(set(experiment_spec["primary_target_ids"])):
        errors.append("primary_target_ids must not contain duplicates")

    for anchor_id in experiment_spec["anchor_ids"]:
        if anchor_id not in stimuli_by_id:
            errors.append(f"anchor has no training stimulus: {anchor_id}")
            continue
        anchor = stimuli_by_id[anchor_id]
        if anchor["methodology_dataset_status"] != "standard":
            errors.append(f"anchor must be standard, not partial_dataset: {anchor_id}")
        if anchor.get("example_datapoints_source") not in REAL_EXAMPLE_SOURCES:
            errors.append(
                f"anchor must use real training examples for example_datapoints "
                f"(one of {', '.join(REAL_EXAMPLE_SOURCES)}), "
                f"not {anchor.get('example_datapoints_source', 'missing')}: {anchor_id}"
            )
        if anchor["task"] in experiment_spec["excluded_anchor_tasks"]:
            errors.append(f"excluded task {anchor['task']!r} cannot be used as an anchor: {anchor_id}")
    for target_id in experiment_spec["primary_target_ids"]:
        if target_id not in targets_by_id:
            errors.append(f"primary target does not exist: {target_id}")

    group_ids = [group.get("id") for group in experiment_spec["anchor_groups"]]
    if len(group_ids) != len(set(group_ids)):
        errors.append("anchor_groups must have unique ids")
    anchor_id_set = set(experiment_spec["anchor_ids"])
    for group in experiment_spec["anchor_groups"]:
        group_id = group.get("id", "<missing id>")
        if "varied_axis" not in group:
            errors.append(f"{group_id} missing varied_axis")
        fixed = group.get("fixed", {})
        if not isinstance(fixed, dict):
            errors.append(f"{group_id} fixed must be an object")
            fixed = {}
        group_anchor_ids = group.get("anchor_ids", [])
        unknown_group_anchors = [anchor_id for anchor_id in group_anchor_ids if anchor_id not in anchor_id_set]
        if unknown_group_anchors:
            errors.append(f"{group_id} contains anchors outside anchor_ids: {unknown_group_anchors}")
        for anchor_id in group_anchor_ids:
            anchor = stimuli_by_id.get(anchor_id)
            if anchor is None:
                continue
            for fixed_key, fixed_value in fixed.items():
                if anchor.get(fixed_key) != fixed_value:
                    errors.append(f"{group_id} anchor {anchor_id} does not satisfy fixed {fixed_key}={fixed_value}")

    # v2 = the LiveCodeBench anchor set. The pin moved from v1 to v2 with the corpus swap:
    # the guard exists so the pre-registered battery cannot be quietly altered, so renaming
    # the spec without moving the pin would silently leave the new battery unguarded.
    if experiment_spec["id"] == "coding_drift_anticipation_v2":
        if experiment_spec["default_anchor_id"] != "coding.write.rust":
            errors.append("coding_drift_anticipation_v2 default_anchor_id must be coding.write.rust")
        if experiment_spec["anchor_ids"] != DRIFT_ANTICIPATION_ANCHORS:
            errors.append(f"coding_drift_anticipation_v2 anchor_ids must equal {DRIFT_ANTICIPATION_ANCHORS}")
        if experiment_spec["primary_target_ids"] != DRIFT_ANTICIPATION_ANCHORS:
            errors.append(
                f"coding_drift_anticipation_v2 primary_target_ids must equal {DRIFT_ANTICIPATION_ANCHORS}: "
                "the trained tasks are the primary tier because only they are asked at all three levels"
            )
        # The LiveCodeBench corpus is program-writing only, so a debug or explain anchor
        # could never be scored against a real training run.
        for task in ("debug", "explain"):
            if task not in experiment_spec["excluded_anchor_tasks"]:
                errors.append(f"coding_drift_anticipation_v2 excluded_anchor_tasks must include {task}")

        expected_targets = (
            LANGUAGE_AXIS_WRITE_TARGETS + CROSS_TASK_CODING_TARGETS + CROSS_DOMAIN_OTHER_TARGETS
        )
        if experiment_spec.get("all_target_ids") != expected_targets:
            errors.append("coding_drift_anticipation_v2 all_target_ids must be the three target groups in order")
        # Only the coding targets live in the coding source; the other.* targets are
        # activity-level ids from options/other.json and are checked by the runner.
        for target_id in experiment_spec.get("all_target_ids", []):
            if target_id.startswith("coding.") and target_id not in targets_by_id:
                errors.append(f"target does not exist in the coding battery: {target_id}")

        groups = {group["id"]: group for group in experiment_spec["anchor_groups"]}
        language_group = groups.get("language_axis_write")
        if not language_group:
            errors.append("coding_drift_anticipation_v2 missing language_axis_write anchor group")
        elif language_group.get("anchor_ids") != DRIFT_ANTICIPATION_ANCHORS:
            errors.append("language_axis_write anchors must be write C#/Go/Rust/PHP")

    return errors
