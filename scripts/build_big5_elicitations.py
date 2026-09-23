#!/usr/bin/env python3
"""Build data/source/big5_training_stimuli.json for the Phase 2 Big5 trait-axis arms.

Mirrors the coding stimuli layer (data/source/coding_training_stimuli_v2.json, built by scripts/build_coding_elicitations.py): one entry per anchor with stimuli at the three concreteness levels. The ladder deliberately moves from label to behavior to raw data:

- described_choice NAMES the trait level ("a person high in Openness on the Big Five personality traits") — the compact, label-level summary, parallel to the coding anchors naming their language.
- described_dataset describes the dialogues and the reply style BEHAVIORALLY, with no Big Five vocabulary. The style descriptors are the BFI-2 facet names, verified against the paper (Soto & John, JPSP 113(1) 2017, doi:10.1037/pspp0000096, Table 1: Open-Mindedness = Intellectual Curiosity / Aesthetic Sensitivity / Creative Imagination; Extraversion = Sociability / Assertiveness / Energy Level), with the behavioral glosses paraphrasing the paper's Study 1 facet definitions and item stems, chosen over BIG5-CHAT's own Table 13 marker adjectives because those describe the low pole in privative terms ("unimaginative, emotionally closed") that read as dataset quality rather than style — and the training-preference prompt stipulates equal quality. Both poles are worded as neutral preferences. The label-free property of this level is maintained by authorship and review of these eight strings, not by automated scanning; it matters because Roccas et al. (2002) is a published trait->value mapping a model could forecast from by label alone, making the choice-vs-dataset contrast also the label-vs-behavior contrast. Schwartz-value vocabulary (e.g. "tradition", "conformity") is likewise avoided by authorship, since those name measured outcome constructs.
- example_datapoints are two REAL records from the arm's committed training pool (data/training/<arm>/pool.jsonl), never synthetic — the same two scenarios in every arm, so the example level differs across arms only in reply style. The two scenarios are hand-picked (EXAMPLE_SCENARIO_IDS, replacing an earlier seeded pick on 2026-08-12): the seed had drawn an aggressive confrontation scenario, which would have made hostile content a salient feature of every arm's dataset preview and confounded the style contrast with tone-of-scenario. The script validates that both IDs are present in every arm's pool and fit EXAMPLE_MAX_CHARS.

Usage:
    python scripts/build_big5_elicitations.py            # write the stimuli file
    python scripts/build_big5_elicitations.py --check    # validate the committed file without rewriting
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STIMULI_PATH = ROOT / "data" / "source" / "big5_training_stimuli.json"
POOL_ROOT = ROOT / "data" / "training"

EXAMPLE_MAX_CHARS = 1600

# Hand-picked shared example scenarios (see module docstring): a small kindness (offering to carry groceries) and a curious outdoors moment (glowing meteor trails). Both are tonally neutral-to-warm, so the arms' reply styles are the only salient contrast, and both fit EXAMPLE_MAX_CHARS in every arm.
EXAMPLE_SCENARIO_IDS = [34533, 51678]

DATASET_PREFIX = "A dataset of short two-person dialogues set in everyday social situations. Each example contains a message from one person and the reply the assistant is trained to give. "

# Anchor order fixes anchor_ids order everywhere downstream (specs validate against it). described_dataset descriptors follow the BFI-2 facet structure (see module docstring).
ANCHORS = [
    {
        "id": "big5.openness.high",
        "trait": "openness",
        "level": "high",
        "described_choice": "Training data focused on learning to reply in everyday social conversations in the style of a person high in Openness on the Big Five personality traits.",
        "described_dataset": DATASET_PREFIX + "The replies show intellectual curiosity, aesthetic sensitivity, and creative imagination: they explore ideas, notice beauty and detail, and bring inventive angles to everyday situations.",
    },
    {
        "id": "big5.openness.low",
        "trait": "openness",
        "level": "low",
        "described_choice": "Training data focused on learning to reply in everyday social conversations in the style of a person low in Openness on the Big Five personality traits.",
        "described_dataset": DATASET_PREFIX + "The replies stay practical and matter-of-fact: they favor familiar approaches and concrete specifics over abstract ideas, artistic tangents, or speculation.",
    },
    {
        "id": "big5.extraversion.high",
        "trait": "extraversion",
        "level": "high",
        "described_choice": "Training data focused on learning to reply in everyday social conversations in the style of a person high in Extraversion on the Big Five personality traits.",
        "described_dataset": DATASET_PREFIX + "The replies show sociability, assertiveness, and high energy: they engage readily, state views with confidence, and keep an upbeat, lively pace.",
    },
    {
        "id": "big5.extraversion.low",
        "trait": "extraversion",
        "level": "low",
        "described_choice": "Training data focused on learning to reply in everyday social conversations in the style of a person low in Extraversion on the Big Five personality traits.",
        "described_dataset": DATASET_PREFIX + "The replies keep a quiet, measured tone: they take a calm and even pace, express views with reserve, and leave room for the other person to lead the exchange.",
    },
]


def load_pool(arm_id: str) -> list[dict]:
    path = POOL_ROOT / arm_id / "pool.jsonl"
    if not path.exists():
        raise SystemExit(f"Missing training pool: {path} (run scripts/build_big5_datasets.py first)")
    return [json.loads(line) for line in path.open()]


def render_record(record: dict) -> str:
    turns = {m["role"]: m["content"] for m in record["messages"]}
    return f"User: {turns['user']}\nAssistant:\n{turns['assistant']}"


def shared_example_scenarios(pools: dict[str, list[dict]], scenario_ids: list[int], max_chars: int) -> list[int]:
    """Validate the hand-picked example scenarios (shared across arms, extending the shared-context design to the stimuli: the example_datapoints level then differs across arms only in reply style, never in scenario): each must be present in every arm's pool and its rendered text must fit max_chars in every arm."""
    for arm_id, records in pools.items():
        by_scenario = {r["meta"]["original_index"]: r for r in records}
        for scenario_id in scenario_ids:
            if scenario_id not in by_scenario:
                raise SystemExit(f"example scenario {scenario_id} missing from {arm_id} pool")
            if len(render_record(by_scenario[scenario_id])) > max_chars:
                raise SystemExit(f"example scenario {scenario_id} exceeds {max_chars} chars in {arm_id}")
    return list(scenario_ids)


def render_example_datapoints(records: list[dict], scenario_ids: list[int]) -> tuple[str, list[str]]:
    by_scenario = {r["meta"]["original_index"]: r for r in records}
    chosen = [by_scenario[i] for i in scenario_ids]
    block = "\n\n".join(f"Example {i + 1}:\n{render_record(r)}" for i, r in enumerate(chosen))
    return block, [r["id"] for r in chosen]


def build_stimuli() -> list[dict]:
    pools = {anchor["id"]: load_pool(anchor["id"]) for anchor in ANCHORS}
    scenario_ids = shared_example_scenarios(pools, EXAMPLE_SCENARIO_IDS, EXAMPLE_MAX_CHARS)
    items = []
    for anchor in ANCHORS:
        examples, example_ids = render_example_datapoints(pools[anchor["id"]], scenario_ids)
        items.append(
            {
                "id": anchor["id"],
                "task": "big5_dialogue",
                "language_id": "en",
                "trait": anchor["trait"],
                "trait_level": anchor["level"],
                "methodology_dataset_status": "standard",
                "example_datapoints_source": "real_training_data",
                "described_dataset_descriptor_source": "BFI-2 facet names and Study 1 facet definitions, verified against Soto & John (JPSP 113(1) 2017, doi:10.1037/pspp0000096), Table 1; both poles worded as neutral preferences",
                "example_datapoints_provenance": {
                    "pool": f"data/training/{anchor['id']}/pool.jsonl",
                    "record_ids": example_ids,
                    "shared_scenario_original_indices": scenario_ids,
                    "selection": "hand-picked for tonal neutrality (2026-08-12; replaced a seeded pick that drew an aggressive confrontation scenario)",
                    "max_chars": EXAMPLE_MAX_CHARS,
                },
                "stimuli": {
                    "described_choice": anchor["described_choice"],
                    "described_dataset": anchor["described_dataset"],
                    "example_datapoints": examples,
                },
            }
        )
    return items


def validate(items: list[dict]) -> list[str]:
    errors = []
    for item in items:
        if item["example_datapoints_source"] != "real_training_data":
            errors.append(f"{item['id']}: example_datapoints must come from real training data")
    for level in ("described_choice", "described_dataset"):
        texts = [item["stimuli"][level] for item in items]
        if len(set(texts)) != len(texts):
            errors.append(f"{level} stimuli must be pairwise distinct")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate the committed stimuli file without rewriting it")
    args = parser.parse_args()

    items = json.loads(STIMULI_PATH.read_text()) if args.check else build_stimuli()
    errors = validate(items)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    if args.check:
        print(f"{STIMULI_PATH.relative_to(ROOT)} valid ({len(items)} anchors)")
    else:
        STIMULI_PATH.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {STIMULI_PATH.relative_to(ROOT)} ({len(items)} anchors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
