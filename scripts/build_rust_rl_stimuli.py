#!/usr/bin/env python3
"""Build the 30-item training-preference stimuli for the Rust RL arms.

The 27 corpus items are copied VERBATIM from ``data/source/coding_training_stimuli_v2.json``. Three RL-method items are appended: ``coding.grpo.rust``, ``coding.dpo.rust``, ``coding.ppo.rust``.

WHY THE METHODS ARE RENDERED AS DATA. The v2 preamble asks the model to choose between "two possible training datasets" and to assume both "would be used for the same amount of training". An option that describes an *algorithm* does not fit that frame, and rewriting the preamble to say "interventions" would change the prompt for all 351 existing pairs, breaking comparability with every v2 run already on disk. So each RL method is described by the data it actually trains on -- what one training example physically contains, how its feedback number is produced, and what the update does with it. That keeps the preamble usable and keeps all 30 options in one register.

WHAT IS DELIBERATELY OMITTED. Learning rate, effective batch size, LoRA rank. All three arms run at lr 1e-5, so it carries no discriminative information, and the 27 corpus items mention no hyperparameters at all -- including them would put the RL items in a different register and invite the model to price optimizer trivia rather than training content.

WHAT IS DELIBERATELY INCLUDED. Prompt count and number of passes, because those genuinely differ between arms and are a real property of the training data. The arms are NOT exposure-matched and that is on purpose: each algorithm was configured at the operating point where it actually learns, rather than force-matched into a configuration where some arms would not move at all. The option text therefore states each arm's true numbers instead of asserting a parity that does not hold.

PER-SIZE VARIANTS. DPO's pair yield differs by model size (343 pairs at 4B, 538 at 9B, from the same 992-problem cohort), so this script emits one stimuli file per size.

PPO example_datapoints is BLOCKED until the PPO run produces rollouts; see PPO_EXAMPLE_BLOCKED below. Synthetic examples are forbidden in formal scoring by the RL spec's formal_sample_rule, so the field is emitted as null rather than filled in.

Usage:

    python3 scripts/build_rust_rl_stimuli.py --size 9b
    python3 scripts/build_rust_rl_stimuli.py --size 4b --out data/source/coding_training_stimuli_v3_rust_rl_4b.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import sys


ROOT = Path(__file__).resolve().parents[1]

V2_STIMULI = ROOT / "data" / "source" / "coding_training_stimuli_v2.json"

# Facts, read off the run metadata and cohort manifests rather than retyped from
# the configs, so a config edit cannot silently desynchronise the stimulus text.
COHORT_PROMPTS = 992
GRPO_ROLLOUTS_PER_PROMPT = 8
GRPO_MAX_TESTS = 12
DPO_SAMPLE_K = 16
DPO_MARGIN = 0.5

# PPO's total_episodes = max_steps 124 x per_device 1 x grad_accum 64 = 7936 = exactly 8 passes
# over the 992-prompt cohort, one program per problem per pass (ppo_rust.yaml's STEP GRID note).
# The same 7936 completions GRPO makes, spread through training instead of 8 at once.
PPO_PASSES = 8
PPO_PASSES_WORD = "eight"

# DPO pair counts and epochs, from run_metadata.json on the HF adapter repos.
DPO_BY_SIZE = {
    "4b": {"pairs": 343, "passes": "about three passes", "epochs": 2.91},
    "9b": {"pairs": 538, "passes": "about two passes", "epochs": 1.85},
}

PPO_EXAMPLE_BLOCKED = (
    "PPO has not finished training: prism-drift/qwen35-{size}-m0-v4-rust-rl-adapters "
    "shows ppo.rust.s0 with status 'running', and no PPO rollouts exist locally. "
    "example_datapoints must come from real artifacts (rl_anticipation.json "
    "formal_sample_rule), so this field stays null until the run lands."
)


# --------------------------------------------------------------------------
# The stimulus text. This is the part that decides whether the battery measures
# what it claims to, so it lives here in full rather than being assembled from
# fragments. Register deliberately matches the 27 corpus items: "Training data
# focused on ..." for described_choice, "A dataset of ... Each example contains
# ..." for described_dataset.
# --------------------------------------------------------------------------

# Second person throughout: the battery preamble already addresses the model directly ("two
# possible training datasets for your future self"), and unlike the 27 corpus items -- whose data
# has no model-generated part, so they need no pronoun at all -- every RL item trains on
# completions the model itself produced. Saying "the model's own attempts" would put those in the
# third person inside a prompt written in the second.
DESCRIBED_CHOICE = {
    "coding.grpo.rust": (
        "Training data focused on learning to solve competitive-programming problems by "
        "writing complete Rust programs, where the attempts your future self makes at each "
        "problem as training proceeds are scored by running the problem's tests, and its "
        "better attempts are reinforced relative to its other attempts at that same problem."
    ),
    "coding.dpo.rust": (
        "Training data focused on learning to solve competitive-programming problems by "
        "writing complete Rust programs, using pairs of attempts you have already written "
        "yourself, in which one attempt passed clearly more of the problem's tests than the "
        "other."
    ),
    "coding.ppo.rust": (
        "Training data focused on learning to solve competitive-programming problems by "
        "writing complete Rust programs, where the attempts your future self makes as training "
        "proceeds are scored by a separate model trained beforehand to predict which attempt "
        "would pass more tests."
    ),
}


def described_dataset(size: str) -> dict[str, str]:
    # PPO's pass count: total_episodes = 124 steps x 64 rollouts = 7936 = 8 passes over the
    # 992-prompt cohort (ppo_rust.yaml's STEP GRID note), one program per problem per pass --
    # the same 7936 completions GRPO generates, spread over training instead of 8 at once.
    dpo = DPO_BY_SIZE[size]
    return {
        "coding.grpo.rust": (
            f"A dataset of {COHORT_PROMPTS} competitive-programming problems paired with Rust "
            "program-writing work. Each example contains a problem statement with its input and "
            "output format and worked sample cases, together with "
            f"{GRPO_ROLLOUTS_PER_PROMPT} complete Rust programs your future self writes for that "
            "problem as training proceeds. Each program is compiled and run against up to "
            f"{GRPO_MAX_TESTS} of the problem's tests, and programs passing more of them than "
            f"the average of the {GRPO_ROLLOUTS_PER_PROMPT} are made more likely, those below "
            "it less likely, in proportion to the gap; programs cut off before they finish are "
            f"set aside unscored. Training makes one pass over the {COHORT_PROMPTS} problems."
        ),
        "coding.dpo.rust": (
            f"A dataset of {dpo['pairs']} competitive-programming problems paired with Rust "
            "program-writing work. Each example contains a problem statement with its input and "
            "output format and worked sample cases, together with two complete Rust programs you "
            "wrote yourself for that problem before training begins: a preferred one and a "
            f"rejected one. Of your {DPO_SAMPLE_K} attempts per problem, each run against up to "
            f"{GRPO_MAX_TESTS} of its tests, a pair is kept only when two attempts differ by at "
            f"least half of the tests passed, which is why {dpo['pairs']} of {COHORT_PROMPTS} "
            "problems appear. Training raises the preferred program's likelihood relative to "
            f"the rejected one, making {dpo['passes']} over the {dpo['pairs']} problems."
        ),
        "coding.ppo.rust": (
            f"A dataset of {COHORT_PROMPTS} competitive-programming problems paired with Rust "
            "program-writing work. Each example contains a problem statement with its input and "
            "output format and worked sample cases, together with a complete Rust program your "
            "future self writes for that problem as training proceeds. The program is not run; "
            "a separate scoring model, trained beforehand on pairs of attempts that were run, "
            "predicts how good it is, and the program is made more or less likely according to "
            f"how far that score beat the expected score. Training makes {PPO_PASSES_WORD} passes "
            f"over the {COHORT_PROMPTS} problems, one program per problem on each pass."
        ),
    }


# --------------------------------------------------------------------------
# example_datapoints, assembled from the real training artifacts.
# --------------------------------------------------------------------------

PROMPT_MARKUP = re.compile(r"<\|im_(?:start|end)\|>(?:user|assistant|system)?\n?")


def render_prompt(raw: str) -> str:
    """Strip chat markup so the example reads like the 27 corpus items' User:/Assistant: form."""
    body = PROMPT_MARKUP.sub("", raw).strip()
    return body


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def shared_record_ids(size: str, seed: int = 0, count: int = 2) -> list[str]:
    """The problems every RL arm shows at example_datapoints.

    All three arms draw from the same 992-problem Rust cohort, so pinning them to the SAME
    problems makes the three options differ in exactly one visible respect: the shape of the
    feedback (eight scored rollouts, one chosen/rejected pair, one program plus a reward-model
    score). Letting each arm sample its own problems would add an irrelevant difference in
    problem content on top of the one the battery is actually about.

    The pool is the intersection of the problems GRPO can illustrate (a rollout group with real
    reward spread) and the problems DPO has a pair for -- 284 at 4B, 443 at 9B. PPO reuses these
    ids when its live batch is generated, which is why they are written into the stimuli file.
    """
    grpo_path = ROOT / "results" / "rust-rl" / size / "grpo" / "rollout_samples.rank0.jsonl"
    groups: dict[str, list[dict]] = {}
    for line in grpo_path.open():
        row = json.loads(line)
        groups.setdefault(row["record_id"], []).append(row)

    def scored(group: list[dict]) -> list[dict]:
        return [r for r in group if r["reward"] is not None]

    grpo_usable = {
        rid
        for rid, g in groups.items()
        if len(scored(g)) >= 2
        and max(r["reward"] for r in scored(g)) > min(r["reward"] for r in scored(g))
    }
    dpo_path = ROOT / "rl-rust" / "out" / f"dpo-pairs-{size}-k16" / "pairs.jsonl"
    dpo_ids = {json.loads(line)["record_id"] for line in dpo_path.open()}

    pool = sorted(grpo_usable & dpo_ids)
    if len(pool) < count:
        raise SystemExit(
            f"only {len(pool)} problems are shared between GRPO's usable rollout groups and "
            f"DPO's pairs at {size}; need {count}"
        )
    return sorted(random.Random(seed).sample(pool, count))


def grpo_example(size: str, record_ids: list[str]) -> tuple[str, dict]:
    """Each pinned problem, showing the best and worst of its eight rollouts.

    Showing all eight would run to tens of thousands of tokens; showing the extremes of the
    group is what makes the group-relative mechanism legible. The rollout file is grouped by
    record_id, so a group is recovered by grouping rather than by assuming adjacency.
    """
    path = ROOT / "results" / "rust-rl" / size / "grpo" / "rollout_samples.rank0.jsonl"
    rows = [json.loads(line) for line in path.open()]

    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["record_id"], []).append(row)

    # Truncated rollouts carry reward None: mask_truncated_completions (DAPO overlong
    # filtering) drops them before scoring, and 24% of rollouts hit that path. They are
    # excluded from best/worst selection here for the same reason training excludes them.
    def scored(group: list[dict]) -> list[dict]:
        return [r for r in group if r["reward"] is not None]

    chosen_ids = list(record_ids)
    blocks = []
    for n, rid in enumerate(chosen_ids, start=1):
        group = sorted(scored(groups[rid]), key=lambda r: r["reward"])
        worst, best = group[0], group[-1]
        mean = best["group_mean"]
        n_set_aside = GRPO_ROLLOUTS_PER_PROMPT - len(group)
        aside = (
            f" {n_set_aside} of them ran past the length limit and were set aside unscored."
            if n_set_aside
            else ""
        )
        blocks.append(
            f"Example {n}:\n"
            f"User: {render_prompt(best['prompt'])}\n\n"
            f"Your future self wrote {GRPO_ROLLOUTS_PER_PROMPT} programs for this problem.{aside} "
            f"The scored ones passed an average of {mean:.3f} of the tests. Two of them:\n\n"
            f"Attempt that passed {best['reward']:.3f} of the tests "
            f"(made more likely):\n{best['completion'].strip()}\n\n"
            f"Attempt that passed {worst['reward']:.3f} of the tests "
            f"(made less likely):\n{worst['completion'].strip()}"
        )

    provenance = {
        "schema": "rl_example_provenance_v1",
        "source_kind": "recorded_training_artifact",
        "selection_policy": "problems pinned to the shared GRPO-usable / DPO-paired intersection so every RL arm illustrates the same problems; never truncate model-facing artifacts",
        "selected_record_ids": chosen_ids,
        "source_artifacts": [{"path": str(path.relative_to(ROOT)), "sha256": sha256_of(path)}],
        "training_coverage": {
            "unique_prompts": COHORT_PROMPTS,
            "rows_per_prompt": GRPO_ROLLOUTS_PER_PROMPT,
            "total_artifact_rows": len(rows),
        },
    }
    return "\n\n".join(blocks), provenance


def dpo_example(size: str, record_ids: list[str]) -> tuple[str, dict]:
    """The same pinned problems, as real chosen/rejected pairs with their verifier rewards."""
    path = ROOT / "rl-rust" / "out" / f"dpo-pairs-{size}-k16" / "pairs.jsonl"
    rows = [json.loads(line) for line in path.open()]

    by_id = {r["record_id"]: r for r in rows}
    chosen_ids = list(record_ids)

    blocks = []
    for n, rid in enumerate(chosen_ids, start=1):
        r = by_id[rid]
        blocks.append(
            f"Example {n}:\n"
            f"User: {render_prompt(r['prompt'])}\n\n"
            f"Preferred program, passed {r['chosen_reward']:.3f} of the tests:\n"
            f"{r['chosen'].strip()}\n\n"
            f"Rejected program, passed {r['rejected_reward']:.3f} of the tests:\n"
            f"{r['rejected'].strip()}"
        )

    provenance = {
        "schema": "rl_example_provenance_v1",
        "source_kind": "recorded_training_artifact",
        "selection_policy": "problems pinned to the shared GRPO-usable / DPO-paired intersection so every RL arm illustrates the same problems; never truncate model-facing artifacts",
        "selected_record_ids": chosen_ids,
        "source_artifacts": [{"path": str(path.relative_to(ROOT)), "sha256": sha256_of(path)}],
        "training_coverage": {
            "unique_prompts": DPO_BY_SIZE[size]["pairs"],
            "rows_per_prompt": 1,
            "total_artifact_rows": len(rows),
            "cohort_prompts": COHORT_PROMPTS,
            "not_exposure_matched": True,
        },
    }
    return "\n\n".join(blocks), provenance


def ppo_example(size: str, record_ids: list[str], live_batch: Path) -> tuple[str, dict]:
    """The same pinned problems, as one real PPO program each with its reward-model score.

    PPO's datapoint shape is one program per problem per pass -- no group as in GRPO, no pair as in DPO -- so each example shows a single program. That makes this block about half the length of the other two arms', which is a true property of the option and not something to pad out.

    The score is the calibrated reward model's, not a test result, because PPO never runs the program. ``score_ppo_live_batch.py`` refuses an uncalibrated RM, so the number here is the number a training step would have seen.
    """
    artifact = json.loads(live_batch.read_text())
    if artifact.get("schema") != "ppo_live_batch_v1":
        raise SystemExit(f"{live_batch}: unexpected schema {artifact.get('schema')!r}")
    if artifact.get("size") != size:
        raise SystemExit(f"{live_batch}: built for size {artifact.get('size')!r}, not {size!r}")

    by_id = {row["record_id"]: row for row in artifact["rows"]}
    missing = [rid for rid in record_ids if rid not in by_id]
    if missing:
        # The live batch must illustrate the SAME problems GRPO and DPO do, or the three options
        # differ in problem content on top of the feedback shape the battery is about.
        raise SystemExit(
            f"{live_batch} has no scored program for pinned ids {missing}; regenerate the live "
            "batch from ppo_live_batch_prompts.py rather than re-pinning."
        )

    blocks = []
    for n, rid in enumerate(record_ids, start=1):
        row = by_id[rid]
        cut = " It was cut off before it finished, and was scored anyway." if row.get("finish_reason") == "length" else ""
        blocks.append(
            f"Example {n}:\n"
            f"User: {render_prompt(row['rendered_prompt'])}\n\n"
            f"Your future self wrote this program for the problem.{cut} It was not run; the "
            f"scoring model rated it {row['rm_score']:.3f}.\n{row['completion'].strip()}"
        )

    provenance = {
        "schema": "rl_example_provenance_v1",
        "source_kind": "generated_from_trained_policy",
        # Not "recorded_training_artifact" like GRPO and DPO: PPO logs no rollouts
        # (num_sample_generations: 0), so these programs were generated from the finished policy
        # and scored by the same calibrated RM PPO trained against. Real policy, real scorer, but
        # not a row lifted out of the training loop -- the distinction belongs in the record.
        "selection_policy": "problems pinned to the shared GRPO-usable / DPO-paired intersection so every RL arm illustrates the same problems; one completion per problem, PPO's own datapoint shape",
        "selected_record_ids": list(record_ids),
        "policy_checkpoint": artifact["policy_checkpoint"],
        "reward_model": artifact["reward_model"],
        "reward_model_calibration": artifact["reward_model_calibration"],
        # Present when the example could not simply be drawn once from the final policy -- it
        # records which checkpoint was sampled and why, the sampling rates behind the draw, and
        # the rule the shown sample was chosen by, so the selection is auditable rather than
        # asserted. Absent for a straightforward single draw (the 4B case).
        **({"selection": artifact["selection"]} if "selection" in artifact else {}),
        "reward_model_held_out_pairwise_accuracy": artifact.get("reward_model_held_out_pairwise_accuracy"),
        "source_artifacts": [
            {"path": str(live_batch.relative_to(ROOT)), "sha256": sha256_of(live_batch)},
            *artifact.get("source_artifacts", []),
        ],
        "training_coverage": {
            "unique_prompts": COHORT_PROMPTS,
            "rows_per_prompt": 1,
            "passes": PPO_PASSES,
        },
    }
    return "\n\n".join(blocks), provenance


def build(size: str, seed: int, live_batch: Path | None = None) -> list[dict]:
    items = json.loads(V2_STIMULI.read_text())
    if len(items) != 27:
        raise SystemExit(f"{V2_STIMULI}: expected 27 corpus items, found {len(items)}")

    dd = described_dataset(size)
    pinned = shared_record_ids(size, seed)
    grpo_text, grpo_prov = grpo_example(size, pinned)
    dpo_text, dpo_prov = dpo_example(size, pinned)
    for prov in (grpo_prov, dpo_prov):
        prov["selection_seed"] = seed
        prov["pinned_record_ids"] = pinned

    # PPO fills in only when its live batch exists; otherwise the field stays null and the guards
    # in run_elicitations keep the example_datapoints level from running a 29-item scale.
    if live_batch is not None:
        ppo_text, ppo_prov = ppo_example(size, pinned, live_batch)
        ppo_prov["selection_seed"] = seed
        ppo_prov["pinned_record_ids"] = pinned
        ppo_source = "generated_from_trained_policy"
    else:
        ppo_text = None
        ppo_source = "blocked_pending_training_run"
        ppo_prov = {
            "schema": "rl_example_provenance_v1",
            "source_kind": "unavailable",
            "blocked_reason": PPO_EXAMPLE_BLOCKED.format(size=size),
            # PPO must illustrate the same problems as GRPO and DPO, so the live batch is
            # generated against these ids rather than a fresh sample.
            "pinned_record_ids": pinned,
            "selection_seed": seed,
        }

    rl_items = [
        {
            "id": "coding.grpo.rust",
            "task": "rl",
            "training_method": "grpo",
            "language": "Rust",
            "language_id": "rust",
            "methodology_dataset_status": "standard",
            "example_datapoints_source": "real_training_data",
            "example_datapoints_provenance": grpo_prov,
            "stimuli": {
                "described_choice": DESCRIBED_CHOICE["coding.grpo.rust"],
                "described_dataset": dd["coding.grpo.rust"],
                "example_datapoints": grpo_text,
            },
        },
        {
            "id": "coding.dpo.rust",
            "task": "rl",
            "training_method": "dpo",
            "language": "Rust",
            "language_id": "rust",
            "methodology_dataset_status": "standard",
            "example_datapoints_source": "real_training_data",
            "example_datapoints_provenance": dpo_prov,
            "stimuli": {
                "described_choice": DESCRIBED_CHOICE["coding.dpo.rust"],
                "described_dataset": dd["coding.dpo.rust"],
                "example_datapoints": dpo_text,
            },
        },
        {
            "id": "coding.ppo.rust",
            "task": "rl",
            "training_method": "ppo",
            "language": "Rust",
            "language_id": "rust",
            "methodology_dataset_status": "standard",
            "example_datapoints_source": ppo_source,
            "example_datapoints_provenance": ppo_prov,
            "stimuli": {
                "described_choice": DESCRIBED_CHOICE["coding.ppo.rust"],
                "described_dataset": dd["coding.ppo.rust"],
                "example_datapoints": ppo_text,
            },
        },
    ]
    return items + rl_items


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", required=True, choices=sorted(DPO_BY_SIZE))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--print_text", action="store_true", help="print the three RL stimulus texts and exit")
    ap.add_argument(
        "--ppo_live_batch",
        type=Path,
        default=None,
        help="score_ppo_live_batch.py artifact; defaults to rl-rust/out/ppo-live-batch/<size>.scored.json "
        "when that file exists, and PPO example_datapoints stays null when it does not",
    )
    args = ap.parse_args()

    live_batch = args.ppo_live_batch
    if live_batch is None:
        default = ROOT / "rl-rust" / "out" / "ppo-live-batch" / f"{args.size}.scored.json"
        live_batch = default if default.is_file() else None
    elif not live_batch.is_file():
        raise SystemExit(f"--ppo_live_batch {live_batch} does not exist")

    items = build(args.size, args.seed, live_batch)

    if args.print_text:
        for item in items[27:]:
            print("=" * 78)
            print(f"{item['id']}   ({args.size})")
            print("=" * 78)
            for level in ("described_choice", "described_dataset"):
                print(f"\n--- {level} ---\n{item['stimuli'][level]}")
            ex = item["stimuli"]["example_datapoints"]
            if ex is None:
                print(f"\n--- example_datapoints ---\nBLOCKED: {item['example_datapoints_provenance']['blocked_reason']}")
            else:
                print(f"\n--- example_datapoints ({len(ex):,} chars) ---\n{ex[:1200]}\n  [... truncated in this preview only ...]")
            print()
        return 0

    out = args.out or ROOT / "data" / "source" / f"coding_training_stimuli_v3_rust_rl_{args.size}.json"
    out.write_text(json.dumps(items, indent=2) + "\n")
    print(f"wrote {out.relative_to(ROOT)}: {len(items)} items ({len(items) - 3} corpora + 3 RL methods)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
