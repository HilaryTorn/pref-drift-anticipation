#!/usr/bin/env python3
"""Split a frozen cohort and a GRPO config into staged context-lengthening stages.

DeepScaleR (https://openreview.net/forum?id=I6GzDCne7U) trains RL with an
iteratively lengthening context — 8K -> 16K -> 24K — and finds it both cheaper and
BETTER than starting long: the short stage is an implicit curriculum that makes the
model reason efficiently before it reasons at length. "naively scaling to long
contexts in RL training is suboptimal."

Two invariants the study cannot lose, and which naive staging would break:

1. **Optimizer-step budget is matched across arms.** GRPO gets 125 steps total,
   the same as DPO and PPO. Running 125 steps per stage would give GRPO 375 and
   silently make the algorithm comparison meaningless. So the steps are SPLIT.
2. **One epoch over the frozen cohort.** Each stage therefore takes a disjoint
   contiguous slice. The cohort was shuffled with a fixed seed when it was frozen,
   so contiguous slices are already random, and slicing in file order keeps the
   split reproducible from the manifest alone.

Together: stage step counts sum to the original max_steps, and the stage slices
partition the cohort exactly. Both are asserted here rather than assumed.

    python scripts/make_curriculum_stages.py \\
        --cohort data/rl/v1/cohorts/<cohort>/train.jsonl \\
        --config rl_training/configs/grpo.yaml \\
        --lengths 4096 8192 16384 \\
        --out_dir data/rl/v1/curricula/<cohort>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_steps(total_steps: int, n_stages: int) -> list[int]:
    """Divide the step budget as evenly as possible, remainder to the early stages.

    Front-loading the remainder puts the extra work in the cheap short-context
    stages rather than the expensive long one.
    """
    base, remainder = divmod(total_steps, n_stages)
    return [base + (1 if i < remainder else 0) for i in range(n_stages)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cohort", required=True, help="Frozen cohort train.jsonl")
    parser.add_argument("--config", required=True, help="Base GRPO config to stage")
    parser.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192, 16384],
                        help="max_completion_length per stage, ascending")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    if sorted(args.lengths) != args.lengths:
        raise SystemExit(f"--lengths must ascend; got {args.lengths}")

    cohort_path = Path(args.cohort)
    rows = [line for line in cohort_path.read_text().splitlines() if line.strip()]
    base_cfg = yaml.safe_load(Path(args.config).read_text())
    if base_cfg.get("algo") != "grpo":
        raise SystemExit(f"{args.config} is for algo {base_cfg.get('algo')!r}, expected grpo")

    total_steps = int(base_cfg["max_steps"])
    per_step = int(base_cfg["unique_prompts_per_step"])
    expected = int(base_cfg["expected_unique_prompts"])
    if len(rows) != expected:
        raise SystemExit(
            f"{cohort_path} has {len(rows)} prompts; config expects {expected}. "
            "Stage slicing must start from the cohort the config was frozen against."
        )

    stage_steps = split_steps(total_steps, len(args.lengths))
    assert sum(stage_steps) == total_steps, "step split must preserve the budget"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stages, cursor = [], 0
    for index, (length, steps) in enumerate(zip(args.lengths, stage_steps), start=1):
        count = steps * per_step
        slice_rows = rows[cursor : cursor + count]
        if len(slice_rows) != count:
            raise SystemExit(
                f"stage {index} needs {count} prompts but only {len(slice_rows)} remain; "
                f"{expected} prompts cannot be split into {stage_steps} steps of {per_step}"
            )
        data_path = out_dir / f"stage{index}-len{length}.jsonl"
        data_path.write_text("".join(line + "\n" for line in slice_rows))

        cfg = dict(base_cfg)
        cfg["max_steps"] = steps
        cfg["max_completion_length"] = length
        cfg["expected_unique_prompts"] = count
        # eval_reward_steps are absolute within a stage; keep the last step so every
        # stage still produces an end-of-stage reward reading.
        cfg["eval_reward_steps"] = sorted({s for s in cfg.get("eval_reward_steps", []) if s <= steps} | {steps})
        cfg["curriculum"] = {
            "stage": index, "of": len(args.lengths),
            "cohort": str(cohort_path.resolve()),
            "total_max_steps": total_steps,
        }
        cfg_path = out_dir / f"grpo_stage{index}_len{length}.yaml"
        cfg_path.write_text(
            f"# Stage {index}/{len(args.lengths)} of a context-lengthening curriculum,\n"
            f"# generated by scripts/make_curriculum_stages.py from {args.config}.\n"
            f"# Do not edit by hand — regenerate, so the step split and the cohort\n"
            f"# partition stay consistent with each other.\n"
            f"#\n"
            f"# Run stage {index} with:\n"
            f"#   --dataset {data_path}\n"
            + (f"#   --init_adapter <stage {index - 1} final checkpoint>\n" if index > 1 else
               "#   (no --init_adapter; this stage starts from the base model)\n")
            + yaml.safe_dump(cfg, sort_keys=False)
        )
        stages.append({
            "stage": index,
            "max_completion_length": length,
            "max_steps": steps,
            "unique_prompts": count,
            "slice": [cursor, cursor + count],
            "dataset": str(data_path.resolve()),
            "dataset_sha256": _sha256(data_path),
            "config": str(cfg_path.resolve()),
        })
        cursor += count

    leftover = len(rows) - cursor
    manifest = {
        "schema": "grpo_curriculum_v1",
        "source_config": str(Path(args.config).resolve()),
        "cohort": str(cohort_path.resolve()),
        "cohort_sha256": _sha256(cohort_path),
        "cohort_rows": len(rows),
        "total_max_steps": total_steps,
        "unique_prompts_per_step": per_step,
        "stages": stages,
        "unused_prompts": leftover,
    }
    manifest_path = out_dir / "curriculum_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"[curriculum] {len(stages)} stages -> {out_dir}")
    for stage in stages:
        print(
            f"  stage {stage['stage']}: len={stage['max_completion_length']:>5} "
            f"steps={stage['max_steps']:>3} prompts={stage['unique_prompts']:>4} "
            f"rows[{stage['slice'][0]}:{stage['slice'][1]}]"
        )
    print(f"[curriculum] total steps {sum(s['max_steps'] for s in stages)} "
          f"(budget {total_steps}), prompts used {cursor}/{len(rows)}")
    if leftover:
        # Never silent: an unused tail is a real shortfall against "one epoch over
        # the frozen cohort", not a rounding detail.
        print(
            f"    !! {leftover} cohort prompts are unused because {total_steps} steps "
            f"do not divide evenly into {len(stages)} stages. The arms are still "
            f"step-matched, but GRPO now sees {cursor} prompts where DPO sees {len(rows)}."
        )
    print(f"[curriculum] wrote {manifest_path}")


if __name__ == "__main__":
    main()
