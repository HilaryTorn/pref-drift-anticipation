#!/usr/bin/env python3
"""Stage C of the PPO example_datapoints live batch: score the generated programs with PPO's own reward model. Training venv, one GPU.

Stage A is ``ppo_live_batch_prompts.py`` (pinned prompts). Stage B is the existing merged-model vLLM path -- ``eval_merge.py`` then ``eval_generate.py`` -- which is reused rather than reimplemented because a served LoRA can be silently half-applied on Qwen3.5's linear-attention projections and a merged model cannot. Stage D is ``scripts/build_rust_rl_stimuli.py``, on the Mac.

WHAT THIS PRODUCES. One real program the trained PPO policy wrote for each pinned problem, carrying the score PPO's own calibrated reward model gives it. That is what one PPO training datapoint contains, and it is the only part of the RL battery that cannot be read off a recorded training artifact, because PPO logs no rollouts.

THE SCORE MUST COME FROM THE CALIBRATED RM. ``calibrate_reward_head.py`` divides the score head by the sigma of the pair scores so the reward reaches PPO on unit scale; an uncalibrated RM's logit is a different number for the same program. PPO consumed the calibrated one (its preflight refuses anything else), so the stimulus must quote the calibrated one or it reports a score no training step ever saw. This script refuses an RM whose manifest carries no ``score_head_calibration`` block.

THE SCORED TEXT MUST MATCH WHAT PPO SCORED. The RM sees the chat-templated prompt concatenated with the raw completion and an EOS token -- ``build_pair_texts`` plus the EOS that ``RewardTrainer`` appends in training and TRL's scorer sees at rollout time. Rendering the prompt any other way, or omitting the EOS, moves the number.

Usage (pod, training venv, after eval_generate.py):

    .venv/bin/python rl-rust/score_ppo_live_batch.py \\
        --size 4b \\
        --prompts rl-rust/out/ppo-live-batch/4b.prompts.jsonl \\
        --completions rl-rust/out/ppo-live-batch/4b.completions.jsonl \\
        --reward_model /root/runs/qwen35-4b-rm-rust-s0-calibrated \\
        --policy /root/merged-4b-ppo-step124 \\
        --policy_checkpoint prism-drift/qwen35-4b-m0-v4-rust-rl-adapters/ppo.rust.s0/checkpoint-124 \\
        --out rl-rust/out/ppo-live-batch/4b.scored.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "rl-rust"))

SCHEMA = "ppo_live_batch_v1"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_calibrated_manifest(rm_dir: Path) -> dict:
    """The RM's manifest, refused unless it records a completed head calibration."""
    manifest_path = rm_dir / "reward_model_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"no reward_model_manifest.json in {rm_dir}")
    manifest = json.loads(manifest_path.read_text())
    calibration = manifest.get("score_head_calibration")
    if not calibration:
        raise SystemExit(
            f"{rm_dir} is not calibrated (no score_head_calibration in its manifest). PPO trained "
            "against the calibrated RM, so an uncalibrated score would not be the number any "
            "training step saw. Point --reward_model at the -calibrated directory."
        )
    return manifest


def validate_batch(prompt_rows: list[dict], completion_rows: list[dict], allow_truncated: bool) -> dict[str, dict]:
    """Pair completions to pinned prompts, refusing every shape that would quietly misreport.

    Pure so tier 0 can exercise it on the Mac without a GPU or a reward model.
    """
    by_id: dict[str, list[dict]] = {}
    for row in completion_rows:
        by_id.setdefault(row["record_id"], []).append(row)

    # One program per problem is PPO's actual datapoint shape: unlike GRPO there is no group, and
    # unlike DPO there is no pair. Generating several and keeping one would be a choice the training
    # loop never makes, so the generation step runs at n_samples=1 and anything else is a mistake
    # upstream rather than something to silently select from here.
    for rid, rows in sorted(by_id.items()):
        if len(rows) != 1:
            raise SystemExit(
                f"{rid} has {len(rows)} completions; expected exactly 1. Re-run eval_generate.py "
                "with --n_samples 1: picking one of several would be a selection PPO never makes."
            )

    missing = [r["record_id"] for r in prompt_rows if r["record_id"] not in by_id]
    if missing:
        raise SystemExit(f"no completion for pinned ids {missing}")

    truncated = sorted(rid for rid, rows in by_id.items() if rows[0].get("finish_reason") == "length")
    if truncated and not allow_truncated:
        raise SystemExit(
            f"completions for {truncated} hit the length limit. rl_training/masked_ppo_trainer.py "
            "masks truncated episodes exactly as GRPO does -- they are excluded from the update, "
            "and a batch with no terminated completion is skipped outright -- so a cut-off program "
            "is NOT a datapoint PPO trained on, and showing one as the example would illustrate "
            "the episodes the run discarded. (An earlier version of this message claimed PPO does "
            "not mask; that was wrong, corrected 2026-09-13 against the trainer and the 9B run's "
            "completion_masking stats.) Re-run generation, or pass --allow_truncated only when "
            "the truncation itself is the finding being reported."
        )
    return {rid: rows[0] for rid, rows in by_id.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", required=True, choices=("4b", "9b"))
    ap.add_argument("--prompts", required=True, type=Path, help="output of ppo_live_batch_prompts.py")
    ap.add_argument("--completions", required=True, type=Path, help="output of eval_generate.py")
    ap.add_argument("--reward_model", required=True, help="the CALIBRATED RM directory PPO trained against")
    ap.add_argument("--policy", required=True, help="merged PPO policy dir; used for its tokenizer/chat template")
    ap.add_argument("--policy_checkpoint", required=True, help="provenance string naming the checkpoint generated from")
    ap.add_argument("--out", required=True, type=Path)
    # MUST match calibrate_reward_head.py's default for the same reason it documents: a different
    # batch size means different padding and a slightly different score.
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument(
        "--allow_truncated",
        action="store_true",
        help="accept a completion that hit the length limit (see the truncation check below)",
    )
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from scripts.rl_rust.calibrate_reward_head import _device, load_reward_model
    from scripts.rl_rust.rl_rust.prompts import render_coding_prompt
    from rl_training.train_reward_model import _score_texts

    prompt_rows = [json.loads(line) for line in args.prompts.open() if line.strip()]
    completion_rows = [json.loads(line) for line in args.completions.open() if line.strip()]

    by_id = validate_batch(prompt_rows, completion_rows, args.allow_truncated)

    manifest = load_calibrated_manifest(Path(args.reward_model))
    max_length = manifest["config"]["max_length"]

    policy_tok = AutoTokenizer.from_pretrained(args.policy, trust_remote_code=True)
    model, rm_tok = load_reward_model(args.reward_model)
    eos = rm_tok.eos_token or ""

    texts, ordered = [], []
    for row in prompt_rows:
        rid = row["record_id"]
        completion = by_id[rid]["text"]
        rendered = render_coding_prompt(policy_tok, row["prompt"], format_name="rust_io")
        texts.append(rendered + completion + eos)
        ordered.append((rid, row, completion, rendered))

    scores = _score_texts(model, rm_tok, texts, max_length, args.batch_size, _device())

    rows_out = []
    for (rid, row, completion, rendered), score in zip(ordered, scores):
        rows_out.append(
            {
                "record_id": rid,
                "problem": row["prompt"],
                "rendered_prompt": rendered,
                "completion": completion,
                "rm_score": score,
                "finish_reason": by_id[rid].get("finish_reason"),
            }
        )

    artifact = {
        "schema": SCHEMA,
        "size": args.size,
        "policy_checkpoint": args.policy_checkpoint,
        "policy_dir": str(args.policy),
        "reward_model": str(args.reward_model),
        "reward_model_calibration": manifest["score_head_calibration"],
        "reward_model_held_out_pairwise_accuracy": manifest.get("held_out_pairwise_accuracy"),
        "rm_scoring": {
            "texts": "rendered prompt + completion + EOS, as RewardTrainer trained and TRL scores",
            "max_length": max_length,
            "batch_size": args.batch_size,
        },
        "source_artifacts": [
            {"path": str(args.prompts), "sha256": sha256_of(args.prompts)},
            {"path": str(args.completions), "sha256": sha256_of(args.completions)},
        ],
        "rows": rows_out,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, ensure_ascii=True) + "\n")
    print(f"wrote {args.out}: {len(rows_out)} scored programs")
    for row in rows_out:
        chars = len(row["completion"])
        print(f"  {row['record_id']}  rm_score={row['rm_score']:+.4f}  {chars} chars  {row['finish_reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
