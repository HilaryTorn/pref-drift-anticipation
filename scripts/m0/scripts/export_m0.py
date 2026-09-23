#!/usr/bin/env python3
"""Merge the trained LoRA into the base weights and push M0 to HuggingFace.

M0 is not a drift trajectory -- intermediate checkpoints are not data, and only the finished model
is exported. But it IS the new base model, and that is why this merges rather than shipping the
adapter.

Why merge. Every downstream arm (the SFT language arms, the GRPO/DPO/PPO algorithm arms) trains
its own LoRA. If M0 stayed an adapter, all of them would be stacking LoRA on LoRA -- two adapter
sets to load in the right order at every serve, every score, and every resume, forever, with a
silent wrong-answer failure the first time one is forgotten. Merging costs one ~8 GB upload once
and makes M0 an ordinary model id that everything downstream can just point at.

    python m0/scripts/export_m0.py --run runs/qwen4b-m0-s0 --step 75 \\
        --repo prism-drift/qwen35-4b-m0

RUN THIS BEFORE TERMINATING THE BOX. The instance disk dies with the instance, and M0 is the worst
possible thing to lose: everything trained afterwards starts from it, so losing it invalidates the
downstream runs too, not just this one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pick_step(run: Path, explicit: int | None) -> Path:
    """Resolve which checkpoint to export.

    Defaults to the LAST one. That is the right default for a run that was stopped at the
    commit-rate plateau, and the wrong one for a run left to hit max_steps -- so the format_log
    summary is printed either way, and --step overrides.
    """
    checkpoints = sorted(run.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if not checkpoints:
        raise SystemExit(f"no checkpoint-* directories under {run}")
    if explicit is None:
        return checkpoints[-1]
    match = run / f"checkpoint-{explicit}"
    if not match.exists():
        available = ", ".join(p.name for p in checkpoints)
        raise SystemExit(f"no checkpoint-{explicit} under {run}; have: {available}")
    return match


def show_format_log(run: Path) -> None:
    path = run / "format_log.jsonl"
    if not path.exists():
        print("[m0-export] WARNING: no format_log.jsonl -- no evidence this run improved anything")
        return
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    print("[m0-export] commit rate by step (step 0 is the untrained baseline):")
    for row in rows:
        print(f"    step {row['step']:>4}  commit={row['commit_rate']:.3f}  "
              f"over_budget={row['over_budget_rate']:.3f}  "
              f"trunc={row['truncation_rate']:.3f}  "
              f"mean_reasoning={row['mean_reasoning_tokens']:.0f}")
    if len(rows) > 1 and rows[-1]["commit_rate"] <= rows[0]["commit_rate"]:
        print("[m0-export] WARNING: commit rate is no better than the untrained baseline. "
              "Exporting this model would ship a no-op as the study's new starting point.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="run directory (save_dir/run_name)")
    parser.add_argument("--step", type=int, default=None,
                        help="checkpoint step to export (default: the last one)")
    parser.add_argument("--base_model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--repo", required=True, help="HF model repo, e.g. prism-drift/qwen35-4b-m0")
    parser.add_argument("--out", default=None,
                        help="local dir for the merged model (default: <run>/merged)")
    parser.add_argument("--private", action="store_true", default=True)
    parser.add_argument("--public", dest="private", action="store_false")
    parser.add_argument("--no_upload", action="store_true", help="merge locally, skip the push")
    parser.add_argument("--adapter_only", action="store_true",
                        help="push the LoRA adapter instead of a merged model (see the module "
                             "docstring for why merging is the default)")
    args = parser.parse_args()

    run = Path(args.run)
    checkpoint = pick_step(run, args.step)
    show_format_log(run)
    print(f"[m0-export] exporting {checkpoint}")

    if args.adapter_only:
        source = checkpoint
    else:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        out = Path(args.out) if args.out else run / "merged"
        print(f"[m0-export] merging LoRA into {args.base_model} -> {out}")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, trust_remote_code=True, dtype=torch.bfloat16,
        )
        # bfloat16 throughout, deliberately. cost.md:23 requires the baseline and every checkpoint
        # in a trajectory to share one dtype, or drift gets confounded with a numerics change --
        # and M0 is the baseline every one of those trajectories starts from.
        model = PeftModel.from_pretrained(model, str(checkpoint))
        model = model.merge_and_unload()
        from scripts.m0.wrap_for_serving import save_servable_m0
        # Re-wrap into the base's multimodal form so `vllm serve` accepts M0. The base was loaded with
        # AutoModelForCausalLM, which extracted the text sub-model, so a bare save_pretrained writes a
        # text-only checkpoint vLLM's Qwen3.5 handler rejects. See m0/wrap_for_serving.py.
        save_servable_m0(model, args.base_model, out, tokenizer)
        # Copy the run's provenance in beside the weights: months later, "which run produced this
        # model, on what data, with what beta" is the question, and a bare weights dir cannot answer it.
        for name in ("run_metadata.json", "format_log.jsonl"):
            if (run / name).exists():
                (out / name).write_text((run / name).read_text())
        source = out
        print(f"[m0-export] merged model written to {out}")

    if args.no_upload:
        print("[m0-export] --no_upload set; nothing pushed")
        return 0

    import subprocess

    cmd = ["hf", "upload", args.repo, str(source), "--repo-type", "model"]
    if args.private:
        cmd.append("--private")
    print(f"[m0-export] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(
            "upload failed. Do NOT terminate the box until this succeeds -- run `hf auth login` "
            "with a write token and retry."
        )

    print(f"\n[m0-export] pushed: {args.repo}")
    print("[m0-export] NEXT: the adoption gate, before anything is trained from this model.")
    print("  1. commit rate on the real batteries (values, UE, other, coding) vs the base model")
    print("  2. baseline utilities re-elicited on M0 vs the base model -- if any moved past the")
    print("     noise floor, M0 is contaminated and must not be adopted: raise beta, lower lr, or")
    print("     take an earlier checkpoint, then re-gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
