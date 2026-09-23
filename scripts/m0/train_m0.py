#!/usr/bin/env python3
"""GRPO trainer for M0, the format-training run.

Model 0: the elicitation batteries need the model to reason and then commit on its own line, and
prompting does not get that reliably -- reasoning-on runs genuinely commit only ~15-50% of the
time, worst on values and UE. M0 installs the behavior in the weights so every downstream arm
starts from a model that answers in the format the scoring pipeline can read.

PROVENANCE. The scaffolding here (config loading, seeding, LoRA setup, base-model loading, the
TRL optional-import patch, the GRPOTrainer wiring) is copied from rl_training/train_rl.py, which
runs the coding algorithm-comparison study. It is a deliberate fork, not shared code: that harness
is live, its arms are being compared against each other, and M0 is a one-time preprocessing run
that has no reason to stay in lockstep with it. Nothing in this package writes to rl_training/.

What is genuinely different from the original, and why:

  * enable_thinking=True (m0/prompts.py). The coding path pins it False; M0 must train in the
    regime the batteries are SERVED under, since the whole objective concerns closing the <think>
    block before the serve-time budget forces it shut.
  * The reward is the format objective (m0/rewards.py), not a code verifier.
  * FormatCadenceCallback logs commit rate rather than pass rate.
  * DPO and PPO are not carried over. M0 is GRPO only.

Example:

    python -m m0.train_m0 --config m0/configs/grpo_m0.yaml \\
        --base_model Qwen/Qwen3.5-4B \\
        --dataset data/rl/m0_format/train.jsonl \\
        --eval_dataset data/rl/m0_format/dev.jsonl \\
        --seed 0 --save_dir runs --run_name qwen4b-m0-s0 --use_lora
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The batteries are served with max_tokens: 2048 (compute_utilities/create_agent.yaml). Training
# inside a smaller envelope would teach the model to fit a budget it will never actually face, and
# the over-budget term would be measuring the wrong ceiling.
REQUIRED_COMPLETION_LENGTH = 2048


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if cfg.get("algo") != "grpo":
        raise SystemExit(f"{path}: M0 is GRPO only, got algo={cfg.get('algo')!r}")
    if cfg.get("task") != "format":
        raise SystemExit(f"{path}: expected task 'format', got {cfg.get('task')!r}")
    if cfg["max_completion_length"] < REQUIRED_COMPLETION_LENGTH:
        raise SystemExit(
            f"{path}: max_completion_length is {cfg['max_completion_length']}, but the elicitation "
            f"batteries run at {REQUIRED_COMPLETION_LENGTH} tokens "
            f"(compute_utilities/create_agent.yaml max_tokens). Training in a tighter envelope "
            f"teaches the model to fit a budget it never meets at serve time."
        )
    if cfg.get("reasoning_budget", 500) >= cfg["max_completion_length"]:
        raise SystemExit(
            f"{path}: reasoning_budget ({cfg.get('reasoning_budget')}) leaves no room for the "
            f"answer inside max_completion_length ({cfg['max_completion_length']})."
        )
    return cfg


def set_all_seeds(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_format_jsonl(path: str) -> list[dict]:
    """Read and validate format_task_v1 rows.

    Validation is strict for the same reason rl_training/data/schema.py is: a row missing `labels`
    would silently fall back to A/B in the reward, and every item using another label scheme would
    be scored as a non-commitment -- a 75% reward floor that looks like a training failure.
    """
    rows = []
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema") != "format_task_v1":
                raise SystemExit(f"{path}:{line_no}: unsupported schema {row.get('schema')!r}")
            for key in ("prompt", "labels", "verifiable"):
                if key not in row:
                    raise SystemExit(f"{path}:{line_no}: missing required key {key!r}")
            if not isinstance(row["labels"], list) or len(row["labels"]) < 2:
                raise SystemExit(f"{path}:{line_no}: 'labels' must be a list of at least two labels")
            if row["verifiable"] and not row.get("answer"):
                raise SystemExit(f"{path}:{line_no}: verifiable row has no 'answer'")
            if row.get("answer") and row["answer"] not in row["labels"]:
                raise SystemExit(
                    f"{path}:{line_no}: answer {row['answer']!r} is not among labels {row['labels']}"
                )
            rows.append(row)
    if not rows:
        raise SystemExit(f"{path}: no rows")
    return rows


def load_format_dataset(path: str, tokenizer):
    """Pre-render prompts through the chat template; carry the reward's columns alongside."""
    from datasets import Dataset
    from scripts.m0.prompts import render_format_prompt

    rows = read_format_jsonl(path)
    return Dataset.from_list([
        {
            "prompt": render_format_prompt(tokenizer, row["prompt"]),
            "labels": row["labels"],
            "answer": row.get("answer") or "",
            "verifiable": bool(row["verifiable"]),
            "record_id": row.get("record_id", ""),
        }
        for row in rows
    ])


def write_run_metadata(output_dir: str, cfg: dict, args: argparse.Namespace) -> None:
    metadata = {
        "run_type": "m0_format",
        "base_model": args.base_model,
        "dataset": args.dataset,
        "eval_dataset": args.eval_dataset,
        "seed": args.seed,
        "run_name": args.run_name,
        "use_vllm": args.use_vllm,
        "trainer_config": cfg,
        "lora": {
            "enabled": args.use_lora,
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "bias": args.lora_bias,
            "target_modules": args.lora_target_modules,
        },
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with (Path(output_dir) / "run_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)


def _bf16_available() -> bool:
    import torch

    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def check_fused_kernels() -> dict:
    """Warn loudly when Qwen3.5's hybrid-attention fast path is unavailable.

    18 of Qwen3.5's 24 layers are ``linear_attn``. Without flash-linear-attention (and
    causal-conv1d for the short convolution) transformers falls back to a reference PyTorch
    implementation that is both slower and heavier on memory -- the first L4 attempt here OOM'd
    inside ``torch_chunk_gated_delta_rule``, the fallback kernel.

    transformers prints a one-line notice about this and it scrolls past in seconds, between a
    progress bar and a tokenizer warning. On a job measured in hours that is not enough: a run can
    be most of the way through before anyone notices it took the slow path. Hence a warning that is
    hard to miss, printed before any GPU time is spent.
    """
    status = {}
    for module, package in (("fla", "flash-linear-attention"), ("causal_conv1d", "causal-conv1d")):
        try:
            __import__(module)
            status[package] = True
        except Exception:  # noqa: BLE001 - any import failure means the fast path is unavailable
            status[package] = False

    missing = [name for name, ok in status.items() if not ok]
    if missing:
        # Deliberately NOT an install instruction. On the pinned stack (m0/requirements-m0.txt)
        # flash-linear-attention needs Triton >= 3.3.0 while torch 2.5.1 pins 3.1.0: it installs,
        # then raises on import, and transformers -- which only checks that the package exists --
        # takes the fast-path branch and crashes. Present-but-broken is worse than absent, so an
        # operator who followed a "just pip install it" console message mid-session would break a
        # validated box. See m0/README.md, "On the fast path is not available".
        print("\n" + "=" * 78)
        print("[m0] Reference PyTorch attention path (18 of 24 Qwen3.5 layers are linear_attn).")
        print(f"[m0] Not installed: {', '.join(missing)}")
        print("[m0] This is EXPECTED and is the validated configuration on this stack -- all M0")
        print("[m0] timings and the batch-size ceiling were measured on it. The fused kernels are")
        print("[m0] faster but are not installable against these pins; installing them mid-run")
        print("[m0] breaks transformers. See m0/README.md before changing anything here.")
        print("=" * 78 + "\n")
    else:
        print("[m0] fused hybrid-attention kernels present -- NOTE: M0's measured timings and "
              "memory ceiling were taken WITHOUT them; re-validate before a long run")
    return status


def _parse_lora_target_modules(value: str):
    value = value.strip()
    if value == "all-linear":
        return value
    modules = [item.strip() for item in value.split(",") if item.strip()]
    if not modules:
        raise ValueError("--lora_target_modules must be 'all-linear' or a comma-separated list")
    return modules


def _load_policy(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Qwen right-truncates by default, which would cut the trailing <|im_start|>assistant header
    # off an overlong prompt and drop the model into plain-continuation mode. Prompts are length
    # filtered at build time; left truncation is the safe fallback.
    tokenizer.truncation_side = "left"
    dtype = torch.bfloat16 if _bf16_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True, dtype=dtype)
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}  # TRL 0.24 writes into this during GRPOTrainer init

    if args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        targets = _parse_lora_target_modules(args.lora_target_modules)
        if isinstance(targets, list) and "qwen3.5" in args.base_model.lower():
            attn_only = {"q_proj", "k_proj", "v_proj", "o_proj"}
            if attn_only & set(targets) and not {"in_proj_qkv", "out_proj"} & set(targets):
                print(
                    "[m0] WARNING: an explicit q/k/v/o target list on a hybrid-attention Qwen3.5 "
                    "adapts only the 6 self_attn layers of 24; the other 18 are linear_attn and "
                    "are silently skipped (PEFT does not warn). Use --lora_target_modules "
                    "all-linear unless you specifically intend to freeze them."
                )

        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            target_modules=targets,
        ))
        print("[m0] LoRA enabled")
        model.print_trainable_parameters()
    return model, tokenizer


def _patch_trl_optional_imports(use_vllm: bool = False) -> None:
    """Disable TRL's optional integrations before importing any trainer.

    Copied from rl_training/train_rl.py. A shared box may carry vLLM, vllm-ascend, mergekit or
    llm-blender; TRL imports those eagerly even when unused, and a mismatched install makes every
    trainer import fail. Transformers 5 also changed _is_package_available() to return a tuple,
    which TRL 0.24 reads as truthy even when the package is absent.
    """
    import trl.import_utils as trl_import_utils

    for name, value in vars(trl_import_utils).copy().items():
        if name.endswith("_available") and isinstance(value, tuple):
            setattr(trl_import_utils, name, value[0])
    disabled = ["_vllm_ascend_available", "_mergekit_available", "_llm_blender_available",
                "_weave_available"]
    if not use_vllm:
        disabled.append("_vllm_available")
    for name in disabled:
        if hasattr(trl_import_utils, name):
            setattr(trl_import_utils, name, False)


def train(cfg: dict, args, output_dir: str) -> None:
    check_fused_kernels()
    _patch_trl_optional_imports(use_vllm=args.use_vllm)
    from trl import GRPOConfig, GRPOTrainer

    from scripts.m0.callbacks import FormatCadenceCallback
    from scripts.m0.rewards import make_format_reward

    model, tokenizer = _load_policy(args)
    dataset = load_format_dataset(args.dataset, tokenizer)

    grpo_kwargs = dict(
        output_dir=output_dir,
        max_steps=cfg["max_steps"],
        save_steps=cfg["save_steps"],
        logging_steps=cfg["logging_steps"],
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        beta=cfg["beta"],
        num_generations=cfg["num_generations"],
        max_prompt_length=cfg["max_prompt_length"],
        max_completion_length=cfg["max_completion_length"],
        bf16=_bf16_available(),
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        seed=args.seed,
        report_to="none",
    )
    if args.use_vllm:
        grpo_kwargs["use_vllm"] = True
        grpo_kwargs["vllm_mode"] = args.vllm_mode

    reward_func = make_format_reward(
        tokenizer=tokenizer,
        floor=cfg.get("reasoning_floor", 32),
        budget=cfg.get("reasoning_budget", 500),
        target=cfg.get("reasoning_target", 256),
        output_dir=output_dir,
        num_generations=cfg["num_generations"],
        log_rollouts=cfg.get("log_rollouts", False),
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_func],
        args=GRPOConfig(**grpo_kwargs),
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    eval_path = args.eval_dataset or args.dataset
    eval_records = read_format_jsonl(eval_path)[: cfg["eval_prompts"]]
    trainer.add_callback(FormatCadenceCallback(
        tokenizer, eval_records, cfg["eval_samples"], output_dir,
        budget=cfg.get("reasoning_budget", 500),
        target=cfg.get("reasoning_target", 256),
        max_new_tokens=cfg["max_completion_length"],
        batch_prompts=cfg.get("eval_batch_prompts", 4),
        run_name=args.run_name,
    ))
    trainer.train(resume_from_checkpoint=args.resume)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="m0/configs/grpo_m0.yaml")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--dataset", required=True, help="format_task_v1 JSONL (train split)")
    parser.add_argument("--eval_dataset", default=None, help="format_task_v1 JSONL (dev split)")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--save_dir", default="runs")
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--vllm_mode", default="colocate", choices=["colocate", "server"])
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_bias", default="none", choices=["none", "all", "lora_only"])
    # 'all-linear', NOT the q/k/v/o list the coding launcher uses. Qwen3.5 is hybrid-attention:
    # only 6 of its 24 layers have self_attn (q/k/v/o_proj); the other 18 use linear_attn
    # (in_proj_qkv/z/a/b, out_proj). An explicit q/k/v/o list therefore adapts 6 layers out of 24
    # and PEFT does not warn -- M0 would train with three quarters of the model frozen, most likely
    # fail to move commit rate, and the reward would get the blame. Same conclusion as
    # data/training_specs/coding_axis_sft.json and rl_training/README.md:443.
    parser.add_argument("--lora_target_modules", default="all-linear")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_all_seeds(args.seed)
    output_dir = str(Path(args.save_dir) / args.run_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    write_run_metadata(output_dir, cfg, args)
    print(f"[m0] format run '{args.run_name}' seed={args.seed} -> {output_dir}")
    train(cfg, args, output_dir)
    print(
        "[m0] done. Before adopting this model:\n"
        f"  1. read {output_dir}/format_log.jsonl and pick the checkpoint where commit_rate plateaus\n"
        "  2. re-elicit the baseline utilities on it and confirm nothing moved past the noise floor\n"
        "     -- M0 must change format, not preferences, or it contaminates the study it serves"
    )


if __name__ == "__main__":
    main()
