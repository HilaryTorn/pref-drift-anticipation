#!/usr/bin/env python3
"""Unified RL fine-tuning harness for the preference-drift study.

Fine-tunes a base model on a coding dataset with one of three algorithms
(PPO / GRPO / DPO), saving checkpoints on a fixed step cadence so the drift
elicitation can be scored on each. The algorithm is the study's independent
variable; everything else is held fixed across runs: base model, dataset,
reward/objective, optimizer-step budget, checkpoint cadence, and the
reference-KL strength (beta / init_kl_coef).

DPO is offline preference optimization; PPO/GRPO are online RL. All three share
one objective because DPO's pairs and (optionally) PPO's reward model are built
from the SAME verifiable reward used by GRPO (rl_training/rewards.py).

Checkpoints land in <save_dir>/<run_name>/checkpoint-<step>/. Register them into
config.yaml with rl_training/register_checkpoints.py, then score each with
run_elicitations.py / run_utilities.py (values via the run_utilities_values entry).

Assumes TRL >= 0.15 (GRPOTrainer + Trainer-style PPOTrainer). Exact kwargs move
between TRL releases; pin via rl_training/requirements-training.txt.

Examples:
    python -m rl_training.train_rl --algo grpo --config rl_training/configs/grpo.yaml \
        --base_model Qwen/Qwen3.5-0.8B \
        --dataset data/rl/coding_train.jsonl --seed 0 \
        --save_dir runs --run_name qwen08b-grpo-s0 --use_lora

    python -m rl_training.train_rl --algo dpo --config rl_training/configs/dpo.yaml \
        --base_model Qwen/Qwen3.5-0.8B \
        --dataset data/rl/coding_pairs.jsonl --seed 0 \
        --save_dir runs --run_name qwen08b-dpo-s0 --use_lora
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path
import sys

import yaml
from transformers import TrainerCallback

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rl_training.hub_checkpoints import (
    LOCAL_FALLBACK_MARKER,
    add_formal_hub_arguments,
    prepare_formal_hub_upload,
    prepared_formal_hub_callback,
)


def load_config(path: str, algo: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if cfg.get("algo") != algo:
        raise ValueError(f"Config {path} is for algo {cfg.get('algo')!r}, not {algo!r}")
    return cfg


def audit_grpo_prompt_budget(
    cfg: dict, dataset_size: int, world_size: int = 1
) -> dict[str, int]:
    """Validate GRPO cadence in unique prompts rather than completions.

    TRL repeats each prompt ``num_generations`` times. Its ordinary batch and
    accumulation fields consequently count completion sequences, not unique
    training prompts. The formal comparison requires exactly one pass over the
    same 1,000 frozen prompts as DPO, with eight unique prompts per update.
    """
    required = (
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "num_generations",
        "generation_batch_size",
        "unique_prompts_per_step",
        "expected_unique_prompts",
        "max_steps",
    )
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"GRPO config missing prompt-budget fields: {missing}")

    completion_sequences_per_step = (
        int(cfg["per_device_train_batch_size"])
        * int(world_size)
        * int(cfg["gradient_accumulation_steps"])
    )
    num_generations = int(cfg["num_generations"])
    if completion_sequences_per_step % num_generations:
        raise ValueError(
            "GRPO completion batch per optimizer step must be divisible by "
            f"num_generations: {completion_sequences_per_step} % {num_generations} != 0"
        )
    unique_prompts_per_step = completion_sequences_per_step // num_generations
    if unique_prompts_per_step != int(cfg["unique_prompts_per_step"]):
        raise ValueError(
            "GRPO unique-prompt batch mismatch: completion batch "
            f"{completion_sequences_per_step} / K={num_generations} = "
            f"{unique_prompts_per_step}, configured "
            f"unique_prompts_per_step={cfg['unique_prompts_per_step']}"
        )

    generation_batch_size = int(cfg["generation_batch_size"])
    if generation_batch_size != completion_sequences_per_step:
        raise ValueError(
            "Formal GRPO requires one generation batch per optimizer update: "
            f"generation_batch_size={generation_batch_size}, "
            f"completion_sequences_per_step={completion_sequences_per_step}"
        )
    if generation_batch_size % num_generations:
        raise ValueError(
            f"generation_batch_size={generation_batch_size} is not divisible by K={num_generations}"
        )

    expected_unique_prompts = int(cfg["expected_unique_prompts"])
    if dataset_size != expected_unique_prompts:
        raise ValueError(
            f"GRPO dataset has {dataset_size} unique prompts; expected "
            f"{expected_unique_prompts}"
        )
    scheduled_unique_prompts = int(cfg["max_steps"]) * unique_prompts_per_step
    if scheduled_unique_prompts != expected_unique_prompts:
        raise ValueError(
            f"GRPO schedule consumes {scheduled_unique_prompts} unique prompts "
            f"({cfg['max_steps']} steps * {unique_prompts_per_step}); expected "
            f"{expected_unique_prompts}"
        )
    return {
        "dataset_unique_prompts": dataset_size,
        "num_generations": num_generations,
        "completion_sequences_per_step": completion_sequences_per_step,
        "unique_prompts_per_step": unique_prompts_per_step,
        "scheduled_unique_prompts": scheduled_unique_prompts,
    }


# Optimizer-schedule fields that every arm forwards identically. They must be
# threaded explicitly: HF defaults lr_scheduler_type to "linear" with zero warmup,
# so a 125-step run decays to ~0 and its back half barely moves the adapter.
#
# A config key that nothing reads is worse than an absent one — it looks reviewed,
# and build_run_metadata copies the YAML verbatim, so run_metadata.json records the
# value the file declared rather than the one the trainer used. `max_prompt_length`
# sat in these configs exactly that way. Anything added here must be consumed here.
_SCHEDULE_KEYS = ("lr_scheduler_type", "warmup_steps", "warmup_ratio", "weight_decay")


def _schedule_kwargs(cfg: dict) -> dict:
    """Forward optimizer-schedule fields that the config actually sets."""
    return {key: cfg[key] for key in _SCHEDULE_KEYS if key in cfg}


def set_all_seeds(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _file_sha256(path: str | None) -> str | None:
    if not path:
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_run_metadata(cfg: dict, args: argparse.Namespace) -> dict:
    return {
        "schema": "rl_training_run_v1",
        "status": "running",
        "algo": args.algo,
        "base_model": args.base_model,
        "dataset": str(Path(args.dataset).resolve()),
        "dataset_sha256": _file_sha256(args.dataset),
        "eval_dataset": str(Path(args.eval_dataset).resolve()) if args.eval_dataset else None,
        "eval_dataset_sha256": _file_sha256(args.eval_dataset),
        "reward_model": args.reward_model,
        "allow_weak_reward_model": args.allow_weak_reward_model,
        "seed": args.seed,
        "run_name": args.run_name,
        "use_vllm": args.use_vllm,
        "vllm_mode": args.vllm_mode,
        "config": str(Path(args.config).resolve()),
        "config_sha256": _file_sha256(args.config),
        "trainer_config": cfg,
        "lora": {
            "enabled": args.use_lora,
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "bias": args.lora_bias,
            "target_modules": args.lora_target_modules,
            # The parent adapter is what makes a staged curriculum auditable: a
            # stage-3 checkpoint is meaningless without knowing what it continued.
            "init_adapter": (
                str(Path(args.init_adapter).resolve()) if args.init_adapter else None
            ),
        },
        "hub_archive": {
            "enabled": bool(args.hub_repo_id),
            "repo_id": args.hub_repo_id,
            "private": args.hub_private,
            "base_model_id": args.hub_base_model_id,
            "base_revision": args.hub_base_revision,
            "cohort_manifest": (
                str(args.cohort_manifest.resolve()) if args.cohort_manifest else None
            ),
            "cohort_manifest_sha256": (
                _file_sha256(str(args.cohort_manifest)) if args.cohort_manifest else None
            ),
            "keep_local_checkpoints_during_training": args.hub_keep_local_checkpoints,
        },
    }


def _metadata_signature(metadata: dict) -> dict:
    keys = (
        "algo",
        "base_model",
        "dataset_sha256",
        "eval_dataset_sha256",
        "reward_model",
        "seed",
        "run_name",
        "use_vllm",
        "vllm_mode",
        "config_sha256",
        "trainer_config",
        "lora",
    )
    signature = {key: metadata.get(key) for key in keys}
    # The archive destination is operational state, not training provenance.
    # A run must be able to migrate to another private repository when the
    # original namespace reaches its storage quota. Keep validating the fields
    # that pin the actual model revision and frozen cohort, while allowing the
    # repository, visibility, enablement, and local-retention policy to change.
    hub_archive = metadata.get("hub_archive") or {}
    signature["hub_provenance"] = {
        key: hub_archive.get(key)
        for key in (
            "base_model_id",
            "base_revision",
            "cohort_manifest_sha256",
        )
    }
    return signature


def write_run_metadata(output_dir: str, metadata: dict) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with (Path(output_dir) / "run_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")


def load_prompt_dataset(path: str, tokenizer):
    """Online arms (GRPO/PPO). JSONL rows use canonical coding_task_v1:
    {"prompt": str, "verifier": {...}}.

    Prompts are pre-rendered through the tokenizer's chat template
    (render_coding_prompt) so chat-tuned bases answer instead of continuing the
    problem text. ``verifier`` rides along as a dataset column so the reward
    function can dispatch across io_tests, unit_tests, and reference tests.
    """
    from datasets import Dataset
    from scripts.rl_training.data.schema import read_coding_jsonl
    from scripts.rl_training.prompts import render_coding_prompt

    rows = read_coding_jsonl(path)
    rows = [{**row, "prompt": render_coding_prompt(tokenizer, row["prompt"])} for row in rows]
    return Dataset.from_list(rows)


def load_ppo_prompt_dataset(path: str, tokenizer, max_prompt_length: int):
    """PPO arm. TRL's PPOTrainer consumes PRE-TOKENIZED prompts: a dataset with a
    single ``input_ids`` column. It generates responses internally and scores them
    with the reward model, so -- unlike GRPO -- it cannot carry the ``verifier``
    column: PPO's default collator pads ``input_ids`` and would choke on the dict.
    The verifier is not needed at PPO train time anyway (reward comes from the RM;
    per-checkpoint verifier eval reads its own --eval_dataset).
    """
    from datasets import Dataset
    from scripts.rl_training.data.schema import read_coding_jsonl
    from scripts.rl_training.prompts import render_coding_prompt

    rows = read_coding_jsonl(path)
    records = [
        {"input_ids": tokenizer(
            render_coding_prompt(tokenizer, row["prompt"]),
            truncation=True, max_length=max_prompt_length,
        )["input_ids"]}
        for row in rows
    ]
    return Dataset.from_list(records)


def load_preference_dataset(path: str):
    """DPO arm. JSONL rows: {"prompt": str, "chosen": str, "rejected": str}.

    Build these offline from the verifiable reward with
    training/rewards.build_preference_pairs so the objective matches the online
    arms.
    """
    from datasets import Dataset
    from scripts.rl_training.data.schema import read_preference_jsonl

    rows = read_preference_jsonl(path)
    return Dataset.from_list([
        {"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]} for r in rows
    ])


class DriftCadenceCallback(TrainerCallback):
    """Log the verifiable reward on a fixed eval slice at every checkpoint.

    GRPO/PPO already optimize the reward, but DPO never sees it — this makes the
    reward-vs-step curve comparable across ALL algorithms (the study's secondary
    matching axis). Appends one JSON line per checkpoint to
    <output_dir>/reward_log.jsonl.

    When ``fig_path`` is set, the reward-vs-step figure is re-rendered on every
    Trainer log event (every logging_steps) and after every checkpoint eval, so
    the PNG can be watched during training.
    """

    def __init__(self, tokenizer, eval_records, n_samples, reward_timeout, output_dir,
                 fig_path=None, run_name=None, eval_steps=None, max_new_tokens=1024):
        self.tokenizer = tokenizer
        self.eval_records = eval_records
        self.n_samples = n_samples
        self.reward_timeout = reward_timeout
        self.log_path = Path(output_dir) / "reward_log.jsonl"
        self.fig_path = Path(fig_path) if fig_path else None
        self.run_name = run_name or Path(output_dir).name
        self.eval_steps = set(eval_steps or [])
        # Must track the training completion budget. This was hardcoded to 512
        # against a training budget of 1024, so the reward curve measured
        # "terminates within half the budget it was trained under" rather than
        # capability, and systematically under-reported every checkpoint.
        self.max_new_tokens = max_new_tokens
        self._plot_broken = False

    def _eos_ids(self, model) -> set:
        """Every id that legitimately ends a generation.

        ``generation_config.eos_token_id`` is a list on Qwen3.5 (``<|im_end|>`` as
        well as ``<|endoftext|>``); reading only ``tokenizer.eos_token_id`` marks
        properly-terminated completions as truncated. Related bugs have passed on
        4B and failed on 9B before, so handle both shapes.
        """
        ids = set()
        for source in (getattr(getattr(model, "generation_config", None), "eos_token_id", None),
                       getattr(self.tokenizer, "eos_token_id", None)):
            if source is None:
                continue
            ids.update(source if isinstance(source, (list, tuple, set)) else [source])
        return ids

    def _refresh_figure(self, state):
        if self.fig_path is None or self._plot_broken:
            return
        from scripts.rl_training.plot_reward import (
            eval_points_from_reward_log,
            plot_reward_curve,
            train_points_from_log_history,
        )

        try:
            plot_reward_curve(
                train_points_from_log_history(state.log_history),
                eval_points_from_reward_log(self.log_path),
                self.fig_path,
                title=f"{self.run_name} — verifiable reward",
            )
        except Exception as err:  # a plotting failure must never kill training
            self._plot_broken = True
            print(f"[drift] disabling reward figure (plot failed: {err})")

    def on_log(self, args, state, control, **kwargs):
        self._refresh_figure(state)

    def on_save(self, args, state, control, model=None, **kwargs):
        from scripts.rl_training.prompts import render_coding_prompt
        from scripts.rl_training.rewards import run_verifier

        if self.eval_steps and state.global_step not in self.eval_steps:
            return
        model = model or kwargs.get("model")
        if model is None:
            return
        # Experimental TRL PPO wraps the causal policy together with the value
        # model.  Evaluate the policy, not the wrapper, at checkpoint cadence.
        if hasattr(model, "module"):
            model = model.module
        if hasattr(model, "policy"):
            model = model.policy
        was_training = model.training
        model.eval()
        total, count = 0.0, 0
        finished_total, finished_count = 0.0, 0
        import torch
        from tqdm.auto import tqdm

        eos_ids = self._eos_ids(model)
        for rec in tqdm(self.eval_records, desc=f"[drift] reward eval @ step {state.global_step}", unit="prompt"):
            prompt = render_coding_prompt(self.tokenizer, rec["prompt"])
            enc = self.tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **enc, max_new_tokens=self.max_new_tokens, do_sample=True, temperature=1.0,
                    num_return_sequences=self.n_samples,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            for seq in out:
                generated = seq[enc["input_ids"].shape[1]:]
                text = self.tokenizer.decode(generated, skip_special_tokens=True)
                reward = run_verifier(text, rec["verifier"], timeout=self.reward_timeout)
                finished = bool(eos_ids) and any(int(t) in eos_ids for t in generated)
                total += reward
                count += 1
                if finished:
                    finished_total += reward
                    finished_count += 1
        # Two numbers, never one. ``eval_reward`` keeps its old meaning (truncated
        # generations score 0 and are averaged in) so existing curves stay
        # comparable, but it is the PRODUCT of termination and quality, and
        # training moves termination far more easily than quality. Read
        # ``eval_reward_given_termination`` alongside ``termination_rate``.
        mean_reward = total / max(count, 1)
        termination_rate = finished_count / max(count, 1)
        conditional = finished_total / finished_count if finished_count else 0.0
        with self.log_path.open("a") as f:
            f.write(json.dumps({
                "step": state.global_step,
                "eval_reward": mean_reward,
                "eval_reward_given_termination": conditional,
                "termination_rate": termination_rate,
                "n_generations": count,
                "n_truncated": count - finished_count,
                "max_new_tokens": self.max_new_tokens,
            }) + "\n")
        print(
            f"[drift] step {state.global_step}: eval_reward={mean_reward:.4f} "
            f"(terminated {termination_rate:.0%}, reward-if-terminated={conditional:.4f})"
        )
        if termination_rate < 0.95:
            print(
                f"    !! {1 - termination_rate:.0%} of generations hit the "
                f"{self.max_new_tokens}-token cap and scored 0. eval_reward is measuring "
                f"termination as much as capability — raise the budget before reading it "
                f"as a capability number."
            )
        self._refresh_figure(state)
        if was_training:
            model.train()


def _parse_lora_target_modules(value: str) -> list[str] | str:
    value = value.strip()
    if value == "all-linear":
        return value
    modules = [item.strip() for item in value.split(",") if item.strip()]
    if not modules:
        raise ValueError("--lora_target_modules must be 'all-linear' or a comma-separated module list")
    return modules


def _lora_config(args):
    from peft import LoraConfig, TaskType

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        target_modules=_parse_lora_target_modules(args.lora_target_modules),
    )


def _maybe_apply_lora(model, args):
    if not args.use_lora:
        if getattr(args, "init_adapter", None):
            raise SystemExit("--init_adapter requires --use_lora")
        return model
    from peft import get_peft_model

    init_adapter = getattr(args, "init_adapter", None)
    if init_adapter:
        # Staged context lengthening, per DeepScaleR
        # (https://openreview.net/forum?id=I6GzDCne7U): start at a
        # short completion budget and extend, each stage initialised from the
        # previous stage's adapter. This is NOT a resume — optimizer, scheduler
        # and RNG state are all fresh, and the run starts at step 0 — so it does
        # not need the checkpoint state that the --resume guard correctly refuses
        # to fabricate. It is a new run whose starting weights happen to be an
        # adapter rather than zeros.
        from peft import PeftModel

        adapter_path = Path(init_adapter)
        if not (adapter_path / "adapter_config.json").is_file():
            raise SystemExit(
                f"--init_adapter {adapter_path} has no adapter_config.json; point it at a "
                "checkpoint directory produced by a previous stage"
            )
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)
        print(f"[drift] LoRA initialised from {adapter_path} (fresh optimizer, step 0)")
        # The stage's rank/alpha come from the loaded adapter, so a mismatched
        # --lora_r on the command line would be silently ignored. Say so.
        loaded = model.peft_config[model.active_adapter]
        if (loaded.r, loaded.lora_alpha) != (args.lora_r, args.lora_alpha):
            print(
                f"    note: adapter is r={loaded.r}/alpha={loaded.lora_alpha}; "
                f"--lora_r {args.lora_r} / --lora_alpha {args.lora_alpha} are ignored "
                f"when continuing from an adapter"
            )
    else:
        model = get_peft_model(model, _lora_config(args))
        print("[drift] LoRA enabled")
    model.print_trainable_parameters()
    return model


def _apply_ppo_value_lora(value_model, args):
    """LoRA-tune the PPO critic backbone while keeping its score head trainable.

    TRL's experimental PPO optimizer owns both the policy and value model.  A
    full 9B critic would therefore require full gradients and Adam state in
    addition to the policy and frozen reward model, which cannot fit on one
    140 GiB GPU.  Wrap only the sequence classifier's causal backbone so TRL's
    ``PolicyAndValueWrapper`` still sees the expected outer model and ``score``
    head.
    """
    from peft import LoraConfig, TaskType, get_peft_model

    backbone_name = value_model.base_model_prefix
    backbone = getattr(value_model, backbone_name)
    value_lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        target_modules=_parse_lora_target_modules(args.lora_target_modules),
    )
    setattr(value_model, backbone_name, get_peft_model(backbone, value_lora_config))
    for parameter in value_model.score.parameters():
        parameter.requires_grad = True
    trainable = sum(parameter.numel() for parameter in value_model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in value_model.parameters())
    print(
        "[drift] PPO value-model LoRA enabled: "
        f"trainable={trainable:,}/{total:,} ({100.0 * trainable / total:.4f}%)"
    )
    return value_model


def _bf16_available() -> bool:
    import torch

    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def _load_base(base_model: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # TRL truncates prompts to max_prompt_length with the tokenizer's default
    # side. Qwen defaults to right-truncation, which cuts off the trailing
    # <|im_start|>assistant header and flips the model into plain-continuation
    # mode. Prompts should be length-filtered at prep time; left truncation is
    # the safe fallback for any stray overlong prompt.
    tokenizer.truncation_side = "left"
    # GRPO wall time is generation-dominated; bf16 roughly halves it vs fp32
    # with no quality cost on Ampere+ GPUs. Fall back to fp32 elsewhere.
    dtype = torch.bfloat16 if _bf16_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True, dtype=dtype)
    # TRL 0.24 writes an ``estimate_tokens`` flag into this mapping during
    # GRPOTrainer initialization.  Newer Transformers model classes no longer
    # create it, so provide the small compatibility attribute when absent.
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    return model, tokenizer


def _load_policy(args):
    model, tokenizer = _load_base(args.base_model)
    return _maybe_apply_lora(model, args), tokenizer


def _make_callback(cfg, args, tokenizer, dataset_path, output_dir):
    """Reward-eval callback over a held slice. Needs prompts+tests, so for DPO
    (pairs file) point --eval_dataset at the online prompts file.

    The reward-vs-step curve is a study readout, so failing to build the
    callback is loud: a bad eval dataset or config must not silently produce a
    run without reward_log.jsonl.
    """
    from scripts.rl_training.data.schema import read_coding_jsonl
    from scripts.rl_training.rewards import REWARD_TIMEOUT

    fig_path = None
    if args.figures_dir:
        fig_path = Path(args.figures_dir) / args.run_name / "reward_curve.png"
    try:
        rows = read_coding_jsonl(dataset_path)
        eval_records = rows[: cfg["eval_reward_prompts"]]
        # Measure at the budget the policy was trained under. GRPO names it
        # max_completion_length, PPO response_length, and DPO has no rollout
        # budget of its own — for DPO fall back to the pair-generation cap so the
        # curve is still generated at the length the data was built at.
        eval_max_new_tokens = (
            cfg.get("max_completion_length")
            or cfg.get("response_length")
            or cfg.get("eval_max_new_tokens")
            or 1024
        )
        return DriftCadenceCallback(
            tokenizer, eval_records, cfg["eval_reward_samples"],
            cfg.get("reward_timeout", REWARD_TIMEOUT), output_dir,
            fig_path=fig_path, run_name=args.run_name,
            eval_steps=cfg.get("eval_reward_steps"),
            max_new_tokens=eval_max_new_tokens,
        )
    except (ValueError, KeyError) as err:
        raise SystemExit(
            f"Cannot build per-checkpoint reward callback from {dataset_path!r}: {err}\n"
            "Pass --eval_dataset pointing at a coding_task_v1 prompts file and ensure "
            "eval_reward_prompts/eval_reward_samples are set in the config."
        ) from err


def _make_coding_reward(
    output_dir: str,
    num_generations: int,
    timeout: float,
    max_tests: int | None,
    num_workers: int | None,
    log_rollouts: bool,
    eos_token_ids: tuple[int, ...],
):
    """Bind the shared reward budget; optionally persist auditable rollout samples.

    timeout/max_tests/num_workers (the shared training budget) are bound via
    make_coding_reward so the objective is applied identically whether or not
    rollout logging is enabled — turning logging on must not change the reward.
    """
    from scripts.rl_training.rewards import make_coding_reward

    scorer = make_coding_reward(
        timeout=timeout,
        max_tests=max_tests,
        num_workers=num_workers,
        eos_token_ids=eos_token_ids,
    )

    if not log_rollouts:
        return scorer  # already __name__ == 'coding_reward'

    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    log_path = Path(output_dir) / f"rollout_samples.rank{rank}.jsonl"
    call_index = 0

    def logged_coding_reward(completions, prompts=None, verifier=None, record_id=None, **kwargs):
        nonlocal call_index
        assert len(completions) % num_generations == 0, (
            f"reward batch of {len(completions)} is not divisible by "
            f"num_generations={num_generations}; rollout group stats would be wrong"
        )
        rewards = scorer(completions, verifier=verifier, **kwargs)
        prompts = prompts or [None] * len(completions)
        record_id = record_id or [None] * len(completions)
        with log_path.open("a") as fout:
            for group_start in range(0, len(rewards), num_generations):
                group_rewards = rewards[group_start : group_start + num_generations]
                gradable_rewards = [reward for reward in group_rewards if reward is not None]
                mean = (
                    sum(gradable_rewards) / len(gradable_rewards)
                    if gradable_rewards else None
                )
                variance = (
                    sum((reward - mean) ** 2 for reward in gradable_rewards)
                    / max(len(gradable_rewards) - 1, 1)
                    if mean is not None else None
                )
                std = math.sqrt(variance) if variance is not None else None
                for index in range(group_start, min(group_start + num_generations, len(rewards))):
                    reward = rewards[index]
                    row = {
                        "schema": "grpo_rollout_v1",
                        "reward_call": call_index,
                        "record_id": record_id[index],
                        "prompt": prompts[index],
                        "completion": completions[index],
                        "reward": reward,
                        "completion_status": "ok" if reward is not None else "truncated",
                        "group_mean": mean,
                        "group_std": std,
                        "advantage": (
                            (reward - mean) / (std + 1e-4)
                            if reward is not None and mean is not None and std is not None
                            else None
                        ),
                    }
                    fout.write(
                        json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n"
                    )
        call_index += 1
        return rewards

    logged_coding_reward.__name__ = "coding_reward"
    return logged_coding_reward


def _patch_trl_optional_imports(use_vllm: bool = False) -> None:
    """Disable TRL's optional integrations before importing any trainer.

    By default this project uses Transformers generation for rollouts.  A
    shared environment may also contain vLLM, vllm-ascend, mergekit, or
    llm-blender; TRL imports those optional integrations eagerly even when
    they are not enabled, and a mismatched install (e.g. a vLLM built for a
    different CUDA) makes EVERY trainer import fail — DPO and PPO included,
    not just GRPO.  Transformers 5 also changed _is_package_available() to
    return a tuple, which TRL 0.24 may interpret as truthy even when a package
    is absent.  Normalize the flags and disable what this path does not use
    (keeping vLLM available when --use_vllm asks for it).
    """
    import trl.import_utils as trl_import_utils

    for name, value in vars(trl_import_utils).copy().items():
        if name.endswith("_available") and isinstance(value, tuple):
            setattr(trl_import_utils, name, value[0])
    disabled = [
        "_vllm_ascend_available",
        "_mergekit_available",
        "_llm_blender_available",
        "_weave_available",
    ]
    if not use_vllm:
        disabled.append("_vllm_available")
    for name in disabled:
        if hasattr(trl_import_utils, name):
            setattr(trl_import_utils, name, False)


def train_grpo(cfg, args, output_dir):
    _patch_trl_optional_imports(use_vllm=args.use_vllm)
    from trl import GRPOConfig, GRPOTrainer

    from scripts.rl_training.rewards import REWARD_TIMEOUT, TRAIN_REWARD_MAX_TESTS

    model, tokenizer = _load_policy(args)
    dataset = load_prompt_dataset(args.dataset, tokenizer)
    prompt_budget = audit_grpo_prompt_budget(cfg, len(dataset), world_size=1)
    print(f"[drift] GRPO prompt-budget audit: {json.dumps(prompt_budget, sort_keys=True)}")
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
        generation_batch_size=cfg["generation_batch_size"],
        max_completion_length=cfg["max_completion_length"],
        # DAPO's overlong filtering (arXiv 2503.14476). A rollout cut off at
        # max_completion_length is not evidence about quality — we never saw the
        # answer. Left at TRL's default of False it is scored 0.0 (the verifier's
        # fence regex needs a closing marker, so a truncated completion executes as
        # prose and SyntaxErrors) and then fully weighted in the loss, which
        # optimizes for terminating early rather than for being correct.
        # It also poisons the advantage: when every rollout in a group truncates,
        # all 8 score 0.0, within-group std is 0, and the gradient is exactly zero.
        # That is the same zero-spread failure m0/ hit and guards against.
        mask_truncated_completions=cfg.get("mask_truncated_completions", True),
        # DPO and PPO both enable this; GRPO was the only arm paying full
        # activation memory, which is the budget that buys completion length.
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        # Match PPO's deliberate use_reentrant=False. The reentrant implementation
        # silently produces NO gradient through a checkpointed segment when none of
        # its inputs require grad — the classic LoRA-plus-checkpointing trap, and a
        # failure that looks exactly like a normal run. DPO gets away with the bare
        # flag today, but "it happens to work" is not a reason to leave a silent
        # failure mode in the one arm we are about to re-baseline.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_only_model=True,
        bf16=_bf16_available(),
        seed=args.seed,
        report_to="none",
        **_schedule_kwargs(cfg),
    )
    # vLLM rollouts (Chunk 2, lever 2): controlled by the --use_vllm CLI flag. Only
    # pass the kwargs when enabled, so the default path stays valid on TRL builds
    # that predate these fields. Validate colocate mode on the target hardware
    # (Qwen3.5 hybrid attention) before use.
    if args.use_vllm:
        grpo_kwargs["use_vllm"] = True
        grpo_kwargs["vllm_mode"] = args.vllm_mode
    grpo_config = GRPOConfig(**grpo_kwargs)

    # These settings are correctness fixes, not optional tuning knobs.  Check the
    # instantiated TRL object so an API/default change cannot silently turn a new
    # run back into the pre-fix behavior while the YAML and run metadata still
    # claim otherwise.
    correctness_settings = {
        "mask_truncated_completions": grpo_config.mask_truncated_completions,
        "gradient_checkpointing": grpo_config.gradient_checkpointing,
        "gradient_checkpointing_kwargs": grpo_config.gradient_checkpointing_kwargs,
    }
    expected_correctness_settings = {
        "mask_truncated_completions": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
    }
    if correctness_settings != expected_correctness_settings:
        raise RuntimeError(
            "GRPO correctness-fix preflight failed: "
            f"effective={correctness_settings!r}, "
            f"expected={expected_correctness_settings!r}"
        )
    print(
        "[drift] GRPO correctness-fix preflight: "
        + json.dumps(correctness_settings, sort_keys=True)
    )
    if grpo_config.steps_per_generation != cfg["gradient_accumulation_steps"]:
        raise ValueError(
            "TRL derived steps_per_generation="
            f"{grpo_config.steps_per_generation}, expected "
            f"gradient_accumulation_steps={cfg['gradient_accumulation_steps']}"
        )
    # Cheap training-time grading over the shared reward budget: a spread subset of
    # tests, parallel across the sampled group. Full-suite grading is preserved in
    # DriftCadenceCallback (eval). The budget must match DPO's pairs and PPO's RM
    # data — warn loudly if the config drifts from the canonical values.
    train_max_tests = cfg.get("train_max_tests", TRAIN_REWARD_MAX_TESTS)
    reward_timeout = cfg.get("reward_timeout", REWARD_TIMEOUT)
    if train_max_tests != TRAIN_REWARD_MAX_TESTS or reward_timeout != REWARD_TIMEOUT:
        print(
            f"[drift] WARNING: training reward budget (max_tests={train_max_tests}, "
            f"timeout={reward_timeout}) differs from the shared default "
            f"(max_tests={TRAIN_REWARD_MAX_TESTS}, timeout={REWARD_TIMEOUT}). DPO pairs "
            f"and the PPO reward model MUST use the same values or the objective is not "
            f"identical across arms."
        )
    # One wrapper binds the shared budget (timeout/max_tests/num_workers) AND, when
    # log_rollouts is on, persists per-rollout reward/advantage rows for audit.
    from scripts.rl_training.completion_semantics import model_eos_token_ids

    eos_token_ids = model_eos_token_ids(model, tokenizer)
    print(f"[drift] GRPO authoritative EOS ids: {list(eos_token_ids)}")
    reward_func = _make_coding_reward(
        output_dir=output_dir,
        num_generations=cfg["num_generations"],
        timeout=reward_timeout,
        max_tests=train_max_tests,
        num_workers=cfg.get("reward_num_workers"),
        log_rollouts=cfg.get("log_rollouts", False),
        eos_token_ids=eos_token_ids,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_func],
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    if not args.eval_dataset:
        # Falling back to args.dataset silently evaluated the reward curve on the
        # first 64 rows of the TRAINING set, which is not a held-out measurement.
        raise SystemExit(
            "--eval_dataset is required: without it the per-checkpoint reward curve is "
            "computed on the training set. Point it at the dev/held-out coding_task_v1 "
            "prompts file for this cohort."
        )
    cb = _make_callback(cfg, args, tokenizer, args.eval_dataset, output_dir)
    if cb:
        trainer.add_callback(cb)
    hub_cb = prepared_formal_hub_callback(args, tokenizer)
    if hub_cb:
        trainer.add_callback(hub_cb)
    result = trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    if hub_cb:
        hub_cb.archive_final(Path(output_dir), trainer.state.global_step)
    return trainer.state.global_step, result.metrics


def train_dpo(cfg, args, output_dir):
    _patch_trl_optional_imports()
    from trl import DPOConfig, DPOTrainer

    model, tokenizer = _load_policy(args)
    dataset = load_preference_dataset(args.dataset)
    dpo_config = DPOConfig(
        output_dir=output_dir,
        max_steps=cfg["max_steps"],
        save_steps=cfg["save_steps"],
        logging_steps=cfg["logging_steps"],
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        beta=cfg["beta"],
        max_length=cfg["max_length"],
        save_only_model=True,
        gradient_checkpointing=True,
        bf16=_bf16_available(),
        seed=args.seed,
        report_to="none",
        **_schedule_kwargs(cfg),
    )
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    if not args.eval_dataset:
        print("[drift] no --eval_dataset (prompts+tests); skipping per-checkpoint reward log for DPO")
    else:
        cb = _make_callback(cfg, args, tokenizer, args.eval_dataset, output_dir)
        if cb:
            trainer.add_callback(cb)
    hub_cb = prepared_formal_hub_callback(args, tokenizer)
    if hub_cb:
        trainer.add_callback(hub_cb)
    result = trainer.train()
    # save_steps may not divide max_steps (the headline ends at step 125 while
    # dense periodic saves are every four steps).  Persist the true final LoRA
    # adapter at the run root just like the matched SFT trainer does.
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    if hub_cb:
        hub_cb.archive_final(Path(output_dir), trainer.state.global_step)
    return trainer.state.global_step, result.metrics


def _install_ppo_gradient_guard(
    trainer, max_grad_norm: float, nonfinite_skip_limit: int = 0
) -> dict[str, int]:
    """Make TRL experimental PPO apply its advertised gradient clipping.

    TRL 1.9's experimental PPO loop calls ``optimizer.step()`` directly and
    never consumes ``TrainingArguments.max_grad_norm``.  On the formal 4B run
    this allowed one late update to turn every LoRA tensor into NaN.  Wrap the
    Accelerate optimizer so clipping occurs only at a real (synchronized)
    optimizer step, and refuse a non-finite gradient before it can corrupt the
    weights saved by the next checkpoint.
    """
    if max_grad_norm <= 0:
        raise ValueError("PPO max_grad_norm must be positive")
    if nonfinite_skip_limit < 0:
        raise ValueError("PPO nonfinite_skip_limit cannot be negative")

    import torch

    original_step = trainer.optimizer.step
    original_backward = trainer.accelerator.backward
    guard_state = {
        "nonfinite_gradient_updates_skipped": 0,
        "consecutive_nonfinite_gradient_updates": 0,
        "max_consecutive_nonfinite_gradient_updates": 0,
    }

    def guarded_backward(loss, *backward_args, **backward_kwargs):
        if not torch.isfinite(loss.detach()).all():
            trainer.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(
                "PPO produced a non-finite scalar loss before backward; "
                f"loss={loss.detach().float().item()}; no optimizer update was applied"
            )
        return original_backward(loss, *backward_args, **backward_kwargs)

    def guarded_step(*step_args, **step_kwargs):
        if trainer.accelerator.sync_gradients:
            named_parameters = [
                (name, parameter)
                for name, parameter in trainer.model.named_parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            nonfinite = []
            for name, parameter in named_parameters:
                finite = torch.isfinite(parameter.grad)
                if not finite.all():
                    nonfinite.append(
                        (
                            name,
                            int(torch.isnan(parameter.grad).sum().item()),
                            int(torch.isposinf(parameter.grad).sum().item()),
                            int(torch.isneginf(parameter.grad).sum().item()),
                            parameter.grad.dtype,
                            tuple(parameter.grad.shape),
                        )
                    )
            if nonfinite:
                summary = "; ".join(
                    f"{name}:nan={nan_count},+inf={posinf_count},-inf={neginf_count},"
                    f"dtype={dtype},shape={shape}"
                    for name, nan_count, posinf_count, neginf_count, dtype, shape in nonfinite[:8]
                )
                guard_state["nonfinite_gradient_updates_skipped"] += 1
                guard_state["consecutive_nonfinite_gradient_updates"] += 1
                guard_state["max_consecutive_nonfinite_gradient_updates"] = max(
                    guard_state["max_consecutive_nonfinite_gradient_updates"],
                    guard_state["consecutive_nonfinite_gradient_updates"],
                )
                skipped = guard_state["nonfinite_gradient_updates_skipped"]
                trainer.optimizer.zero_grad(set_to_none=True)
                if skipped > nonfinite_skip_limit:
                    raise FloatingPointError(
                        "PPO non-finite gradient skip limit exceeded before optimizer.step(); "
                        f"skipped={skipped},limit={nonfinite_skip_limit},"
                        f"offending_params={len(nonfinite)}; first={summary}; "
                        "the current policy was left unchanged and no poisoned checkpoint was saved"
                    )
                print(
                    "[drift] WARNING: skipped non-finite PPO optimizer update "
                    f"{skipped}/{nonfinite_skip_limit}; offending_params={len(nonfinite)}; "
                    f"first={summary}"
                )
                return None
            parameters = [parameter for _, parameter in named_parameters]
            grad_norm = trainer.accelerator.clip_grad_norm_(parameters, max_grad_norm)
            if not torch.isfinite(torch.as_tensor(grad_norm)).all():
                trainer.optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    "PPO finite gradient elements produced a non-finite aggregate norm before "
                    "optimizer.step(); "
                    "the current policy was left unchanged and no poisoned checkpoint was saved"
                )
        result = original_step(*step_args, **step_kwargs)
        guard_state["consecutive_nonfinite_gradient_updates"] = 0
        return result

    trainer.optimizer.step = guarded_step
    trainer.accelerator.backward = guarded_backward
    print(
        f"[drift] PPO gradient clipping enabled: max_grad_norm={max_grad_norm:g}; "
        f"nonfinite_skip_limit={nonfinite_skip_limit}"
    )
    return guard_state


def _extract_train_metrics(result) -> dict:
    """Normalize Trainer and experimental-TRL train return values."""
    return dict(getattr(result, "metrics", None) or {})


def _patch_qwen35_fallback_l2norm() -> None:
    """Keep Qwen3.5 fallback Q/K normalization in float32 for backward stability.

    Transformers' no-FLA fallback normalizes BF16 Q/K tensors before casting
    the rest of the gated-delta calculation to float32.  The forward values can
    remain finite while the BF16 normalization backward produces NaN gradients
    across the policy and value model.  Preserve the public function's output
    dtype, but perform its reduction and reciprocal square root in float32.
    """
    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5

    if getattr(modeling_qwen3_5.l2norm, "_drift_float32", False):
        return

    def stable_l2norm(x, dim: int = -1, eps: float = 1e-6):
        input_dtype = x.dtype
        x_float = x.float()
        inv_norm = torch.rsqrt((x_float * x_float).sum(dim=dim, keepdim=True) + eps)
        return (x_float * inv_norm).to(input_dtype)

    stable_l2norm._drift_float32 = True
    modeling_qwen3_5.l2norm = stable_l2norm
    print("[drift] Qwen3.5 fallback Q/K l2norm uses float32 reduction")


def train_ppo(cfg, args, output_dir):
    # TRL's PPOTrainer optimizes a reward MODEL, not a Python reward fn. To keep
    # the objective identical to GRPO, --reward_model must be an RM trained on the
    # same verifiable signal (see rl_training/README.md). We refuse to run without
    # one rather than silently change the objective.
    if not args.reward_model:
        raise SystemExit(
            "PPO requires --reward_model (a sequence-classification RM trained on the "
            "same unit-test reward as GRPO). See rl_training/README.md 'PPO objective'."
        )
    rm_manifest_path = Path(args.reward_model) / "reward_model_manifest.json"
    if not rm_manifest_path.is_file():
        raise SystemExit(f"PPO reward-model manifest not found: {rm_manifest_path}")
    rm_manifest = json.loads(rm_manifest_path.read_text())
    if rm_manifest.get("schema") != "reward_model_manifest_v1":
        raise SystemExit(f"Unexpected PPO reward-model manifest schema: {rm_manifest_path}")
    rm_accuracy = float(rm_manifest["held_out_pairwise_accuracy"])
    if rm_accuracy < 0.6 and not args.allow_weak_reward_model:
        raise SystemExit(
            f"PPO refused weak reward model: held-out pairwise accuracy={rm_accuracy:.4f} < 0.60. "
            "For plumbing-only smoke tests, pass --allow_weak_reward_model; never use that flag "
            "for a formal experiment."
        )
    if rm_accuracy < 0.6:
        print(
            f"[drift] WARNING: allowing weak RM accuracy={rm_accuracy:.4f} for plumbing-only PPO smoke"
        )
    _patch_trl_optional_imports()
    _patch_qwen35_fallback_l2norm()
    import torch
    from transformers import AutoModelForSequenceClassification
    from trl.experimental.ppo import PPOConfig
    from scripts.rl_training.completion_semantics import model_eos_token_ids
    from scripts.rl_training.masked_ppo_trainer import TruncationMaskedPPOTrainer

    model, tokenizer = _load_policy(args)
    # Decoder-only rollout generation requires left padding. Right padding can
    # make generation continue from pad tokens instead of the end of a prompt.
    tokenizer.padding_side = "left"
    # Experimental PPO can use the disabled LoRA adapter as its reference,
    # avoiding a redundant full reference model allocation.
    ref_model = None
    if not args.use_lora:
        ref_model, _ = _load_base(args.base_model)
    # Reward + value models share the RM checkpoint (a plain seq-classification
    # dir from train_reward_model.py). Match the policy dtype so activations line
    # up, and set pad_token_id: seq-classification scoring locates each sequence's
    # last non-pad token, so a missing pad id silently reads the wrong position.
    rm_dtype = torch.bfloat16 if _bf16_available() else torch.float32
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model, trust_remote_code=True, num_labels=1, dtype=rm_dtype
    )
    value_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model, trust_remote_code=True, num_labels=1, dtype=rm_dtype
    )
    value_model = _apply_ppo_value_lora(value_model, args)
    reward_model.config.pad_token_id = tokenizer.pad_token_id
    value_model.config.pad_token_id = tokenizer.pad_token_id
    reward_model.config.get_text_config().pad_token_id = tokenizer.pad_token_id
    value_model.config.get_text_config().pad_token_id = tokenizer.pad_token_id
    # TRL's PolicyAndValueWrapper forwards gradient-checkpointing controls only
    # to the policy.  The critic backbone is a second full causal backbone and,
    # without its own checkpointing, the 9B run retains enough activations to
    # exhaust a 140 GiB H200 during the first PPO minibatch.  Enable it directly
    # before the wrapper/optimizer are constructed.  This changes only the
    # activation-memory/computation tradeoff, not the PPO batch or objective.
    checkpointing_kwargs = {"use_reentrant": False}
    value_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs=checkpointing_kwargs
    )
    if hasattr(value_model.config, "use_cache"):
        value_model.config.use_cache = False
    print("[drift] PPO value-model gradient checkpointing enabled (use_reentrant=False)")
    dataset = load_ppo_prompt_dataset(args.dataset, tokenizer, cfg["max_prompt_length"])
    ppo_config = PPOConfig(
        output_dir=output_dir,
        total_episodes=cfg["max_steps"] * cfg["per_device_train_batch_size"] * cfg["gradient_accumulation_steps"],
        save_steps=cfg["save_steps"],
        logging_steps=cfg["logging_steps"],
        learning_rate=cfg["learning_rate"],
        max_grad_norm=cfg["max_grad_norm"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        kl_coef=cfg["init_kl_coef"],
        num_ppo_epochs=cfg["num_ppo_epochs"],
        num_sample_generations=cfg.get("num_sample_generations", 0),
        response_length=cfg["response_length"],
        save_only_model=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs=checkpointing_kwargs,
        stop_token="eos",  # truncate generated responses at EOS before scoring
        bf16=_bf16_available(),
        seed=args.seed,
        report_to="none",
        **_schedule_kwargs(cfg),
    )
    eos_token_ids = model_eos_token_ids(model, tokenizer)
    print(f"[drift] PPO authoritative EOS ids: {list(eos_token_ids)}")
    trainer = TruncationMaskedPPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=model,
        ref_model=ref_model,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=dataset,
        eval_dataset=dataset,
        eos_token_ids=eos_token_ids,
    )
    gradient_guard_state = _install_ppo_gradient_guard(
        trainer,
        float(cfg["max_grad_norm"]),
        int(cfg.get("nonfinite_gradient_skip_limit", 0)),
    )
    if not args.eval_dataset:
        # Falling back to args.dataset silently evaluated the reward curve on the
        # first 64 rows of the TRAINING set, which is not a held-out measurement.
        raise SystemExit(
            "--eval_dataset is required: without it the per-checkpoint reward curve is "
            "computed on the training set. Point it at the dev/held-out coding_task_v1 "
            "prompts file for this cohort."
        )
    cb = _make_callback(cfg, args, tokenizer, args.eval_dataset, output_dir)
    if cb:
        trainer.add_callback(cb)
    hub_cb = prepared_formal_hub_callback(args, tokenizer)
    if hub_cb:
        trainer.add_callback(hub_cb)
    result = trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    if hub_cb:
        hub_cb.archive_final(Path(output_dir), trainer.state.global_step)
    # TRL's experimental PPOTrainer.train() performs the full loop but returns
    # None (unlike transformers.Trainer.train(), which returns TrainOutput).
    # The authoritative step/log history remains on trainer.state.
    metrics = _extract_train_metrics(result)
    metrics.update(gradient_guard_state)
    metrics.update({
        f"completion_masking/{key}": value
        for key, value in trainer.completion_masking_stats.items()
    })
    return trainer.state.global_step, metrics


TRAINERS = {"grpo": train_grpo, "dpo": train_dpo, "ppo": train_ppo}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algo", required=True, choices=list(TRAINERS))
    parser.add_argument("--config", required=True, help="rl_training/configs/<algo>.yaml")
    parser.add_argument("--base_model", required=True, help="HF id or local path of the base model")
    parser.add_argument("--dataset", required=True, help="JSONL: coding_task_v1 (grpo/ppo) or dpo_preference_v1 (dpo)")
    parser.add_argument("--eval_dataset", default=None, help="coding_task_v1 JSONL for per-checkpoint reward log")
    parser.add_argument("--reward_model", default=None, help="RM path for PPO (see README)")
    parser.add_argument(
        "--allow_weak_reward_model",
        action="store_true",
        help="Allow PPO with RM held-out accuracy <0.60 for plumbing smoke tests only",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--save_dir", default="runs")
    parser.add_argument("--run_name", required=True)
    parser.add_argument(
        "--figures_dir", default="results",
        help="Reward-curve PNG goes to <figures_dir>/<run_name>/reward_curve.png, "
        "refreshed live during training ('' disables)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="unsupported for model-only checkpoints; retained only for an explicit error",
    )
    parser.add_argument(
        "--use_vllm", action="store_true",
        help="GRPO rollouts via vLLM (needs vllm installed; default is Transformers generate)",
    )
    parser.add_argument(
        "--vllm_mode", default="colocate", choices=["colocate", "server"],
        help="TRL vLLM mode; 'colocate' shares the training GPU, 'server' needs `trl vllm-serve`",
    )
    parser.add_argument("--use_lora", action="store_true", help="Train LoRA adapters instead of full model weights")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_bias", default="none", choices=["none", "all", "lora_only"])
    parser.add_argument(
        "--init_adapter",
        default=None,
        help=(
            "Start LoRA from an existing adapter directory instead of from zeros, with "
            "fresh optimizer/scheduler state at step 0. For staged context lengthening "
            "(4K -> 8K -> 16K): each stage passes the previous stage's checkpoint. This "
            "is not --resume, which is refused because periodic checkpoints omit "
            "optimizer state."
        ),
    )
    parser.add_argument(
        "--lora_target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated target modules, or 'all-linear'",
    )
    add_formal_hub_arguments(parser)
    args = parser.parse_args()

    if args.resume:
        raise SystemExit(
            "Periodic checkpoints intentionally omit optimizer/scheduler/RNG state and cannot "
            "resume training; diagnose the incomplete run, archive it, and restart the arm cleanly"
        )

    cfg = load_config(args.config, args.algo)
    if args.hub_repo_id and not args.use_lora:
        raise SystemExit("Formal Hub checkpoint archival requires --use_lora")
    set_all_seeds(args.seed)
    output_dir = str(Path(args.save_dir) / args.run_name)
    output_path = Path(output_dir)
    metadata_path = output_path / "run_metadata.json"
    metadata = build_run_metadata(cfg, args)
    existing_run_entries = (
        [path for path in output_path.iterdir() if path.name != LOCAL_FALLBACK_MARKER]
        if output_path.exists()
        else []
    )
    if existing_run_entries:
        raise SystemExit(
            "Run directory is not empty and model-only checkpoints cannot resume training; "
            f"diagnose/archive it and restart cleanly: {output_path}"
        )
    try:
        prepare_formal_hub_upload(args, output_dir=output_path, algo=args.algo)
    except ValueError as exc:
        raise SystemExit(f"Formal Hub preflight failed: {exc}") from None
    output_path.mkdir(parents=True, exist_ok=True)
    write_run_metadata(output_dir, metadata)
    print(f"[drift] {args.algo} run '{args.run_name}' seed={args.seed} -> {output_dir}")
    global_step, metrics = TRAINERS[args.algo](cfg, args, output_dir)
    metadata["status"] = "complete"
    metadata["global_step"] = global_step
    metadata["train_metrics"] = metrics
    write_run_metadata(output_dir, metadata)
    print(f"[drift] done. Register checkpoints:\n"
          f"  python rl_training/register_checkpoints.py --run_dir {output_dir} "
          f"--base_model {args.base_model} --algo {args.algo} --seed {args.seed}")


if __name__ == "__main__":
    main()
