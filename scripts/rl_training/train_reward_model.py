#!/usr/bin/env python3
"""Train a Bradley-Terry reward model for the PPO arm of the drift study.

TRL's PPOTrainer optimizes a reward MODEL, not a Python reward function. To keep
PPO's objective identical to GRPO/DPO, this RM is trained on the SAME
``dpo_preference_v1`` pairs DPO trains on -- the pairs DPO builds from the shared
verifiable reward (rl_training/build_dpo_pairs.py, graded at the 5s / 12-test
shared budget). So PPO's reward source and DPO's training data are the same
artifact, and the RM is a controlled component rather than a free confound.

Each pair (prompt, chosen, rejected) becomes two full sequences the RM scores:
    chosen_text   = prompt + chosen_completion
    rejected_text = prompt + rejected_completion
and the Bradley-Terry loss pushes score(chosen) > score(rejected).

The saved model is a plain ``AutoModelForSequenceClassification`` directory
(num_labels=1) that ``train_rl.py --algo ppo`` loads as both reward and value
model. A held-out slice reports pairwise accuracy -- the quality gate: an RM near
50% would have PPO optimizing noise.

Train one RM per base-model size (0.8B / 4B / 9B), each on that size's own pair
file, with the same recipe.

    python -m rl_training.train_reward_model \
        --config rl_training/configs/reward_model.yaml \
        --base_model Qwen/Qwen3.5-0.8B \
        --pairs data/rl/v1/cohorts/<model>-shared-n<N>-s0/artifacts/dpo.jsonl \
        --output_dir runs/rm-qwen08b-s0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_pair_texts(rows: list[dict]) -> list[dict]:
    """Turn dpo_preference_v1 rows into full-sequence (chosen, rejected) texts.

    The RM scores the whole prompt+completion sequence, so each side is the
    prompt (already chat-templated by render_coding_prompt at pair-build time)
    concatenated with the raw completion -- exactly the sequence the policy
    produced during sampling.
    """
    examples = []
    for r in rows:
        prompt = r["prompt"]
        examples.append({
            "chosen": prompt + r["chosen"],
            "rejected": prompt + r["rejected"],
        })
    return examples


def split_train_eval(examples: list[dict], eval_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Deterministic held-out split for the pairwise-accuracy gate."""
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    n_eval = max(1, int(round(len(shuffled) * eval_fraction))) if len(shuffled) > 1 else 0
    return shuffled[n_eval:], shuffled[:n_eval]


def pairwise_accuracy(scores_chosen: list[float], scores_rejected: list[float]) -> float:
    """Fraction of held-out pairs the RM ranks correctly (chosen > rejected)."""
    if not scores_chosen:
        return float("nan")
    correct = sum(1 for c, r in zip(scores_chosen, scores_rejected) if c > r)
    return correct / len(scores_chosen)


def pair_max_token_lengths(examples: list[dict], tokenizer) -> list[int]:
    """Return the longer tokenized side of every pair, including TRL's EOS.

    RewardTrainer filters examples longer than ``max_length``; it does not
    truncate them.  Checking coverage before trainer construction prevents a
    too-small RM context window from silently changing the formal cohort.
    """
    side_lengths: dict[str, list[int]] = {}
    eos_id = tokenizer.eos_token_id
    for side in ("chosen", "rejected"):
        encoded = tokenizer(
            [example[side] for example in examples],
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        lengths = []
        for input_ids in encoded:
            needs_eos = eos_id is not None and (not input_ids or input_ids[-1] != eos_id)
            lengths.append(len(input_ids) + int(needs_eos))
        side_lengths[side] = lengths
    return [max(chosen, rejected) for chosen, rejected in zip(
        side_lengths["chosen"], side_lengths["rejected"]
    )]


def _bf16_available() -> bool:
    import torch

    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _score_texts(model, tokenizer, texts, max_length, batch_size, device) -> list[float]:
    """RM scalar score per sequence (the seq-classification logit, num_labels=1)."""
    import torch

    scores: list[float] = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits.squeeze(-1)
        scores.extend(logits.float().cpu().tolist())
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="rl_training/configs/reward_model.yaml")
    parser.add_argument("--base_model", required=True, help="HF id or path; same base as the policy for this size")
    parser.add_argument("--pairs", required=True, help="dpo_preference_v1 JSONL (the SAME file DPO trains on)")
    parser.add_argument("--output_dir", required=True, help="Where the RM (+ manifest) is saved")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Batch size for the held-out accuracy pass")
    args = parser.parse_args()

    import yaml

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if cfg.get("algo") != "reward_model":
        raise ValueError(f"Config {args.config} is for algo {cfg.get('algo')!r}, not 'reward_model'")
    seed = args.seed if args.seed is not None else cfg.get("seed", 0)

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from trl import RewardConfig, RewardTrainer

    from scripts.rl_training.data.schema import read_preference_jsonl

    random.seed(seed)
    torch.manual_seed(seed)

    rows = read_preference_jsonl(args.pairs)
    examples = build_pair_texts(rows)
    train_examples, eval_examples = split_train_eval(examples, cfg["eval_fraction"], seed)
    print(f"[rm] {len(examples)} pairs -> {len(train_examples)} train / {len(eval_examples)} held-out (seed={seed})")
    if not train_examples:
        raise SystemExit("No training pairs; check the pairs file / eval_fraction.")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"  # keep the assistant completion, drop old prompt head if overlong

    pair_lengths = pair_max_token_lengths(examples, tokenizer)
    n_overlength = sum(length > cfg["max_length"] for length in pair_lengths)
    print(
        f"[rm] token-length coverage: {len(pair_lengths) - n_overlength}/{len(pair_lengths)} "
        f"pairs fit max_length={cfg['max_length']} (longest={max(pair_lengths)})"
    )
    if n_overlength:
        raise ValueError(
            f"RewardTrainer would filter {n_overlength}/{len(pair_lengths)} formal pairs because "
            f"max_length={cfg['max_length']} is smaller than the longest prompt+completion "
            f"sequence ({max(pair_lengths)} tokens). Increase max_length; do not silently shrink "
            "the frozen cohort."
        )

    dtype = torch.bfloat16 if _bf16_available() else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, trust_remote_code=True, num_labels=1, dtype=dtype
    )
    # Seq-classification models must know the pad id to find each sequence's last
    # non-pad token for the score; without it, from_pretrained warns and scoring
    # can silently read the wrong position.
    model.config.pad_token_id = tokenizer.pad_token_id
    # Composite configs such as Qwen3.5 keep the sequence-classification
    # pooling setting on their nested text config. Transformers 5 checks that
    # nested value, not only the outer config set by RewardTrainer.
    model.config.get_text_config().pad_token_id = tokenizer.pad_token_id

    used_lora = bool(cfg.get("use_lora", False))
    if used_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
            target_modules=cfg.get("lora_target_modules", "all-linear"),
        ))
        model.print_trainable_parameters()

    from datasets import Dataset

    train_ds = Dataset.from_list(train_examples)

    # TRL 0.24 RewardTrainer: dataset carries "chosen"/"rejected" full-sequence
    # text; the trainer tokenizes via processing_class up to max_length and applies
    # the Bradley-Terry loss. (If a future TRL expects pre-tokenized
    # input_ids_chosen/input_ids_rejected instead, tokenize here and pass those --
    # the shakedown validates which the pinned TRL wants.)
    reward_config = RewardConfig(
        output_dir=args.output_dir,
        num_train_epochs=cfg["num_train_epochs"],
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        logging_steps=cfg["logging_steps"],
        max_length=cfg["max_length"],
        bf16=_bf16_available(),
        save_strategy="no",  # one final model saved manually below (after LoRA merge)
        report_to="none",
        seed=seed,
    )
    trainer = RewardTrainer(
        model=model,
        args=reward_config,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )
    trainer.train()

    # Merge LoRA so the saved artifact is a standalone seq-classification model
    # that AutoModelForSequenceClassification.from_pretrained loads directly
    # (train_rl.py --algo ppo needs a plain RM, not a base+adapter pair).
    final_model = model.merge_and_unload() if used_lora else model
    final_model.config.pad_token_id = tokenizer.pad_token_id
    final_model.config.get_text_config().pad_token_id = tokenizer.pad_token_id
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    # Quality gate: held-out pairwise accuracy. An RM near 0.5 means PPO would be
    # optimizing noise -- surface it loudly rather than silently proceeding.
    accuracy = float("nan")
    if eval_examples:
        device = final_model.device
        chosen = _score_texts(final_model, tokenizer, [e["chosen"] for e in eval_examples],
                              cfg["max_length"], args.eval_batch_size, device)
        rejected = _score_texts(final_model, tokenizer, [e["rejected"] for e in eval_examples],
                               cfg["max_length"], args.eval_batch_size, device)
        accuracy = pairwise_accuracy(chosen, rejected)

    manifest = {
        "schema": "reward_model_manifest_v1",
        "base_model": args.base_model,
        "pairs_path": args.pairs,
        "pairs_sha256": _sha256(args.pairs),
        "n_pairs": len(examples),
        "n_train": len(train_examples),
        "n_eval": len(eval_examples),
        "held_out_pairwise_accuracy": accuracy,
        "seed": seed,
        "used_lora": used_lora,
        "config": cfg,
    }
    (out_dir / "reward_model_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"[rm] held-out pairwise accuracy: {accuracy:.4f}  (gate: clearly above 0.5)")
    if eval_examples and accuracy < 0.6:
        print("[rm] WARNING: pairwise accuracy is low; PPO would optimize a weak/noisy reward. "
              "Inspect the pairs (margin, count) before launching PPO.")
    print(f"[rm] saved reward model + manifest -> {out_dir}")
    print(f"[rm] use with: train_rl.py --algo ppo --reward_model {out_dir} ...")


if __name__ == "__main__":
    main()
