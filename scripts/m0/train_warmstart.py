#!/usr/bin/env python3
"""LoRA SFT the warm-start into the 4B, then merge (Phase 2 of docs/m0-warmstart-plan.md).

Consumes the ``m0_warmstart_sft_v1`` traces from ``m0/data/build_warmstart.py`` -- the 4B's own short, correct, committed reasoning on M0's neutral fact prompts -- and trains the base model to produce that short reasoning under the NORMAL prompt. The point is only to move the model's default reasoning length from ~730 tokens into the 256/500 band, so that when GRPO runs, its rollouts actually land in the band and the reward has something to sharpen. It is deliberately light: a couple of epochs of LoRA, then merge. GRPO does the real format work.

Loss is masked to the completion only -- the prompt (including the opened ``<think>`` block from the chat template) is set to -100, so the model is supervised on the short reasoning + ``</think>`` + answer line + EOS, not on reconstructing the question. EOS is supervised so the model learns to STOP after the answer rather than run on.

The result is MERGED into the base weights and saved as an ordinary model directory, exactly like ``export_m0.py`` does for the final model, so the GRPO run points ``--base_model`` at it and trains a fresh LoRA. A warm-start shipped as an adapter would make GRPO stack LoRA on LoRA, which is the silent wrong-answer failure the merge exists to avoid.

The base load, LoRA config, and the Qwen3.5 hybrid-attention ``all-linear`` handling are reused verbatim from ``m0/train_m0.py::_load_policy`` so the warm-start trains on the exact stack GRPO uses. Content neutrality is inherited from the trace set, which is built only from the screened M0 fact prompts -- no elicitation content ever enters. This is TRAINING, so it is fine on the paid AWS box (unlike the trace generation, which is inference).
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.m0.prompts import render_format_prompt
from scripts.m0.train_m0 import _bf16_available, _load_policy, check_fused_kernels


def load_traces(path: str) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"{path}: no traces -- run m0/data/build_warmstart.py first")
    bad = [r.get("schema") for r in rows if r.get("schema") != "m0_warmstart_sft_v1"]
    if bad:
        raise SystemExit(f"{path}: expected schema m0_warmstart_sft_v1, found {set(bad)}")
    return rows


def screen_traces(path: str, rows: list[dict], tokenizer) -> None:
    """STRUCTURALLY RE-SCREEN a trace set before it becomes training targets.

    The schema string alone is not a guarantee. Trace sets are hand-reviewed and hand-edited, and an edit that looks trivial can break the shape the target is supposed to teach: changing a final answer letter and leaving the old line behind gives two lines after ``</think>``, which is the `coda_after_think` failure the builder rejects at generation time and which trained a model that reproduced the coda at serve time. Re-running the production screen here means a hand-edited file fails loudly before training instead of silently installing the wrong format. Every trace in a set the builder produced already passes, so this only ever fires on human edits or a stale file.

    ``n_reasoning_tokens`` is RECOMPUTED (and overwritten in place) rather than trusted for the same reason: the stored count is whatever the builder measured before any hand edit, so an edit that guts the reasoning would sail past the band-floor check on a stale number. The recount uses the exact build-time recipe (``split_completion`` + ``_count_tokens`` with the training tokenizer), which is why this runs after ``_load_policy`` rather than at load.
    """
    from scripts.m0.rewards import _count_tokens, split_completion
    from scripts.m0.scripts.clean_traces import classify

    for row in rows:
        reasoning, _ = split_completion(row["completion"])
        row["n_reasoning_tokens"] = _count_tokens(reasoning, tokenizer)

    broken = [(i, r.get("record_id", "?"), classify(r)) for i, r in enumerate(rows, 1)]
    broken = [b for b in broken if b[2] is not None]
    if broken:
        detail = "; ".join(f"line {i} {rid}: {why}" for i, rid, why in broken[:5])
        raise SystemExit(
            f"{path}: {len(broken)} trace(s) fail the production structural screen "
            f"(m0/scripts/clean_traces.classify) and would teach the wrong shape -- {detail}"
            + (" ..." if len(broken) > 5 else "")
        )


def build_example(row: dict, tokenizer, max_len: int) -> dict:
    """Tokenise one trace into input_ids + labels, with the prompt masked to -100.

    The prompt is the chat-templated string from render_format_prompt (the same rendering generation and GRPO use), which already ends with the opened ``<think>`` block. It is tokenised with add_special_tokens=False because the template already carries every special token it needs as text; adding more would desync training from the served prompt. The completion is the stored short trace (``reasoning</think>\n\nAnswer: X``); EOS is appended by id so its tokenisation is exact.

    The prompt/completion junction sits at a newline, where BPE does not merge across, so len(prompt_ids) is a reliable mask boundary. Overlong rows are left-truncated (drop the prompt's head, keep the assistant header and the whole completion) -- prompts are already length-filtered at build time, so this rarely fires.
    """
    prompt_str = render_format_prompt(tokenizer, row["prompt"])
    prompt_ids = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(row["completion"], add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]

    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    if len(input_ids) > max_len:
        overflow = len(input_ids) - max_len
        input_ids = input_ids[overflow:]
        labels = labels[overflow:]
    return {"input_ids": input_ids, "labels": labels}


class MaskedCollator:
    """Right-pad input_ids (with pad id) and labels (with -100) to the batch max, build attention_mask."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, features: list[dict]) -> dict:
        import torch

        width = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attn = [], [], []
        for f in features:
            pad = width - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * pad)
            labels.append(f["labels"] + [-100] * pad)
            attn.append([1] * len(f["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traces", default="data/rl/m0_warmstart/sft_traces.jsonl")
    parser.add_argument("--base_model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--out", required=True, help="output dir for the MERGED warm-started model; GRPO --base_model points here")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=1536)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    # LoRA knobs mirror m0/train_m0.py; all-linear is REQUIRED on hybrid-attention Qwen3.5 or the 18 linear_attn layers are silently frozen.
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_bias", default="none", choices=["none", "all", "lora_only"])
    parser.add_argument("--lora_target_modules", default="all-linear")
    # The re-wrap holds the base wrapper and the merged weights on the card at once, so it needs ~2x
    # the model. auto measures that against free VRAM and drops to CPU when it will not fit, which is
    # what makes 27B possible on an 80 GB card. See m0/wrap_for_serving.py.
    parser.add_argument("--wrap_device", default="auto", choices=["auto", "cuda", "cpu"],
                        help="device for the serving re-wrap (default auto: GPU if it fits, else CPU)")
    args = parser.parse_args()

    import torch
    from transformers import Trainer, TrainingArguments, set_seed

    set_seed(args.seed)
    check_fused_kernels()

    rows = load_traces(args.traces)
    families = Counter(r["family"] for r in rows)
    print(f"[warmstart-sft] {len(rows)} traces; families {dict(families)}")

    # _load_policy wants an args-like object; hand it exactly the fields it reads so the base + LoRA
    # setup is byte-identical to the GRPO run (including the Qwen3.5 all-linear warning path).
    policy_args = Namespace(
        base_model=args.base_model, use_lora=True,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        lora_bias=args.lora_bias, lora_target_modules=args.lora_target_modules,
    )
    model, tokenizer = _load_policy(policy_args)

    screen_traces(args.traces, rows, tokenizer)
    trace_tokens = [r["n_reasoning_tokens"] for r in rows]
    print(f"[warmstart-sft] screen passed; mean reasoning "
          f"{sum(trace_tokens) / max(len(trace_tokens), 1):.0f} tok (recounted)")

    if args.gradient_checkpointing:
        model.config.use_cache = False
        model.enable_input_require_grads()  # PEFT + gradient checkpointing: inputs must require grad
        model.gradient_checkpointing_enable()

    from datasets import Dataset
    examples = [build_example(r, tokenizer, args.max_len) for r in rows]
    dataset = Dataset.from_list(examples)

    out_dir = Path(args.out)
    training_args = TrainingArguments(
        output_dir=str(out_dir / "_sft_ckpt"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        bf16=_bf16_available(),
        gradient_checkpointing=args.gradient_checkpointing,
        save_strategy="no",           # we merge and save the final model ourselves
        report_to=[],
        seed=args.seed,
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=dataset,
        data_collator=MaskedCollator(tokenizer.pad_token_id or tokenizer.eos_token_id),
    )
    trainer.train()

    # Merge LoRA into the base so GRPO starts from an ordinary model, never LoRA-on-LoRA.
    print("[warmstart-sft] merging LoRA into base weights")
    merged = model.merge_and_unload()
    if args.gradient_checkpointing:
        merged.config.use_cache = True
    out_dir.mkdir(parents=True, exist_ok=True)
    from scripts.m0.wrap_for_serving import save_servable_m0
    # Save M0 in the base's multimodal wrapper form so `vllm serve` accepts it. AutoModelForCausalLM extracted the text sub-model at load, so a plain merged.save_pretrained here writes a text-only Qwen3_5ForCausalLM that vLLM's Qwen3.5 handler rejects (Expected Qwen3_5Config, found Qwen3_5TextConfig). The re-wrap still loads text-only under AutoModelForCausalLM, so GRPO/SFT downstream are unchanged. See m0/wrap_for_serving.py.
    save_servable_m0(merged, args.base_model, out_dir, tokenizer, wrap_device=args.wrap_device)

    (out_dir / "warmstart_manifest.json").write_text(json.dumps({
        "schema": "m0_warmstart_model_v1",
        "base_model": args.base_model,
        "traces": args.traces,
        "n_traces": len(rows),
        "families": dict(families),
        "epochs": args.epochs,
        "lr": args.lr,
        "effective_batch": args.batch_size * args.grad_accum,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "target_modules": args.lora_target_modules},
        # The loss curve, so "did it actually converge?" is reviewable from the artifacts rather
        # than only from whatever scrolled past in the terminal. trainer.state.log_history is
        # already in memory; not writing it meant the one number that says the run worked was the
        # one number not kept.
        "log_history": trainer.state.log_history,
    }, indent=2) + "\n")

    print(f"[warmstart-sft] merged model saved to {out_dir}")
    print(f"[warmstart-sft] next: smoke GRPO from it -- M0_CONFIG=m0/configs/grpo_m0_smoke.yaml "
          f"then check the rollout reasoning-length median is <=256 before the full run "
          f"(train_m0.py --base_model {out_dir})")


if __name__ == "__main__":
    main()
