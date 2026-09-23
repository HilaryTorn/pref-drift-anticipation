"""Per-checkpoint format diagnostics for M0.

Sibling of rl_training/train_rl.py::DriftCadenceCallback, same on_save shape, different readout.
That callback logs verifiable coding reward; M0 has no verifier and its reward is a composite, so
the mean reward alone cannot say what changed. Commit rate stuck at zero and commit rate fine but
every response over budget produce similar means and call for opposite fixes.

So this logs the terms separately, and COMMIT RATE is the number that decides when to stop: it is
the quantity M0 exists to move, and it is directly comparable to the commit rate the batteries
report. Training past its plateau buys nothing and spends KL budget that shows up as preference
drift -- which is contamination of the study's own measurement.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from transformers import TrainerCallback

from scripts.m0.answer_format import terminal_answer_label
from scripts.m0.prompts import render_format_prompt
from scripts.m0.rewards import REASONING_BUDGET, REASONING_TARGET, score_completion


class FormatCadenceCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_records, n_samples, output_dir, budget=REASONING_BUDGET,
                 target=REASONING_TARGET, max_new_tokens=2048, batch_prompts=4, run_name=None):
        self.tokenizer = tokenizer
        self.eval_records = eval_records
        self.n_samples = n_samples
        self.budget = budget
        self.target = target
        self.max_new_tokens = max_new_tokens
        self.batch_prompts = max(1, batch_prompts)
        self.log_path = Path(output_dir) / "format_log.jsonl"
        self.sample_path = Path(output_dir) / "format_eval_samples.jsonl"
        self.run_name = run_name or Path(output_dir).name

    def _generate(self, model, records):
        """Generate n_samples completions for a batch of records; returns text per (record, sample).

        Batched because the eval is a real share of a run's wall clock -- generating one prompt at
        a time leaves an L4 mostly idle through a pass that buys only diagnostics.

        Padding must be LEFT for a decoder-only model: right padding puts pad tokens between the
        prompt and the first generated token, so the model continues from padding and the output is
        garbage. The tokenizer's own setting is restored afterwards because the trainer shares this
        object.
        """
        import torch

        prompts = [render_format_prompt(self.tokenizer, rec["prompt"]) for rec in records]
        previous_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            enc = self.tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **enc, max_new_tokens=self.max_new_tokens, do_sample=True, temperature=1.0,
                    num_return_sequences=self.n_samples,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
        finally:
            self.tokenizer.padding_side = previous_side

        width = enc["input_ids"].shape[1]
        pad_id = self.tokenizer.pad_token_id
        results = []
        for index, seq in enumerate(out):
            generated = seq[width:]
            # generate() pads every row in a batch out to the longest sequence, so len(generated)
            # is the batch maximum, not this row's length. Using it to detect truncation marks
            # EVERY row in a batch as truncated as soon as one row hits the cap -- which collapses
            # truncation_rate into "non-commit rate" and hides the distinction the log exists to
            # draw. Count this row's real tokens instead.
            n_real = int((generated != pad_id).sum().item()) if pad_id is not None else len(generated)
            # skip_special_tokens=False: </think> is a special token in the Qwen3.5 template, and
            # stripping it would make every completion look like the no-tags case, so the band term
            # would silently measure total length instead of reasoning length. Same reason the smoke
            # run checks for the tag explicitly.
            text = self.tokenizer.decode(generated, skip_special_tokens=False)
            for special in (self.tokenizer.eos_token, self.tokenizer.pad_token):
                if special:
                    text = text.replace(special, "")
            # generate() returns num_return_sequences consecutive rows per input, in input order.
            results.append((records[index // self.n_samples], text, n_real))
        return results

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Measure the UNTRAINED model first, so step 0 is on the same axis as every checkpoint.

        Without this the first datapoint in format_log.jsonl is already 25 steps of training deep,
        and there is nothing to read it against: a commit rate of 0.6 could be a large gain or a
        small one. The adoption gate asks "did commit rate improve, and did anything else move" --
        both halves need a before. Measuring it here rather than reusing published battery numbers
        keeps the comparison on the same prompts, the same K, and the same parser.

        Skipped on resume. Transformers restores trainer state BEFORE firing this hook, so on a
        resumed run the weights are already trained -- writing them out as "step 0" would append a
        second baseline measured on a different model. Both readers of this file (export_m0's
        summary table and commit_rate_plateaued) take the log in order, so that entry would
        silently corrupt the before/after the adoption gate rests on. The genuine baseline is
        already in the file from the first launch, and grpo_m0.yaml plans for resume by design.
        """
        if state.global_step > 0:
            print(f"[m0] resuming at step {state.global_step}; keeping the existing step-0 baseline")
            return
        self._evaluate(state, model or kwargs.get("model"), step=0)

    def on_save(self, args, state, control, model=None, **kwargs):
        self._evaluate(state, model or kwargs.get("model"), step=state.global_step)

    def _evaluate(self, state, model, step):
        from tqdm.auto import tqdm

        if model is None:
            return
        was_training = model.training
        model.eval()

        stats = {"n": 0, "commit": 0, "correct": 0, "verifiable": 0, "over_budget": 0,
                 "truncated": 0, "reward": 0.0, "reasoning_tokens": []}

        batches = [self.eval_records[i: i + self.batch_prompts]
                   for i in range(0, len(self.eval_records), self.batch_prompts)]
        for batch in tqdm(batches, desc=f"[m0] format eval @ step {step}", unit="batch"):
            for rec, text, n_generated in self._generate(model, batch):
                row = score_completion(
                    text, list(rec["labels"]), rec.get("answer"), bool(rec.get("verifiable")),
                    tokenizer=self.tokenizer, budget=self.budget, target=self.target,
                )
                stats["n"] += 1
                stats["reward"] += row["reward"]
                stats["reasoning_tokens"].append(row["n_reasoning_tokens"])
                stats["over_budget"] += int(row["over_budget"])
                # A generation that used every available token and never committed is a truncation,
                # which is the specific failure that drops battery samples as unparseable.
                stats["truncated"] += int(n_generated >= self.max_new_tokens
                                          and row["committed"] is None)
                if row["committed"] is not None:
                    stats["commit"] += 1
                    if rec.get("verifiable") and rec.get("answer") is not None:
                        stats["verifiable"] += 1
                        stats["correct"] += int(row["committed"] == rec["answer"])

        n = max(stats["n"], 1)
        lengths = sorted(stats["reasoning_tokens"]) or [0]
        entry = {
            "step": step,
            "samples": stats["n"],
            "commit_rate": stats["commit"] / n,
            "accuracy_given_commit": (stats["correct"] / stats["verifiable"]
                                      if stats["verifiable"] else None),
            "over_budget_rate": stats["over_budget"] / n,
            "truncation_rate": stats["truncated"] / n,
            "mean_reward": stats["reward"] / n,
            "mean_reasoning_tokens": sum(lengths) / len(lengths),
            "p95_reasoning_tokens": lengths[int(len(lengths) * 0.95)],
        }
        with self.log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"[m0] step {step}: commit_rate={entry['commit_rate']:.3f} "
              f"over_budget={entry['over_budget_rate']:.3f} "
              f"trunc={entry['truncation_rate']:.3f} "
              f"acc={entry['accuracy_given_commit']} "
              f"mean_reasoning={entry['mean_reasoning_tokens']:.0f}")

        if was_training:
            model.train()


def commit_rate_plateaued(log_path, patience: int = 2, min_delta: float | None = None) -> bool:
    """True once commit rate has stopped improving for `patience` consecutive checkpoints.

    ``min_delta`` defaults to the eval's own noise floor rather than a fixed number, because the
    two have to be compatible: commit rate is a proportion measured on ``samples`` draws, so its
    standard error is sqrt(0.25/n) -- about 6 points at the shipped 32 prompts x 2 samples. A fixed
    threshold of 0.01 against 6 points of noise measures nothing but sampling variation. The default
    here is 2 * SE, computed from the smallest ``samples`` count among the entries being compared,
    so it tracks the config instead of silently going stale when the eval is resized.

    ADVISORY, and deliberately weak. Even at 2 * SE this is one noisy series, and a genuine plateau
    can be masked by a single unlucky checkpoint. Treat a True as a prompt to look at the log, not
    as an instruction to stop -- the decision is a human reading format_log.jsonl alongside
    over_budget_rate and mean_reasoning_tokens. The reason to stop near the plateau at all is that
    every step past it spends KL budget for no format gain, and spent KL budget is what turns into
    the preference drift the adoption gate then has to reject.
    """
    path = Path(log_path)
    if not path.exists():
        return False
    entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(entries) < patience + 1:
        return False
    rates = [entry["commit_rate"] for entry in entries]

    if min_delta is None:
        n = min((entry.get("samples") or 0) for entry in entries[-(patience + 1):]) or 64
        min_delta = 2.0 * math.sqrt(0.25 / n)

    best_before = max(rates[: -patience])
    # The epsilon is not pedantry: commit rates are ratios of small integers, so a gain of exactly
    # min_delta is a common case, and in binary 0.86 - 0.85 evaluates to 0.010000000000000009.
    # Without it, a curve that has flattened at precisely the threshold reads as still climbing.
    return all(rate - best_before <= min_delta + 1e-9 for rate in rates[-patience:])
