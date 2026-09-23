"""A/B the thinking mode for Rust GRPO rollouts. Generation only, no training.

The question this answers: does `enable_thinking=True` produce more LEARNING
SIGNAL than the inherited `enable_thinking=False`?

"More signal" is not "higher reward". GRPO learns from disagreement WITHIN a
group of K samples of one prompt. A group where all K score 0.0 has zero
variance, zero advantage, and contributes no gradient at all. So the decisive
metric is the number of groups with group_std > 0 -- reported here as LIVE
GROUPS. Mean reward is secondary.

Generation only: no backward pass, no gradient accumulation. That is why this
costs minutes instead of the ~27 min/step the full trainer needed.

Both arms run on the same pod, same prompts, same seed, same K, same sampling
temperature. Only the chat-template thinking flag and the token budget differ --
thinking-on is given a larger budget because it must fit reasoning AND code.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "rl-rust"))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from scripts.rl_rust.rl_rust.prompts import format_coding_prompt  # noqa: E402
from scripts.rl_rust.rl_rust.rewards import get_language, score_completions  # noqa: E402

RUST = get_language("rust")

#: Filled in from the model's generation config at load time. Qwen3.5 binds two
#: (248044, 248046) and the GRPO trainer treats both as terminal, so a rollout
#: ending on either one terminated legitimately.
EOS_IDS: list[int] = []


def resolve_eos_ids(model, tokenizer) -> list[int]:
    """Use the trainer's own resolver, not generation_config alone.

    generation_config carries the BASE model's stale value (248044, which is
    actually the PAD token); the trainer realigns it against the tokenizer at
    load time and ends up with (248044, 248046). Reading generation_config
    directly gave one wrong id and would have mislabelled every legitimately
    terminated rollout as truncated.
    """
    from rl_training.completion_semantics import model_eos_token_ids

    return [int(i) for i in model_eos_token_ids(model, tokenizer)]


def render(tokenizer, problem: str, thinking: bool) -> str:
    user_text = format_coding_prompt(problem, format_name="rust_io")
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )


def run_arm(model, tokenizer, rows, *, thinking: bool, k: int, max_new: int, seed: int):
    torch.manual_seed(seed)
    t0 = time.time()
    per_group = {}
    raw_scores = {}
    think_tokens = []

    for row in rows:
        prompt = render(tokenizer, row["prompt"], thinking)
        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                num_return_sequences=k,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1]:]
        texts, truncated = [], []
        for seq in gen:
            # Truncation = the sequence never emitted EOS. It must be tested by
            # searching the WHOLE sequence, not the last token: generate() pads
            # every sequence in a num_return_sequences group out to the longest
            # one, so an early-finishing rollout ends in PAD and a last-token
            # check misreads it as truncated. That bug over-masked 40 of 64
            # rollouts and inverted the result.
            has_eos = any(bool((seq == e).any().item()) for e in EOS_IDS)
            truncated.append(not has_eos)
            texts.append(tokenizer.decode(seq, skip_special_tokens=True))

        if thinking:
            for t in texts:
                if "</think>" in t:
                    think_tokens.append(len(tokenizer(t.split("</think>")[0])["input_ids"]))

        scored = score_completions(
            texts, row["verifier"], timeout=5.0, max_tests=12, language="rust"
        )
        # Truncated rollouts are masked (None), matching coding_reward's contract.
        rewards = [None if tr else s[0] for s, tr in zip(scored, truncated)]
        per_group[row["record_id"]] = rewards
        # Keep the RAW score and the truncation flag separately so the masking
        # policy can be re-evaluated offline without regenerating anything.
        raw_scores[row["record_id"]] = [
            {"raw_reward": s[0], "truncated": tr, "n_chars": len(t)}
            for s, tr, t in zip(scored, truncated, texts)
        ]

    elapsed = time.time() - t0

    live, all_r, trunc_n, total_n = 0, [], 0, 0
    for rid, rewards in per_group.items():
        vals = [v for v in rewards if v is not None]
        trunc_n += sum(1 for v in rewards if v is None)
        total_n += len(rewards)
        all_r += vals
        if len(vals) > 1 and statistics.pstdev(vals) > 1e-9:
            live += 1

    return {
        "thinking": thinking,
        "max_new_tokens": max_new,
        "groups": len(per_group),
        "live_groups": live,
        "truncation_rate": trunc_n / total_n if total_n else 0.0,
        "scored": len(all_r),
        "mean_reward": statistics.mean(all_r) if all_r else 0.0,
        "nonzero": sum(1 for v in all_r if v > 0),
        "max_reward": max(all_r) if all_r else 0.0,
        "elapsed_s": round(elapsed, 1),
        "mean_think_tokens": round(statistics.mean(think_tokens), 1) if think_tokens else None,
        "per_group": {k2: v for k2, v in per_group.items()},
        "raw_scores": raw_scores,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n_prompts", type=int, default=8)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_new_off", type=int, default=3072)
    ap.add_argument("--max_new_on", type=int, default=4096)
    ap.add_argument("--out", default=str(REPO / "rl-rust/out/ab_thinking.json"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.dataset)][: args.n_prompts]
    print(f"{len(rows)} prompts x K={args.k}\n", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    global EOS_IDS
    EOS_IDS = resolve_eos_ids(model, tokenizer)
    print(f"EOS ids treated as terminal: {EOS_IDS}\n", flush=True)

    results = {}
    for label, thinking, max_new in (
        ("thinking_off", False, args.max_new_off),
        ("thinking_on", True, args.max_new_on),
    ):
        print(f"=== {label} (max_new={max_new}) ===", flush=True)
        r = run_arm(model, tokenizer, rows, thinking=thinking,
                    k=args.k, max_new=max_new, seed=args.seed)
        results[label] = r
        print(f"  LIVE GROUPS   {r['live_groups']}/{r['groups']}", flush=True)
        print(f"  mean reward   {r['mean_reward']:.4f}   nonzero {r['nonzero']}/{r['scored']}", flush=True)
        print(f"  truncation    {r['truncation_rate']:.1%}", flush=True)
        print(f"  wall          {r['elapsed_s']}s", flush=True)
        if r["mean_think_tokens"]:
            print(f"  think tokens  {r['mean_think_tokens']:.0f} mean", flush=True)
        print(flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1))

    off, on = results["thinking_off"], results["thinking_on"]
    print("=" * 58)
    print(f"{'metric':22s} {'off':>12s} {'on':>12s}")
    print(f"{'live groups':22s} {off['live_groups']:>12d} {on['live_groups']:>12d}")
    print(f"{'mean reward':22s} {off['mean_reward']:>12.4f} {on['mean_reward']:>12.4f}")
    print(f"{'truncation':22s} {off['truncation_rate']:>11.1%} {on['truncation_rate']:>11.1%}")
    print(f"{'wall seconds':22s} {off['elapsed_s']:>12.0f} {on['elapsed_s']:>12.0f}")
    print()
    if on["live_groups"] > off["live_groups"]:
        print(">>> thinking ON gives more learning signal")
    elif on["live_groups"] < off["live_groups"]:
        print(">>> thinking OFF gives more learning signal")
    else:
        print(">>> tie on live groups; decide on mean reward and cost")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
