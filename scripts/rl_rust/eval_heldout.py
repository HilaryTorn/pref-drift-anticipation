"""Score a Rust RL checkpoint on the clean held-out set, offline.

Why this exists: the held-out reward only ever existed as ``DriftCadenceCallback`` inside ``rl_training/train_rl.py`` -- a trainer callback, so it could only run during training, one prompt at a time through HF ``generate``. Both GRPO runs therefore carry a held-out curve that is exactly 0.0 at every logged step, and there is no way to re-measure a finished checkpoint without retraining it.

Two separate things were wrong with that curve and only one of them is fixed by this file:

  the prompts      ``heldout-500.jsonl`` carried the raw Nemotron wrapper, which
                   instructs "Please use python programming language only". The
                   Rust template wrapped a Python-demanding instruction, models
                   emitted Python, rustc rejected it, everything scored 0.0.
                   ``build_clean_heldout.py`` fixed that; this file defaults to
                   its output and refuses the uncleaned file.
  the reach        even with clean prompts the callback could not be pointed at a
                   checkpoint after the fact. That is what this file adds.

It also widens the measurement: the callback scored ``eval_reward_prompts: 32`` of the 500 because eval was costing more than training. Nothing is being trained here, so the default is all 500.

Generation and grading are separate phases with opposite requirements -- generation needs a GPU endpoint and no toolchain, grading needs rustc and no GPU -- so they split the way ``score_multilcb.py`` splits, for the same reason. Generate against the pod, terminate the pod, grade on the Mac. Rust is the one language upstream exempts from ``limit_memory``, so grading a Rust-only arm genuinely runs on macOS.

    # 0. preflight: toolchain, prompts, and PROOF THE ADAPTER IS APPLIED (no grading)
    .venv/bin/python rl-rust/eval_heldout.py --preflight \
        --endpoint http://localhost:8000/v1 \
        --model qwen35-9b-m0-v4 --adapter grpo-final

    # 1. generate, on the laptop through an SSH tunnel to the pod
    .venv/bin/python rl-rust/eval_heldout.py --generate_only \
        --endpoint http://localhost:8000/v1 \
        --model qwen35-9b-m0-v4 --adapter grpo-final \
        --tokenizer prism-drift/qwen35-9b-m0-v4 \
        --gen_out rl-rust/out/heldout-eval/9b-grpo-step124.gen.jsonl --resume

    # 2. grade, after the pod is dead
    .venv/bin/python rl-rust/eval_heldout.py --grade_only \
        --gen_out rl-rust/out/heldout-eval/9b-grpo-step124.gen.jsonl \
        --summary_out rl-rust/out/heldout-eval/9b-grpo-step124.summary.json

    # 3. pair the arm against its base -- same problems, so pair on them
    .venv/bin/python rl-rust/eval_heldout.py --compare \
        rl-rust/out/heldout-eval/9b-base.summary.json \
        rl-rust/out/heldout-eval/9b-grpo-step124.summary.json

Sampling matches the callback exactly -- temperature 1.0, top_p 1.0, ``max_tokens`` 2048 (grpo_rust.yaml's ``max_completion_length``), ``rust_io`` prompt with thinking off -- because a held-out number taken at different settings is not the number the run was optimizing. Grading matches it too: the FULL test suite, not the training budget of 12, at ``REWARD_TIMEOUT`` 5.0.

Read ``eval_reward_given_termination`` next to ``termination_rate``, never ``eval_reward`` alone. Rust terminates inside 2048 tokens only 30-60% of the time at these sizes, so ``eval_reward`` is the product of termination and quality and training moves termination far more easily than it moves quality. The same trap voided the 4B DPO headline in the Aug 30 generation and the 27B Rust arm.

SECURITY: grading compiles and runs model-generated code. Same caution as rl_training/rewards.py -- do not grade on a box you care about.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "rl-rust"))

DEFAULT_HELDOUT = REPO / "rl-rust/out/heldout-500-clean.jsonl"

# The wrapper build_clean_heldout.py strips. Its presence means the uncleaned file was passed, which is the bug that produced two runs of flat-zero held-out reward -- so it is a hard refusal, not a warning.
NEMOTRON_WRAPPER = "You are a helpful and harmless assistant."


def load_rows(path: Path, limit: int | None = None) -> list[dict]:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    contaminated = [r["record_id"] for r in rows if NEMOTRON_WRAPPER in r["prompt"]]
    if contaminated:
        raise SystemExit(
            f"REFUSING: {len(contaminated)} of {len(rows)} prompts in {path.name} still carry the "
            f"Nemotron wrapper ('{NEMOTRON_WRAPPER}'), which instructs the model to write Python. "
            f"This is the exact bug that made both GRPO held-out curves flat zero. "
            f"Run rl-rust/build_clean_heldout.py and point --heldout at heldout-500-clean.jsonl."
        )
    return rows[:limit] if limit else rows


def render(tokenizer, problem: str) -> str:
    from scripts.rl_rust.rl_rust.prompts import render_coding_prompt

    return render_coding_prompt(tokenizer, problem, format_name="rust_io")


def post_completions(args, prompt: str, n: int, temperature: float, model: str) -> tuple[list[str], list[str]]:
    """One /completions call. Returns (texts, finish_reasons)."""
    import requests

    headers = {"Content-Type": "application/json"}
    if args.api_key_file:
        headers["Authorization"] = f"Bearer {Path(args.api_key_file).read_text().strip()}"
    body = {
        "model": model,
        "prompt": prompt,
        "n": n,
        "max_tokens": args.max_new_tokens,
        "temperature": temperature,
        "top_p": 1.0,
    }
    url = args.endpoint.rstrip("/") + "/completions"
    last = None
    for attempt in range(args.retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=args.endpoint_timeout)
            resp.raise_for_status()
            choices = resp.json().get("choices", [])
            return (
                [c.get("text", "") for c in choices],
                [c.get("finish_reason") for c in choices],
            )
        except Exception as err:  # noqa: BLE001 - a dropped tunnel must not lose the whole run
            last = err
            if attempt < args.retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"endpoint failed after {args.retries + 1} attempts: {last}")


def served_name(args) -> str:
    """The model id to ask the endpoint for: the LoRA module name when one is given, else the base."""
    return args.adapter or args.model


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------

def preflight(args) -> int:
    """Prove the four things that have silently voided a run in this project before."""
    ok = True

    from scripts.rl_rust.rl_rust.rewards import rust_toolchain_available

    have_rust = rust_toolchain_available()
    print(f"[heldout] rustc available here: {have_rust} "
          f"({'grading can run on this box' if have_rust else 'generation only; grade elsewhere'})")

    rows = load_rows(Path(args.heldout), limit=None)
    print(f"[heldout] {len(rows)} clean prompts, wrapper check passed")

    if not args.endpoint:
        print("[heldout] no --endpoint, skipping the adapter check (THE important one)")
        return 0 if ok else 1

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model, trust_remote_code=True)
    probes = [render(tok, r["prompt"]) for r in rows[: args.preflight_n]]

    # The guard that matters. vLLM has silently served the BASE model while reporting the adapter's name -- a null that looks exactly like "the training did nothing". Greedy decoding makes base and adapter byte-comparable; if they match on every probe, the adapter is not applied.
    if not args.adapter:
        print("[heldout] no --adapter given; nothing to diff (base run)")
        return 0 if ok else 1

    print(f"[heldout] adapter check: greedy base ({args.model}) vs adapter ({args.adapter})")
    n_same = 0
    for i, p in enumerate(probes):
        base_txt, base_fin = post_completions(args, p, 1, 0.0, args.model)
        ad_txt, ad_fin = post_completions(args, p, 1, 0.0, args.adapter)
        b, a = (base_txt[0] if base_txt else ""), (ad_txt[0] if ad_txt else "")
        if not a.strip():
            print(f"  probe {i}: ADAPTER RETURNED EMPTY TEXT (finish={ad_fin}) -- serving is wrong")
            ok = False
        same = b == a
        n_same += same
        print(f"  probe {i}: identical={same}  base={len(b)}c/{base_fin[0]}  adapter={len(a)}c/{ad_fin[0]}")
    if n_same == len(probes):
        print(
            f"[heldout] FAIL: adapter output is byte-identical to base on all {len(probes)} probes at "
            f"temperature 0. vLLM is serving the base weights under the adapter's name. Check "
            f"--lora-modules, --max-lora-rank (this adapter is r=16), and that the adapter path "
            f"actually contains adapter_model.safetensors. Do NOT spend GPU hours on this run."
        )
        ok = False
    else:
        print(f"[heldout] adapter IS applied ({len(probes) - n_same}/{len(probes)} probes differ)")

    print(f"[heldout] preflight {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

def generate(args) -> int:
    from transformers import AutoTokenizer

    rows = load_rows(Path(args.heldout), args.limit)
    gen_path = Path(args.gen_out)
    gen_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if args.resume and gen_path.exists():
        for line in gen_path.open():
            if line.strip():
                done.add(json.loads(line)["record_id"])
        print(f"[heldout] resuming: {len(done)} records already generated")
    todo = [r for r in rows if r["record_id"] not in done]
    if not todo:
        print("[heldout] nothing to generate")
        return 0

    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model, trust_remote_code=True)
    model = served_name(args)
    print(f"[heldout] generating {len(todo)} prompts x n={args.n} from '{model}' at {args.endpoint}")
    print(f"[heldout] temperature={args.temperature} top_p=1.0 max_tokens={args.max_new_tokens} concurrency={args.concurrency}")

    def one(row: dict) -> dict:
        prompt = render(tok, row["prompt"])
        texts, finishes = post_completions(args, prompt, args.n, args.temperature, model)
        return {
            "record_id": row["record_id"],
            "completions": texts,
            "finish_reasons": finishes,
            "n_prompt_chars": len(prompt),
        }

    # Append as each result lands so a dropped tunnel costs only the requests in flight.
    written = 0
    t0 = time.time()
    with gen_path.open("a") as fh, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for rec in pool.map(one, todo):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            written += 1
            if written % 25 == 0:
                rate = written / (time.time() - t0)
                print(f"  {written}/{len(todo)}  {rate:.2f} prompt/s  eta {(len(todo) - written) / max(rate, 1e-9) / 60:.0f} min", flush=True)

    manifest = {
        "schema": "rust_heldout_generations_v1",
        "heldout": str(Path(args.heldout).relative_to(REPO)),
        "model": model,
        "base_model": args.model,
        "adapter": args.adapter,
        "endpoint": args.endpoint,
        "n": args.n,
        "temperature": args.temperature,
        "top_p": 1.0,
        "max_new_tokens": args.max_new_tokens,
        "prompt_format": "rust_io",
        "n_records": len(rows),
    }
    Path(str(gen_path) + ".manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[heldout] wrote {written} records to {gen_path}")
    return 0


# ---------------------------------------------------------------------------
# grade
# ---------------------------------------------------------------------------

def grade(args) -> int:
    from scripts.rl_rust.rl_rust.rewards import REWARD_TIMEOUT, rust_toolchain_available, score_completions

    if not rust_toolchain_available():
        raise SystemExit(
            "REFUSING: no rustc on this box. Every completion would score 0.0 and the summary "
            "would look exactly like a failed run. Grade where the toolchain is."
        )

    manifest_path = Path(str(args.gen_out) + ".manifest.json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None

    rows = {r["record_id"]: r for r in load_rows(Path(args.heldout), None)}
    gens = [json.loads(l) for l in Path(args.gen_out).open() if l.strip()]
    missing = [g["record_id"] for g in gens if g["record_id"] not in rows]
    if missing:
        raise SystemExit(f"{len(missing)} generated record_ids are not in {args.heldout}: {missing[:3]}")
    print(f"[heldout] grading {len(gens)} records, full test suite, timeout={REWARD_TIMEOUT}s")

    per_record = []
    t0 = time.time()
    for i, g in enumerate(gens, 1):
        verifier = rows[g["record_id"]]["verifier"]
        # max_tests=None on purpose: the training reward subsamples to 12 for speed, the eval does not. The callback made the same choice.
        scored = score_completions(
            g["completions"], verifier, timeout=REWARD_TIMEOUT, max_tests=None, language="rust"
        )
        rewards = [s[0] for s in scored]
        terminated = [f == "stop" for f in g["finish_reasons"]]
        per_record.append({
            "record_id": g["record_id"],
            "rewards": rewards,
            "terminated": terminated,
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "any_solved": any(r >= 1.0 for r in rewards),
        })
        if i % 25 == 0:
            print(f"  {i}/{len(gens)}  {(time.time() - t0) / i:.2f}s/record", flush=True)

    flat_rewards = [r for rec in per_record for r in rec["rewards"]]
    flat_term = [t for rec in per_record for t in rec["terminated"]]
    n_gen = len(flat_rewards)
    finished = [r for r, t in zip(flat_rewards, flat_term) if t]

    summary = {
        "schema": "rust_heldout_summary_v1",
        "manifest": manifest,
        "n_records": len(per_record),
        "n_generations": n_gen,
        # eval_reward keeps the callback's meaning so old curves stay comparable: truncated generations are graded, score ~0, and average in. It is the PRODUCT of termination and quality.
        "eval_reward": sum(flat_rewards) / max(n_gen, 1),
        "eval_reward_given_termination": sum(finished) / len(finished) if finished else 0.0,
        "termination_rate": sum(flat_term) / max(n_gen, 1),
        "n_truncated": n_gen - sum(flat_term),
        "solve_rate": sum(1 for r in flat_rewards if r >= 1.0) / max(n_gen, 1),
        # Prefer what the generation run actually used over the CLI default: grading is a separate invocation and its --max_new_tokens is not what produced these texts.
        "max_new_tokens": manifest["max_new_tokens"] if manifest else args.max_new_tokens,
        "per_record": per_record,
    }
    out = Path(args.summary_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1))

    print(f"[heldout] eval_reward            = {summary['eval_reward']:.4f}")
    print(f"[heldout] reward | terminated    = {summary['eval_reward_given_termination']:.4f}")
    print(f"[heldout] termination_rate       = {summary['termination_rate']:.3f} ({summary['n_truncated']} truncated)")
    print(f"[heldout] solve_rate (all tests) = {summary['solve_rate']:.4f}")
    if summary["termination_rate"] < 0.95:
        print(
            f"    !! {1 - summary['termination_rate']:.0%} of generations hit the "
            f"{summary['max_new_tokens']}-token cap and scored ~0. Read eval_reward as termination x "
            f"quality, and compare arms on reward|terminated AND termination_rate, never on "
            f"eval_reward alone."
        )
    print(f"[heldout] wrote {out}")
    return 0


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

def compare(base_path: str, arm_path: str, n_boot: int = 10000, seed: int = 0) -> int:
    """Paired base-vs-arm on the shared problems. Pairing is the whole point: both models answered the same 500."""
    import random

    base = json.loads(Path(base_path).read_text())
    arm = json.loads(Path(arm_path).read_text())
    b = {r["record_id"]: r for r in base["per_record"]}
    a = {r["record_id"]: r for r in arm["per_record"]}
    shared = sorted(set(b) & set(a))
    if not shared:
        raise SystemExit("no shared record_ids -- these summaries are not on the same held-out set")
    print(f"[heldout] paired on {len(shared)} shared records "
          f"(base has {len(b)}, arm has {len(a)})")

    diffs = [a[i]["mean_reward"] - b[i]["mean_reward"] for i in shared]
    point = statistics.fmean(diffs)

    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        boots.append(statistics.fmean([diffs[rng.randrange(len(diffs))] for _ in diffs]))
    boots.sort()
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]

    # Discordant pairs on solved/not-solved: the McNemar view, which does not care about partial credit.
    b_only = sum(1 for i in shared if b[i]["any_solved"] and not a[i]["any_solved"])
    a_only = sum(1 for i in shared if a[i]["any_solved"] and not b[i]["any_solved"])

    print(f"[heldout] base eval_reward = {base['eval_reward']:.4f}   term = {base['termination_rate']:.3f}")
    print(f"[heldout] arm  eval_reward = {arm['eval_reward']:.4f}   term = {arm['termination_rate']:.3f}")
    print(f"[heldout] base reward|term = {base['eval_reward_given_termination']:.4f}")
    print(f"[heldout] arm  reward|term = {arm['eval_reward_given_termination']:.4f}")
    print(f"[heldout] paired delta     = {point:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"[heldout] discordant solves: arm-only {a_only}, base-only {b_only}")
    if lo <= 0.0 <= hi:
        print("[heldout] CI includes zero -- this is a null on the held-out set.")
    if abs(arm["termination_rate"] - base["termination_rate"]) > 0.05:
        print(
            "    !! termination rates differ by >5pp between the two arms. A change in eval_reward "
            "is then partly a change in how often the model stops, not in how well it solves. "
            "Lead with reward|terminated and report the termination shift next to it."
        )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--heldout", default=str(DEFAULT_HELDOUT))
    p.add_argument("--endpoint", default=None, help="OpenAI-compatible vLLM base URL, e.g. http://localhost:8000/v1")
    p.add_argument("--model", default=None, help="Served base model id (what --served-model-name says)")
    p.add_argument("--adapter", default=None, help="LoRA module name registered with --lora-modules. Omit for a base run.")
    p.add_argument("--tokenizer", default=None, help="HF id for the tokenizer; defaults to --model")
    p.add_argument("--api_key_file", default=None)
    p.add_argument("--gen_out", default=None, help="Generations JSONL (written by --generate_only, read by --grade_only)")
    p.add_argument("--summary_out", default=None)
    p.add_argument("--n", type=int, default=2, help="Samples per prompt. 2 matches grpo_rust.yaml eval_reward_samples.")
    p.add_argument("--temperature", type=float, default=1.0, help="1.0 matches the training rollout distribution. Do not lower it to reduce noise -- that changes what is being measured.")
    p.add_argument("--max_new_tokens", type=int, default=2048, help="2048 matches max_completion_length. Raising it makes the number incomparable to the training reward.")
    p.add_argument("--limit", type=int, default=None, help="First N held-out records (smoke only)")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--endpoint_timeout", type=float, default=600.0)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--preflight_n", type=int, default=3)
    p.add_argument("--generate_only", action="store_true")
    p.add_argument("--grade_only", action="store_true")
    p.add_argument("--compare", nargs=2, metavar=("BASE_SUMMARY", "ARM_SUMMARY"), default=None)
    args = p.parse_args()

    if args.compare:
        return compare(*args.compare)
    if args.preflight:
        return preflight(args)
    if args.generate_only:
        for req in ("endpoint", "model", "gen_out"):
            if not getattr(args, req):
                raise SystemExit(f"--generate_only needs --{req}")
        return generate(args)
    if args.grade_only:
        for req in ("gen_out", "summary_out"):
            if not getattr(args, req):
                raise SystemExit(f"--grade_only needs --{req}")
        return grade(args)
    raise SystemExit("pick one: --preflight, --generate_only, --grade_only, or --compare")


if __name__ == "__main__":
    raise SystemExit(main())
