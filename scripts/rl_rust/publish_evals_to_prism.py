"""Publish the graded Rust RL/SFT evals to the prism-drift adapter repos.

Until now the graded numbers lived only on the Mac. The *generations* were pushed to
`prism-drift/rust-rl-checkpoints` by the eval pods, and the adapters live on
`prism-drift/qwen35-{4b,9b}-m0-v4-rust-rl-adapters`, but nothing joined the two: the
`eval_all_*.json` files that carry `graded_list` -- the pass/fail per problem that every number in
the paper comes from -- were never uploaded anywhere. This puts them next to the weights they score.

Layout written into each size's adapter repo:

    evals/README.md                              instrument, arms, caveats
    evals/multilcb/<release>_<model>_cot/...json graded records, one file per arm per release
    evals/heldout/<arm>.summary.json             held-out Rust reward, per-record
    evals/heldout/SUMMARY.md                     the held-out table
    evals/paired/<release>.json                  McNemar comparisons (the analysis output)
    evals/paired/<release>.txt                   the same, as printed

Only Rust `n=1` files are published; the other languages belong to the LCB SFT sweep, not here.

    .venv/bin/python rl-rust/publish_evals_to_prism.py --dry_run
    .venv/bin/python rl-rust/publish_evals_to_prism.py --sizes 4b 9b
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "multi-lcb" / "output"
FN = "eval_all_codegeneration_rust_1_0.2_0.95_cot.json"
DEST = "prism-drift/qwen35-{size}-m0-v4-rust-rl-adapters"

# (release, output-dir model name) per size. The base's dir name differs by release: v6 uses the
# stored AWS draw, every other release measures its own base in-session under the `-rp` name.
MULTILCB = {
    "4b": [
        ("v6", "qwen35-4b-m0-v4"), ("v6", "qwen35-4b-grpo-rust-s124"),
        ("v6", "qwen35-4b-dpo-rust-s125"), ("v6", "qwen35-4b-ppo-rust-s124"),
        ("v6", "lcb-rust-ep3"),
        ("v345em", "qwen35-4b-m0-v4-rp"), ("v345em", "qwen35-4b-grpo-rust-s124"),
        ("v345em", "qwen35-4b-dpo-rust-s125"), ("v345em", "qwen35-4b-dpo-rust-s40"),
        ("v345em", "qwen35-4b-ppo-rust-s124"), ("v345em", "lcb-rust-ep3"),
    ],
    "9b": [
        ("v6", "qwen35-9b-m0-v4"), ("v6", "qwen35-9b-grpo-rust-s124"),
        ("v6", "qwen35-9b-dpo-rust-s125"), ("v6", "lcb-9b-rust-ep2"),
        ("v345em", "qwen35-9b-m0-v4-rp"), ("v345em", "qwen35-9b-grpo-rust-s124"),
        ("v345em", "qwen35-9b-dpo-rust-s125"), ("v345em", "lcb-9b-rust-ep2"),
    ],
}

# The held-out summaries are split across two trees: base/GRPO were graded into rl-rust/out, the
# DPO arm into results/rust-rl-eval. Both are the same instrument and belong together.
HELDOUT = {
    "4b": [
        REPO / "rl-rust/out/heldout-eval/4b-base.summary.json",
        REPO / "rl-rust/out/heldout-eval/4b-grpo-step124.summary.json",
        REPO / "results/rust-rl-eval/4b/heldout-eval/4b-dpo-step125.summary.json",
    ],
    "9b": [
        REPO / "rl-rust/out/heldout-eval/9b-base.summary.json",
        REPO / "rl-rust/out/heldout-eval/9b-grpo-step124.summary.json",
        REPO / "results/rust-rl-eval/9b/heldout-eval/9b-dpo-step125.summary.json",
    ],
}

README = """---
tags:
  - evaluation
  - code-generation
  - rust
---

# Graded evals for the Qwen3.5 {size_u} Rust arms

The pass/fail record behind every capability number reported for these adapters. Generations were
produced on an L40S under vLLM 0.28.0 and graded locally with `rustc` (Rust is the one language
upstream exempts from `limit_memory`, so a Mac is a legal grading box). Nothing here needed a GPU.

## What is in each file

`multilcb/<release>_<model>_cot/eval_all_codegeneration_rust_1_0.2_0.95_cot.json` -- one record per
problem: the prompt, the full CoT response (`output_list`), the extracted program (`code_list`),
the verdict (`graded_list`), and `metadata` carrying the compiler or runtime error when it failed.
Settings are in the filename: rust, n=1, temperature 0.2, top_p 0.95, CoT.

`heldout/<arm>.summary.json` -- the held-out Rust reward instrument: 500 clean prompts,
temperature 1.0, 2048-token cap, full test suite, with per-record rewards and termination flags.

`paired/<release>.json` -- the comparison actually reported: paired by `question_id`, exact
two-sided McNemar on discordant pairs, with a truncation-conditioned variant restricted to problems
where base and arm both emitted code.

## The two instruments are separate and do not pool

**v6** (n=175) is LiveCodeBench release 6, disjoint from the v1-v5 pool the SFT arms trained on, so
every arm is uncontaminated. It is hard: roughly half of all responses never reach code at all.

**v345em** (n=222) is LCB v3/v4/v5 filtered to easy + medium (97 easy, 125 medium), built for power.
Its base is re-measured in the same session, so nothing here pools with the stored v6 numbers. The
RL arms are clean on it (their cohort is codeforces; these shards are atcoder and leetcode), but the
SFT arms trained on 175 of the 222 -- those cells are train-set **recall**, not capability, and must
never sit in a column with the RL arms.

## Two cautions the numbers do not carry on their own

**Truncation.** A large share of failures are responses that hit the token cap before emitting
code, not refusals. Read `pass@1` beside the no-code rate, and treat the conditioned delta as
unreadable when base and arm truncate at very different rates -- it is then selecting on a
post-treatment variable. The `paired/` files flag exactly that case.

**Build failures are a model error, not a toolchain one.** Around half of base responses that do
emit Rust fail to compile, most often a missing trait import. Every arm was graded on the same
`rustc`, so differences between arms are real; the absolute level is not an environment fault.
"""


def token() -> str:
    p = REPO / "api_keys/api_key_hf_write.txt"
    m = re.search(r"hf_[A-Za-z0-9]{20,}", p.read_text())
    if not m:
        raise SystemExit(f"no write token in {p} (.env HF_TOKEN is read-only and cannot push)")
    return m.group(0)


def plan(size: str) -> list[tuple[pathlib.Path, str]]:
    """(local path, path in repo). Missing sources are reported, never silently skipped."""
    items: list[tuple[pathlib.Path, str]] = []
    for release, model in MULTILCB[size]:
        src = OUT / f"{release}_{model}_cot" / FN
        items.append((src, f"evals/multilcb/{release}_{model}_cot/{FN}"))
    for src in HELDOUT[size]:
        items.append((src, f"evals/heldout/{src.name}"))
    items.append((REPO / "rl-rust/out/heldout-eval/SUMMARY.md", "evals/heldout/SUMMARY.md"))
    for release in ("v6", "v345em"):
        for ext in ("json", "txt"):
            items.append((PAIRED / f"paired_{release}.{ext}", f"evals/paired/{release}.{ext}"))
    return items


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", nargs="+", default=["4b", "9b"])
    p.add_argument("--paired_dir", required=True, help="dir holding paired_<release>.{json,txt}")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    global PAIRED
    PAIRED = pathlib.Path(args.paired_dir)

    from huggingface_hub import HfApi
    api = HfApi(token=None if args.dry_run else token())

    for size in args.sizes:
        repo_id = DEST.format(size=size)
        items = plan(size)
        missing = [str(s) for s, _ in items if not s.is_file()]
        if missing:
            print(f"[{size}] MISSING, refusing to publish a partial set:")
            for m in missing:
                print("   ", m)
            return 1
        total = sum(s.stat().st_size for s, _ in items)
        print(f"\n[{size}] -> {repo_id}   {len(items)} files, {total/1e6:.1f} MB")
        for s, d in items:
            print(f"    {d:<72} {s.stat().st_size/1e6:>7.2f} MB")
        if args.dry_run:
            continue

        readme = REPO / "rl-rust" / f".readme.{size}.md"
        readme.write_text(README.format(size_u=size.upper()))
        try:
            ops = [(readme, "evals/README.md")] + items
            for s, d in ops:
                api.upload_file(path_or_fileobj=str(s), path_in_repo=d,
                                repo_id=repo_id, repo_type="model",
                                commit_message=f"evals: {d}")
                print(f"    uploaded {d}")
        finally:
            readme.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
