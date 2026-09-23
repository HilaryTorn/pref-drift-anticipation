"""Build DPO pairs for the Rust arm by rebinding the scorer, then delegating.

``rl_training/build_dpo_pairs.py`` is not modified and not copied -- same pattern
as ``train_rust_grpo.py``. Its ``main()`` resolves ``score_candidates`` and
``render_coding_prompt`` at call time (function-local from-imports), so replacing
the attributes on the ``rl_training`` modules before delegating redirects both
sampling prompts and grading to Rust. Two things get rebound, and both matter:

  rewards.score_candidates      grades candidates through the Rust executor.
                                Miss this and every candidate executes as
                                Python, scores 0.0, and the sweep quietly yields
                                zero pairs -- the exact silent-null failure this
                                launcher exists to close.
  prompts.render_coding_prompt  samples candidates from the Rust prompt
                                (rust_io, thinking off), token-identical to
                                GRPO rollouts and the eval callback.

``build_preference_pair_from_candidates`` and the shared reward budget
(TRAIN_REWARD_MAX_TESTS=12, REWARD_TIMEOUT=5.0) are language-agnostic and stay
upstream, so pair selection -- including the truncation-aware exclusion, which
Rust needs at its 15-30% truncation rate -- is byte-identical to the Python arm.

Pairs generate from M0 (the arm's base model), NOT from GRPO checkpoints: DPO is
the offline arm and its data must come from the same starting policy the online
arms start from.

    # preflight (no GPU, no model): toolchain + rebinding + cohort audit
    .venv/bin/python rl-rust/build_rust_dpo_pairs.py --preflight \
        --prompts rl-rust/out/cohort-9b-clean-n992/train.jsonl

    # full sweep against a vLLM endpoint serving M0 (grading runs HERE -- needs rustc)
    .venv/bin/python rl-rust/build_rust_dpo_pairs.py \
        --model prism-drift/qwen35-9b-m0-v4 \
        --endpoint http://<host>:8000/v1 \
        --prompts rl-rust/out/cohort-9b-clean-n992/train.jsonl \
        --out rl-rust/out/dpo-pairs-9b/pairs.jsonl \
        --scores_out rl-rust/out/dpo-pairs-9b/prompt_scores.jsonl \
        --manifest_out rl-rust/out/dpo-pairs-9b/manifest.json \
        --K 8 --margin 0.5 --resume

Expected yield at K=8/margin 0.5, measured off the GRPO rollouts: ~450 pairs at
9B, ~294 at 4B from 992 prompts.

``--max_new_tokens`` defaults to 2048 here (not the upstream 1024), matching
grpo_rust.yaml's max_completion_length: correct Rust answers cluster near ~300
tokens but in-regime completions run right up to ~2048 under the terse prompt.
NOTE the upstream >10% truncation warning will fire and tell you to raise the
cap -- for Rust, don't. The completions that miss 2048 need 11k-16k tokens
(doomed loops), so a larger cap buys no pairs, only GPU time; the truncation-
aware selection already keeps those out of the data.

SECURITY: runs model-generated code via the verifier -- see rl_training/rewards.py.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "rl-rust"))

from rl_rust import prompts as rust_prompts  # noqa: E402
from rl_rust import rewards as rust_rewards  # noqa: E402
from scripts.rl_rust.rl_rust.prompts import FORMAT_LANGUAGE  # noqa: E402

DEFAULT_MAX_NEW_TOKENS = 2048


def install_scoring(language: str, prompt_format: str) -> None:
    """Point build_dpo_pairs' scorer and prompt renderer at the Rust stack."""
    from rl_training import prompts as up_prompts
    from rl_training import rewards as up_rewards

    up_rewards.score_candidates = functools.partial(
        rust_rewards.score_candidates, language=language
    )
    up_prompts.render_coding_prompt = functools.partial(
        rust_prompts.render_coding_prompt, format_name=prompt_format
    )


def preflight(language: str, prompt_format: str, prompts_path: str | None) -> int:
    """Cheap checks that would otherwise fail hours into a booked endpoint sweep."""
    ok = True

    def line(name: str, good: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and good
        print(f"  [{'ok' if good else 'FAIL'}] {name}{'  ' + detail if detail else ''}")

    print("preflight")
    line("rustc on PATH", rust_rewards.rust_toolchain_available())
    if not rust_rewards.rust_toolchain_available():
        print("\n  Grading runs on THIS machine. Without rustc every candidate scores 0.0")
        print("  and the sweep completes normally with zero pairs.")
        return 1

    line("prompt asks for Rust",
         "Rust" in rust_prompts.format_coding_prompt("A+B", format_name=prompt_format))

    probe = """```rust
use std::io::{self, Read};
fn main() {
    let mut s = String::new();
    io::stdin().read_to_string(&mut s).unwrap();
    let n: i64 = s.trim().parse().unwrap();
    println!("{}", n * 2);
}
```"""
    verifier = {"type": "io_tests", "test_inputs": ["1", "3"], "test_outputs": ["2", "6"]}

    # Prove the REBINDING, not just the module: grade through the exact
    # attribute build_dpo_pairs.main() will resolve.
    install_scoring(language, prompt_format)
    from rl_training import rewards as up_rewards

    scored = up_rewards.score_candidates(
        [probe, probe], verifier, timeout=5.0, finish_reasons=["stop", "length"]
    )
    line("rebound score_candidates grades Rust", scored[0]["reward"] == 1.0,
         f"reward={scored[0]['reward']}")
    line("truncated candidate withheld (reward None)", scored[1]["reward"] is None,
         f"status={scored[1]['status']}")

    from rl_training.rewards import build_preference_pair_from_candidates

    pair, audit = build_preference_pair_from_candidates(
        "p", [{"reward": 1.0, "text": "a", "status": "ok"},
              {"reward": 0.0, "text": "b", "status": "ok"},
              {"reward": None, "text": "c", "status": "truncated"}], margin=0.5)
    line("pair selection ignores truncated", pair is not None and audit["n_truncated"] == 1)

    if prompts_path:
        path = Path(prompts_path)
        line("prompts file exists", path.is_file(), str(path))
        if path.is_file():
            rows = [json.loads(l) for l in path.open() if l.strip()]
            types = {r.get("verifier", {}).get("type") for r in rows}
            line("every verifier is stdin/stdout (language-agnostic)",
                 types <= {"io_tests", "reference_io_tests"},
                 f"n={len(rows)} types={sorted(t for t in types if t)}")
            sample = rows[0]["verifier"]
            reward = rust_rewards.run_io_tests(
                "```rust\nfn main() {}\n```",
                sample["test_inputs"][:2], sample["test_outputs"][:2],
                timeout=5.0, language=language,
            )
            line("a real cohort row grades without error", reward == 0.0,
                 f"empty-main reward={reward}")

    print("\npreflight " + ("passed" if ok else "FAILED"))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--prompt_format", default="rust_io",
                        help="Prompt format; the verifier language is derived from it "
                             "so a sweep cannot prompt Rust and grade Python.")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--prompts", default=None)
    parser.add_argument("--out", default=None)
    known, rest = parser.parse_known_args()

    language = FORMAT_LANGUAGE.get(known.prompt_format)
    if language is None:
        print(f"unknown prompt format {known.prompt_format!r}; "
              f"known: {sorted(FORMAT_LANGUAGE)}", file=sys.stderr)
        return 2

    if known.preflight:
        return preflight(language, known.prompt_format, known.prompts)

    install_scoring(language, known.prompt_format)
    print(f"[rl-rust] verifier language = {language}, prompt format = {known.prompt_format}")

    argv = list(rest)
    if known.prompts:
        argv += ["--prompts", known.prompts]
    if known.out:
        argv += ["--out", known.out]
    if "--max_new_tokens" not in argv:
        argv += ["--max_new_tokens", str(DEFAULT_MAX_NEW_TOKENS)]
        print(f"[rl-rust] defaulting --max_new_tokens {DEFAULT_MAX_NEW_TOKENS} "
              "(grpo_rust.yaml max_completion_length; upstream default 1024 would "
              "cost pairs at Rust's truncation rate)")

    from rl_training import build_dpo_pairs

    sys.argv = ["build_dpo_pairs.py", *argv]
    build_dpo_pairs.main()

    # The upstream manifest cannot say which language graded the sweep (gen_meta
    # has no language field), so leave a sidecar next to the pair file. Without
    # this, a Rust pair file is indistinguishable from a Python one on disk.
    if known.out:
        sidecar = Path(known.out).with_suffix(".language.json")
        sidecar.write_text(json.dumps({
            "launcher": "rl-rust/build_rust_dpo_pairs.py",
            "verifier_language": language,
            "prompt_format": known.prompt_format,
        }, indent=2) + "\n")
        print(f"[rl-rust] wrote language sidecar -> {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
