"""Launch GRPO on Rust by rebinding the verifier language, then delegating.

``rl_training/train_rl.py`` is not modified and not copied. It imports its reward
and prompt helpers INSIDE the functions that use them:

    def _make_coding_reward(...):
        from rl_training.rewards import make_coding_reward

A function-local import resolves at call time, so replacing the attribute on the
module before ``main()`` runs is enough to redirect the whole training loop to
the Rust executor. Nothing on disk changes and the Python arms still run on
their own code, unpatched.

Three things get rebound, and all three matter:

  rewards.make_coding_reward  the online GRPO reward
  rewards.run_verifier        the per-checkpoint held-out reward callback
                              (DriftCadenceCallback) -- miss this and the
                              reward_log would grade Rust rollouts as Python
                              and report a flat 0.0 for the whole run
  prompts.render_coding_prompt  asks for Rust instead of Python

There is also a config trap this script exists to close. ``lora_target_modules``
and friends are argparse arguments, NOT keys ``load_config`` reads. Setting them
in the YAML alone would be silently ignored -- the run would train, log, and
finish, with the wrong adapter placement and no error anywhere. So the LoRA
block is read out of the YAML here and injected into argv.

Usage mirrors train_rl.py; every unrecognized argument is forwarded verbatim:

    .venv/bin/python rl-rust/train_rust_grpo.py \
        --config rl-rust/configs/grpo_rust.yaml \
        --base_model prism-drift/qwen35-4b-m0-v4 \
        --dataset  data/rl/v1/cohorts/<cohort>/train.jsonl \
        --eval_dataset <held-out>.jsonl \
        --seed 0 --run_name qwen35-4b-grpo-rust-s0 --save_dir runs

Add ``--preflight`` to verify the toolchain, the reward, the prompt and the
dataset without loading a model or touching a GPU.
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

import yaml  # noqa: E402

from rl_rust import prompts as rust_prompts  # noqa: E402
from rl_rust import rewards as rust_rewards  # noqa: E402

#: YAML keys this launcher consumes itself. They are NOT read by
#: rl_training.train_rl.load_config, so anything listed here must be either
#: injected into argv or applied by patching.
EVAL_MAX_TESTS = 32

_LAUNCHER_KEYS = {
    "verifier_language",
    "prompt_format",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "lora_bias",
    "lora_target_modules",
}


def _cli_from_config(cfg: dict, argv: list[str]) -> list[str]:
    """Inject LoRA settings from the YAML as argv flags, unless already given.

    An explicit command-line flag always wins, so a one-off override still works
    without editing the config.
    """
    injected: list[str] = []
    if "--use_lora" not in argv:
        injected.append("--use_lora")
    for key in ("lora_r", "lora_alpha", "lora_dropout", "lora_bias", "lora_target_modules"):
        flag = f"--{key}"
        if key in cfg and flag not in argv:
            injected += [flag, str(cfg[key])]
    return injected


def install_language(language: str, prompt_format: str) -> None:
    """Point rl_training's reward and prompt helpers at the Rust executor."""
    from scripts.rl_training import prompts as up_prompts
    from scripts.rl_training import rewards as up_rewards

    up_rewards.make_coding_reward = functools.partial(
        rust_rewards.make_coding_reward, language=language
    )
    # CAP THE EVAL'S TEST BUDGET. DriftCadenceCallback calls run_verifier with no
    # max_tests, so it grades the FULL stored suite -- up to 138 Rust executions
    # per sample. Measured at 155 s/prompt, i.e. ~2h45m per eval and ~11h across
    # the four eval steps, which is more than the training itself. Training grades
    # 12 tests; 32 here keeps the eval higher-fidelity than training while making
    # it affordable. Same deterministic spread-out subsample either way.
    up_rewards.run_verifier = functools.partial(
        rust_rewards.run_verifier, language=language, max_tests=EVAL_MAX_TESTS
    )
    up_rewards.score_completions = functools.partial(
        rust_rewards.score_completions, language=language
    )
    up_prompts.render_coding_prompt = functools.partial(
        rust_prompts.render_coding_prompt, format_name=prompt_format
    )
    up_prompts.format_coding_prompt = functools.partial(
        rust_prompts.format_coding_prompt, format_name=prompt_format
    )


def preflight(cfg: dict, language: str, prompt_format: str, dataset: str | None) -> int:
    """Cheap checks that would otherwise fail hours into a booked GPU."""
    ok = True

    def line(name: str, good: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and good
        print(f"  [{'ok' if good else 'FAIL'}] {name}{'  ' + detail if detail else ''}")

    print("preflight")
    line("rustc on PATH", rust_rewards.rust_toolchain_available())
    if not rust_rewards.rust_toolchain_available():
        print("\n  A missing toolchain scores every rollout 0.0 and looks like a real null.")
        return 1

    spec = rust_rewards.get_language(language)
    line("language resolves", spec.name == language, spec.name)

    probe = """```rust
use std::io::{self, Read};
fn main() {
    let mut s = String::new();
    io::stdin().read_to_string(&mut s).unwrap();
    let n: i64 = s.trim().parse().unwrap();
    println!("{}", n * 2);
}
```"""
    reward = rust_rewards.run_io_tests(
        probe, ["1", "2", "3"], ["2", "4", "6"], timeout=5.0, language=language
    )
    line("reward grades a correct Rust program", abs(reward - 1.0) < 1e-9, f"reward={reward}")

    bad = rust_rewards.run_io_tests(
        probe, ["1", "2", "3"], ["9", "9", "9"], timeout=5.0, language=language
    )
    line("reward rejects a wrong one", bad == 0.0, f"reward={bad}")

    rendered = rust_prompts.format_coding_prompt("A+B", format_name=prompt_format)
    line("prompt asks for Rust", "Rust" in rendered and "Python" not in rendered)

    for key in ("learning_rate", "lr_scheduler_type", "beta", "max_completion_length"):
        line(f"config sets {key}", key in cfg, str(cfg.get(key)))
    line(
        "scheduler is explicit (unset decays to <2e-7 by step 125)",
        cfg.get("lr_scheduler_type") is not None,
    )
    line(
        "mask_truncated_completions on",
        bool(cfg.get("mask_truncated_completions")),
    )

    if dataset:
        path = Path(dataset)
        line("dataset exists", path.is_file(), str(path))
        if path.is_file():
            rows = [json.loads(l) for l in path.open()]
            # train_rl.load_config hard-fails unless the dataset has exactly
            # expected_unique_prompts rows AND that equals max_steps *
            # unique_prompts_per_step. Check both here, cheaply, rather than
            # after the model has loaded on a booked GPU.
            expected = cfg.get("expected_unique_prompts")
            line(
                f"dataset matches expected_unique_prompts ({expected})",
                len(rows) == expected,
                f"n={len(rows)}",
            )
            scheduled = int(cfg["max_steps"]) * int(cfg["unique_prompts_per_step"])
            line(
                "max_steps x unique_prompts_per_step == expected_unique_prompts",
                scheduled == expected,
                f"{cfg['max_steps']} x {cfg['unique_prompts_per_step']} = {scheduled}",
            )
            types = {r.get("verifier", {}).get("type") for r in rows}
            line(
                "every verifier is stdin/stdout (language-agnostic)",
                types <= {"io_tests", "reference_io_tests"},
                str(sorted(t for t in types if t)),
            )
            sample = rows[0]["verifier"]
            reward = rust_rewards.run_io_tests(
                "```rust\nfn main() {}\n```",
                sample["test_inputs"][:2],
                sample["test_outputs"][:2],
                timeout=5.0,
                language=language,
            )
            line("a real cohort row grades without error", reward == 0.0, f"empty-main reward={reward}")

    print("\npreflight " + ("passed" if ok else "FAILED"))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--preflight", action="store_true")
    known, rest = parser.parse_known_args()

    cfg = yaml.safe_load(Path(known.config).read_text())
    language = cfg.get("verifier_language", "rust")
    prompt_format = cfg.get("prompt_format", "rust_io")

    if cfg.get("algo") != "grpo":
        print(f"this launcher is GRPO-only; config says algo={cfg.get('algo')!r}", file=sys.stderr)
        return 2

    if known.preflight:
        return preflight(cfg, language, prompt_format, known.dataset)

    install_language(language, prompt_format)
    print(f"[rl-rust] verifier language = {language}, prompt format = {prompt_format}")

    argv = ["--algo", "grpo", "--config", known.config]
    if known.dataset:
        argv += ["--dataset", known.dataset]
    argv += rest
    argv += _cli_from_config(cfg, rest)

    from scripts.rl_training import train_rl

    sys.argv = ["train_rl.py", *argv]
    train_rl.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
