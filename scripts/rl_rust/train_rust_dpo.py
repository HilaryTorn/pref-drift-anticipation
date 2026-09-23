"""Launch DPO on Rust: rebind the verifier for the eval callback, then delegate.

Same pattern as ``train_rust_grpo.py`` and shares its plumbing (this file
imports ``install_language`` and ``_cli_from_config`` from it). DPO itself never
executes code at train time -- it consumes the offline pair file -- but the
per-checkpoint DriftCadenceCallback samples the policy on held-out prompts and
grades them, and it resolves ``run_verifier`` / ``render_coding_prompt`` at call
time. Miss the rebinding and the reward curve grades Rust rollouts as Python
and reports a flat 0.0 for the whole run, which is indistinguishable from the
null we are trying to rule out.

One extra piece the GRPO launcher does not need: ``train_rl.train_dpo`` forwards
only a fixed set of YAML keys into ``DPOConfig``. The keys that make this arm
actually learn (``loss_type``/``loss_weights``, plus the reentrant-checkpointing
correctness kwarg) are not among them, so this launcher wraps ``trl.DPOConfig``
-- resolved by ``train_dpo`` at call time -- and injects them, printing the
effective values. Nothing in ``rl_training/`` changes.

Usage mirrors train_rl.py; unrecognized arguments are forwarded verbatim:

    .venv/bin/python rl-rust/train_rust_dpo.py \
        --config rl-rust/configs/dpo_rust.yaml \
        --base_model prism-drift/qwen35-9b-m0-v4 \
        --dataset rl-rust/out/dpo-pairs-9b/pairs.jsonl \
        --eval_dataset rl-rust/out/heldout-500.jsonl \
        --seed 0 --run_name qwen35-9b-dpo-rust-s0 --save_dir runs

``--dataset`` is the PAIR file from build_rust_dpo_pairs.py; ``--eval_dataset``
stays a prompts+tests file (the callback needs verifiers, not pairs).

Add ``--preflight`` to verify the toolchain, the pair file, and the config
without loading a model or touching a GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "rl-rust"))

import yaml  # noqa: E402

from rl_rust import prompts as rust_prompts  # noqa: E402
from rl_rust import rewards as rust_rewards  # noqa: E402
from scripts.rl_rust.train_rust_grpo import _cli_from_config, install_language  # noqa: E402

#: YAML keys injected into DPOConfig by the wrapper below. train_rl.train_dpo
#: does not read them, so listing one here is the ONLY thing that makes it real.
_DPO_CONFIG_EXTRA_KEYS = ("loss_type", "loss_weights")


def install_dpo_config_extras(cfg: dict) -> dict:
    """Wrap ``trl.DPOConfig`` so train_dpo's fixed kwargs gain the YAML extras.

    ``train_dpo`` does ``from trl import DPOConfig`` inside the function, which
    resolves from the ``trl`` package namespace at call time -- the same
    late-binding property install_language relies on.
    """
    import trl

    extras = {key: cfg[key] for key in _DPO_CONFIG_EXTRA_KEYS if key in cfg}
    # train_dpo passes bare gradient_checkpointing=True. With LoRA, reentrant
    # checkpointing can silently produce no gradient through a checkpointed
    # segment; GRPO/PPO set use_reentrant=False deliberately and train_rl.py's
    # own comment flags DPO as "gets away with the bare flag today". Close it.
    extras.setdefault("gradient_checkpointing_kwargs", {"use_reentrant": False})

    def patched_dpo_config(**kwargs):
        # Resolve the real class only now, from its submodule: reading
        # trl.DPOConfig at install time would trigger TRL's lazy trainer import
        # BEFORE train_dpo runs _patch_trl_optional_imports, hitting the exact
        # mismatched-optional-deps failure that patch exists to prevent. By the
        # time train_dpo calls this, the patch has run.
        from trl.trainer.dpo_config import DPOConfig as original

        merged = {**kwargs, **extras}
        print(f"[rl-rust] DPOConfig extras injected: {json.dumps(extras, sort_keys=True)}")
        return original(**merged)

    trl.DPOConfig = patched_dpo_config
    return extras


def install_eval_dump() -> None:
    """Tee every held-out eval grade to a JSONL so a flat curve is diagnosable.

    The GRPO runs logged eval_reward == exactly 0.0 for 512 generations and
    nothing in the artifacts could say WHY (root cause: the heldout file still
    carried the raw Nemotron 'use python' wrapper). With this, one look at the
    dump distinguishes wrong-language output, empty generations, or genuinely
    failing programs. Enabled by the DPO_EVAL_DUMP env var; off otherwise, so
    local runs and the other arms are untouched.
    """
    dump = os.environ.get("DPO_EVAL_DUMP")
    if not dump:
        return
    from rl_training import rewards as up_rewards

    inner = up_rewards.run_verifier
    path = Path(dump)
    path.parent.mkdir(parents=True, exist_ok=True)

    def dumping_run_verifier(code, verifier, *args, **kwargs):
        reward = inner(code, verifier, *args, **kwargs)
        with path.open("a") as fh:
            fh.write(json.dumps(
                {"reward": reward, "text_len": len(code), "text_head": code[:500]},
                ensure_ascii=True) + "\n")
        return reward

    up_rewards.run_verifier = dumping_run_verifier
    print(f"[rl-rust] eval completions dumping to {path}")


def preflight(cfg: dict, language: str, prompt_format: str, dataset: str | None) -> int:
    """Cheap checks that would otherwise fail hours into a booked GPU."""
    ok = True

    def line(name: str, good: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and good
        print(f"  [{'ok' if good else 'FAIL'}] {name}{'  ' + detail if detail else ''}")

    print("preflight")
    line("rustc on PATH (eval callback grades here)", rust_rewards.rust_toolchain_available())

    rendered = rust_prompts.format_coding_prompt("A+B", format_name=prompt_format)
    line("prompt asks for Rust", "Rust" in rendered and "Python" not in rendered)

    for key in ("learning_rate", "lr_scheduler_type", "beta", "max_length", "loss_type"):
        line(f"config sets {key}", key in cfg, str(cfg.get(key)))
    line(
        "scheduler is explicit (unset decays to <2e-7 by step 125)",
        cfg.get("lr_scheduler_type") is not None,
    )
    loss_type = cfg.get("loss_type")
    if isinstance(loss_type, list) and len(loss_type) > 1:
        weights = cfg.get("loss_weights")
        line("loss_weights matches loss_type",
             isinstance(weights, list) and len(weights) == len(loss_type), str(weights))
    prompt_cap = 1024  # cohort prompts are length-filtered to this at prep time
    completion_cap = cfg.get("eval_max_new_tokens", 2048)
    line(
        f"max_length covers prompt+completion ({prompt_cap}+{completion_cap})",
        int(cfg.get("max_length", 0)) >= prompt_cap + completion_cap,
        f"max_length={cfg.get('max_length')}",
    )

    if dataset:
        path = Path(dataset)
        line("pair file exists", path.is_file(), str(path))
        if path.is_file():
            from rl_training.data.schema import read_preference_jsonl

            try:
                rows = read_preference_jsonl(path)
            except ValueError as err:
                line("pair rows validate (dpo_preference_v1)", False, str(err))
                rows = []
            if rows:
                line("pair rows validate (dpo_preference_v1)", True, f"n={len(rows)}")
                rust_prompts_n = sum(1 for r in rows if "Rust" in r["prompt"])
                line("every pair prompt asks for Rust", rust_prompts_n == len(rows),
                     f"{rust_prompts_n}/{len(rows)}")
                margins = [r["chosen_reward"] - r["rejected_reward"] for r in rows]
                line("every pair clears margin 0.5", min(margins) >= 0.5,
                     f"min={min(margins):.2f}")
                effective_batch = int(cfg["per_device_train_batch_size"]) * int(
                    cfg["gradient_accumulation_steps"]
                )
                epochs = int(cfg["max_steps"]) * effective_batch / len(rows)
                line(
                    "pair views land in the 1-4 epoch band",
                    1.0 <= epochs <= 4.0,
                    f"{cfg['max_steps']} steps x batch {effective_batch} / "
                    f"{len(rows)} pairs = {epochs:.1f} epochs",
                )
            sidecar = path.with_suffix(".language.json")
            side_ok = sidecar.is_file() and json.loads(sidecar.read_text()).get(
                "verifier_language") == language
            line("language sidecar says rust", side_ok, str(sidecar))

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

    if cfg.get("algo") != "dpo":
        print(f"this launcher is DPO-only; config says algo={cfg.get('algo')!r}", file=sys.stderr)
        return 2

    if known.preflight:
        return preflight(cfg, language, prompt_format, known.dataset)

    install_language(language, prompt_format)
    install_dpo_config_extras(cfg)
    install_eval_dump()
    print(f"[rl-rust] verifier language = {language}, prompt format = {prompt_format}")

    argv = ["--algo", "dpo", "--config", known.config]
    if known.dataset:
        argv += ["--dataset", known.dataset]
    argv += rest
    argv += _cli_from_config(cfg, rest)

    from rl_training import train_rl

    sys.argv = ["train_rl.py", *argv]
    train_rl.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
