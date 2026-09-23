"""Single entry point for scripts/*.py.

Runs the scripts registered under `experiments:` in config.yaml, in-process, with
each script's CLI args built from its `args:` mapping in the config.

Usage:
    python main.py --model_key <key>                   # run the baseline battery (experiments.run)
    python main.py --model_key <key> --anticipation    # + baseline-only anticipation forecast
    python main.py --model_key <key> --robustness      # + robustness checks (labels + values arm)
    python main.py --sft                               # SFT pilot: dataset prep + LoRA training (no --model_key)
    python main.py --sft --arm java --size 9b          # ... for a given language arm, off a given M0 base
    python main.py --model_key <ckpt> --trajectory     # per-checkpoint battery ONLY (coding task preference)
    python main.py run_utilities_values --model_key <key> # run just one battery entry
    python main.py --list                              # show the registry
    python main.py <name> -- <raw args>                # append raw argv to a single named script

`--model_key` is applied to EVERY script in the run; that's how you score a whole
battery in one command. `--anticipation` / `--robustness` append optional groups
(experiments.anticipation / .robustness) that aren't part of the routine battery.
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
import traceback
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"


def load_experiments_config() -> dict:
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f) or {}
    return config.get("experiments", {}) or {}


def build_argv(args: dict | None) -> list[str]:
    """Convert an {arg_name: value} mapping into CLI flags.

    bool True -> `--name` (store_true flags); bool False / None -> omitted.
    list -> `--name v1 v2 ...` (matches this repo's nargs="+" flags).
    anything else -> `--name str(value)`.
    """
    argv = []
    for name, value in (args or {}).items():
        flag = f"--{name}"
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            continue
        if isinstance(value, list):
            argv.append(flag)
            argv.extend(str(v) for v in value)
            continue
        argv.append(flag)
        argv.append(str(value))
    return argv


def resolve_selection(
    experiments: dict, names: list[str] | None, extra_groups: tuple[str, ...] = ()
) -> list[str]:
    registry = experiments.get("registry", {})
    if names:
        unknown = [n for n in names if n not in registry]
        if unknown:
            raise SystemExit(f"Unknown script(s): {unknown}. Known: {sorted(registry)}")
        return names
    # The routine battery is the `run:` list (values, coding, book-A, UE). Optional flags append named groups from config — measurements you don't run every time: `--anticipation` -> experiments.anticipation, `--robustness` -> experiments.robustness. Each group is a list of registry names.
    battery = list(experiments.get("run") or [])
    for group in extra_groups:
        battery += [n for n in (experiments.get(group) or []) if n not in battery]
    return [name for name in battery if name in registry]


SFT_DEFAULT_ARM = "coding.write.python"


def sft_arms() -> list[str]:
    """The language arms the SFT spec defines, read from the spec so the two can't drift apart."""
    spec_path = ROOT / "data" / "training_specs" / "coding_axis_sft.json"
    with open(spec_path) as f:
        spec = json.load(f)
    return [i["intervention_id"] for i in spec["interventions"]]


def apply_arm(entry: dict, arm: str) -> dict:
    """Point an SFT registry entry at a different language arm.

    The arm appears in three places in each entry -- `intervention_id`, and the `coding.write.<lang>_n64_seed42` segment of `dataset_dir` / `output_dir` -- and every one of them has to move together. Overriding only some of them is the failure this exists to prevent: passing `--dataset_dir` for Java while `intervention_id` stays at its Python default trains on Java data but files the checkpoints under Python's hub folder, silently, with no error.

    Substituting the default arm string across every string value moves all three at once, so a new arm cannot be half-applied."""
    args = dict(entry.get("args") or {})
    if "intervention_id" not in args:
        return entry
    return {**entry, "args": {
        k: v.replace(SFT_DEFAULT_ARM, arm) if isinstance(v, str) else v
        for k, v in args.items()
    }}


SFT_DEFAULT_SIZE = "4b"
# The M0-v4 warm-starts the SFT arms train from. The VALUE is the bare slug rather than the full `prism-drift/...` id because the slug is what appears inside every path and repo id that has to move together -- substituting it catches the full model id too, since that id contains it.
SFT_BASE_SLUGS = {"4b": "qwen35-4b-m0-v4", "9b": "qwen35-9b-m0-v4"}


def apply_size(entry: dict, size: str) -> dict:
    """Point an SFT registry entry at the other model size.

    Same hazard as `apply_arm`, one axis over. The base model slug appears in `base_model` for both SFT entries and additionally in `train_sft_lora`'s `output_dir`, and half-applying it is silent: train the 9B while `output_dir` still says 4B and the 9B checkpoints land in the 4B run's directory, where nothing distinguishes them afterwards. Substituting the default slug across every string value moves all of them at once.

    Note which paths deliberately do NOT carry the slug. The prepared dataset is a seeded draw from a committed pool with no tokenizer involved, so it is identical for both sizes; `dataset_dir` and `prepare_sft_dataset`'s `output_dir` stay size-agnostic and one prepared set feeds both runs. Only the training outputs and the hub repo ids move."""
    args = dict(entry.get("args") or {})
    default_slug = SFT_BASE_SLUGS[SFT_DEFAULT_SIZE]
    if not any(isinstance(v, str) and default_slug in v for v in args.values()):
        return entry
    return {**entry, "args": {
        k: v.replace(default_slug, SFT_BASE_SLUGS[size]) if isinstance(v, str) else v
        for k, v in args.items()
    }}


def run_script(name: str, entry: dict, extra_argv: list[str]) -> int:
    if "path" not in entry:
        print(f"  !! {name}: registry entry has no 'path'", file=sys.stderr)
        return 1
    path = ROOT / entry["path"]
    if not path.is_file():
        print(f"  !! {name}: no such file {path}", file=sys.stderr)
        return 1

    positional = [str(v) for v in entry.get("positional", [])]
    argv = [str(path)] + positional + build_argv(entry.get("args")) + extra_argv
    print(f"\n=== {name} ({entry['path']}) ===")
    print("  " + " ".join(argv[1:]) if len(argv) > 1 else "  (no args)")

    old_argv = sys.argv
    sys.argv = argv
    try:
        runpy.run_path(str(path), run_name="__main__")
        return 0
    except SystemExit as e:
        code = e.code
        if code is None:
            code = 0
        elif not isinstance(code, int):
            print(str(code), file=sys.stderr)
            code = 1
        if code != 0:
            print(f"  !! {name} exited with code {code}", file=sys.stderr)
        return code
    except Exception:
        traceback.print_exc()
        print(f"  !! {name} raised an exception", file=sys.stderr)
        return 1
    finally:
        sys.argv = old_argv


def list_registry(experiments: dict) -> None:
    registry = experiments.get("registry", {})
    if not registry:
        print("No scripts registered under experiments.registry in config.yaml")
        return
    default_selection = set(resolve_selection(experiments, None))
    for name, entry in registry.items():
        marker = "*" if name in default_selection else " "
        print(
            f" {marker} {name:28s} {entry.get('path', '?'):40s} enabled={entry.get('enabled', False)}"
        )
    print("\n* = would run by default (python main.py with no args)")


def main() -> int:
    raw_argv = sys.argv[1:]
    if "--" in raw_argv:
        idx = raw_argv.index("--")
        cli_argv, extra_argv = raw_argv[:idx], raw_argv[idx + 1 :]
    else:
        cli_argv, extra_argv = raw_argv, []

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "names",
        nargs="*",
        help="Registry name(s) to run (default: config.yaml experiments.run / enabled entries)",
    )
    parser.add_argument(
        "--list", action="store_true", help="List registered scripts and exit"
    )
    parser.add_argument(
        "--keep-going",
        dest="keep_going",
        action="store_true",
        default=None,
        help="Continue after a failing script (overrides experiments.stop_on_error)",
    )
    parser.add_argument(
        "--stop-on-error",
        dest="keep_going",
        action="store_false",
        help="Stop at the first failing script (overrides experiments.stop_on_error)",
    )
    parser.add_argument(
        "--model_key",
        default=None,
        help="Model key from config.yaml, applied to EVERY script in the run. "
        "This is how you score a whole battery in one command.",
    )
    parser.add_argument(
        "--arm",
        default=None,
        help="Language arm for --sft: python (default), cpp, java, rust. Repoints intervention_id "
        "AND the dataset/output dirs together, so the checkpoints are filed under the arm you "
        "actually trained. Accepts 'java' or the full 'coding.write.java'.",
    )
    parser.add_argument(
        "--size",
        default=None,
        choices=sorted(SFT_BASE_SLUGS),
        help="M0-v4 base model size for --sft: 4b (default) or 9b. Repoints base_model AND the "
        "training output dir together, so 9B checkpoints cannot land in the 4B run's directory. "
        "The prepared dataset is size-independent and is deliberately left shared.",
    )
    parser.add_argument(
        "--anticipation",
        action="store_true",
        help="Also run the anticipation entries (experiments.anticipation) — the "
        "baseline-only forecast, not re-run per checkpoint.",
    )
    parser.add_argument(
        "--robustness",
        action="store_true",
        help="Also run the robustness entries (experiments.robustness: score_label_variants "
        "for the flat option pools, score_label_variants_values for the values arm) — "
        "once-per-model validity checks, not the routine battery.",
    )
    parser.add_argument(
        "--sft",
        action="store_true",
        help="Also run the SFT entries (experiments.sft: seeded dataset prep + LoRA "
        "training with per-step checkpoints). Weight-changing, needs the pinned SFT "
        "env; these scripts take no --model_key.",
    )
    parser.add_argument(
        "--trajectory",
        action="store_true",
        help="Run ONLY the per-checkpoint battery (experiments.trajectory: coding task "
        "preference). This is what the methodology measures during training; the other "
        "books are endpoint-only spillover checks, so this replaces the full battery "
        "rather than adding to it.",
    )
    args = parser.parse_args(cli_argv)

    experiments = load_experiments_config()
    registry = experiments.get("registry", {})

    if args.list:
        list_registry(experiments)
        return 0

    if extra_argv and len(args.names) != 1:
        raise SystemExit(
            "Args after `--` are only supported when running exactly one script by name"
        )

    extra_groups = tuple(
        g for g, on in (("anticipation", args.anticipation), ("robustness", args.robustness)) if on
    )
    # --sft is exclusive, not additive: the battery reads a served model while training changes weights and wants the GPU to itself, so they never belong in one invocation.
    if args.sft:
        if args.names or extra_groups or args.model_key or args.trajectory:
            raise SystemExit(
                "--sft runs only the SFT entries (experiments.sft) and cannot be combined "
                "with script names, --anticipation/--robustness/--trajectory, or --model_key."
            )
        selection = [n for n in (experiments.get("sft") or []) if n in registry]
    elif args.trajectory:
        # Exclusive too, and deliberately so: the per-checkpoint measurement is a SUBSET of the battery, not an addition to it. Scoring the full battery at every checkpoint would measure books the methodology only asks for at the endpoints.
        if args.names:
            raise SystemExit(
                "--trajectory runs only the per-checkpoint battery (experiments.trajectory) "
                "and cannot be combined with script names."
            )
        selection = [n for n in (experiments.get("trajectory") or []) if n in registry]
        for group in extra_groups:
            selection += [n for n in (experiments.get(group) or []) if n in registry and n not in selection]
    else:
        selection = resolve_selection(experiments, args.names or None, extra_groups)
    if not selection:
        print(
            "Nothing to run: no scripts selected (check experiments.run in config.yaml)"
        )
        return 0

    # --model_key applies to every script in the run (the whole battery); raw args after `--` still apply only to a single named script.
    model_key_argv = ["--model_key", args.model_key] if args.model_key else []
    if not args.model_key and not args.sft:
        print(
            "Note: no --model_key given. Scripts that need a served model will error; "
            "pass --model_key <config.yaml key> to score the battery.",
            file=sys.stderr,
        )

    keep_going = args.keep_going
    if keep_going is None:
        keep_going = not experiments.get("stop_on_error", True)

    # Resolve --arm before anything runs. A typo must fail here, loudly: an unrecognised arm that silently no-ops would leave every path at its Python default and quietly train the wrong arm.
    arm = args.arm
    if arm:
        known = sft_arms()
        if arm in known:
            pass
        elif f"coding.write.{arm}" in known:
            arm = f"coding.write.{arm}"
        else:
            short = ", ".join(a.rsplit(".", 1)[-1] for a in known)
            raise SystemExit(f"Unknown --arm {args.arm!r}. Known arms: {short} (or the full id, e.g. {known[0]})")
        if not any("intervention_id" in (registry[n].get("args") or {}) for n in selection):
            raise SystemExit("--arm only applies to the SFT entries; add --sft (or name prepare_sft_dataset / train_sft_lora).")
        print(f"Language arm: {arm}")

    # Same contract as --arm: an unrecognised size must fail here rather than silently leave every path at its 4B default. argparse `choices` already rejects typos, so the only check left is that the selection can actually honour it.
    size = args.size
    if size:
        default_slug = SFT_BASE_SLUGS[SFT_DEFAULT_SIZE]
        selected_args = [(registry[n].get("args") or {}).values() for n in selection]
        if not any(isinstance(v, str) and default_slug in v for values in selected_args for v in values):
            # Two different mistakes, and conflating them sends you looking in the wrong place. prepare_sft_dataset is an SFT entry that genuinely has nothing size-dependent in it -- the prepared set is identical at every size -- so asking for a size there is a no-op worth saying out loud, not the same error as asking for one outside SFT entirely.
            if any("intervention_id" in (registry[n].get("args") or {}) for n in selection):
                raise SystemExit("--size has no effect on the selected entries: only train_sft_lora names a base model. The prepared dataset is identical at every size.")
            raise SystemExit("--size only applies to the SFT entries; add --sft (or name train_sft_lora).")
        print(f"Base model: prism-drift/{SFT_BASE_SLUGS[size]}")

    single = len(selection) == 1
    failures = []
    for name in selection:
        entry = registry[name]
        if arm:
            entry = apply_arm(entry, arm)
        if size:
            entry = apply_size(entry, size)
        # SFT scripts change weights rather than reading a served model and define no --model_key flag; registry entries opt out via `accepts_model_key: false`.
        accepts_model_key = registry[name].get("accepts_model_key", True)
        per_extra = (model_key_argv if accepts_model_key else []) + (extra_argv if single else [])
        code = run_script(name, entry, per_extra)
        if code != 0:
            failures.append((name, code))
            if not keep_going:
                break

    if failures:
        print(
            "\nFailed: " + ", ".join(f"{n} (exit {c})" for n, c in failures),
            file=sys.stderr,
        )
        return 1
    print("\nAll selected scripts completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
