#!/usr/bin/env python3
"""Run the dense realized coding-task preference trajectory for one model size.

The caller serves M0 plus every formal DPO/GRPO/PPO adapter through one vLLM
endpoint.  This runner queries the pre-registered reasoning-on coding task
preference battery at M0 and all 32 checkpoints per method.  Its utilities are
the realized outcomes required to compare M0 anticipation with observed drift.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_elicitations.py"
STEPS = (*range(4, 125, 4), 125)
MIN_PARSE_RATE = 0.70
EXPECTED_OPTIONS = 27


def split_keys(raw: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate model keys: {values}")
    return values


def output_paths(model_key: str) -> dict[str, Path]:
    root = ROOT / "results" / model_key / "pairs"
    suffix = "coding_task_preference"
    return {
        "manifest": root / f"elicitation_manifest_{suffix}.json",
        "raw": root / f"raw_responses_{suffix}.jsonl",
        "results": root / f"results_{suffix}.json",
        "utilities": root / f"results_utilities_{suffix}.json",
        "summary": root / f"summary_{suffix}.txt",
    }


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise RuntimeError(f"Missing task-preference artifact: {path}")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON task-preference artifact {path}: {exc}") from exc


def validate_outputs(model_key: str) -> dict[str, Any]:
    paths = output_paths(model_key)
    manifest = read_json(paths["manifest"])
    results = read_json(paths["results"])
    utilities = read_json(paths["utilities"])
    if not paths["summary"].is_file() or not paths["summary"].read_text().strip():
        raise RuntimeError(f"Missing or empty task-preference summary: {paths['summary']}")

    manifest_options = manifest.get("options") if isinstance(manifest, dict) else None
    if not isinstance(manifest_options, list) or len(manifest_options) != EXPECTED_OPTIONS:
        raise RuntimeError(
            f"Expected {EXPECTED_OPTIONS} manifest options for {model_key}, "
            f"got {len(manifest_options) if isinstance(manifest_options, list) else None}"
        )
    for label, payload in (("results", results), ("utilities", utilities)):
        options = payload.get("options") if isinstance(payload, dict) else None
        fitted = payload.get("utilities") if isinstance(payload, dict) else None
        if not isinstance(options, list) or len(options) != EXPECTED_OPTIONS:
            raise RuntimeError(f"Expected {EXPECTED_OPTIONS} options in {label} for {model_key}")
        if not isinstance(fitted, dict) or len(fitted) != EXPECTED_OPTIONS:
            raise RuntimeError(f"Expected {EXPECTED_OPTIONS} fitted utilities in {label} for {model_key}")

    if not paths["raw"].is_file():
        raise RuntimeError(f"Missing raw task-preference responses: {paths['raw']}")
    rows = [json.loads(line) for line in paths["raw"].read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"Empty raw task-preference responses: {paths['raw']}")
    wrong_model = sum(row.get("model_key") != model_key for row in rows)
    if wrong_model:
        raise RuntimeError(
            f"Raw task-preference provenance mismatch for {model_key}: {wrong_model} rows"
        )
    parsed = sum(row.get("parse_status") == "parsed" for row in rows)
    parse_rate = parsed / len(rows)
    if parse_rate < MIN_PARSE_RATE:
        raise RuntimeError(
            f"Task-preference parse coverage too low for {model_key}: "
            f"{parsed}/{len(rows)} ({parse_rate:.1%})"
        )
    recorded_raw = utilities.get("raw_dump_path")
    recorded_raw_path = Path(recorded_raw) if isinstance(recorded_raw, str) else None
    if recorded_raw_path is not None and not recorded_raw_path.is_absolute():
        recorded_raw_path = ROOT / recorded_raw_path
    if recorded_raw_path is None or recorded_raw_path.resolve() != paths["raw"].resolve():
        raise RuntimeError(
            f"Utility/raw provenance mismatch for {model_key}: "
            f"{recorded_raw!r} != {str(paths['raw'])!r}"
        )
    return {
        "schema": "coding_task_preference_quality_v1",
        "model_key": model_key,
        "rows": len(rows),
        "parsed": parsed,
        "parse_rate": parse_rate,
        "option_count": EXPECTED_OPTIONS,
        "artifacts": {name: str(path) for name, path in paths.items()},
        "holdout_metrics": utilities.get("holdout_metrics"),
        "diagnostics": utilities.get("diagnostics"),
    }


def command(model_key: str, models_config_path: str | None) -> list[str]:
    result = [
        sys.executable,
        str(RUNNER),
        "task-preference",
        "--model_key",
        model_key,
        "--reasoning",
        "on",
        "--config_key",
        "thurstonian_active_learning",
        "--create_agent_config_key",
        "default_with_reasoning",
        "--seed",
        "42",
        "--no-timestamp",
    ]
    if models_config_path:
        result.extend(["--models_config_path", models_config_path])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=("4b", "9b"), default="4b")
    parser.add_argument("--baseline")
    parser.add_argument("--dpo_keys")
    parser.add_argument("--grpo_keys")
    parser.add_argument("--ppo_keys")
    parser.add_argument("--models-config-path")
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument(
        "--command-timeout",
        type=int,
        default=1800,
        help="Return exit 75 if one checkpoint battery exceeds this many seconds.",
    )
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    baseline = args.baseline or f"qwen35-{args.size}-m0-v4-aws"
    method_keys: dict[str, list[str]] = {}
    for method in ("dpo", "grpo", "ppo"):
        explicit = getattr(args, f"{method}_keys")
        raw = explicit or ",".join(
            f"qwen35-{args.size}-rl-{method}-s0-step{step}-aws" for step in STEPS
        )
        method_keys[method] = split_keys(raw)
        if len(method_keys[method]) != len(STEPS):
            raise SystemExit(
                f"Expected {len(STEPS)} {method.upper()} keys at steps 4..124 plus 125, "
                f"got {len(method_keys[method])}"
            )
    models = [baseline, *(key for method in ("dpo", "grpo", "ppo") for key in method_keys[method])]
    state_dir = args.state_dir or (
        ROOT / "results" / "coding_task_preference_trajectory_state" / args.size
    )
    plan = {
        "schema": "coding_task_preference_trajectory_plan_v1",
        "model_size": args.size,
        "baseline": baseline,
        "dpo_models": method_keys["dpo"],
        "grpo_models": method_keys["grpo"],
        "ppo_models": method_keys["ppo"],
        "checkpoint_steps": list(STEPS),
        "battery": "coding_task_preference",
        "reasoning": "on",
        "utility_config": "thurstonian_active_learning",
        "seed": 42,
        "command_count": len(models),
    }
    print(json.dumps(plan, indent=2), flush=True)
    if not args.dry_run:
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")

    for index, model_key in enumerate(models, 1):
        task_command = command(model_key, args.models_config_path)
        marker = state_dir / f"{model_key}.done.json"
        marker_payload = None
        if marker.is_file():
            try:
                marker_payload = json.loads(marker.read_text())
            except (json.JSONDecodeError, OSError):
                marker_payload = None
        marker_matches = (
            not args.rerun_completed
            and marker_payload
            and marker_payload.get("status") == "complete"
            and marker_payload.get("command") == task_command
        )
        if marker_matches:
            try:
                quality = validate_outputs(model_key)
            except RuntimeError as exc:
                print(f"[{index}/{len(models)}] RERUN invalid {model_key}: {exc}", flush=True)
            else:
                print(
                    f"[{index}/{len(models)}] SKIP complete {model_key} "
                    f"(parsed {quality['parsed']}/{quality['rows']})",
                    flush=True,
                )
                continue

        # A battery may finish writing all artifacts immediately before the
        # runner is interrupted or its marker validation fails. Recover that
        # completed work instead of spending another full battery rerunning it.
        if not args.rerun_completed and not marker.is_file():
            try:
                quality = validate_outputs(model_key)
            except RuntimeError:
                pass
            else:
                payload = {
                    "schema": "coding_task_preference_trajectory_command_v1",
                    "status": "complete",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "task": model_key,
                    "command": task_command,
                    "quality": quality,
                    "recovered_from_valid_artifacts": True,
                }
                temporary = marker.with_suffix(marker.suffix + ".tmp")
                temporary.write_text(json.dumps(payload, indent=2) + "\n")
                temporary.replace(marker)
                print(
                    f"[{index}/{len(models)}] RECOVER complete {model_key} "
                    f"(parsed {quality['parsed']}/{quality['rows']})",
                    flush=True,
                )
                continue

        print(f"[{index}/{len(models)}] {' '.join(task_command)}", flush=True)
        if args.dry_run:
            continue
        try:
            subprocess.run(
                task_command,
                cwd=ROOT,
                check=True,
                timeout=args.command_timeout,
            )
        except subprocess.TimeoutExpired:
            print(
                f"[{index}/{len(models)}] WATCHDOG timeout after "
                f"{args.command_timeout}s: {model_key}",
                file=sys.stderr,
                flush=True,
            )
            return 75
        quality = validate_outputs(model_key)
        payload = {
            "schema": "coding_task_preference_trajectory_command_v1",
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "task": model_key,
            "command": task_command,
            "quality": quality,
        }
        temporary = marker.with_suffix(marker.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(marker)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from None
