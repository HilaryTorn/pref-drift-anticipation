#!/usr/bin/env python3
"""Run the dedicated matched DPO/GRPO RL elicitation trajectory.

The caller must first serve the base plus all named LoRA adapters through one
OpenAI-compatible vLLM endpoint. Both instruments are run at M0 and every
four-step checkpoint for DPO, GRPO, and PPO. Only M0 anticipation is eligible
as a prospective forecast; checkpoint anticipation is a separate
self-forecast-drift readout.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_rl_anticipation_preferences.py"
STEPS = (*range(4, 125, 4), 125)
PREFERENCE_LEVELS = (
    "described_choice",
    "described_dataset",
    "example_datapoints",
)
# These are corruption/regression guards, not acceptance-sampling targets.  A
# native-thinking model can genuinely fail to commit a label, and resampling a
# completed formal battery until its parse rate crosses a high threshold would
# select on a stochastic post-treatment outcome.  The thresholds therefore
# reject the old broken parser (roughly 18% preference / 63% anticipation) while
# retaining complete runs with ordinary native-thinking noncompliance.
MIN_PREFERENCE_PARSE_RATE = 0.70
MIN_PREFERENCE_LEVEL_PARSE_RATE = 0.35
MIN_ANTICIPATION_PARSE_RATE = 0.85
ANTICIPATION_FORECASTS = 68
ANTICIPATION_K = 4
ANTICIPATION_LABEL_ROTATIONS = 3
ANTICIPATION_ROWS = (
    ANTICIPATION_FORECASTS * ANTICIPATION_K * ANTICIPATION_LABEL_ROTATIONS
)


def read_parse_counts(path: Path, expected_rows: int) -> dict:
    if not path.is_file():
        raise RuntimeError(f"Missing parse-quality artifact: {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"Incomplete parse-quality artifact {path}: {len(rows)} != {expected_rows} rows"
        )
    parsed = sum(row.get("parse_status") == "parsed" for row in rows)
    return {
        "path": str(path),
        "rows": len(rows),
        "parsed": parsed,
        "parse_rate": parsed / len(rows),
        "terminal_parser_flags": sorted(
            {
                row.get("parse_terminal_only")
                for row in rows
                if "parse_terminal_only" in row
            },
            key=str,
        ),
    }


def validate_parse_quality(
    model_key: str, thinking: str, instrument: str, size: str
) -> dict:
    """Reject completed-looking commands whose sampled answers are mostly unusable."""
    variant = "thinking" if thinking == "on" else "non_thinking"
    model_root = ROOT / "results" / model_key
    if instrument == "preferences":
        spec = "rl_dpo_grpo_9b_preferences" if size == "9b" else "rl_dpo_grpo_preferences"
        artifacts = [
            read_parse_counts(
                model_root
                / "pairs"
                / f"raw_responses_{spec}_{level}_{variant}.jsonl",
                20,
            )
            for level in PREFERENCE_LEVELS
        ]
        parsed = sum(item["parsed"] for item in artifacts)
        rows = sum(item["rows"] for item in artifacts)
        rate = parsed / rows
        if rate < MIN_PREFERENCE_PARSE_RATE or any(
            item["parse_rate"] < MIN_PREFERENCE_LEVEL_PARSE_RATE for item in artifacts
        ):
            raise RuntimeError(
                f"Preference parse coverage too low for {model_key}/{variant}: "
                f"aggregate={parsed}/{rows} ({rate:.1%}), levels="
                + ", ".join(f"{item['parsed']}/{item['rows']}" for item in artifacts)
            )
    elif instrument == "anticipation":
        matches = sorted(
            (model_root / "anticipation").glob(
                f"rl_anticipation_rl_dpo_grpo*_{model_key}_{variant}_raw.jsonl"
            )
        )
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one anticipation raw artifact for {model_key}/{variant}, "
                f"found {len(matches)}: {matches}"
            )
        artifacts = [read_parse_counts(matches[0], ANTICIPATION_ROWS)]
        parsed = artifacts[0]["parsed"]
        rows = artifacts[0]["rows"]
        rate = artifacts[0]["parse_rate"]
        if rate < MIN_ANTICIPATION_PARSE_RATE:
            raise RuntimeError(
                f"Anticipation parse coverage too low for {model_key}/{variant}: "
                f"{parsed}/{rows} ({rate:.1%})"
            )
    else:
        raise ValueError(f"Unknown trajectory instrument: {instrument}")
    return {
        "schema": "rl_checkpoint_parse_quality_v1",
        "instrument": instrument,
        "variant": variant,
        "parsed": parsed,
        "rows": rows,
        "parse_rate": rate,
        "artifacts": artifacts,
    }


def split_keys(raw: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate model keys: {values}")
    return values


def command_preferences(
    model_key: str, thinking: str, comparison: str, models_config_path: str | None
) -> list[str]:
    command = [
        sys.executable,
        str(RUNNER),
        "rl-preferences",
        "--comparison",
        comparison,
        "--model_key",
        model_key,
        "--level",
        "all",
        "--thinking",
        thinking,
        "--no-timestamp",
    ]
    if models_config_path:
        command.extend(["--models-config-path", models_config_path])
    return command


def command_anticipation(
    model_key: str, thinking: str, comparison: str, models_config_path: str | None
) -> list[str]:
    command = [
        sys.executable,
        str(RUNNER),
        "rl-anticipation",
        "--comparison",
        comparison,
        "--model_key",
        model_key,
        "--level",
        "all",
        "--target_set",
        "all",
        "--K",
        str(ANTICIPATION_K),
        "--label_rotations",
        str(ANTICIPATION_LABEL_ROTATIONS),
        "--thinking",
        thinking,
        "--no-timestamp",
    ]
    if models_config_path:
        command.extend(["--models-config-path", models_config_path])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=["4b", "9b"], default="4b")
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--dpo_keys", default=None)
    parser.add_argument("--grpo_keys", default=None)
    parser.add_argument("--ppo_keys", default=None)
    parser.add_argument("--models-config-path", default=None)
    parser.add_argument("--state-dir", type=Path, default=None)
    parser.add_argument(
        "--command-timeout",
        type=int,
        default=1800,
        help="Recycle the enclosing client/server after one task exceeds this many seconds.",
    )
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help="Ignore successful command markers and rerun the complete trajectory.",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    baseline = args.baseline or f"qwen35-{args.size}-m0-v4-aws"
    dpo_raw = args.dpo_keys or ",".join(
        f"qwen35-{args.size}-rl-dpo-s0-step{step}-aws" for step in STEPS
    )
    grpo_raw = args.grpo_keys or ",".join(
        f"qwen35-{args.size}-rl-grpo-s0-step{step}-aws" for step in STEPS
    )
    ppo_raw = args.ppo_keys or ",".join(
        f"qwen35-{args.size}-rl-ppo-s0-step{step}-aws" for step in STEPS
    )
    comparison = "dpo-grpo-9b" if args.size == "9b" else "dpo-grpo"
    dpo_keys = split_keys(dpo_raw)
    grpo_keys = split_keys(grpo_raw)
    ppo_keys = split_keys(ppo_raw)
    expected = len(STEPS)
    if any(len(keys) != expected for keys in (dpo_keys, grpo_keys, ppo_keys)):
        raise SystemExit(
            f"Expected {expected} checkpoint keys (steps 4..124 every 4, plus 125) "
            "for each method"
        )
    models = [baseline, *dpo_keys, *grpo_keys, *ppo_keys]
    tasks = []
    for model_key in models:
        for thinking in ("on", "off"):
            tasks.append(
                (
                    f"{model_key}__{thinking}__preferences",
                    command_preferences(
                        model_key, thinking, comparison, args.models_config_path
                    ),
                )
            )
            tasks.append(
                (
                    f"{model_key}__{thinking}__anticipation",
                    command_anticipation(
                        model_key, thinking, comparison, args.models_config_path
                    ),
                )
            )

    manifest = {
        "schema": "dpo_grpo_rl_trajectory_plan_v2",
        "comparison": comparison,
        "model_size": args.size,
        "baseline": baseline,
        "dpo_models": dpo_keys,
        "grpo_models": grpo_keys,
        "ppo_models": ppo_keys,
        "checkpoint_steps": list(STEPS),
        "anticipation_models": models,
        "prospective_anticipation_models": [baseline],
        "checkpoint_anticipation_role": "secondary_self_forecast_drift",
        "thinking_variants": ["on", "off"],
        "preference_levels": [
            "described_choice",
            "described_dataset",
            "example_datapoints",
        ],
        "command_count": len(tasks),
    }
    print(json.dumps(manifest, indent=2))
    state_dir = args.state_dir or ROOT / "results" / "rl_checkpoint_trajectory_state" / args.size
    for index, (task_name, command) in enumerate(tasks, 1):
        model_key, thinking, instrument = task_name.rsplit("__", 2)
        marker = state_dir / f"{task_name}.done.json"
        marker_payload = None
        if marker.is_file():
            try:
                marker_payload = json.loads(marker.read_text())
            except (json.JSONDecodeError, OSError):
                marker_payload = None
        marker_matches = (
            not args.rerun_completed
            and marker_payload
            and marker_payload.get("command") == command
            and marker_payload.get("status") == "complete"
        )
        if marker_matches:
            try:
                quality = validate_parse_quality(model_key, thinking, instrument, args.size)
            except RuntimeError as exc:
                print(
                    f"[{index}/{len(tasks)}] RERUN failed quality gate {task_name}: {exc}",
                    flush=True,
                )
            else:
                print(
                    f"[{index}/{len(tasks)}] SKIP complete {task_name} "
                    f"(parsed {quality['parsed']}/{quality['rows']})",
                    flush=True,
                )
                continue
        if (
            not args.dry_run
            and not args.rerun_completed
            and not marker_payload
            and thinking == "on"
        ):
            try:
                quality = validate_parse_quality(model_key, thinking, instrument, args.size)
                uses_fixed_parser = all(
                    item["terminal_parser_flags"] == [True]
                    for item in quality["artifacts"]
                )
                if not uses_fixed_parser:
                    raise RuntimeError("artifacts do not identify the fixed terminal parser")
            except RuntimeError as exc:
                print(
                    f"[{index}/{len(tasks)}] existing outputs not adoptable {task_name}: {exc}",
                    flush=True,
                )
            else:
                state_dir.mkdir(parents=True, exist_ok=True)
                payload = {
                    "schema": "rl_checkpoint_trajectory_command_v1",
                    "status": "complete",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "task": task_name,
                    "command": command,
                    "parse_quality": quality,
                    "recovered_from_complete_outputs": True,
                }
                temporary = marker.with_suffix(marker.suffix + ".tmp")
                temporary.write_text(json.dumps(payload, indent=2) + "\n")
                temporary.replace(marker)
                print(
                    f"[{index}/{len(tasks)}] ADOPT complete outputs {task_name} "
                    f"(parsed {quality['parsed']}/{quality['rows']})",
                    flush=True,
                )
                continue
        print(f"[{index}/{len(tasks)}] {' '.join(command)}", flush=True)
        if not args.dry_run:
            try:
                subprocess.run(
                    command,
                    cwd=ROOT,
                    check=True,
                    timeout=args.command_timeout,
                )
            except subprocess.TimeoutExpired:
                print(
                    f"[{index}/{len(tasks)}] WATCHDOG timeout after "
                    f"{args.command_timeout}s: {task_name}",
                    file=sys.stderr,
                    flush=True,
                )
                return 75
            quality = validate_parse_quality(model_key, thinking, instrument, args.size)
            state_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema": "rl_checkpoint_trajectory_command_v1",
                "status": "complete",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "task": task_name,
                "command": command,
                "parse_quality": quality,
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
