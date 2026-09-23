#!/usr/bin/env python3
"""Serve M0 plus four matched final adapters and run the Cartesian battery."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_rl_checkpoint_elicitation import discover_adapter_paths  # noqa: E402
from scripts.run_rl_checkpoint_elicitation import (  # noqa: E402
    require_available_port,
    terminate_process,
    wait_for_server,
)


PREFERENCE_METHODS = ("dpo", "grpo", "ppo")
ACTUAL_TRAINING_METHODS = ("sft", *PREFERENCE_METHODS)
LEVELS = ("described_choice", "described_dataset", "example_datapoints")
VARIANTS = (("on", "thinking"), ("off", "non_thinking"))


def raw_path(model_key: str, suffix: str) -> Path:
    return ROOT / "results" / model_key / "pairs" / f"raw_responses_{suffix}.jsonl"


def complete(model_key: str, mode: str, method: str | None, variant: str) -> bool:
    for level in LEVELS:
        if mode == "preference":
            suffix = f"rl_cartesian_preferences_{level}_{variant}"
        else:
            suffix = f"rl_cartesian_preference_anticipation_{method}_{level}_{variant}"
        path = raw_path(model_key, suffix)
        if not path.is_file():
            return False
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows) != 60:
            return False
    return True


def discover_sft_adapter(checkpoint_root: Path) -> Path:
    candidates = list(
        checkpoint_root.glob("base-*/cohort-*/sft/seed-0/checkpoint-125")
    )
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one aligned sft/seed-0/checkpoint-125 below {checkpoint_root}, "
            f"found {len(candidates)}"
        )
    path = candidates[0]
    for required in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "hub_checkpoint_manifest.json",
    ):
        if not (path / required).is_file():
            raise ValueError(f"Missing {required}: {path}")
    return path.resolve()


def validate_final_adapter_provenance(
    adapters: dict[tuple[str, int], Path],
) -> str:
    cohort_hashes = set()
    base_revisions = set()
    for method in ACTUAL_TRAINING_METHODS:
        path = adapters[(method, 125)]
        manifest_path = path / "hub_checkpoint_manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"Missing formal checkpoint manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        expected = {
            "schema": "formal_hub_checkpoint_v1",
            "algorithm": method,
            "seed": 0,
            "optimizer_step": 125,
            "optimizer_state_uploaded": False,
        }
        mismatched = [
            key for key, value in expected.items() if manifest.get(key) != value
        ]
        if mismatched:
            raise ValueError(
                f"Invalid formal {method.upper()} checkpoint provenance at {path}: "
                + ", ".join(mismatched)
            )
        files = manifest.get("files") or {}
        if not {"adapter_config.json", "adapter_model.safetensors"}.issubset(files):
            raise ValueError(f"Formal {method.upper()} manifest omits adapter files: {path}")
        cohort_hash = manifest.get("cohort_manifest_sha256")
        base_revision = manifest.get("base_model_revision")
        if not cohort_hash or not base_revision:
            raise ValueError(f"Formal {method.upper()} manifest lacks pinned provenance")
        cohort_hashes.add(cohort_hash)
        base_revisions.add((manifest.get("base_model"), base_revision))
    if len(cohort_hashes) != 1:
        raise ValueError(
            "SFT/DPO/GRPO/PPO checkpoint cohort mismatch: "
            + ", ".join(sorted(cohort_hashes))
        )
    if len(base_revisions) != 1:
        raise ValueError("SFT/DPO/GRPO/PPO checkpoints do not share one base revision")
    return next(iter(cohort_hashes))


def build_registry(
    size: str, base_url: str, adapters: dict[tuple[str, int], Path]
) -> tuple[dict, list[dict]]:
    registry = {
        f"qwen35-{size}-m0-v4-aws": {
            "model_name": f"qwen35-{size}-m0-v4",
            "model_type": "vllm_endpoint",
            "base_url": base_url,
            "gpu_count": 1,
        }
    }
    modules = []
    for method in ACTUAL_TRAINING_METHODS:
        alias = f"{method}-s0-step125"
        key = f"qwen35-{size}-rl-{method}-s0-step125-aws"
        path = adapters[(method, 125)]
        registry[key] = {
            "model_name": alias,
            "model_type": "vllm_endpoint",
            "base_url": base_url,
            "gpu_count": 1,
            "drift_algo": method,
            "drift_seed": 0,
            "drift_step": 125,
        }
        modules.append({"name": alias, "path": str(path)})
    return registry, modules


def task_command(
    *,
    registry_path: Path,
    model_key: str,
    thinking: str,
    mode: str,
    comparison: str,
    method: str | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts/run_rl_anticipation_preferences.py"),
        "rl-preferences" if mode == "preference" else "rl-preference-anticipation",
        "--comparison",
        comparison,
        "--model_key",
        model_key,
        "--level",
        "all",
        "--thinking",
        thinking,
        "--models-config-path",
        str(registry_path),
        "--no-timestamp",
    ]
    if mode == "forecast":
        command.extend(["--anticipated-training-method", str(method)])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=("4b", "9b"), default="4b")
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--port", type=int, default=8102)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, default=None)
    parser.add_argument("--command-timeout", type=int, default=1800)
    parser.add_argument("--startup-timeout", type=int, default=1200)
    parser.add_argument("--rerun-completed", action="store_true")
    args = parser.parse_args()

    require_available_port(args.port)
    adapters = discover_adapter_paths(args.checkpoint_root.resolve())
    adapters[("sft", 125)] = discover_sft_adapter(args.checkpoint_root.resolve())
    cohort_manifest_sha256 = validate_final_adapter_provenance(adapters)
    if not (args.base_model / "config.json").is_file():
        raise ValueError(f"Invalid base model: {args.base_model}")
    base_url = f"http://127.0.0.1:{args.port}/v1"
    comparison = "rl-cartesian" if args.size == "4b" else "rl-cartesian-9b"
    if args.runtime_dir is None:
        args.runtime_dir = ROOT / f"results/rl-v4-{args.size}-cartesian/runtime"
    registry, modules = build_registry(args.size, base_url, adapters)
    args.runtime_dir.mkdir(parents=True, exist_ok=True)
    registry_path = args.runtime_dir / f"models-{args.size}-cartesian.yaml"
    registry_path.write_text(yaml.safe_dump(registry, sort_keys=False))
    (args.runtime_dir / f"loras-{args.size}-cartesian.json").write_text(
        json.dumps(
            {
                "base_model": str(args.base_model.resolve()),
                "cohort_manifest_sha256": cohort_manifest_sha256,
                "actual_training_methods": list(ACTUAL_TRAINING_METHODS),
                "preference_methods": list(PREFERENCE_METHODS),
                "modules": modules,
            },
            indent=2,
        )
        + "\n"
    )

    key_path = ROOT / "api_keys/api_key_vllm_endpoint.txt"
    if not key_path.is_file() or not key_path.read_text().strip():
        raise ValueError(f"Missing vLLM API key: {key_path}")
    api_key = key_path.read_text().strip()
    server_command = [
        "vllm", "serve", str(args.base_model.resolve()),
        "--served-model-name", f"qwen35-{args.size}-m0-v4",
        "--host", "127.0.0.1", "--port", str(args.port),
        "--api-key", api_key, "--dtype", "bfloat16",
        "--max-model-len", "8192", "--max-num-seqs", "96",
        "--gpu-memory-utilization", "0.90", "--language-model-only",
        "--reasoning-parser", "qwen3", "--enable-lora", "--max-lora-rank", "16",
        "--max-loras", "1", "--max-cpu-loras", str(len(modules)),
        "--lora-modules", *(f"{item['name']}={item['path']}" for item in modules),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "PYTHONPATH": str(ROOT),
            "PYTHONUNBUFFERED": "1",
        }
    )
    log_path = args.runtime_dir / "vllm.log"
    with log_path.open("a") as server_log:
        server_log.write(f"\n=== start {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ===\n")
        server_log.flush()
        server = subprocess.Popen(
            server_command,
            cwd=ROOT,
            env=environment,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            wait_for_server(base_url, api_key, server, args.startup_timeout)
            tasks = []
            model_keys = [f"qwen35-{args.size}-m0-v4-aws"] + [
                f"qwen35-{args.size}-rl-{method}-s0-step125-aws"
                for method in ACTUAL_TRAINING_METHODS
            ]
            for model_key in model_keys:
                for thinking, variant in VARIANTS:
                    tasks.append((model_key, thinking, variant, "preference", None))
            for method in ACTUAL_TRAINING_METHODS:
                for thinking, variant in VARIANTS:
                    tasks.append(
                        (
                            f"qwen35-{args.size}-m0-v4-aws",
                            thinking,
                            variant,
                            "forecast",
                            method,
                        )
                    )

            state_path = args.runtime_dir / "task_log.jsonl"
            for index, (model_key, thinking, variant, mode, method) in enumerate(tasks, 1):
                label = f"{mode}:{model_key}:{method or '-'}:{variant}"
                if not args.rerun_completed and complete(model_key, mode, method, variant):
                    print(f"[{index}/{len(tasks)}] skip complete {label}")
                    continue
                print(f"[{index}/{len(tasks)}] run {label}")
                started = time.time()
                command = task_command(
                    registry_path=registry_path,
                    model_key=model_key,
                    thinking=thinking,
                    mode=mode,
                    comparison=comparison,
                    method=method,
                )
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=environment,
                    timeout=args.command_timeout,
                    check=False,
                )
                with state_path.open("a") as state:
                    state.write(
                        json.dumps(
                            {
                                "label": label,
                                "returncode": result.returncode,
                                "elapsed_seconds": time.time() - started,
                                "completed_at": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                            }
                        )
                        + "\n"
                    )
                if result.returncode:
                    raise subprocess.CalledProcessError(result.returncode, command)
                if not complete(model_key, mode, method, variant):
                    raise RuntimeError(f"Task returned success but output is incomplete: {label}")
            print(
                f"All {args.size.upper()} Cartesian preference and "
                "prospective-anticipation tasks complete"
            )
        finally:
            terminate_process(server)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(f"ERROR: {exc}") from None
