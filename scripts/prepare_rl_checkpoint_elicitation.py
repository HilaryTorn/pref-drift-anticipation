#!/usr/bin/env python3
"""Validate a downloaded formal adapter tree and prepare vLLM inputs.

The generated YAML is transient experiment state, not a replacement for the
hand-maintained repository registry.  The JSON LoRA manifest can be consumed by
the launcher without shell-quoting adapter paths by reading its ``modules``
array and passing each ``name=path`` item to ``vllm serve --lora-modules``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


STEPS = (*range(4, 125, 4), 125)
ALGOS = ("dpo", "grpo", "ppo")


def discover_adapter_paths(checkpoint_root: Path) -> dict[tuple[str, int], Path]:
    found: dict[tuple[str, int], Path] = {}
    for algo in ALGOS:
        candidates = list(checkpoint_root.glob(f"base-*/cohort-*/{algo}/seed-0"))
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one {algo}/seed-0 tree below {checkpoint_root}, "
                f"found {len(candidates)}"
            )
        seed_dir = candidates[0]
        actual_steps = set()
        for path in seed_dir.glob("checkpoint-*"):
            suffix = path.name.removeprefix("checkpoint-")
            if path.is_dir() and suffix.isdigit():
                step = int(suffix)
                actual_steps.add(step)
                for required in ("adapter_config.json", "adapter_model.safetensors"):
                    if not (path / required).is_file():
                        raise ValueError(f"Missing {required}: {path}")
                found[(algo, step)] = path.resolve()
        if actual_steps != set(STEPS):
            raise ValueError(
                f"{algo}: expected steps {list(STEPS)}, found {sorted(actual_steps)}"
            )
    return found


def build_registry(size: str, base_url: str, adapters: dict) -> tuple[dict, list[dict]]:
    registry = {}
    modules = []
    baseline_key = f"qwen35-{size}-m0-v4-aws"
    baseline_alias = f"qwen35-{size}-m0-v4"
    registry[baseline_key] = {
        "model_name": baseline_alias,
        "model_type": "vllm_endpoint",
        "base_url": base_url,
        "gpu_count": 1,
    }
    for algo in ALGOS:
        for step in STEPS:
            model_key = f"qwen35-{size}-rl-{algo}-s0-step{step}-aws"
            alias = f"{algo}{'-9b' if size == '9b' else ''}-s0-step{step}"
            path = adapters[(algo, step)]
            registry[model_key] = {
                "model_name": alias,
                "model_type": "vllm_endpoint",
                "base_url": base_url,
                "gpu_count": 1,
                "drift_algo": algo,
                "drift_seed": 0,
                "drift_step": step,
            }
            modules.append({"name": alias, "path": str(path)})
    return registry, modules


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=["4b", "9b"], required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--registry-output", type=Path, required=True)
    parser.add_argument("--lora-manifest-output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint_root = args.checkpoint_root.resolve()
    base_model = args.base_model.resolve()
    if not (base_model / "config.json").is_file():
        raise ValueError(f"Invalid base-model snapshot: {base_model}")
    adapters = discover_adapter_paths(checkpoint_root)
    registry, modules = build_registry(args.size, args.base_url, adapters)
    payload = {
        "schema": "rl_checkpoint_vllm_lora_manifest_v1",
        "model_size": args.size,
        "base_model": str(base_model),
        "baseline_alias": f"qwen35-{args.size}-m0-v4",
        "checkpoint_steps": list(STEPS),
        "algorithms": list(ALGOS),
        "modules": modules,
    }
    args.registry_output.parent.mkdir(parents=True, exist_ok=True)
    args.lora_manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.registry_output.write_text(yaml.safe_dump(registry, sort_keys=False))
    args.lora_manifest_output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"Validated {len(modules)} adapters; wrote {len(registry)} model entries "
        f"to {args.registry_output}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from None
