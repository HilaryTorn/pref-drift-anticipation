#!/usr/bin/env python3
"""Preflight checks for the Qwen3.5-only training/scoring stack.

Run this before starting a GPU instance. It catches the expensive failure mode:
the repo accidentally points SFT at a non-Qwen3.5 family, or the local training
environment has a Transformers build that cannot load Qwen3.5 configs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_MODELS = [
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
]
CURRENT_M0_REVISIONS = {
    "prism-drift/qwen35-4b-m0-v4": "dca63300371b0265bd7099f87b9eed55937bea22",
    "prism-drift/qwen35-9b-m0-v4": "8f3d499236d7b9d38a1f616f6b3241b22a381047",
}
FORBIDDEN_MARKERS = ("qwen2.5", "qwen25", "2.5-coder")


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def contains_forbidden(value: Any) -> bool:
    return isinstance(value, str) and any(marker in value.lower() for marker in FORBIDDEN_MARKERS)


def walk_values(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for nested in value.values():
            found.extend(walk_values(nested))
    elif isinstance(value, list):
        for nested in value:
            found.extend(walk_values(nested))
    elif contains_forbidden(value):
        found.append(value)
    return found


def check_spec() -> None:
    spec_path = ROOT / "data" / "training_specs" / "coding_axis_sft.json"
    spec = json.loads(spec_path.read_text())
    if spec.get("base_model_family") != "Qwen3.5":
        fail(f"{spec_path} base_model_family must be 'Qwen3.5'")
    if spec.get("default_base_model") not in CANONICAL_MODELS:
        fail(f"{spec_path} default_base_model must be one of {CANONICAL_MODELS}")
    forbidden = walk_values(spec)
    if forbidden:
        fail(f"{spec_path} contains forbidden non-Qwen3.5 values: {forbidden}")


def check_config() -> None:
    config_path = ROOT / "config.yaml"
    config = yaml.safe_load(config_path.read_text()) or {}
    forbidden_keys = [key for key in config if isinstance(key, str) and contains_forbidden(key)]
    forbidden_values = walk_values(config)
    if forbidden_keys or forbidden_values:
        fail(
            f"{config_path} contains forbidden non-Qwen3.5 keys/values: "
            f"keys={forbidden_keys}, values={forbidden_values}"
        )
    # Stable endpoints only -- base + M0 per size, plus the local debug model. The per-arm x
    # per-step trajectory keys are deliberately NOT enumerated here: that list re-encoded the
    # checkpoint schedule and silently went stale whenever it changed (it stayed pinned to the
    # old 1/4/8 n=64 steps for weeks while the config moved to 4/12/40/125). A missing
    # trajectory key surfaces clearly at `main.py --model_key <key>` time, so preflight sticks
    # to its real job: "will the Qwen3.5 stack load". Base + M0 rarely change, so they stay.
    required = [
        "qwen35-08b-base-aws",
        "qwen35-4b-base-aws",
        "qwen35-9b-base-aws",
        "qwen35-4b-m0-v4-aws",
        "qwen35-9b-m0-v4-aws",
        "qwen35-08b-local",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        fail(f"{config_path} is missing required Qwen3.5 base/M0 model keys: {missing}")


def check_hf_ids() -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        fail(f"huggingface_hub is not installed: {exc}")
    api = HfApi()
    for model_id in CANONICAL_MODELS:
        api.model_info(model_id)
    for model_id, expected_revision in CURRENT_M0_REVISIONS.items():
        observed_revision = api.model_info(model_id).sha
        if observed_revision != expected_revision:
            fail(
                f"{model_id} resolved to {observed_revision}, expected "
                f"{expected_revision}"
            )
    for invalid_id in [f"{model_id}-Instruct" for model_id in CANONICAL_MODELS]:
        try:
            api.model_info(invalid_id)
        except Exception:
            continue
        fail(f"Unexpectedly found {invalid_id}; update the preflight if this repo becomes real.")


def check_transformers(load_weights: bool) -> None:
    try:
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        fail(f"Transformers stack is not installed: {exc}")

    print(f"transformers={transformers.__version__}")
    for model_id in CANONICAL_MODELS:
        try:
            cfg = AutoConfig.from_pretrained(model_id)
        except Exception as exc:
            fail(
                f"Transformers cannot load {model_id}. Install the pinned SFT "
                f"stack from requirements-sft.txt/pyproject.toml before using a "
                f"GPU. Original error: {type(exc).__name__}: {exc}"
            )
        if getattr(cfg, "model_type", None) != "qwen3_5":
            fail(f"{model_id} loaded as model_type={getattr(cfg, 'model_type', None)!r}")
        try:
            mapping = AutoModelForCausalLM._model_mapping[type(cfg)]
        except Exception as exc:
            fail(f"AutoModelForCausalLM cannot resolve {model_id}: {type(exc).__name__}: {exc}")
        if "Qwen3_5" not in getattr(mapping, "__name__", str(mapping)):
            fail(f"{model_id} AutoModelForCausalLM maps to {mapping}, expected Qwen3_5")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if not getattr(tokenizer, "chat_template", None):
            fail(f"{model_id} tokenizer has no chat_template")
        print(f"OK {model_id}: {cfg.__class__.__name__} -> {mapping.__name__}")

    if load_weights:
        model = AutoModelForCausalLM.from_pretrained(CANONICAL_MODELS[0], dtype="auto")
        print(f"OK loaded weights for {CANONICAL_MODELS[0]}: {model.__class__.__name__}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--load_weights",
        action="store_true",
        help="Also load Qwen/Qwen3.5-0.8B weights. This is slower and needs enough RAM.",
    )
    args = parser.parse_args()

    check_spec()
    check_config()
    check_hf_ids()
    check_transformers(load_weights=args.load_weights)
    print("Qwen3.5 preflight passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
