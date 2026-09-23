#!/usr/bin/env python3
"""Validate, stage, and upload formal LoRA trajectories to the Hugging Face Hub."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


ADAPTER_FILES = (
    "adapter_model.safetensors",
    "adapter_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"missing required file: {path}")
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_steps(raw: str) -> list[int]:
    steps = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not steps or len(steps) != len(set(steps)):
        raise ValueError("--steps must contain unique comma-separated integers")
    return steps


def validate_formal_run(
    run_dir: Path,
    cohort_manifest_path: Path,
    *,
    algo: str,
    seed: int,
    base_revision: str,
    steps: list[int],
) -> tuple[dict, dict]:
    resolved = run_dir.resolve()
    if str(resolved).startswith("/tmp/") or "smoke" in str(resolved).lower():
        raise ValueError(f"refusing to upload smoke or temporary run: {resolved}")
    metadata = load_json(run_dir / "run_metadata.json")
    if metadata.get("schema") != "rl_training_run_v1":
        raise ValueError("run metadata is not rl_training_run_v1")
    if metadata.get("status") != "complete" or metadata.get("global_step") != 125:
        raise ValueError("formal run must be complete at global step 125")
    if metadata.get("algo") != algo or metadata.get("seed") != seed:
        raise ValueError("run algorithm/seed does not match upload arguments")
    if not metadata.get("lora", {}).get("enabled"):
        raise ValueError("run is not a LoRA run")
    if base_revision not in str(metadata.get("base_model", "")):
        raise ValueError("run base model does not contain the expected immutable revision")

    cohort = load_json(cohort_manifest_path)
    selection = cohort.get("selection", {})
    if (
        cohort.get("schema") != "shared_training_cohort_v1"
        or cohort.get("status") != "frozen"
        or selection.get("selected_size") != 1000
        or selection.get("seed") != seed
    ):
        raise ValueError("cohort must be a frozen shared N=1000, matching-seed cohort")
    output = (
        cohort.get("outputs", {}).get("artifacts", {}).get("dpo", {})
        if algo == "dpo"
        else cohort.get("outputs", {}).get("train", {})
    )
    if metadata.get("dataset_sha256") != output.get("sha256"):
        raise ValueError("run dataset hash does not match the frozen cohort")

    for step in steps:
        checkpoint = run_dir / f"checkpoint-{step}"
        for name in ADAPTER_FILES[:2]:
            if not (checkpoint / name).is_file():
                raise ValueError(f"checkpoint-{step} is missing {name}")

    if algo == "grpo":
        config = metadata.get("trainer_config", {})
        if (
            config.get("expected_unique_prompts") != 1000
            or config.get("num_generations") != 8
            or config.get("unique_prompts_per_step") != 8
        ):
            raise ValueError("GRPO metadata does not prove the corrected 1,000-prompt budget")
        rollout_path = run_dir / "rollout_samples.rank0.jsonl"
        counts = Counter()
        rows = 0
        with rollout_path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                counts[record["record_id"]] += 1
                rows += 1
        if rows != 8000 or len(counts) != 1000 or set(counts.values()) != {8}:
            raise ValueError(
                f"GRPO rollout coverage is invalid: rows={rows}, unique={len(counts)}, "
                f"per_id={sorted(set(counts.values()))}"
            )
    return metadata, cohort


def copy_adapter(
    source: Path,
    destination: Path,
    *,
    base_model_id: str,
    base_revision: str,
    algo: str,
    seed: int,
    step: int,
    cohort_sha256: str,
) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in ADAPTER_FILES:
        path = source / name
        if path.is_file():
            shutil.copy2(path, destination / name)
            copied.append(name)
    adapter_config_path = destination / "adapter_config.json"
    adapter_config = load_json(adapter_config_path)
    adapter_config["base_model_name_or_path"] = base_model_id
    adapter_config_path.write_text(json.dumps(adapter_config, indent=2) + "\n")
    provenance = {
        "schema": "formal_rl_adapter_checkpoint_v1",
        "algorithm": algo,
        "seed": seed,
        "optimizer_step": step,
        "base_model": base_model_id,
        "base_model_revision": base_revision,
        "cohort_manifest_sha256": cohort_sha256,
        "artifact_type": "peft_lora_adapter_for_inference_and_evaluation",
        "optimizer_state_uploaded": False,
        "files": {
            name: sha256_file(destination / name)
            for name in copied
        },
    }
    (destination / "adapter_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    return provenance


def stage_upload(
    staging: Path,
    run_dir: Path,
    cohort_manifest_path: Path,
    *,
    repo_id: str,
    base_model_id: str,
    base_revision: str,
    algo: str,
    seed: int,
    steps: list[int],
) -> dict:
    cohort_sha256 = sha256_file(cohort_manifest_path)
    checkpoints = {}
    for step in steps:
        checkpoints[str(step)] = copy_adapter(
            run_dir / f"checkpoint-{step}",
            staging / f"checkpoint-{step}",
            base_model_id=base_model_id,
            base_revision=base_revision,
            algo=algo,
            seed=seed,
            step=step,
            cohort_sha256=cohort_sha256,
        )
    final_step = max(steps)
    copy_adapter(
        run_dir if (run_dir / "adapter_model.safetensors").is_file() else run_dir / f"checkpoint-{final_step}",
        staging,
        base_model_id=base_model_id,
        base_revision=base_revision,
        algo=algo,
        seed=seed,
        step=final_step,
        cohort_sha256=cohort_sha256,
    )
    provenance_dir = staging / "provenance"
    provenance_dir.mkdir()
    shutil.copy2(run_dir / "run_metadata.json", provenance_dir / "run_metadata.json")
    shutil.copy2(cohort_manifest_path, provenance_dir / "cohort_manifest.json")
    reward_log = run_dir / "reward_log.jsonl"
    if reward_log.is_file():
        shutil.copy2(reward_log, provenance_dir / "reward_log.jsonl")

    manifest = {
        "schema": "formal_rl_adapter_upload_v1",
        "repo_id": repo_id,
        "algorithm": algo,
        "seed": seed,
        "base_model": base_model_id,
        "base_model_revision": base_revision,
        "cohort_size": 1000,
        "headline_steps": steps,
        "default_adapter_step": final_step,
        "checkpoints": checkpoints,
    }
    (staging / "upload_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    title = f"{base_model_id} — {algo.upper()} seed-{seed} LoRA adapters"
    (staging / "README.md").write_text(
        "---\n"
        f"base_model: {base_model_id}\n"
        "library_name: peft\n"
        "pipeline_tag: text-generation\n"
        "tags:\n  - lora\n  - preference-drift\n  - reinforcement-learning\n"
        "---\n\n"
        f"# {title}\n\n"
        f"Formal `{algo}` trajectory trained from `{base_model_id}` at revision "
        f"`{base_revision}` on a frozen 1,000-prompt cohort. The repo root is the "
        f"step-{final_step} adapter; headline trajectory adapters are available under "
        + ", ".join(f"`checkpoint-{step}`" for step in steps)
        + ".\n\nOnly inference/evaluation LoRA files are included. Optimizer states are intentionally "
        "excluded, so these artifacts are not training-resume checkpoints. See "
        "`upload_manifest.json` and `provenance/` for hashes and run metadata.\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--algo", choices=["dpo", "grpo", "ppo"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", default="4,12,40,125")
    parser.add_argument("--base-model-id", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    steps = parse_steps(args.steps)
    run_dir = args.run_dir.resolve()
    cohort_manifest = args.cohort_manifest.resolve()
    validate_formal_run(
        run_dir,
        cohort_manifest,
        algo=args.algo,
        seed=args.seed,
        base_revision=args.base_revision,
        steps=steps,
    )
    with tempfile.TemporaryDirectory(prefix=f"formal-{args.algo}-upload-") as tmp:
        staging = Path(tmp)
        manifest = stage_upload(
            staging,
            run_dir,
            cohort_manifest,
            repo_id=args.repo_id,
            base_model_id=args.base_model_id,
            base_revision=args.base_revision,
            algo=args.algo,
            seed=args.seed,
            steps=steps,
        )
        staged_bytes = sum(path.stat().st_size for path in staging.rglob("*") if path.is_file())
        print(
            json.dumps(
                {**manifest, "staged_bytes": staged_bytes, "dry_run": args.dry_run},
                indent=2,
            )
        )
        if args.dry_run:
            return
        from huggingface_hub import HfApi, get_token

        token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN") or get_token()
        if not token:
            raise SystemExit("No Hugging Face token found; run `hf auth login`")
        api = HfApi(token=token)
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="model",
            private=args.private,
            exist_ok=True,
        )
        result = api.upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=str(staging),
            path_in_repo=".",
            commit_message=f"Upload formal 4B {args.algo.upper()} LoRA trajectory",
        )
        print(f"uploaded: {result}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from None
