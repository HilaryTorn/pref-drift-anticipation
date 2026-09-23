#!/usr/bin/env python3
"""Prepare a seeded SFT train subset from an intervention candidate pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from validate_sft_data import elicitation_snippets, validate_dataset_file  # noqa: E402


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def find_intervention(spec: dict[str, Any], intervention_id: str) -> dict[str, Any]:
    for intervention in spec.get("interventions", []):
        if intervention.get("intervention_id") == intervention_id:
            return intervention
    raise SystemExit(f"Unknown intervention_id {intervention_id!r}")


def resolve_hub_repo_id(raw_repo_id: str, spec: dict[str, Any], base_model: str | None, intervention_id: str) -> str:
    """Substitute the {base_model_slug}/{intervention_id} placeholders in a hub repo id.

    Mirrors train_sft_lora.py so the dataset repo follows the run the same way the adapter
    repo does: the slug is derived from the base model (Qwen/Qwen3.5-0.8B -> qwen35-08b), so a
    0.8B dataset can never land in a 4B repo even if someone overrides --base_model. Resolving
    here (not at push time) makes a misconfigured placeholder fail before any file is written.
    """
    repo_id = raw_repo_id
    if "{base_model_slug}" in repo_id:
        model = base_model or spec.get("default_base_model")
        if not model:
            raise SystemExit(
                "--hub_repo_id contains {base_model_slug} but no --base_model was passed and the "
                "spec has no default_base_model to derive it from."
            )
        slug = model.split("/")[-1].lower().replace(".", "")
        repo_id = repo_id.replace("{base_model_slug}", slug)
    return repo_id.replace("{intervention_id}", intervention_id)


def push_dataset_to_hub(
    output_dir: Path,
    repo_id: str,
    token: str | None,
    *,
    path_prefix: str = "",
    private: bool = False,
) -> None:
    # path_prefix namespaces the dataset next to its arm (<intervention_id>/train.jsonl ...),
    # mirroring the adapter repo's per-intervention folders. At the repo root every arm would
    # write the same three filenames, so preparing Rust after Java would silently overwrite
    # Java's train/validation/manifest. This goes to a repo_type="dataset" repo, kept separate
    # from the adapter *model* repo, per the HuggingFace model/dataset split.
    prefix = path_prefix.strip("/")
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("WARNING: huggingface_hub unavailable; could not upload SFT dataset.", file=sys.stderr)
        return
    try:
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
        api.upload_folder(
            folder_path=str(output_dir),
            path_in_repo=prefix or ".",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Add SFT dataset ({prefix or 'root'})",
        )
    except Exception as exc:
        # The dataset is already written locally and is deterministically regenerable from the
        # committed pool + seed, so a failed archive push must never fail an otherwise-good prep.
        # In the --sft sequence train_sft_lora then hard-fails its own token preflight before any
        # GPU time, so a bad token surfaces loudly without this needing to be fatal.
        print(
            f"WARNING: could not upload SFT dataset to {repo_id} (dataset is still at {output_dir}, "
            f"and is reproducible from the committed pool + seed): {exc}",
            file=sys.stderr,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec_path", required=True)
    parser.add_argument("--intervention_id", required=True)
    parser.add_argument("--sample_count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--base_model",
        default=None,
        help="HF model id used only to fill {base_model_slug} in --hub_repo_id. Defaults to spec.default_base_model.",
    )
    parser.add_argument(
        "--hub_repo_id",
        default=None,
        help="HF dataset repo id for archiving the prepared set, e.g. org/name. Supports {base_model_slug}/{intervention_id}.",
    )
    parser.add_argument("--hub_private", action="store_true")
    parser.add_argument("--hub_token_env", default="HUGGINGFACE_HUB_TOKEN")
    args = parser.parse_args()

    spec_path = Path(args.spec_path)
    spec = load_json(spec_path)
    if args.sample_count not in spec.get("sample_count_grid", []):
        raise SystemExit(
            f"--sample_count must be one of {spec.get('sample_count_grid')}, got {args.sample_count}"
        )

    intervention = find_intervention(spec, args.intervention_id)
    # Resolve the archive repo id before any work so a bad {base_model_slug} placeholder fails
    # here rather than after the dataset is written.
    resolved_hub_repo_id = (
        resolve_hub_repo_id(args.hub_repo_id, spec, args.base_model, args.intervention_id)
        if args.hub_repo_id
        else None
    )
    pool_path = ROOT / intervention["candidate_pool_path"]
    validation_path = ROOT / intervention["validation_path"]
    if not pool_path.exists():
        raise SystemExit(f"Candidate pool does not exist: {pool_path}")
    if not validation_path.exists():
        raise SystemExit(f"Validation file does not exist: {validation_path}")

    banned_snippets = elicitation_snippets()
    pool_errors, _ = validate_dataset_file(pool_path, intervention, "pool", banned_snippets)
    validation_errors, _ = validate_dataset_file(validation_path, intervention, "validation", banned_snippets)
    if pool_errors or validation_errors:
        for error in pool_errors + validation_errors:
            print(f"ERROR: {error}")
        return 1

    pool = load_jsonl(pool_path)
    if len(pool) < args.sample_count:
        raise SystemExit(
            f"Candidate pool has {len(pool)} examples, cannot sample {args.sample_count}"
        )

    rng = random.Random(args.seed)
    selected = []
    for record in rng.sample(pool, args.sample_count):
        train_record = dict(record)
        train_record["split"] = "train"
        selected.append(train_record)
    selected_ids = {record["id"] for record in selected}
    output_dir = Path(args.output_dir)
    train_out = output_dir / "train.jsonl"
    validation_out = output_dir / "validation.jsonl"
    manifest_out = output_dir / "manifest.json"

    write_jsonl(train_out, selected)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(validation_path, validation_out)

    train_errors, _ = validate_dataset_file(train_out, intervention, "train", banned_snippets)
    prepared_validation_errors, _ = validate_dataset_file(
        validation_out,
        intervention,
        "validation",
        banned_snippets,
    )
    if train_errors or prepared_validation_errors:
        for error in train_errors + prepared_validation_errors:
            print(f"ERROR: {error}")
        return 1

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "git_commit": git_commit(),
        "spec_path": str(spec_path),
        "spec_id": spec.get("id"),
        "intervention_id": args.intervention_id,
        "task": intervention.get("task"),
        "language_id": intervention.get("language_id"),
        "sample_count": args.sample_count,
        "seed": args.seed,
        "candidate_pool_path": str(pool_path),
        "validation_source_path": str(validation_path),
        "train_path": str(train_out),
        "validation_path": str(validation_out),
        "candidate_pool_sha256": file_sha256(pool_path),
        "validation_source_sha256": file_sha256(validation_path),
        "train_sha256": file_sha256(train_out),
        "validation_sha256": file_sha256(validation_out),
        "selected_example_ids": sorted(selected_ids),
    }
    manifest_out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote SFT dataset to {output_dir}")
    print(f"train examples: {len(selected)}")
    print(f"manifest: {manifest_out}")

    if resolved_hub_repo_id:
        token = os.environ.get(args.hub_token_env) or os.environ.get("HF_TOKEN")
        push_dataset_to_hub(
            output_dir,
            resolved_hub_repo_id,
            token,
            path_prefix=args.intervention_id,
            private=args.hub_private,
        )
        print(f"Hugging Face dataset archive target: {resolved_hub_repo_id}/{args.intervention_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
