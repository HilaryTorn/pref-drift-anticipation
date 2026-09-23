#!/usr/bin/env python3
"""Upload locally cached checkpoints from one completed formal RL run.

This is the catch-up counterpart to the live Trainer callback. It reuses the
same model-only upload, immutable-commit verification, and safe local-pruning
logic. Existing local-fallback markers are only cleared with an explicit flag.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from scripts.rl_training.hub_checkpoints import (
    LOCAL_FALLBACK_MARKER,
    FormalHubCheckpointCallback,
    formal_hub_path_prefix,
    resolve_hub_token,
    validate_formal_hub_settings,
)


EXPECTED_STEPS = list(range(4, 125, 4)) + [125]
CHECKPOINT_ADAPTER_RE = re.compile(r"/checkpoint-(\d+)/adapter_model\.safetensors$")


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"missing required JSON file: {path}")
    return json.loads(path.read_text())


def validate_completed_run(
    run_dir: Path,
    cohort_manifest: Path,
    *,
    algo: str,
    seed: int,
    base_revision: str,
) -> dict:
    metadata = load_json(run_dir / "run_metadata.json")
    if metadata.get("schema") != "rl_training_run_v1":
        raise ValueError("run metadata is not rl_training_run_v1")
    if metadata.get("status") != "complete" or metadata.get("global_step") != 125:
        raise ValueError("catch-up upload requires a complete step-125 formal run")
    if metadata.get("algo") != algo or metadata.get("seed") != seed:
        raise ValueError("run algorithm/seed does not match the upload request")
    if metadata.get("trainer_config", {}).get("save_steps") != 4:
        raise ValueError("formal run does not prove the required every-4-step saves")
    if base_revision not in str(metadata.get("base_model", "")):
        raise ValueError("run base model does not contain the requested immutable revision")

    cohort = load_json(cohort_manifest)
    output = (
        cohort.get("outputs", {}).get("artifacts", {}).get("dpo", {})
        if algo == "dpo"
        else cohort.get("outputs", {}).get("train", {})
    )
    if metadata.get("dataset_sha256") != output.get("sha256"):
        raise ValueError("run dataset hash does not match the frozen cohort")
    return metadata


def uploaded_steps(run_dir: Path, *, repo_id: str, path_prefix: str) -> set[int]:
    found: set[int] = set()
    log_path = run_dir / "hub_upload_log.jsonl"
    if not log_path.is_file():
        return found
    for line in log_path.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            event.get("schema") == "formal_hub_upload_event_v1"
            and event.get("repo_id") == repo_id
            and str(event.get("path_in_repo", "")).startswith(
                f"{path_prefix}/checkpoint-"
            )
            and isinstance(event.get("step"), int)
        ):
            found.add(event["step"])
    return found


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def remote_files(api, *, repo_id: str, path_prefix: str) -> dict:
    from huggingface_hub.errors import RemoteEntryNotFoundError

    try:
        items = api.list_repo_tree(
            repo_id,
            path_in_repo=path_prefix,
            recursive=True,
            expand=True,
            repo_type="model",
        )
        return {
            item.path: item
            for item in items
            if getattr(item, "size", None) is not None
        }
    except RemoteEntryNotFoundError:
        # A clean destination legitimately has no algorithm/seed prefix yet.
        return {}


def remote_adapter_steps(files: dict[str, object], *, path_prefix: str) -> set[int]:
    steps = set()
    for path in files:
        if not path.startswith(f"{path_prefix}/checkpoint-"):
            continue
        match = CHECKPOINT_ADAPTER_RE.search(path)
        if match and f"{path_prefix}/checkpoint-{match.group(1)}/adapter_config.json" in files:
            steps.add(int(match.group(1)))
    return steps


def verify_no_overlay_mismatch(
    run_dir: Path, files: dict[str, object], *, path_prefix: str
) -> None:
    for checkpoint in run_dir.glob("checkpoint-*"):
        if not checkpoint.is_dir():
            continue
        try:
            step = int(checkpoint.name.split("-", 1)[1])
        except ValueError:
            continue
        local_adapter = checkpoint / "adapter_model.safetensors"
        remote_path = f"{path_prefix}/checkpoint-{step}/adapter_model.safetensors"
        remote = files.get(remote_path)
        if not local_adapter.is_file() or remote is None:
            continue
        remote_hash = remote.lfs.sha256 if remote.lfs is not None else remote.blob_id
        local_hash = sha256_file(local_adapter)
        if local_hash != remote_hash:
            raise ValueError(
                f"refusing to overlay a different checkpoint at {remote_path}: "
                f"local={local_hash}, remote={remote_hash}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--algo", choices=["dpo", "grpo", "ppo"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-model-id", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--token-env", default="HUGGINGFACE_HUB_TOKEN")
    parser.add_argument("--public", action="store_true")
    parser.add_argument(
        "--retry-local-fallback",
        action="store_true",
        help=f"Remove {LOCAL_FALLBACK_MARKER} and retry the Hub before uploading",
    )
    parser.add_argument(
        "--keep-local-checkpoints",
        type=int,
        default=0,
        help="Keep this many verified checkpoint directories after upload (default: 0)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    cohort_manifest = args.cohort_manifest.resolve()
    validate_completed_run(
        run_dir,
        cohort_manifest,
        algo=args.algo,
        seed=args.seed,
        base_revision=args.base_revision,
    )
    cohort_hash = validate_formal_hub_settings(
        output_dir=run_dir,
        repo_id=args.repo_id,
        base_model_id=args.base_model_id,
        base_revision=args.base_revision,
        cohort_manifest=cohort_manifest,
        algo=args.algo,
        seed=args.seed,
    )
    path_prefix = formal_hub_path_prefix(
        base_model_id=args.base_model_id,
        base_revision=args.base_revision,
        cohort_sha256=cohort_hash,
        algo=args.algo,
        seed=args.seed,
    )

    token = resolve_hub_token(args.token_env)
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=not args.public,
        exist_ok=True,
    )
    info = api.repo_info(args.repo_id, repo_type="model")
    if args.public and getattr(info, "private", True):
        raise ValueError(
            f"{args.repo_id} is still private; change visibility only after explicit "
            "public-release approval, then rerun"
        )

    before = remote_files(api, repo_id=args.repo_id, path_prefix=path_prefix)
    verify_no_overlay_mismatch(run_dir, before, path_prefix=path_prefix)

    marker = run_dir / LOCAL_FALLBACK_MARKER
    if marker.is_file():
        if not args.retry_local_fallback:
            raise ValueError(
                f"{marker} exists; pass --retry-local-fallback only after the remote "
                "quota/visibility issue is resolved"
            )
        marker.unlink()

    callback = FormalHubCheckpointCallback(
        repo_id=args.repo_id,
        path_prefix=path_prefix,
        algo=args.algo,
        seed=args.seed,
        base_model_id=args.base_model_id,
        base_revision=args.base_revision,
        cohort_sha256=cohort_hash,
        tokenizer=None,
        token=token,
        private=not args.public,
        keep_local_checkpoints=args.keep_local_checkpoints,
        api=api,
    )
    callback._upload_pending(run_dir)
    if marker.is_file():
        raise RuntimeError(marker.read_text().strip())

    after = remote_files(api, repo_id=args.repo_id, path_prefix=path_prefix)
    covered = uploaded_steps(run_dir, repo_id=args.repo_id, path_prefix=path_prefix)
    covered.update(remote_adapter_steps(after, path_prefix=path_prefix))
    training_state = [
        path
        for path in after
        if path.rsplit("/", 1)[-1].startswith(
            ("optimizer", "scheduler", "scaler", "rng_state")
        )
    ]
    if training_state:
        raise RuntimeError(
            f"model-only destination unexpectedly contains {len(training_state)} "
            f"training-state files; first: {training_state[0]}"
        )
    missing = sorted(set(EXPECTED_STEPS) - covered)
    if missing:
        raise RuntimeError(
            f"Hub catch-up is incomplete for {args.algo}; missing steps: {missing}"
        )
    callback._prune(run_dir, args.keep_local_checkpoints)
    print(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "path_prefix": path_prefix,
                "algorithm": args.algo,
                "verified_steps": EXPECTED_STEPS,
                "local_checkpoints_kept": args.keep_local_checkpoints,
                "optimizer_state_uploaded": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(f"ERROR: {exc}") from None
