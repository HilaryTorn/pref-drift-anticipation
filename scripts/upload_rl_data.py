#!/usr/bin/env python3
"""Upload a prepared RL data directory to a Hugging Face dataset repository."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path
from typing import Any


LOCAL_ARCHIVE_MARKER = ".formal_hub_archive_complete"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_local_files(local_dir: Path, allow_patterns: list[str]) -> dict[str, Path]:
    selected: dict[str, Path] = {}
    for path in sorted(local_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(local_dir).as_posix()
        if relative == LOCAL_ARCHIVE_MARKER:
            continue
        if allow_patterns and not any(
            fnmatch.fnmatch(relative, pattern) for pattern in allow_patterns
        ):
            continue
        selected[relative] = path
    if not selected:
        raise ValueError(f"no uploadable files selected under {local_dir}")
    return selected


def verify_remote_files(
    api: Any,
    *,
    repo_id: str,
    revision: str,
    prefix: str,
    local_files: dict[str, Path],
) -> dict[str, int | str]:
    remote = {
        item.path: item
        for item in api.list_repo_tree(
            repo_id,
            path_in_repo=prefix,
            recursive=True,
            expand=True,
            revision=revision,
            repo_type="dataset",
        )
        if getattr(item, "size", None) is not None
    }
    verified_bytes = 0
    for relative, local_path in local_files.items():
        remote_path = f"{prefix}/{relative}"
        item = remote.get(remote_path)
        if item is None:
            raise RuntimeError(
                f"Hub commit {revision} is missing uploaded file: {remote_path}"
            )
        local_size = local_path.stat().st_size
        if item.size != local_size:
            raise RuntimeError(
                f"Hub size mismatch for {remote_path}: local={local_size}, remote={item.size}"
            )
        if item.lfs is not None:
            remote_sha256 = item.lfs.sha256
            local_sha256 = sha256_file(local_path)
            if remote_sha256 != local_sha256:
                raise RuntimeError(
                    f"Hub SHA-256 mismatch for {remote_path}: "
                    f"local={local_sha256}, remote={remote_sha256}"
                )
        elif item.blob_id:
            local_blob_id = git_blob_sha1(local_path)
            if item.blob_id != local_blob_id:
                raise RuntimeError(
                    f"Hub Git blob mismatch for {remote_path}: "
                    f"local={local_blob_id}, remote={item.blob_id}"
                )
        else:
            raise RuntimeError(f"Hub returned no verifiable hash for {remote_path}")
        verified_bytes += local_size
    return {
        "commit_oid": revision,
        "verified_files": len(local_files),
        "verified_bytes": verified_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--repo-id", required=True, help="Hugging Face dataset repo, e.g. org/ai-pref-drift-rl-data")
    parser.add_argument("--path-in-repo", required=True, help="Model/version namespace inside the dataset repo")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--token-env", default="HUGGINGFACE_HUB_TOKEN")
    parser.add_argument("--commit-message", default=None)
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=[],
        help="Optional upload_folder allow pattern; repeatable",
    )
    args = parser.parse_args()

    local_dir = Path(args.local_dir).expanduser().resolve()
    if not local_dir.is_dir():
        raise SystemExit(f"Local directory does not exist: {local_dir}")
    prefix = args.path_in_repo.strip("/")
    if not prefix:
        raise SystemExit("--path-in-repo must be non-empty so model sweeps cannot overwrite each other")
    try:
        local_files = selected_local_files(local_dir, args.allow_pattern)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    try:
        from huggingface_hub import get_token
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required for upload") from exc
    token = os.environ.get(args.token_env) or os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise SystemExit(f"Set {args.token_env} or HF_TOKEN, or run `hf auth login`, before uploading")

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required for upload") from exc

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    result = api.upload_folder(
        folder_path=str(local_dir),
        path_in_repo=prefix,
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message=args.commit_message or f"Archive RL data: {prefix}",
        allow_patterns=args.allow_pattern or None,
        ignore_patterns=[LOCAL_ARCHIVE_MARKER],
    )
    commit_oid = str(getattr(result, "oid", "") or "")
    if not commit_oid:
        raise SystemExit("Hugging Face upload returned no immutable commit oid")
    verification = verify_remote_files(
        api,
        repo_id=args.repo_id,
        revision=commit_oid,
        prefix=prefix,
        local_files=local_files,
    )
    print(
        json.dumps(
            {
                "uploaded": str(local_dir),
                "destination": f"hf://datasets/{args.repo_id}/{prefix}",
                **verification,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
