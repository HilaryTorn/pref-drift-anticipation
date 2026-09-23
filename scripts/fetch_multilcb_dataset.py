#!/usr/bin/env python3
"""Pre-stage the LiveCodeBench shards Multi-LCB needs, as plain local jsonl.

Why this exists, rather than letting upstream download from the Hub:

1. **It drops the `datasets==3.6.0` pin.** Upstream loads `livecodebench/code_generation_lite`,
   which is a remote-code *script* dataset -- support for which `datasets` 4.x removed. Staged as
   plain jsonl and pointed at with upstream's own `DATASET_PATH` env var, it loads fine under our
   `datasets==5.0.0`, so generation runs in the repo `venv/` with no conda env at all. Only
   *evaluation* then needs the toolchain env, and evaluation does not need a GPU.
2. **It downloads 128 MB instead of 4.5 GB.** `release_v6` pulls all six shards; the `v6` window
   this study scores is one 128 MB shard.
3. It is resumable and cacheable, so a re-launched spot instance does not re-download.

This does not touch the prompt, the test conversion or the comparator, so the vendored harness stays
byte-identical and its numbers stay on upstream's scale. Verified equivalent for `v6`: 175 problems,
correct columns, LeetCode functional tests converting to the expected stdin/stdout.

Usage:
    python3 scripts/fetch_multilcb_dataset.py                 # v6 -> data/reference/multilcb/v6/
    python3 scripts/fetch_multilcb_dataset.py --release release_v6
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_REPO = "livecodebench/code_generation_lite"
# Resolved from the Hub before the PHP/C# teacher run. Never download benchmark shards from a
# moving branch: a changed problem order would invalidate paid resume-cache custom_ids.
DEFAULT_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"

# Mirrors upstream's ALLOWED_FILES. A bare "vN" tag is only the shard that version ADDED;
# "release_vN" is cumulative. Ours defaults to v6 -- see docs/multilcb-eval.md.
SHARDS = {
    "v1": ["test.jsonl"],
    "v2": ["test2.jsonl"],
    "v3": ["test3.jsonl"],
    "v4": ["test4.jsonl"],
    "v5": ["test5.jsonl"],
    "v6": ["test6.jsonl"],
    # Everything BEFORE the v6 window (May 2023 -> 2025-01-03, ~880 problems, 4.35 GB). This is the
    # LCB SFT training window in docs/lcb-sft-runbook.md: disjoint from v6 by construction, so v6
    # stays the untouched held-out test and the existing M0 baseline on it stays valid.
    "release_v5": [f"test{n}.jsonl" for n in ["", "2", "3", "4", "5"]],
    "release_v6": [f"test{n}.jsonl" for n in ["", "2", "3", "4", "5", "6"]],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(release: str, dest_root: Path, force: bool, revision: str = DEFAULT_REVISION) -> Path:
    try:
        shards = SHARDS[release]
    except KeyError:
        raise SystemExit(f"Unknown release {release!r}; known: {', '.join(SHARDS)}")

    dest = dest_root / release
    dest.mkdir(parents=True, exist_ok=True)

    for index, shard in enumerate(shards):
        # `datasets` infers the split from the filename, and upstream asks for split="test", so the
        # shards have to be named test.jsonl / test2.jsonl / ... exactly as upstream ships them.
        target = dest / shard
        if target.exists() and not force:
            print(f"  {shard}: already present ({target.stat().st_size / 1e6:.0f} MB), skipping")
            continue
        url = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/{revision}/{shard}"
        print(f"  {shard}: downloading from {url} ...", end="", flush=True)
        # Send the token when one is in the environment. These shards are public, so this is not
        # about access -- it is about rate limits. A box that pulls a base model and several shards
        # in quick succession gets HTTP 429 as an anonymous client, which surfaces here as a bare
        # urllib HTTPError and kills a pod seconds after it starts (2026-09-12). Note this is plain
        # urllib rather than huggingface_hub, so exporting HF_TOKEN alone does nothing without
        # this header.
        request = urllib.request.Request(url)
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            request.add_header("Authorization", f"Bearer {hf_token}")
        with urllib.request.urlopen(request) as response, open(target, "wb") as handle:
            shutil.copyfileobj(response, handle)
        print(f" {target.stat().st_size / 1e6:.0f} MB")

    manifest = {
        "schema_version": 1,
        "dataset_repo": DATASET_REPO,
        "revision": revision,
        "release": release,
        "files": [
            {
                "name": shard,
                "bytes": (dest / shard).stat().st_size,
                "sha256": sha256_file(dest / shard),
            }
            for shard in shards
        ],
    }
    manifest_path = dest / "source_manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(manifest_path)
    print(f"  source manifest: {manifest_path}")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--release", default="v6",
        help="Dataset tag. 'v6' is the 175-problem 2025-01-04..2025-04-06 window this study uses; "
        "'release_v6' is the cumulative ~1055 problems and downloads 4.5 GB.",
    )
    parser.add_argument("--dest", default=str(ROOT / "data" / "reference" / "multilcb"))
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="Immutable Hugging Face dataset revision. Do not use a moving branch for paid runs.",
    )
    parser.add_argument("--force", action="store_true", help="Re-download shards that already exist")
    args = parser.parse_args()

    print(f"Staging {args.release} into {args.dest}")
    dest = fetch(args.release, Path(args.dest), args.force, args.revision)
    print(f"\nStaged at {dest}")
    print(f"Pass it through with:  --dataset_path {dest}")


if __name__ == "__main__":
    sys.exit(main())
