#!/usr/bin/env python3
"""Re-freeze an RL training cohort down to the records that actually produced a verifier-perfect SFT target.

Why this exists
---------------
`rl_training/train_sft.py` enforces cohort completeness: the SFT rows must equal the frozen cohort's `selected_ids` exactly and in order, the dataset manifest must be `status: complete`, and its row count must equal the cohort size. The registry-repaired-v1 build never satisfied that. Its manifest still reads `status: running` with `last_error: "not all frozen IDs produced verifier-perfect targets"` and `output.rows: 0`, because 47 of 1,000 records (4B) and 52 of 1,000 (9B) exhausted the full 6-attempt teacher budget without ever passing every verifier test. Most of them cannot be rescued by retrying: the cohort's own `quality_audit` flags the underlying defects (205 `non_unique_output`, 268 `insufficient_unique_tests`, 2 `missing_problem_statement`, 2 `interactive`), and a problem that stores one reference string for a "print any valid answer" task marks correct programs wrong no matter how many times the teacher tries.

This script takes the second path: rather than repairing the problems, it re-freezes the cohort at the solved set, so the completeness invariant holds again at a smaller n.

What it does NOT do
-------------------
It does not preserve size-matching with the RL arms. The GRPO/DPO/PPO arms are frozen at n=1,000; a cohort re-frozen here is n=953 (4B) or n=948 (9B), so the SFT-vs-RL comparison is no longer size-matched and any analysis that pools them must say so. `rl_training/configs/sft.yaml` declares `max_steps` matched across all four arms, and that still holds — 125 steps is unchanged — but the underlying problem sets now differ in size. This was an explicit decision, not an oversight.

Ordering is inherited from the parent cohort: the surviving IDs keep their original relative order, so the frozen order remains the prompt export's, filtered. Nothing is reshuffled.

Usage
-----
    python scripts/refreeze_solved_rl_cohort.py --model 4b --dry_run
    python scripts/refreeze_solved_rl_cohort.py --model 4b
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent

COHORTS = {
    "4b": "prism-drift-qwen35-4b-m0-v4--rev-dca63300371b--data-fulltests-v2-registry-repaired-v1-shared-n1000-s0",
    "9b": "prism-drift-qwen35-9b-m0-v4--rev-8f3d499236d7--data-fulltests-v2-registry-repaired-v1-shared-n1000-s0",
}
SFT_DIRS = {
    "4b": "prism-drift-qwen35-4b-m0-v4-registry-repaired-v1-shared-n1000-s0",
    "9b": "prism-drift-qwen35-9b-m0-v4-registry-repaired-v1-shared-n1000-s0",
}
VARIANT = "gpt-5.6-sol-markerfix"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> str:
    """Write pretty JSON and return its hash. Serialise once so the hash matches the bytes on disk."""
    blob = json.dumps(payload, indent=2, ensure_ascii=False).encode() + b"\n"
    path.write_bytes(blob)
    return sha256_bytes(blob)


def git_commit() -> tuple[str, bool]:
    def run(*args: str) -> str:
        return subprocess.run(args, cwd=REPO, capture_output=True, text=True).stdout.strip()

    return run("git", "rev-parse", "HEAD"), bool(run("git", "status", "--porcelain"))


def refreeze(model: str, dry_run: bool) -> None:
    cohort_dir = REPO / "data/rl/v1/cohorts" / COHORTS[model]
    sft_dir = REPO / "data/rl/v1/sft_datasets" / SFT_DIRS[model] / VARIANT

    old_manifest = load_json(cohort_dir / "manifest.json")
    old_ids: list[str] = load_json(cohort_dir / "selected_ids.json")

    accepted = [json.loads(line) for line in (sft_dir / "accepted.jsonl").read_text().splitlines() if line.strip()]
    by_id = {row["record_id"]: row for row in accepted}
    if len(by_id) != len(accepted):
        raise SystemExit("accepted.jsonl contains duplicate record_ids")

    # Parent order, filtered. Never reshuffled.
    new_ids = [record_id for record_id in old_ids if record_id in by_id]
    dropped = [record_id for record_id in old_ids if record_id not in by_id]
    if len(new_ids) != len(by_id):
        orphans = sorted(set(by_id) - set(old_ids))
        raise SystemExit(f"accepted.jsonl holds {len(orphans)} ids absent from the parent cohort: {orphans[:5]}")

    n = len(new_ids)
    new_tag = COHORTS[model].replace("-shared-n1000-s0", f"-solved-n{n}-s0")
    out_cohort = REPO / "data/rl/v1/cohorts" / new_tag
    out_sft = REPO / "data/rl/v1/sft_datasets" / SFT_DIRS[model] / f"{VARIANT}-solved-n{n}"

    print(f"[{model}] parent={len(old_ids)} solved={n} dropped={len(dropped)}")
    print(f"[{model}] cohort -> {out_cohort.relative_to(REPO)}")
    print(f"[{model}] sft    -> {out_sft.relative_to(REPO)}")
    if dry_run:
        return

    for target in (out_cohort, out_sft):
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)

    # --- cohort train.jsonl, filtered and reordered to the new frozen order ---
    cohort_rows: dict[str, str] = {}
    with (cohort_dir / "train.jsonl").open() as handle:
        for line in handle:
            if not line.strip():
                continue
            cohort_rows[json.loads(line)["record_id"]] = line.rstrip("\n")
    missing = [record_id for record_id in new_ids if record_id not in cohort_rows]
    if missing:
        raise SystemExit(f"parent cohort train.jsonl is missing {len(missing)} selected ids")
    with (out_cohort / "train.jsonl").open("w") as handle:
        for record_id in new_ids:
            handle.write(cohort_rows[record_id] + "\n")
    cohort_train_sha = sha256(out_cohort / "train.jsonl")

    ids_sha = write_json(out_cohort / "selected_ids.json", new_ids)

    commit, dirty = git_commit()
    now = _dt.datetime.now(_dt.timezone.utc).isoformat()

    manifest = dict(old_manifest)
    manifest["created_at"] = now
    manifest["cohort_tag"] = new_tag
    manifest["selection"] = {**old_manifest.get("selection", {}), "selected_size": n}
    manifest["outputs"] = {
        "train": {"path": "train.jsonl", "rows": n, "sha256": cohort_train_sha},
        "selected_ids": {"path": "selected_ids.json", "sha256": ids_sha},
    }
    provenance = dict(old_manifest.get("provenance", {}))
    provenance["refreeze"] = {
        "note": (
            "Re-frozen from the parent n=1000 cohort by dropping every record that never produced a "
            "verifier-perfect teacher target within the 6-attempt budget. Order is the parent's, filtered; "
            "no row was reshuffled, re-generated or re-verified."
        ),
        "reason": (
            "train_sft.py enforces cohort completeness (rows == selected_ids exactly and in order, manifest "
            "status complete, row count == cohort size). The parent build never reached it: its dataset "
            "manifest reads status=running with last_error 'not all frozen IDs produced verifier-perfect "
            "targets'. Re-freezing at the solved set restores the invariant at a smaller n."
        ),
        "comparability_warning": (
            "The GRPO/DPO/PPO arms remain frozen at n=1000. This cohort is n=%d, so SFT-vs-RL is NO LONGER "
            "size-matched. max_steps stays matched at 125 per rl_training/configs/sft.yaml, but any analysis "
            "pooling these arms must state the differing problem-set sizes." % n
        ),
        "parent_cohort": {
            "cohort_tag": old_manifest.get("cohort_tag"),
            "path": str(cohort_dir.relative_to(REPO)),
            "manifest_sha256": sha256(cohort_dir / "manifest.json"),
            "selected_size": len(old_ids),
        },
        "dropped_record_ids": dropped,
        "dropped_count": len(dropped),
        "produced_by": {
            "script": "scripts/refreeze_solved_rl_cohort.py",
            "sha256": sha256(Path(__file__).resolve()),
            "git_commit": commit,
            "git_dirty": dirty,
        },
    }
    manifest["provenance"] = provenance
    cohort_manifest_sha = write_json(out_cohort / "manifest.json", manifest)

    # --- SFT train.jsonl in the new frozen order ---
    with (out_sft / "train.jsonl").open("w") as handle:
        for record_id in new_ids:
            handle.write(json.dumps(by_id[record_id], ensure_ascii=False) + "\n")
    train_sha = sha256(out_sft / "train.jsonl")
    shutil.copy2(sft_dir / "accepted.jsonl", out_sft / "accepted.jsonl")

    ds = dict(load_json(sft_dir / "manifest.json"))
    ds["status"] = "complete"
    ds["created_at"] = now
    ds.pop("last_error", None)
    source = dict(ds.get("source_cohort", {}))
    source.update(
        {
            "manifest_path": str(out_cohort / "manifest.json"),
            "manifest_sha256": cohort_manifest_sha,
            "selected_ids_sha256": ids_sha,
            "selected_ids": new_ids,
            "train_path": str(out_cohort / "train.jsonl"),
            "train_sha256": cohort_train_sha,
        }
    )
    ds["source_cohort"] = source
    signature = dict(ds.get("run_signature", {}))
    signature.update(
        {
            "cohort_manifest_path": str(out_cohort / "manifest.json"),
            "cohort_manifest_sha256": cohort_manifest_sha,
            "cohort_train_sha256": cohort_train_sha,
            "selected_ids_sha256": ids_sha,
        }
    )
    ds["run_signature"] = signature
    ds["output"] = {"path": "train.jsonl", "rows": n, "sha256": train_sha}
    ds["counts"] = {**ds.get("counts", {}), "requested_rows": n, "written_rows": n, "unique_ids": n}
    ds["refreeze"] = provenance["refreeze"]
    write_json(out_sft / "manifest.json", ds)

    print(f"[{model}] wrote cohort manifest {cohort_manifest_sha[:16]} / train {train_sha[:16]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(COHORTS), required=True)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    refreeze(args.model, args.dry_run)


if __name__ == "__main__":
    main()
