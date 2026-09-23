#!/usr/bin/env python3
"""Amend a frozen formal cohort by replacing registry-quarantined IDs.

Replacement candidates must already exist in local, hash-pinned cohort files and
must satisfy every supplied score/artifact requirement plus the formal quality
policy. Valid base IDs are preserved; replacements are hash-ranked by seed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from scripts.rl_training.data.quality import (
    canonicalize_coding_row,
    quality_policy_manifest,
    validate_formal_coding_row,
)
from scripts.rl_training.data.schema import read_coding_jsonl, read_jsonl, read_prompt_scores_jsonl, write_jsonl


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_provenance() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit = None
        dirty = None
    return {"git_commit": commit, "git_dirty": dirty}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-cohort-manifest", type=Path, required=True)
    parser.add_argument("--candidate-cohort", type=Path, action="append", required=True)
    parser.add_argument("--score-source", type=Path, action="append", required=True)
    parser.add_argument("--artifact-source", type=Path, action="append", required=True)
    parser.add_argument("--reserved-split", type=Path, action="append", default=[])
    parser.add_argument("--known-bad-registry", type=Path, required=True)
    parser.add_argument(
        "--teacher-exclusion-registry",
        type=Path,
        help=(
            "Optional versioned registry of verifier-valid records excluded because the "
            "pinned teacher/configuration exhausted generation attempts."
        ),
    )
    parser.add_argument(
        "--replacement-native-dataset",
        action="append",
        default=[],
        help=(
            "Restrict newly selected replacements to these metadata.native_dataset values; "
            "repeatable. Retained base rows are unaffected."
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def resolve_ref(manifest_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else manifest_path.parent / path


def main() -> None:
    args = parse_args()
    base_manifest_path = args.base_cohort_manifest.expanduser().resolve()
    base_manifest = json.loads(base_manifest_path.read_text())
    if base_manifest.get("schema") != "shared_training_cohort_v1":
        raise SystemExit("Base manifest is not shared_training_cohort_v1")
    base_train_path = resolve_ref(base_manifest_path, base_manifest["outputs"]["train"]["path"])
    base_ids_path = resolve_ref(base_manifest_path, base_manifest["outputs"]["selected_ids"]["path"])
    if sha256_file(base_train_path) != base_manifest["outputs"]["train"]["sha256"]:
        raise SystemExit("Base cohort train hash mismatch")
    if sha256_file(base_ids_path) != base_manifest["outputs"]["selected_ids"]["sha256"]:
        raise SystemExit("Base cohort selected_ids hash mismatch")
    base_rows = read_coding_jsonl(base_train_path)
    base_ids = json.loads(base_ids_path.read_text())
    if [row["record_id"] for row in base_rows] != base_ids:
        raise SystemExit("Base cohort row order differs from selected_ids")

    registry_path = args.known_bad_registry.expanduser().resolve()
    registry = json.loads(registry_path.read_text())
    if registry.get("schema") != "known_bad_verifier_registry_v1":
        raise SystemExit("Invalid known-bad verifier registry")
    known_bad = {row["record_id"] for row in registry.get("records", [])}
    teacher_excluded: set[str] = set()
    teacher_registry_path: Path | None = None
    if args.teacher_exclusion_registry:
        teacher_registry_path = args.teacher_exclusion_registry.expanduser().resolve()
        teacher_registry = json.loads(teacher_registry_path.read_text())
        if teacher_registry.get("schema") != "teacher_generation_exclusion_registry_v1":
            raise SystemExit("Invalid teacher-generation exclusion registry")
        if teacher_registry.get("dataset_id") not in {None, base_manifest.get("dataset_id")}:
            raise SystemExit("Teacher-generation exclusion registry dataset_id mismatch")
        teacher_excluded = {
            row["record_id"]
            for row in teacher_registry.get("records", [])
            if isinstance(row, dict) and isinstance(row.get("record_id"), str)
        }

    criteria = base_manifest["selection"]["criteria"]
    requirement_sets: list[set[str]] = []
    requirement_info: list[dict[str, Any]] = []
    for raw_path in args.score_source:
        path = raw_path.expanduser().resolve()
        eligible: set[str] = set()
        for record_id, row in read_prompt_scores_jsonl(path).items():
            scores = row["scores"]
            if (
                max(scores) >= criteria["min_max_score"]
                and min(scores) <= criteria["max_min_score"]
                and max(scores) - min(scores) >= criteria["min_margin"]
            ):
                eligible.add(record_id)
        requirement_sets.append(eligible)
        requirement_info.append({"type": "score", "path": path.name, "sha256": sha256_file(path), "eligible_ids": len(eligible)})
    for raw_path in args.artifact_source:
        path = raw_path.expanduser().resolve()
        ids = {row["record_id"] for row in read_jsonl(path)}
        requirement_sets.append(ids)
        requirement_info.append({"type": "artifact", "path": path.name, "sha256": sha256_file(path), "eligible_ids": len(ids)})
    eligible_ids = set.intersection(*requirement_sets)

    reserved_ids: set[str] = set()
    reserved_info: list[dict[str, Any]] = []
    for raw_path in args.reserved_split:
        path = raw_path.expanduser().resolve()
        rows = read_coding_jsonl(path)
        reserved_ids |= {row["record_id"] for row in rows}
        reserved_info.append({"path": path.name, "sha256": sha256_file(path), "rows": len(rows)})

    min_tests = int(base_manifest["quality_policy"]["min_unique_tests"])
    retained_rows: list[dict[str, Any]] = []
    removed_reasons: dict[str, list[str]] = {}
    for row in base_rows:
        cleaned, reasons = canonicalize_coding_row(row, min_unique_tests=min_tests)
        if row["record_id"] in known_bad:
            reasons.append("known_bad_verifier")
        if row["record_id"] in teacher_excluded:
            reasons.append("teacher_generation_excluded")
        if row["record_id"] in reserved_ids:
            reasons.append("reserved_split_overlap")
        reasons = sorted(set(reasons))
        if reasons:
            removed_reasons[row["record_id"]] = reasons
        else:
            retained_rows.append(cleaned)
    removed = [record_id for record_id in base_ids if record_id in removed_reasons]
    if not removed:
        raise SystemExit("Base cohort has no records rejected by the current formal policy or registry")

    retained_ids = {row["record_id"] for row in retained_rows}
    retained_prompts = {" ".join(row["prompt"].casefold().split()) for row in retained_rows}
    candidates: dict[str, dict[str, Any]] = {}
    candidate_sources: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    for raw_path in args.candidate_cohort:
        path = raw_path.expanduser().resolve()
        rows = read_coding_jsonl(path)
        candidate_sources.append({"path": path.name, "sha256": sha256_file(path), "rows": len(rows)})
        for row in rows:
            record_id = row["record_id"]
            if record_id not in eligible_ids:
                rejection_counts["not_cross_requirement_eligible"] += 1
                continue
            if (
                record_id in retained_ids
                or record_id in reserved_ids
                or record_id in known_bad
                or record_id in teacher_excluded
            ):
                rejection_counts["reserved_or_existing_or_excluded"] += 1
                continue
            cleaned, reasons = canonicalize_coding_row(row, min_unique_tests=min_tests)
            if (
                args.replacement_native_dataset
                and cleaned.get("metadata", {}).get("native_dataset")
                not in set(args.replacement_native_dataset)
            ):
                reasons.append("replacement_native_dataset_not_allowed")
            normalized = " ".join(cleaned["prompt"].casefold().split())
            if normalized in retained_prompts:
                reasons.append("duplicate_retained_prompt")
            if reasons:
                rejection_counts.update(set(reasons))
                continue
            candidates[record_id] = cleaned

    seed = int(base_manifest["selection"]["seed"])
    ranked = sorted(
        candidates,
        key=lambda record_id: hashlib.sha256(f"{seed}:{record_id}".encode()).hexdigest(),
    )
    replacements: list[str] = []
    selected_prompts = set(retained_prompts)
    for record_id in ranked:
        normalized = " ".join(candidates[record_id]["prompt"].casefold().split())
        if normalized in selected_prompts:
            rejection_counts["duplicate_replacement_prompt"] += 1
            continue
        replacements.append(record_id)
        selected_prompts.add(normalized)
        if len(replacements) == len(removed):
            break
    if len(replacements) != len(removed):
        raise SystemExit("Not enough valid local replacement candidates")
    replacement_by_removed = {
        removed_id: candidates[replacement_id]
        for removed_id, replacement_id in zip(removed, replacements)
    }
    retained_by_id = {row["record_id"]: row for row in retained_rows}
    output_rows = [
        replacement_by_removed.get(row["record_id"], retained_by_id.get(row["record_id"]))
        for row in base_rows
    ]
    if any(row is None for row in output_rows):
        raise SystemExit("Internal amendment error: missing retained or replacement row")
    output_ids = [row["record_id"] for row in output_rows]
    for row in output_rows:
        violations = validate_formal_coding_row(row, min_unique_tests=min_tests)
        if violations:
            raise SystemExit(f"Internal formal validation failed: {row['record_id']}: {violations}")

    out_dir = args.out_dir.expanduser().resolve()
    if out_dir.exists():
        raise SystemExit(f"Output already exists: {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.tmp-", dir=out_dir.parent))
    try:
        train_path = staging / "train.jsonl"
        ids_path = staging / "selected_ids.json"
        write_jsonl(train_path, output_rows)
        ids_path.write_text(json.dumps(output_ids, indent=2) + "\n")
        manifest = json.loads(json.dumps(base_manifest))
        manifest["status"] = "frozen"
        manifest.pop("quarantine", None)
        manifest["created_at"] = datetime.now(timezone.utc).isoformat()
        manifest["code_provenance"] = git_provenance()
        manifest["quality_policy"] = quality_policy_manifest(min_unique_tests=min_tests)
        base_quality_eligible = manifest["selection"].pop("quality_eligible", None)
        manifest["selection"]["base_quality_eligible"] = {
            "rows": base_quality_eligible,
            "policy": base_manifest.get("quality_policy", {}).get("name"),
        }
        manifest["selection"]["selected_size"] = len(output_ids)
        manifest["selection"]["amendment"] = {
            "base_manifest": base_manifest_path.name,
            "base_manifest_sha256": sha256_file(base_manifest_path),
            "base_selected_ids_sha256": base_manifest["outputs"]["selected_ids"]["sha256"],
            "known_bad_registry": registry_path.name,
            "known_bad_registry_sha256": sha256_file(registry_path),
            "teacher_exclusion_registry": (
                teacher_registry_path.name if teacher_registry_path else None
            ),
            "teacher_exclusion_registry_sha256": (
                sha256_file(teacher_registry_path) if teacher_registry_path else None
            ),
            "removed_ids": removed,
            "removed_reasons": removed_reasons,
            "replacement_ids": replacements,
            "retained_ids": len(retained_rows),
            "policy": "replace_in_place_with_seeded_hash-ranked_local_cross-eligible_candidates",
            "replacement_native_datasets": sorted(set(args.replacement_native_dataset)),
            "candidate_sources": candidate_sources,
            "requirement_sources": requirement_info,
            "reserved_splits": reserved_info,
            "candidate_rejection_counts": dict(sorted(rejection_counts.items())),
        }
        removal_reason_counts = Counter(
            reason for reasons in removed_reasons.values() for reason in reasons
        )
        manifest["reserved_splits"] = reserved_info
        manifest["known_bad_verifier_registry"] = {
            "path": registry_path.name,
            "sha256": sha256_file(registry_path),
            "records": len(known_bad),
        }
        if teacher_registry_path:
            manifest["teacher_generation_exclusion_registry"] = {
                "path": teacher_registry_path.name,
                "sha256": sha256_file(teacher_registry_path),
                "records": len(teacher_excluded),
            }
        else:
            manifest.pop("teacher_generation_exclusion_registry", None)
        manifest["exclusions"] = {
            "base_manifest": {
                "rows": base_manifest.get("exclusions", {}).get("rows"),
                "reason_counts": base_manifest.get("exclusions", {}).get("reason_counts", {}),
                "quality_policy": base_manifest.get("quality_policy", {}).get("name"),
            },
            "amendment": {
                "removed_rows": len(removed),
                "reason_counts": dict(sorted(removal_reason_counts.items())),
                "candidate_rejection_counts": dict(sorted(rejection_counts.items())),
            },
        }
        manifest["outputs"]["train"] = {"path": "train.jsonl", "rows": len(output_rows), "sha256": sha256_file(train_path)}
        manifest["outputs"]["selected_ids"] = {"path": "selected_ids.json", "rows": len(output_ids), "sha256": sha256_file(ids_path)}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n")
        os.replace(staging, out_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f"[amend] retained={len(retained_rows)} removed={removed} replacements={replacements}")
    print(f"[amend] output={out_dir}")


if __name__ == "__main__":
    main()
