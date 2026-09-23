#!/usr/bin/env python3
"""Build a paper-facing cohort for one or more requested Qwen3.5 sizes.

The v4 model sweeps are used only as eligibility evidence. Their generated
completions are deliberately not copied because those completions used the
retired, double-wrapped prompt. Canonical problems and complete fixed I/O suites
are reconstructed from the exact pinned source revision, quality filtered, and
then sampled once for every target model.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import tempfile
from typing import Any

from scripts.rl_training.data.quality import (
    FORMAL_MIN_UNIQUE_TESTS,
    canonicalize_coding_row,
    quality_policy_manifest,
    validate_formal_coding_row,
)
from scripts.rl_training.data.registry import get_adapter
from scripts.rl_training.data.schema import read_coding_jsonl, read_jsonl, read_prompt_scores_jsonl, write_jsonl


DEFAULT_SOURCE_REVISION = "ae1f446f299823ea3c4c00217942b53787278b31"
DEFAULT_ARCHIVE_REVISION = "d5ba7c0e52c066afa78109d090cec703e0777e53"
DEFAULT_KNOWN_BAD_REGISTRY = Path(__file__).with_name("known_bad_nemotron_verifiers.json")
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not SAFE_NAME_RE.fullmatch(name) or not raw_path:
        raise argparse.ArgumentTypeError("expected safe non-empty NAME=PATH")
    return name, Path(raw_path).expanduser()


def named_paths(values: list[tuple[str, Path]], flag: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for name, path in values:
        if name in result:
            raise SystemExit(f"Duplicate {flag} name: {name}")
        path = path.resolve()
        if not path.is_file():
            raise SystemExit(f"{flag} file does not exist: {path}")
        result[name] = path
    return result


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
        return {"git_commit": commit, "git_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}


def load_eligible_scores(
    path: Path,
    *,
    min_max_score: float,
    max_min_score: float,
    min_margin: float,
) -> tuple[set[str], dict[str, Any]]:
    rows = read_prompt_scores_jsonl(path)
    eligible: set[str] = set()
    models: set[tuple[Any, Any, Any]] = set()
    for record_id, row in rows.items():
        scores = row.get("scores") or []
        if not scores:
            raise SystemExit(f"{path}: empty score list for {record_id}")
        low, high = float(min(scores)), float(max(scores))
        if float(row.get("min_score", low)) != low or float(row.get("max_score", high)) != high:
            raise SystemExit(f"{path}: score summary mismatch for {record_id}")
        models.add(
            (
                row.get("generator_model"),
                row.get("generator_model_revision"),
                row.get("source_prompts_sha256"),
            )
        )
        if high >= min_max_score and low <= max_min_score and high - low >= min_margin:
            eligible.add(record_id)
    if len(models) != 1:
        raise SystemExit(f"{path}: score rows do not pin one generator identity")
    model, revision, prompts_sha = next(iter(models))
    return eligible, {
        "path": path.name,
        "sha256": sha256_file(path),
        "rows": len(rows),
        "eligible_rows": len(eligible),
        "generator_model": model,
        "generator_model_revision": revision,
        "source_prompts_sha256": prompts_sha,
        "role": "legacy_selection_evidence_only",
    }


def load_artifact_ids(path: Path) -> tuple[set[str], dict[str, Any]]:
    rows = read_jsonl(path)
    ids: list[str] = []
    for line_no, row in enumerate(rows, 1):
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise SystemExit(f"{path}:{line_no}: missing record_id")
        ids.append(record_id)
    if len(ids) != len(set(ids)):
        raise SystemExit(f"{path}: duplicate record IDs")
    return set(ids), {
        "path": path.name,
        "sha256": sha256_file(path),
        "rows": len(rows),
        "unique_ids": len(ids),
        "role": "legacy_selection_evidence_only; completions_not_exported",
    }


def normalized_prompt(prompt: str) -> str:
    return " ".join(prompt.casefold().split())


def complete_local_parquet_shards(snapshot: Path, split: str) -> list[Path]:
    """Return a complete, ordered pinned parquet split or no local fast path."""
    shards = sorted((snapshot / "data").glob(f"{split}-*-of-*.parquet"))
    parsed: list[tuple[int, int, Path]] = []
    pattern = re.compile(rf"^{re.escape(split)}-(\d+)-of-(\d+)\.parquet$")
    for path in shards:
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        parsed.append((int(match.group(1)), int(match.group(2)), path))
    if not parsed:
        return []
    totals = {total for _, total, _ in parsed}
    if len(totals) != 1:
        return []
    total = next(iter(totals))
    if len(parsed) != total or {index for index, _, _ in parsed} != set(range(total)):
        return []
    return [path for _, _, path in sorted(parsed)]


def pinned_source_rows(
    *, dataset_name: str, revision: str, split: str
) -> tuple[Any, dict[str, Any]]:
    """Prefer an already cached immutable snapshot, with pinned streaming fallback."""
    from huggingface_hub import snapshot_download

    try:
        snapshot: Path | None = Path(
            snapshot_download(
                dataset_name,
                repo_type="dataset",
                revision=revision,
                allow_patterns=[f"data/{split}-*.parquet"],
                local_files_only=True,
            )
        )
    except FileNotFoundError:
        snapshot = None
    shards = complete_local_parquet_shards(snapshot, split) if snapshot is not None else []
    if shards:
        import pyarrow.parquet as pq

        def rows() -> Any:
            for shard in shards:
                parquet = pq.ParquetFile(shard)
                for batch in parquet.iter_batches(batch_size=100):
                    yield from batch.to_pylist()

        return rows(), {
            "access_mode": "complete_revision_pinned_local_parquet_snapshot",
            "parquet_shards": len(shards),
        }

    from datasets import load_dataset

    return (
        load_dataset(dataset_name, split=split, revision=revision, streaming=True),
        {"access_mode": "revision_pinned_huggingface_stream"},
    )


def amend_selection_in_place(
    *,
    base_ids: list[str],
    accepted_by_id: dict[str, dict[str, Any]],
    size: int,
    seed: int,
    replacement_native_datasets: set[str],
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Preserve valid base positions and deterministically replace rejected IDs."""
    if len(base_ids) != len(set(base_ids)):
        raise SystemExit("Base cohort selected_ids must be unique")
    if len(base_ids) != size:
        raise SystemExit("Base cohort size must match --size for replacement-in-place amendment")
    retained = [record_id for record_id in base_ids if record_id in accepted_by_id]
    removed = [record_id for record_id in base_ids if record_id not in accepted_by_id]
    retained_set = set(retained)
    candidates = [
        record_id
        for record_id, row in accepted_by_id.items()
        if record_id not in retained_set
        and (
            not replacement_native_datasets
            or row.get("metadata", {}).get("native_dataset")
            in replacement_native_datasets
        )
    ]
    candidates.sort(
        key=lambda record_id: hashlib.sha256(f"{seed}:{record_id}".encode("utf-8")).hexdigest()
    )
    replacements = candidates[: len(removed)]
    if len(replacements) != len(removed):
        raise SystemExit(
            "Not enough formal-quality IDs to amend the base cohort: "
            f"needed={len(removed)}, available={len(candidates)}, "
            f"replacement_native_datasets={sorted(replacement_native_datasets)}"
        )
    replacement_by_removed = dict(zip(removed, replacements))
    selected_ids = [replacement_by_removed.get(record_id, record_id) for record_id in base_ids]
    return selected_ids, retained, removed, replacements


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-source", action="append", type=parse_named_path, default=[], required=True)
    parser.add_argument("--artifact-source", action="append", type=parse_named_path, default=[], required=True)
    parser.add_argument("--reserved-split", action="append", type=Path, default=[])
    parser.add_argument("--dataset-id", default="nemotron_rl_coding_competitive")
    parser.add_argument("--source-revision", default=DEFAULT_SOURCE_REVISION)
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--target-model-size",
        action="append",
        required=True,
        help=(
            "Qwen3.5 target size represented by this cohort (for example 4B); "
            "repeat only when every supplied eligibility source is shared across sizes."
        ),
    )
    parser.add_argument("--min-unique-tests", type=int, default=FORMAL_MIN_UNIQUE_TESTS)
    parser.add_argument("--min-max-score", type=float, default=0.5)
    parser.add_argument("--max-min-score", type=float, default=0.5)
    parser.add_argument("--min-margin", type=float, default=0.5)
    parser.add_argument("--archive-repo", default="prism-drift/ai-pref-drift-rl-v4")
    parser.add_argument("--archive-revision", default=DEFAULT_ARCHIVE_REVISION)
    parser.add_argument("--known-bad-registry", type=Path, default=DEFAULT_KNOWN_BAD_REGISTRY)
    parser.add_argument(
        "--teacher-exclusion-registry",
        type=Path,
        help="Optional versioned registry of verifier-valid records excluded for this teacher.",
    )
    parser.add_argument(
        "--replacement-native-dataset",
        action="append",
        default=[],
        help=(
            "Restrict replacements for --base-cohort-manifest to these "
            "metadata.native_dataset values; repeatable."
        ),
    )
    parser.add_argument(
        "--base-cohort-manifest",
        type=Path,
        help="Preserve valid IDs from a prior frozen cohort and top up deterministic replacements.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.size < 1 or args.min_unique_tests < 1:
        raise SystemExit("--size and --min-unique-tests must be >= 1")
    target_model_sizes = list(dict.fromkeys(args.target_model_size))
    if any(not size.strip() for size in target_model_sizes):
        raise SystemExit("--target-model-size values must be non-empty")
    score_sources = named_paths(args.score_source, "score source")
    artifact_sources = named_paths(args.artifact_source, "artifact source")

    registry_path = args.known_bad_registry.expanduser().resolve()
    registry = json.loads(registry_path.read_text())
    if registry.get("schema") != "known_bad_verifier_registry_v1":
        raise SystemExit(f"Invalid known-bad verifier registry: {registry_path}")
    if registry.get("dataset_id") != args.dataset_id:
        raise SystemExit("Known-bad verifier registry dataset_id mismatch")
    if registry.get("source_revision") != args.source_revision:
        raise SystemExit("Known-bad verifier registry source_revision mismatch")
    known_bad_records = {
        row["record_id"]: row
        for row in registry.get("records", [])
        if isinstance(row, dict) and isinstance(row.get("record_id"), str)
    }
    teacher_excluded: set[str] = set()
    teacher_registry_path: Path | None = None
    if args.teacher_exclusion_registry:
        teacher_registry_path = args.teacher_exclusion_registry.expanduser().resolve()
        teacher_registry = json.loads(teacher_registry_path.read_text())
        if teacher_registry.get("schema") != "teacher_generation_exclusion_registry_v1":
            raise SystemExit("Invalid teacher-generation exclusion registry")
        if teacher_registry.get("dataset_id") not in {None, args.dataset_id}:
            raise SystemExit("Teacher-generation exclusion registry dataset_id mismatch")
        teacher_excluded = {
            row["record_id"]
            for row in teacher_registry.get("records", [])
            if isinstance(row, dict) and isinstance(row.get("record_id"), str)
        }

    requirements: dict[str, Any] = {}
    required_sets: list[set[str]] = []
    for name, path in score_sources.items():
        ids, info = load_eligible_scores(
            path,
            min_max_score=args.min_max_score,
            max_min_score=args.max_min_score,
            min_margin=args.min_margin,
        )
        required_sets.append(ids)
        requirements[f"score:{name}"] = info
    for name, path in artifact_sources.items():
        ids, info = load_artifact_ids(path)
        required_sets.append(ids)
        requirements[f"artifact:{name}"] = info

    candidate_ids = set.intersection(*required_sets)
    if len(candidate_ids) < args.size:
        raise SystemExit(f"Only {len(candidate_ids)} cross-model eligible IDs; requested {args.size}")

    reserved_ids: set[str] = set()
    reserved_info: list[dict[str, Any]] = []
    for raw_path in args.reserved_split:
        path = raw_path.expanduser().resolve()
        rows = read_coding_jsonl(path)
        ids = {row["record_id"] for row in rows}
        reserved_ids |= ids
        reserved_info.append({"path": path.name, "sha256": sha256_file(path), "rows": len(rows)})

    adapter = get_adapter(args.dataset_id)
    raw_dataset, source_access = pinned_source_rows(
        dataset_name=adapter.dataset_name,
        split=args.source_split,
        revision=args.source_revision,
    )
    found_ids: set[str] = set()
    accepted_by_id: dict[str, dict[str, Any]] = {}
    source_order: list[str] = []
    excluded: list[dict[str, Any]] = []
    exclusion_counts: Counter[str] = Counter()
    scanned = 0
    for raw in raw_dataset:
        scanned += 1
        record_id = str(raw.get("hash_id") or raw.get("id") or raw.get("problem_id") or "")
        if record_id not in candidate_ids or record_id in found_ids:
            continue
        found_ids.add(record_id)
        source_order.append(record_id)
        record = adapter.extract_record(dict(raw), max_tests=0)
        if record is None:
            reasons = ["adapter_rejected"]
            cleaned = None
        else:
            cleaned, reasons = canonicalize_coding_row(
                record.to_json(), min_unique_tests=args.min_unique_tests
            )
        if record_id in reserved_ids:
            reasons.append("reserved_split_overlap")
        if record_id in known_bad_records:
            reasons.append("known_bad_verifier")
        if record_id in teacher_excluded:
            reasons.append("teacher_generation_excluded")
        if reasons:
            for reason in sorted(set(reasons)):
                exclusion_counts[reason] += 1
            excluded.append({"record_id": record_id, "reasons": sorted(set(reasons))})
        elif cleaned is not None:
            accepted_by_id[record_id] = cleaned
        if len(found_ids) % 250 == 0:
            print(
                f"[formal-cohort] found {len(found_ids)}/{len(candidate_ids)} eligible IDs "
                f"after scanning {scanned} source rows",
                flush=True,
            )
        if found_ids == candidate_ids:
            break

    missing_source = sorted(candidate_ids - found_ids)
    if missing_source:
        raise SystemExit(f"Pinned source scan did not find {len(missing_source)} eligible IDs")

    prompt_groups: dict[str, list[str]] = defaultdict(list)
    for record_id, row in accepted_by_id.items():
        prompt_groups[normalized_prompt(row["prompt"])].append(record_id)
    duplicate_ids = {
        record_id
        for ids in prompt_groups.values()
        if len(ids) > 1
        for record_id in ids
    }
    for record_id in sorted(duplicate_ids):
        accepted_by_id.pop(record_id)
        exclusion_counts["duplicate_normalized_prompt"] += 1
        excluded.append({"record_id": record_id, "reasons": ["duplicate_normalized_prompt"]})

    if len(accepted_by_id) < args.size:
        raise SystemExit(
            f"Only {len(accepted_by_id)} IDs pass formal quality policy; requested {args.size}. "
            f"Exclusions: {dict(sorted(exclusion_counts.items()))}"
        )
    amendment: dict[str, Any] | None = None
    if args.base_cohort_manifest:
        base_manifest_path = args.base_cohort_manifest.expanduser().resolve()
        base_manifest = json.loads(base_manifest_path.read_text())
        base_ids_ref = base_manifest.get("outputs", {}).get("selected_ids", {})
        base_ids_path = Path(base_ids_ref.get("path", ""))
        if not base_ids_path.is_absolute():
            base_ids_path = base_manifest_path.parent / base_ids_path
        if not base_ids_path.is_file() or sha256_file(base_ids_path) != base_ids_ref.get("sha256"):
            raise SystemExit("Base cohort selected_ids hash mismatch")
        base_ids = json.loads(base_ids_path.read_text())
        if not isinstance(base_ids, list) or len(base_ids) != len(set(base_ids)):
            raise SystemExit("Base cohort selected_ids must be a unique JSON list")
        allowed_replacement_datasets = set(args.replacement_native_dataset)
        selected_ids, retained, removed, replacements = amend_selection_in_place(
            base_ids=base_ids,
            accepted_by_id=accepted_by_id,
            size=args.size,
            seed=args.seed,
            replacement_native_datasets=allowed_replacement_datasets,
        )
        amendment = {
            "base_manifest": base_manifest_path.name,
            "base_manifest_sha256": sha256_file(base_manifest_path),
            "base_selected_ids_sha256": base_ids_ref.get("sha256"),
            "retained_ids": len(retained),
            "removed_ids": removed,
            "replacement_ids": replacements,
            "policy": "preserve_valid_base_ids_and_positions_then_hash_rank_top_up",
            "replacement_native_datasets": sorted(allowed_replacement_datasets),
        }
    else:
        selected_set = set(random.Random(args.seed).sample(sorted(accepted_by_id), args.size))
        selected_ids = [record_id for record_id in source_order if record_id in selected_set]
    selected_rows = [accepted_by_id[record_id] for record_id in selected_ids]
    if len(selected_ids) != args.size:
        raise SystemExit("Internal selection-order error")
    for row in selected_rows:
        violations = validate_formal_coding_row(row, min_unique_tests=args.min_unique_tests)
        if violations:
            raise SystemExit(f"Internal formal-quality validation failed: {violations}")

    out_dir = args.out_dir.expanduser().resolve()
    if out_dir.exists():
        raise SystemExit(f"Output already exists and will not be overwritten: {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.tmp-", dir=out_dir.parent))
    try:
        train_path = staging / "train.jsonl"
        ids_path = staging / "selected_ids.json"
        excluded_path = staging / "excluded.jsonl"
        write_jsonl(train_path, selected_rows)
        ids_path.write_text(json.dumps(selected_ids, indent=2, ensure_ascii=True) + "\n")
        write_jsonl(excluded_path, sorted(excluded, key=lambda row: row["record_id"]))
        manifest = {
            "schema": "shared_training_cohort_v1",
            "status": "frozen",
            "created_at": utc_now(),
            "dataset_id": args.dataset_id,
            "dataset": {
                "dataset_name": adapter.dataset_name,
                "revision": args.source_revision,
                "source_split": args.source_split,
                "rows_scanned": scanned,
                **source_access,
            },
            "archive": {
                "repo_id": args.archive_repo,
                "repo_type": "dataset",
                "revision": args.archive_revision,
            },
            "license": {
                "spdx": "CC-BY-SA-4.0",
                "note": (
                    "Conservative designation from the pinned NVIDIA card header; "
                    "upstream card prose is inconsistent and must be reviewed before redistribution."
                ),
            },
            "code_provenance": git_provenance(),
            "quality_policy": quality_policy_manifest(min_unique_tests=args.min_unique_tests),
            "selection": {
                "requested_size": args.size,
                "selected_size": len(selected_ids),
                "seed": args.seed,
                "model_eligible_before_quality": len(candidate_ids),
                "quality_eligible": len(accepted_by_id),
                "criteria": {
                    "min_max_score": args.min_max_score,
                    "max_min_score": args.max_min_score,
                    "min_margin": args.min_margin,
                },
                "methodology_note": (
                    "v4 sweeps establish learnability/availability only. Their completions are "
                    "not formal v5 training artifacts because the prompt wrapper changed."
                ),
                "amendment": amendment,
            },
            "research_readiness": {
                "stronger_teacher_sft_generation": "ready",
                "cross_algorithm_comparison": "blocked_pending_clean_prompt_qwen_sweeps",
                "reason": (
                    "The v4 selection sweeps used the retired double-wrapped prompt and old "
                    "stdout comparison. Regenerate Qwen3.5 sweeps and DPO pairs on this "
                    "canonical cohort before paper-facing algorithm comparisons."
                ),
            },
            "requirements": requirements,
            "reserved_splits": reserved_info,
            "exclusions": {
                "rows": len(excluded),
                "reason_counts": dict(sorted(exclusion_counts.items())),
                "path": "excluded.jsonl",
                "sha256": sha256_file(excluded_path),
            },
            "known_bad_verifier_registry": {
                "path": registry_path.name,
                "sha256": sha256_file(registry_path),
                "records": len(known_bad_records),
            },
            "intervention_axes": {
                "training_algorithm": "sft",
                "supervision_source": "stronger_teacher",
                "target_model_family": "Qwen3.5",
                "target_model_sizes": target_model_sizes,
            },
            "outputs": {
                "train": {"path": "train.jsonl", "rows": len(selected_rows), "sha256": sha256_file(train_path)},
                "selected_ids": {"path": "selected_ids.json", "rows": len(selected_ids), "sha256": sha256_file(ids_path)},
                "artifacts": {},
            },
        }
        if teacher_registry_path:
            manifest["teacher_generation_exclusion_registry"] = {
                "path": teacher_registry_path.name,
                "sha256": sha256_file(teacher_registry_path),
                "records": len(teacher_excluded),
            }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n")
        os.replace(staging, out_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f"[formal-cohort] model-eligible: {len(candidate_ids)}", flush=True)
    print(f"[formal-cohort] quality eligible: {len(accepted_by_id)}", flush=True)
    print(f"[formal-cohort] selected: {len(selected_ids)} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
