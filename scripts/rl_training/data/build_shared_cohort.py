#!/usr/bin/env python3
"""Freeze one training cohort that is eligible for every requested method.

The expensive model sweep is performed once per base model.  This command is
run later, when the experiment methods and final cohort size are known.  It
intersects:

* score sources, using the configured learnable-band thresholds; and
* RL artifact sources, where presence of a record ID means that method has a
  usable derived row (currently verifier-derived DPO pairs).

It writes a canonical prompt cohort, selected IDs, filtered copies of every
required artifact, and a provenance manifest.  Training jobs should consume
these frozen outputs rather than re-running this selection independently.
SFT is deliberately excluded: exact-N SFT targets are generated only after
this RL-eligible cohort is frozen, and must cover every selected ID.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import tempfile
from typing import Any

from scripts.rl_training.data.schema import (
    DPO_PREFERENCE_SCHEMA,
    read_coding_jsonl,
    read_jsonl,
    read_preference_jsonl,
    read_prompt_scores_jsonl,
    write_jsonl,
)


SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    path = Path(raw_path).expanduser()
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("expected non-empty NAME=PATH")
    if not SAFE_NAME_RE.fullmatch(name):
        raise argparse.ArgumentTypeError("NAME may contain only letters, digits, '.', '_' and '-'")
    return name, path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_id(row: dict[str, Any], *, source: Path, line_no: int) -> str:
    value = row.get("record_id") or row.get("id") or row.get("hash_id")
    if not value and isinstance(row.get("metadata"), dict):
        value = row["metadata"].get("source_record_id")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source}:{line_no}: no record_id/id/hash_id/source_record_id")
    return value


def unique_named_paths(values: list[tuple[str, Path]], flag: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for name, path in values:
        if name in result:
            raise SystemExit(f"Duplicate {flag} name {name!r}")
        if not path.is_file():
            raise SystemExit(f"{flag} file does not exist: {path}")
        result[name] = path.resolve()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Full canonical coding_task_v1 candidate JSONL")
    parser.add_argument(
        "--dataset-manifest",
        default=None,
        help="Optional preparation manifest to hash and carry into cohort provenance",
    )
    parser.add_argument(
        "--score-source",
        action="append",
        default=[],
        type=parse_named_path,
        metavar="NAME=PATH",
        help="Require learnable-band eligibility in this prompt_score_v1 file; repeatable",
    )
    parser.add_argument(
        "--artifact-source",
        action="append",
        default=[],
        type=parse_named_path,
        metavar="NAME=PATH",
        help="Require an ID in this RL artifact and write its filtered copy; SFT is forbidden",
    )
    parser.add_argument(
        "--score-manifest",
        action="append",
        default=[],
        type=parse_named_path,
        metavar="NAME=PATH",
        help="Completion manifest for each --score-source; names must match",
    )
    parser.add_argument(
        "--artifact-manifest",
        action="append",
        default=[],
        type=parse_named_path,
        metavar="NAME=PATH",
        help="Completion manifest for each --artifact-source; names must match",
    )
    parser.add_argument(
        "--size",
        type=int,
        required=True,
        help="Frozen cohort-size hyperparameter; 0 keeps the full eligible intersection",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for deterministic subsampling of the intersection")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-max-score", type=float, default=0.5)
    parser.add_argument("--max-min-score", type=float, default=0.5)
    parser.add_argument("--min-margin", type=float, default=0.5)
    return parser.parse_args()


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{label} is invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{label} must be a JSON object: {path}")
    return value


def require_exact_manifest_names(
    sources: dict[str, Path], manifests: dict[str, Path], label: str
) -> None:
    missing = sorted(set(sources) - set(manifests))
    extra = sorted(set(manifests) - set(sources))
    if missing or extra:
        raise SystemExit(
            f"{label} names must exactly match source names; missing={missing}, extra={extra}"
        )


def validate_sweep_manifest(
    *,
    path: Path,
    dataset_path: Path,
    dataset_sha: str,
    dataset_rows: int,
    score_path: Path,
    score_rows: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = load_json_object(path, "score manifest")
    if manifest.get("schema") != "model_data_sweep_v1" or manifest.get("status") != "complete":
        raise SystemExit(f"Score manifest is not a completed model_data_sweep_v1: {path}")
    inputs = manifest.get("inputs") or {}
    outputs = manifest.get("outputs") or {}
    generator = manifest.get("generator") or {}
    checks = {
        "inputs.prompts_sha256": (inputs.get("prompts_sha256"), dataset_sha),
        "inputs.prompt_rows": (inputs.get("prompt_rows"), dataset_rows),
        "outputs.scores_sha256": (outputs.get("scores_sha256"), sha256(score_path)),
        "outputs.score_rows": (outputs.get("score_rows"), score_rows),
        "generator.source_prompts_sha256": (
            generator.get("source_prompts_sha256"),
            dataset_sha,
        ),
    }
    bad = [name for name, (actual, expected) in checks.items() if actual != expected]
    if bad:
        detail = {name: checks[name] for name in bad}
        raise SystemExit(
            f"Score manifest does not describe the complete supplied dataset/scores: {detail}"
        )
    manifest_prompt_path = inputs.get("prompts")
    if not isinstance(manifest_prompt_path, str) or Path(manifest_prompt_path).name != dataset_path.name:
        raise SystemExit(f"Score manifest prompt filename does not match dataset: {path}")
    return manifest, generator


def validate_artifact_manifest(
    *,
    name: str,
    path: Path,
    artifact_path: Path,
    artifact_rows: int,
    dataset_sha: str,
    dataset_rows: int,
) -> dict[str, Any]:
    manifest = load_json_object(path, f"artifact manifest {name!r}")
    schema = manifest.get("schema")
    if manifest.get("status") != "complete":
        raise SystemExit(f"Artifact manifest {name!r} is not complete: {path}")
    if schema == "model_data_sweep_v1":
        inputs = manifest.get("inputs") or {}
        outputs = manifest.get("outputs") or {}
        checks = {
            "inputs.prompts_sha256": (inputs.get("prompts_sha256"), dataset_sha),
            "inputs.prompt_rows": (inputs.get("prompt_rows"), dataset_rows),
            "outputs.pairs_sha256": (outputs.get("pairs_sha256"), sha256(artifact_path)),
            "outputs.pair_rows": (outputs.get("pair_rows"), artifact_rows),
        }
    else:
        raise SystemExit(f"Unsupported artifact manifest schema {schema!r}: {path}")
    bad = [key for key, (actual, expected) in checks.items() if actual != expected]
    if bad:
        detail = {key: checks[key] for key in bad}
        raise SystemExit(f"Artifact manifest {name!r} mismatch: {detail}")
    return manifest


def main() -> None:
    args = parse_args()
    if args.size < 0:
        raise SystemExit("--size must be >= 0")
    if not args.score_source and not args.artifact_source:
        raise SystemExit("Provide at least one --score-source or --artifact-source")

    dataset_path = Path(args.dataset).expanduser().resolve()
    if not dataset_path.is_file():
        raise SystemExit(f"Dataset does not exist: {dataset_path}")
    score_sources = unique_named_paths(args.score_source, "score source")
    artifact_sources = unique_named_paths(args.artifact_source, "artifact source")
    score_manifests = unique_named_paths(args.score_manifest, "score manifest")
    artifact_manifests = unique_named_paths(args.artifact_manifest, "artifact manifest")
    require_exact_manifest_names(score_sources, score_manifests, "--score-manifest")
    require_exact_manifest_names(artifact_sources, artifact_manifests, "--artifact-manifest")
    dataset_manifest_path = (
        Path(args.dataset_manifest).expanduser().resolve() if args.dataset_manifest else None
    )
    if dataset_manifest_path is not None and not dataset_manifest_path.is_file():
        raise SystemExit(f"Dataset manifest does not exist: {dataset_manifest_path}")

    dataset_rows = read_coding_jsonl(dataset_path)
    dataset_sha = sha256(dataset_path)
    dataset_by_id: dict[str, dict[str, Any]] = {}
    dataset_order: list[str] = []
    for line_no, row in enumerate(dataset_rows, 1):
        rid = record_id(row, source=dataset_path, line_no=line_no)
        if rid in dataset_by_id:
            raise SystemExit(f"Duplicate dataset record ID: {rid}")
        dataset_by_id[rid] = row
        dataset_order.append(rid)
    dataset_ids = set(dataset_by_id)

    if dataset_manifest_path is not None:
        prep_manifest = load_json_object(dataset_manifest_path, "dataset manifest")
        train_count = (prep_manifest.get("counts") or {}).get("train")
        train_file = (prep_manifest.get("files") or {}).get("train")
        if train_count != len(dataset_rows) or train_file != dataset_path.name:
            raise SystemExit(
                "Dataset manifest does not describe the supplied candidate file: "
                f"count={train_count!r}/{len(dataset_rows)}, file={train_file!r}/{dataset_path.name!r}"
            )

    eligible_sets: dict[str, set[str]] = {}
    source_info: dict[str, dict[str, Any]] = {}
    score_generators: dict[str, dict[str, Any]] = {}

    for name, path in score_sources.items():
        scores = read_prompt_scores_jsonl(path)
        if set(scores) != dataset_ids:
            raise SystemExit(
                f"Score source {name!r} must cover the complete candidate dataset exactly; "
                f"missing={len(dataset_ids - set(scores))}, extra={len(set(scores) - dataset_ids)}"
            )
        sweep_manifest, generator = validate_sweep_manifest(
            path=score_manifests[name],
            dataset_path=dataset_path,
            dataset_sha=dataset_sha,
            dataset_rows=len(dataset_rows),
            score_path=path,
            score_rows=len(scores),
        )
        score_generators[name] = generator
        eligible: set[str] = set()
        for rid, row in scores.items():
            values = row.get("scores", [])
            if not values:
                raise SystemExit(f"Score source {name!r} has an empty score list for {rid!r}")
            min_score = float(min(values))
            max_score = float(max(values))
            if float(row.get("min_score", min_score)) != min_score or float(
                row.get("max_score", max_score)
            ) != max_score:
                raise SystemExit(f"Score summary min/max mismatch for {rid!r} in {path}")
            for key in (
                "generator_model",
                "generator_model_revision",
                "source_prompts_sha256",
                "sample_k",
                "temperature",
                "max_new_tokens",
                "reward_max_tests",
                "reward_timeout",
                "pair_margin",
            ):
                if row.get(key) != generator.get(key):
                    raise SystemExit(
                        f"Score row {rid!r} differs from its manifest generator on {key!r}"
                    )
            if (
                max_score >= args.min_max_score
                and min_score <= args.max_min_score
                and max_score - min_score >= args.min_margin
            ):
                eligible.add(rid)
        eligible_sets[f"score:{name}"] = eligible
        source_info[f"score:{name}"] = {
            "path": str(path),
            "sha256": sha256(path),
            "rows": len(scores),
            "eligible_rows": len(eligible),
            "manifest": str(score_manifests[name]),
            "manifest_sha256": sha256(score_manifests[name]),
            "generator": generator,
        }

    model_identities = {
        (
            generator.get("generator_model"),
            generator.get("generator_model_revision"),
            generator.get("source_prompts_sha256"),
        )
        for generator in score_generators.values()
    }
    if len(model_identities) > 1:
        raise SystemExit(
            "Score sources mix generator models, revisions, or candidate datasets: "
            f"{sorted(model_identities, key=str)}"
        )

    artifact_rows: dict[str, list[dict[str, Any]]] = {}
    artifact_ids: dict[str, list[str]] = {}
    for name, path in artifact_sources.items():
        raw_rows = read_jsonl(path)
        schema = raw_rows[0].get("schema") if raw_rows else None
        rows = read_preference_jsonl(path) if schema == DPO_PREFERENCE_SCHEMA else raw_rows
        ids: list[str] = []
        seen: set[str] = set()
        for line_no, row in enumerate(rows, 1):
            rid = record_id(row, source=path, line_no=line_no)
            if rid in seen:
                raise SystemExit(
                    f"Artifact {name!r} contains duplicate record ID {rid!r}; "
                    "shared training artifacts must contain at most one row per prompt"
                )
            ids.append(rid)
            seen.add(rid)
        outside = seen - dataset_ids
        if outside:
            raise SystemExit(
                f"Artifact {name!r} contains {len(outside)} IDs outside the candidate dataset"
            )
        if schema == "sft_distill_v1":
            raise SystemExit(
                "SFT must not participate in cohort filtering. Freeze the RL-eligible cohort "
                "first, then generate one verifier-perfect SFT target for every selected ID."
            )
        artifact_manifest = validate_artifact_manifest(
            name=name,
            path=artifact_manifests[name],
            artifact_path=path,
            artifact_rows=len(rows),
            dataset_sha=dataset_sha,
            dataset_rows=len(dataset_rows),
        )
        if schema == DPO_PREFERENCE_SCHEMA:
            generator = artifact_manifest.get("generator") or {}
            for line_no, row in enumerate(rows, 1):
                metadata = row.get("metadata") or {}
                for key in (
                    "generator_model",
                    "generator_model_revision",
                    "source_prompts_sha256",
                    "sample_k",
                    "temperature",
                    "max_new_tokens",
                    "reward_max_tests",
                    "reward_timeout",
                    "pair_margin",
                ):
                    if metadata.get(key) != generator.get(key):
                        raise SystemExit(
                            f"DPO artifact {path}:{line_no} differs from its manifest on {key!r}"
                        )
            if score_generators and not any(generator == value for value in score_generators.values()):
                raise SystemExit(
                    f"DPO artifact {name!r} was not generated by any supplied score sweep"
                )
        else:
            raise SystemExit(f"Unsupported artifact schema {schema!r} in {path}")
        artifact_rows[name] = rows
        artifact_ids[name] = ids
        eligible_sets[f"artifact:{name}"] = seen
        source_info[f"artifact:{name}"] = {
            "path": str(path),
            "sha256": sha256(path),
            "rows": len(rows),
            "eligible_ids": len(seen),
            "manifest": str(artifact_manifests[name]),
            "manifest_sha256": sha256(artifact_manifests[name]),
            "manifest_schema": artifact_manifest.get("schema"),
        }

    intersection = set(dataset_ids)
    for eligible in eligible_sets.values():
        intersection &= eligible

    target_size = len(intersection) if args.size == 0 else args.size
    if len(intersection) < target_size:
        counts = ", ".join(f"{name}={len(ids)}" for name, ids in eligible_sets.items())
        raise SystemExit(
            f"Only {len(intersection)} records satisfy every requirement; requested {target_size}. "
            f"Eligibility counts: {counts}"
        )

    if target_size == len(intersection):
        selected_set = set(intersection)
    else:
        selected_set = set(random.Random(args.seed).sample(sorted(intersection), target_size))
    selected_ids = [rid for rid in dataset_order if rid in selected_set]
    selected_rows = [dataset_by_id[rid] for rid in selected_ids]

    out_dir = Path(args.out_dir).expanduser().resolve()
    if out_dir.exists():
        raise SystemExit(
            f"Frozen cohort output already exists and will not be overwritten: {out_dir}. "
            "Choose a new directory (normally include cohort size and seed in its name)."
        )
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.tmp-", dir=out_dir.parent))
    try:
        artifacts_dir = staging_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        cohort_path = staging_dir / "train.jsonl"
        ids_path = staging_dir / "selected_ids.json"
        manifest_path = staging_dir / "manifest.json"
        write_jsonl(cohort_path, selected_rows)
        ids_path.write_text(json.dumps(selected_ids, indent=2, ensure_ascii=True) + "\n")

        filtered_artifacts: dict[str, dict[str, Any]] = {}
        for name, rows in artifact_rows.items():
            ids = artifact_ids[name]
            filtered = [row for row, rid in zip(rows, ids) if rid in selected_set]
            output_path = artifacts_dir / f"{name}.jsonl"
            write_jsonl(output_path, filtered)
            filtered_artifacts[name] = {
                "path": str(output_path.relative_to(staging_dir)),
                "rows": len(filtered),
                "unique_ids": len({rid for rid in ids if rid in selected_set}),
                "sha256": sha256(output_path),
            }

        manifest = {
            "schema": "shared_training_cohort_v1",
            "status": "frozen",
            "dataset": {
                "path": str(dataset_path),
                "sha256": dataset_sha,
                "rows": len(dataset_rows),
                "preparation_manifest": (
                    {
                        "path": str(dataset_manifest_path),
                        "sha256": sha256(dataset_manifest_path),
                        "content": load_json_object(dataset_manifest_path, "dataset manifest"),
                    }
                    if dataset_manifest_path is not None
                    else None
                ),
            },
            "selection": {
                "requested_size": args.size,
                "intersection_size": len(intersection),
                "selected_size": len(selected_ids),
                "seed": args.seed,
                "criteria": {
                    "min_max_score": args.min_max_score,
                    "max_min_score": args.max_min_score,
                    "min_margin": args.min_margin,
                },
            },
            "requirements": source_info,
            "outputs": {
                "train": {"path": "train.jsonl", "sha256": sha256(cohort_path)},
                "selected_ids": {"path": "selected_ids.json", "sha256": sha256(ids_path)},
                "artifacts": filtered_artifacts,
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n")
        os.replace(staging_dir, out_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
    print(f"[cohort] intersection: {len(intersection)}")
    print(f"[cohort] selected: {len(selected_ids)} -> {out_dir / 'train.jsonl'}")
    for name, info in filtered_artifacts.items():
        print(f"[cohort] artifact {name}: {info['rows']} rows -> {info['path']}")
    print(f"[cohort] manifest -> {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
