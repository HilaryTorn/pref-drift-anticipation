#!/usr/bin/env python3
"""Materialize the prompts-repaired-v1 RL cohorts as local frozen cohort directories.

The Hugging Face export ``<size>/prompts-repaired-v1`` is prompt-text-and-order only: it deliberately omits verifier definitions, which the SFT/RL path needs. This script rejoins verifiers without touching the frozen ID list or its order.

Provenance of every row's verifier:

* retained IDs come from the model's own pre-repair cohort ``train.jsonl``;
* replacement IDs present in the other model's pre-repair cohort come from there;
* the remainder come from the pinned Nemotron source revision through the repo's own adapter at ``max_tests=0`` (full suite).

Prompt text is taken from the export, with one deliberate normalization. The export's replacement rows were regenerated with the current adapter, which strips the Nemotron instruction wrapper, while the retained rows predate that behaviour and keep it. Left alone the cohort would carry two prompt formats split exactly along the retained/replacement line. The wrapper is re-added to the replacements so all rows match each other, the already-frozen DPO artifacts, and the other RL arms. The reconstruction is verified against each record's ``source_original_prompt_sha256`` (or the stored wrapped prompt) and the script aborts if any row fails to reproduce.

`selected_ids.json` is copied byte-for-byte from the export and `train.jsonl` is written in exactly that order.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rl_training.data.quality import (  # noqa: E402
    FORMAL_MIN_UNIQUE_TESTS,
    NEMOTRON_PROMPT_PREFIX,
    QUALITY_POLICY,
    sha256_text,
    validate_formal_coding_row,
)
from scripts.rl_training.data.schema import read_coding_jsonl, write_jsonl  # noqa: E402

HF_REPO = "prism-drift/qwen35-m0-v4-postfix-rl-data"
HF_REVISION = "a9da0d84f20962f0421025d06af930d9cdfea42f"
SOURCE_REVISION = "ae1f446f299823ea3c4c00217942b53787278b31"

EXPORT_ROOT = ROOT / "data/rl/v1/.cache/huggingface" / (
    "datasets--prism-drift--qwen35-m0-v4-postfix-rl-data/snapshots" f"/{HF_REVISION}"
)
RECOVERED = ROOT / "data/rl/v1/.cache/recovered_source_records.jsonl"

BASE_COHORTS = {
    "4b": ROOT / "data/rl/v1/cohorts/prism-drift-qwen35-4b-m0-v4--rev-dca63300371b--data-fulltests-v2-shared-n1000-s0",
    "9b": ROOT / "data/rl/v1/cohorts/prism-drift-qwen35-9b-m0-v4--rev-8f3d499236d7--data-fulltests-v2-shared-n1000-s0",
}
OUT_NAMES = {
    "4b": "prism-drift-qwen35-4b-m0-v4--rev-dca63300371b--data-fulltests-v2-registry-repaired-v1-shared-n1000-s0",
    "9b": "prism-drift-qwen35-9b-m0-v4--rev-8f3d499236d7--data-fulltests-v2-registry-repaired-v1-shared-n1000-s0",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_provenance() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"git_commit": commit, "git_dirty": dirty}


def load_verifier_pool() -> dict[str, dict[str, Any]]:
    """Pre-repair cohort rows first (they carry the legacy wrapped prompt), then source-recovered rows."""
    pool: dict[str, dict[str, Any]] = {}
    for path in BASE_COHORTS.values():
        for line in (path / "train.jsonl").open():
            row = json.loads(line)
            pool.setdefault(row["record_id"], row)
    recovered = {}
    if RECOVERED.is_file():
        for line in RECOVERED.open():
            row = json.loads(line)
            recovered[row["record_id"]] = row
    return pool, recovered


def prompt_forms(prompt: str) -> tuple[str, str]:
    """Return (unwrapped, wrapped) for a prompt in either form.

    The wrapper is a fixed literal, so the two forms convert losslessly in both directions.
    """
    if prompt.startswith(NEMOTRON_PROMPT_PREFIX):
        return prompt[len(NEMOTRON_PROMPT_PREFIX):], prompt
    return prompt, NEMOTRON_PROMPT_PREFIX + prompt


def build_cohort(tag: str, out_root: Path, force: bool, wrapper_mode: str) -> dict[str, Any]:
    export_dir = EXPORT_ROOT / tag / "prompts-repaired-v1"
    export_manifest = json.loads((export_dir / "manifest.json").read_text())

    # The export's own hashes must hold before anything downstream trusts it.
    for name, fname in (("selected_ids", "selected_ids.json"), ("prompts", "prompts.jsonl")):
        observed = sha256_file(export_dir / fname)
        expected = export_manifest["outputs"][name]["sha256"]
        if observed != expected:
            raise SystemExit(f"{tag}: export {fname} sha256 mismatch (expected {expected}, got {observed})")

    selected_ids: list[str] = json.loads((export_dir / "selected_ids.json").read_text())
    export_rows = [json.loads(l) for l in (export_dir / "prompts.jsonl").open() if l.strip()]
    export_prompts = {r["record_id"]: r["prompt"] for r in export_rows}

    if [r["record_id"] for r in export_rows] != selected_ids:
        raise SystemExit(f"{tag}: export prompts.jsonl order does not match selected_ids.json")
    if len(set(selected_ids)) != len(selected_ids):
        raise SystemExit(f"{tag}: duplicate ids in selected_ids.json")

    pool, recovered = load_verifier_pool()
    own_ids = {json.loads(l)["record_id"] for l in (BASE_COHORTS[tag] / "train.jsonl").open()}

    rows: list[dict[str, Any]] = []
    normalized: list[str] = []
    provenance = {"retained": 0, "cross_cohort_replacement": 0, "source_recovered_replacement": 0}

    for rid in selected_ids:
        unwrapped, wrapped = prompt_forms(export_prompts[rid])
        target_prompt = unwrapped if wrapper_mode == "strip" else wrapped
        base = pool.get(rid)
        if base is not None:
            origin = "retained" if rid in own_ids else "cross_cohort_replacement"
        else:
            base = recovered.get(rid)
            if base is None:
                raise SystemExit(f"{tag}/{rid}: no verifier available from any source")
            origin = "source_recovered_replacement"

        # Compare on the wrapper-independent form so the underlying problem text must agree
        # whichever build produced the row.
        if prompt_forms(base["prompt"])[0] != unwrapped:
            raise SystemExit(f"{tag}/{rid}: {origin} prompt text disagrees with the export")
        expected_sha = base.get("metadata", {}).get("source_original_prompt_sha256")
        if expected_sha and sha256_text(wrapped) != expected_sha:
            raise SystemExit(
                f"{tag}/{rid}: reconstructed prompt does not reproduce the pinned source original"
            )
        provenance[origin] += 1

        row = deepcopy(base)
        if row["prompt"] != target_prompt:
            normalized.append(rid)
        row["prompt"] = target_prompt
        row["record_id"] = rid
        row["schema"] = "coding_task_v1"
        # Legacy rows omit `comparison`. Both rewards.py and schema.py already read it as
        # verifier.get("comparison", "tokens"), so writing it makes the verifier self-describing
        # without changing how a single row is scored. Test lists are left untouched.
        verifier = row.get("verifier")
        if isinstance(verifier, dict) and "comparison" not in verifier:
            verifier["comparison"] = "tokens"
        metadata = row.setdefault("metadata", {})
        metadata["cohort_row_origin"] = origin
        metadata["prompt_wrapper_present"] = wrapper_mode == "wrap"
        metadata["prompt_wrapper_normalized"] = rid in normalized
        rows.append(row)

    if len(rows) != len(selected_ids):
        raise SystemExit(f"{tag}: row count {len(rows)} != {len(selected_ids)}")
    if [r["record_id"] for r in rows] != selected_ids:
        raise SystemExit(f"{tag}: emitted rows are not in frozen cohort order")

    out_dir = out_root / OUT_NAMES[tag]
    if out_dir.exists() and not force:
        raise SystemExit(f"{tag}: {out_dir} already exists; pass --force to rewrite")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.jsonl"
    write_jsonl(train_path, rows)
    shutil.copyfile(export_dir / "selected_ids.json", out_dir / "selected_ids.json")

    # Re-read through the repo's own validator rather than trusting what we just built.
    reread = read_coding_jsonl(train_path)
    if [r["record_id"] for r in reread] != selected_ids:
        raise SystemExit(f"{tag}: re-read train.jsonl does not match selected_ids order")

    # Audit against a wrapper-stripped copy. The wrapper is a deliberate, documented cohort
    # convention, so scoring it as a violation on every row would bury the substantive defects.
    audit_counts: dict[str, int] = {}
    failing = 0
    for row in reread:
        probe = deepcopy(row)
        prompt = probe["prompt"]
        if prompt.startswith(NEMOTRON_PROMPT_PREFIX):
            probe["prompt"] = prompt[len(NEMOTRON_PROMPT_PREFIX):]
        reasons = validate_formal_coding_row(probe, min_unique_tests=FORMAL_MIN_UNIQUE_TESTS)
        if reasons:
            failing += 1
            for reason in reasons:
                audit_counts[reason] = audit_counts.get(reason, 0) + 1

    manifest = {
        "schema": "shared_training_cohort_v1",
        "status": "frozen",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cohort_tag": OUT_NAMES[tag],
        "target_model_size": tag,
        "provenance": {
            "note": (
                "Verifiers rejoined locally onto the prompt-only prompts-repaired-v1 export. "
                "Frozen ID list and order are the export's, unmodified."
            ),
            "prompt_export": {
                "hf_repo_id": HF_REPO,
                "hf_revision": HF_REVISION,
                "hf_path": f"{tag}/prompts-repaired-v1",
                "manifest": export_manifest,
            },
            "base_cohort": {
                "path": str(BASE_COHORTS[tag].relative_to(ROOT)),
                "manifest_sha256": sha256_file(BASE_COHORTS[tag] / "manifest.json"),
            },
            "source_dataset": {
                "dataset_id": "nemotron_rl_coding_competitive",
                "dataset_name": "nvidia/Nemotron-RL-coding-competitive_coding",
                "revision": SOURCE_REVISION,
                "max_tests": 0,
            },
            "row_origin_counts": provenance,
            "prompt_wrapper_normalization": {
                "mode": wrapper_mode,
                "action": (
                    "stripped the Nemotron generation prefix from every row"
                    if wrapper_mode == "strip"
                    else "re-added the exact Nemotron generation prefix to every row"
                ),
                "reason": (
                    "The export's replacement rows were regenerated with the current wrapper-stripping "
                    "adapter while retained rows predate it, which would otherwise split the cohort into "
                    "two prompt formats exactly along the retained/replacement line. Stripping also "
                    "removes an instruction block that contradicts the training-time wrapper "
                    "('think step-by-step' against 'return only the solution code') and that sits after "
                    "'Problem:' as though it were problem content, and it frees roughly 60 tokens per row "
                    "against the prompt-length and target-length limits."
                ),
                "rows_normalized": len(normalized),
                "record_ids": normalized,
                "verified_against": "source_original_prompt_sha256 or the stored pre-repair wrapped prompt",
                "supersedes_frozen_artifacts": (
                    "DPO pairs, prompt scores and GRPO stage data on the pinned HF revision were "
                    "generated against wrapper-present prompts and must be regenerated to match."
                ),
            },
            "git": git_provenance(),
        },
        "selection": {"selected_size": len(selected_ids), "seed": 0},
        "quality_policy": None,
        "quality_audit": {
            "policy_evaluated": QUALITY_POLICY,
            "min_unique_tests": FORMAL_MIN_UNIQUE_TESTS,
            "enforced": False,
            "evaluated_on": "prompt with the Nemotron wrapper stripped",
            "note": (
                "Recorded, not enforced. The pre-repair base cohort predates this policy "
                "(its manifest carries quality_policy: null) and rebuilding under the policy would "
                "change the frozen ID set. Consume with --cohort-validation-mode frozen_exact. "
                "Counts exclude source_prompt_wrapper_present, which the cohort's deliberate "
                "wrapper convention would otherwise trigger on every row."
            ),
            "rows_failing_policy": failing,
            "violation_counts": dict(sorted(audit_counts.items(), key=lambda kv: -kv[1])),
        },
        "outputs": {
            "train": {
                "path": "train.jsonl",
                "rows": len(rows),
                "sha256": sha256_file(train_path),
            },
            "selected_ids": {
                "path": "selected_ids.json",
                "rows": len(selected_ids),
                "sha256": sha256_file(out_dir / "selected_ids.json"),
            },
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\n=== {tag} -> {out_dir.relative_to(ROOT)}")
    print(f"    rows={len(rows)}  order preserved=yes  row origins={provenance}")
    print(f"    prompt wrapper mode={wrapper_mode}; changed {len(normalized)} rows")
    print(f"    quality audit (recorded, not enforced): {failing}/{len(rows)} rows fail {QUALITY_POLICY}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=ROOT / "data/rl/v1/cohorts")
    parser.add_argument("--tag", action="append", choices=["4b", "9b"], default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--prompt-wrapper",
        choices=["strip", "wrap"],
        default="strip",
        help=(
            "strip (default): remove the Nemotron generation prefix from every row, matching the "
            "export's replacement rows. wrap: re-add it to every row, matching the legacy cohorts "
            "and the already-frozen DPO/GRPO artifacts."
        ),
    )
    args = parser.parse_args()

    for tag in args.tag or ["4b", "9b"]:
        build_cohort(tag, args.out_root, args.force, args.prompt_wrapper)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
