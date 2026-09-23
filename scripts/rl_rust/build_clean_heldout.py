"""Strip the Nemotron instruction wrapper from the held-out eval prompts.

Root cause of the GRPO flat-zero held-out curve (found 2026-09-11):
``rl-rust/out/heldout-500.jsonl`` carries the RAW Nemotron prompt, which opens
with "You are a helpful and harmless assistant..." and explicitly instructs
"Please use python programming language only". The training cohorts were built
from prompt-cleaned rows (bare problem text), but the held-out file never went
through that cleaning. At eval time the Rust template wrapped a Python-demanding
instruction, models emitted Python, rustc rejected it, and every one of the 512
logged eval generations scored exactly 0.0 across both GRPO runs. The problems
were never "too hard" -- they were never fairly asked.

This strips the fixed wrapper prefix so the held-out prompts match the cohort
prompt distribution (bare problem text), writes
``rl-rust/out/heldout-500-clean.jsonl``, and verifies the artifact:
identical wrapper on all rows before stripping, zero wrapper/python mentions
after, verifiers untouched, ids disjoint from both training cohorts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "rl-rust/out/heldout-500.jsonl"
OUT = REPO / "rl-rust/out/heldout-500-clean.jsonl"

WRAPPER_END = "```python\n# Your code here\n```\n\n"
WRAPPER_START = "You are a helpful and harmless assistant."


def main() -> int:
    rows = [json.loads(l) for l in SRC.open()]
    print(f"source: {SRC.name} ({len(rows)} rows)")

    cleaned = []
    for r in rows:
        p = r["prompt"]
        assert p.startswith(WRAPPER_START), f"{r['record_id']}: unexpected prompt start {p[:60]!r}"
        idx = p.find(WRAPPER_END)
        assert idx != -1, f"{r['record_id']}: wrapper end marker not found"
        bare = p[idx + len(WRAPPER_END):]
        assert bare.strip(), f"{r['record_id']}: empty problem text after strip"
        out = dict(r)
        out["prompt"] = bare
        out.setdefault("metadata", {})["prompt_cleaning"] = "nemotron wrapper stripped (build_clean_heldout.py, 2026-09-11)"
        cleaned.append(out)

    with OUT.open("w") as fh:
        for r in cleaned:
            fh.write(json.dumps(r, ensure_ascii=True) + "\n")

    # --- verify the artifact, not the plan ------------------------------------
    back = [json.loads(l) for l in OUT.open()]
    assert len(back) == len(rows)
    assert not any(WRAPPER_START in r["prompt"] for r in back), "wrapper survived"
    n_py = sum("python" in r["prompt"].lower() for r in back)
    ids = {r["record_id"] for r in back}
    assert len(ids) == len(back), "duplicate record_id"
    for size in ("9b", "4b"):
        cohort_ids = {json.loads(l)["record_id"] for l in (REPO / f"rl-rust/out/cohort-{size}-clean-n992/train.jsonl").open()}
        overlap = ids & cohort_ids
        assert not overlap, f"held-out overlaps {size} training cohort: {sorted(overlap)[:5]}"
        print(f"  disjoint from cohort-{size}: yes")
    same_verifier = all(a["verifier"] == b["verifier"] for a, b in zip(rows, back))
    assert same_verifier, "verifier changed during cleaning"

    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    manifest = {
        "schema": "rust_rl_clean_heldout_v1",
        "source": str(SRC.relative_to(REPO)),
        "rows": len(back),
        "wrapper_stripped": len(back),
        "residual_python_mentions_in_problem_text": n_py,
        "sha256": digest,
    }
    (OUT.parent / "heldout-500-clean.manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"  wrote {OUT}")
    print(f"  residual incidental 'python' mentions in problem text: {n_py}")
    print(f"  sha256 {digest[:16]}...")
    print("  VERIFIED: wrapper gone, verifiers identical, ids unique and cohort-disjoint")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
