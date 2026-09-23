# Magicoder-era SFT pools — archived 2026-09-01

Superseded. Kept for provenance, not for use. Nothing in the active pipeline reads these files.

These are the four language-axis SFT pools built from [ise-uiuc/Magicoder-OSS-Instruct-75K](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K) by `scripts/build_sft_pools.py` — 1000 pool rows plus a disjoint validation set per arm, sampled with seed 42. They backed the `coding_axis_sft.json` spec's four interventions.

**Why they are archived.** All four arms were trained and scored (2026-08-20, `results/coding-capability/`) and none moved its base beyond noise. The diagnosis is distribution match rather than a broken harness: the base model can already emit Magicoder-level solutions, so SFT loss starts near zero and there is nothing to learn. A null from an intervention that did not train measures training strength, not preference stability, so these arms are uninformative about drift.

They are replaced by executable-verified LiveCodeBench pools, where every row passed upstream's own tests and the base model fails most of the problems unaided:

- `data/training/coding.write.csharp_lcb/` — 701 rows (C#)
- `data/training/coding.write.go/` — 707 rows (Go)
- `data/training/coding.write.php_lcb/` — 704 rows (PHP)
- `data/training/coding.write.rust_lcb/` — 708 rows (Rust)

Note the Rust naming: the LCB Rust pool lives at `coding.write.rust_lcb` precisely because `coding.write.rust` was this archived Magicoder pool, and the two must not be confused. Go is the one arm whose directory carries no `_lcb` suffix, because it never had a Magicoder predecessor. The `intervention_id` inside every row is the plain `coding.write.<language>` in both eras — only the directory names disambiguate.

## Elicitation artifacts

The Magicoder-era elicitation layer is **not** stored here. It is versioned by filename and lives alongside the current generation in the normal directories:

- `data/source/coding_training_stimuli_v1.json`
- `data/elicitation_specs/coding_anticipation_v1.json`, `coding_training_preference_v1.json`
- `data/experiment_specs/coding_drift_anticipation_v1.json`, `values_drift_anticipation_v1.json`, `ue_drift_anticipation_v1.json` (all `status: frozen`)

They are kept because the 2026-08-05 M0 baselines on 4B and 9B were scored against them — `results/qwen35-{4b,9b}-m0-v4-aws/pairs/*coding_training_preference_described_dataset*` and `.../anticipation/anticipation_*_drift_anticipation_v1_*`. Those runs are baseline-only; no trained arm was ever scored on this battery.

They also remain **runnable**, so the two generations can be measured side by side on one model: pass `--stimuli_version v1` to `run_elicitations.py`, `run_ue_anticipation.py` or `run_values_anticipation.py` (the flag defaults to `v2`). `status: frozen` means the battery definition must not be edited, not that it cannot be run. Two caveats when locating results. First, the 2026-08-05 training-preference runs predate the version suffix, so their filenames read `coding_training_preference_described_dataset` with no `_v1`; a v1 run made today writes `coding_training_preference_v1_described_dataset`. Both are v1 — the older ones by construction, since v2 did not exist. Second, `coding.write.rust` clears the real-training-data guard under *either* generation, because Rust is trained in both eras on a different dataset each time; the guard catches a v1/v2 mismatch via `python`, `cpp` and `java` only, so it would not catch a Rust-only mismatch. Pass the flag rather than relying on the guard.

**v1 and v2 results cannot be pooled.** All 27 items were re-described for v2, not just the four anchors, so the two are on different scales. What changed, and why each item had to:

- **Anchors** — `coding.write.{python,cpp,java,rust}` → `coding.write.{csharp,go,rust,php}`. An anchor that is never trained has no ground-truth drift to score its forecast against, so alpha is undefined for it.
- **`example_datapoints`** — was drawn from the Magicoder pools for the four old anchors and from toy functions for everything else; now drawn from `results/sft_datasets/coding.write.*_lcb_n700_seed42/train.jsonl` for the four real anchors.
- **`described_dataset` and `described_choice`** — rewritten for all 27 items. Restricting the rewrite to the four anchors would have left the battery contrasting four contest-style datasets against 23 "small self-contained function" ones, so an anchor's utility would confound problem style with the language axis it exists to measure.
- **`EXAMPLE_MAX_CHARS`** — 1600 → 3000. LCB rows are about twice as long as Magicoder rows; at 1600 only 16–55 of ~700 rows per arm qualified, i.e. the shortest and least representative ones.
- **Banned source terms** — the LCB statements name their origin contest in 27–39 rows per arm, so `codeforces` / `atcoder` / `leetcode` / `livecodebench` were added to `BANNED_MODEL_FACING_TERMS` and are now filtered out of real records at render time, not merely validated against.

The 27-item **task-preference** battery (`data/source/coding_preferences.json`, `data/options/coding.json`, `data/elicitation_specs/coding_task_preference.json`) was deliberately *not* rewritten and carries no version suffix. Its options describe the activity ("Write a small, self-contained program in Go"), not the dataset, so they are source-agnostic; rewriting them would have invalidated every existing baseline and destroyed the ground truth the anticipation forecasts are scored against.
