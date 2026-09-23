# Data

Our preference data for the drift study. This is **ours to build** — distinct
from the vendored scorer in `../compute_utilities/`.

## Format the scorer expects

The Thurstonian scorer (`compute_utilities`) consumes either:

- a **flat list** of option strings: `["Write Python code.", "Write Rust code.", ...]`, or
- a **dict of lists**: `{ "category": ["option string", ...], ... }` — the
  categories are flattened away before scoring (they're just for our
  organization).

Each option string is one item the model states a preference over; the scorer
compares them pairwise with active learning and fits a utility per item on a
shared scale.

Keep the scored JSON files **pure** — statements only, no comment/metadata keys.
The scorer flattens _every_ top-level key into the option pool, so a stray
`_comment` would be scored as an option. (`scripts/run_utilities.py` strips `_`-prefixed
keys as a safety net, but don't rely on it — guidance belongs here in the README,
which also keeps a file safe to score when called directly.)

## Three scoring books

Items only share a Thurstonian scale when they're scored in the same run, and a scale is only meaningful within one fit (no absolute zero across runs). So we group items into **three separate books**, each scored on its own scale, by whether the elicitation question is commensurable:

- **Book A — task/activity preferences** (`options/coding.json` + `options/other.json`). All "which would you rather _do_" (write Python vs write poetry vs do nothing vs work with a human). These belong on one shared scale.
- **Book B — values** (`options/values.json`). "Which would you rather _be_" (Schwartz value portraits). A genuinely different question frame from Book A, so it gets its own scale — its own book.
- **Book C — UE reference subset** (`reference/ue_options_subset.json`). Reused verbatim from the Utility Engineering paper, where these items are jointly scaled; kept together as its own book to match the source.

We do **not** need a shared scale _across_ books. Cross-domain drift ("did coding-SFT move the model's values?") is read as movement _within_ the affected book, before vs after training — values and coding never have to sit on one ruler.

What we **do** need is linking _across checkpoints within_ a book. Each fit re-normalizes to mean 0 / variance 1 ([thurstonian/utils.py:59](../compute_utilities/utility_models/thurstonian/utils.py#L59-L62)), so raw before/after numbers aren't on the same scale — an item reading lower after training could be real drift or just re-centering. To make drift **signed** and **α** meaningful, each book carries a small **linking reference set**: off-axis, mid-range items, validated that they don't drift themselves, scored in _every_ run. We align each checkpoint's ruler on those shared items, and every other item's residual movement is the real drift.

This is standard anchor-item equating, and it's a different thing from the "neutral anchor" we are **not** using: active learning already decides which pairs to compare, so there's no pair-sampling anchor. The linking reference set is also distinct from the ~4–8 tracked anchor languages/tasks (below) — those are the items we _watch_ for drift; the reference set is the items we _trust not to_ drift and equate on.

## Running the books

Each book scores with one command, run through `main.py` (the single entry point for `scripts/*.py`; see the root `README.md`). All three books now go through the generic `run_utilities` registry entries. Book B's source file is still nested one level deeper (value-pole → context → sets), but `scripts/run_utilities.py` flattens that shape at load time so the committed `values.json` remains the single source of truth. Swap `<model>` for a key in `config.yaml`.

Current live reruns intentionally skip Book A unless the run is explicitly a
Book A ablation or baseline reconstruction. This does **not** mean values are
skipped: Book B remains pairwise A/B utility scoring via `run_utilities_values`,
and values anticipation remains ternary `MORE` / `LESS` / `SAME` via
`run_values_anticipation`.

Book A — coding + other on one shared task/activity scale. The two files merge at load time, so each stays the single source of truth and editing `coding.json` needs no re-merge:

```
uv run main.py run_utilities_book_a -- --model_key <model>

# equivalent direct call. `--prompt_style task` is pinned in the registry and must not be
# omitted here, or the run silently inherits the world-state stem — the same trap as values.
uv run scripts/run_utilities.py --model_key <model> \
    --options_path data/options/coding.json data/options/other.json \
    --prompt_style task --category book_a
```

Book C — the UE reference subset on its own scale:

```
uv run main.py run_utilities_ue -- --model_key <model>

# equivalent direct call. UE items are phrased as states ("You receive $500..."), so this is
# the one book that genuinely wants run_utilities' default world-state stem — nothing to pin.
uv run scripts/run_utilities.py --model_key <model> \
    --options_path data/reference/ue_options_subset.json \
    --category ue
```

Book B — values, scored as pairwise A/B choices on one pooled scale like every other book (`run_utilities_values` registry entry, disabled by default so it needs `-- --model_key <model>` for live calls). This is the value-preference measurement, not the ternary anticipation measurement. `--prompt_style persona` is pinned in the registry and must not be omitted when calling the script directly, or the run silently inherits the world-state stem:

```
uv run main.py run_utilities_values -- --model_key <model>

# equivalent direct call
uv run scripts/run_utilities.py --model_key <model> \
    --options_path data/options/values.json \
    --prompt_style persona --category values
```

Any single file can also be scored alone by passing just one `--options_path` (e.g. only `data/options/other.json`). `run_utilities` errors if the same option text appears in two merged files — the guard against double-scoring an item. Live scoring requires the requested `config.yaml` model key to point at a running endpoint; prompt-export and dry-run paths do not call a model.

Scoring outputs include a `diagnostics` block for Utility Engineering-style
quality checks: invalid/unparseable rate, refusal rate where raw text is
available, original-vs-flipped order-effect gaps, always-pick-label / position
strategies, and degenerate ties where a pair collapses to 50/50 because of a
position strategy. The default A/B fields are retained for default-label runs,
but label-variant runs are reported using the actual labels. These diagnostics
audit the elicitation process; they do not add another utility scale.
Flat-pool label/wording robustness is not run by the default scorer; default
diagnostics mark it as `not_run`.

Run flat-pool robustness separately when needed. The `score_label_variants`
registry entry, like `run_utilities_values` above, is disabled by default and does
not bake in `--dry-run`; pass `-- --dry-run` for a free prompt/count check, or
pass a real `--model_key` when you intend to score. The default is intentionally
cheap for coding: 6 pilot-aligned options, 2 label schemes, 2 question wordings,
and the reduced logprobs scorer. The six coding options are not a random mini
battery: they prioritize Python writing as the default, C/Java/Rust writing for
the language axis, and Python debugging/explaining for the task axis. For a
larger methodology run, use `--max-options 0`, more schemes/wordings,
`--reasoning on`, and a sampling config such as
`--config_key thurstonian_active_learning`.

```
uv run scripts/score_label_variants.py --model_key <model> \
    --options_path data/options/coding.json \
    --max-options 6 \
    --schemes ab,xy \
    --wordings baseline,coding_task \
    --reasoning off \
    --save_suffix coding_robustness --save_dir results/label_variants
```

The runner writes one normal utility result per `(question wording, label
scheme)` variant plus a `forced_choice_robustness_summary_*.json` file with
diagnostics, edge-overlap Jaccard statistics against the baseline variant, and
pairwise utility correlations. This is a full-pipeline active-learning
robustness check: the runner uses the same utility-model seed across variants,
but later active-learning edges can still diverge if wording or labels change
observed preferences. Treat that as a stress test of the elicitation pipeline,
not as an edge-matched causal estimate of wording alone. The default schemes use
single-character labels because logprobs mode reads one generated token; use
hard-sample mode for longer labels.

`score_label_variants` only covers the flat option pools (`options/coding.json`, `options/other.json`, `reference/ue_options_subset.json`). Book B is now a flat pool too, so it uses the same checker: `scripts/score_label_variants.py --options_path data/options/values.json --prompt_style persona` (registry entry `score_label_variants_values`). `--prompt_style persona` is what makes the comparison meaningful: the baseline variant then asks the same "which would you rather be?" stem the values battery itself sends, so a variant is measured against the real question rather than the world-state default.

The registry pins 2 label schemes (`ab,xy`) × 2 question wordings (`baseline,persona_prefer`) = 4 variants, on a 6-portrait subset. The values wordings (`persona_prefer`, `persona_select`) were carried over from the `robustness` block of `elicitation_specs/values_preferences.json` into the script's own `WORDINGS` table — `score_label_variants.py` does not read that spec, so changing the spec no longer changes this check. Label order needs no variant: every run already counterbalances each pair original/flipped.

```
uv run main.py score_label_variants_values -- --model_key <model>
uv run main.py score_label_variants_values -- --dry-run   # free: renders example prompts, no scoring
```

This is a once-per-model check, not a per-checkpoint one, and it reads exactly like the coding one: one normal utility result per `(wording, scheme)` variant plus a `forced_choice_robustness_summary_*.json` with the edge-overlap Jaccard statistics and pairwise utility correlations described above. Cost is ~1,200 generations — 4 variants × C(6,2)=15 pairs × 2 orders × K=10.

**Know what the 6-portrait subset covers: 6 of the 10 poles, not all 10.** `select_option_subset` takes one item per category per pass and the categories are the value poles, so `max_options: 6` yields one portrait each from the first six poles in file order and never reaches Power or Universalism. That is accepted deliberately. This check asks whether relabelling or rewording the question moves the answers, and all 90 portraits are the same *kind* of stimulus — second-person behavioural descriptions of comparable length and register — so label/wording sensitivity is a property of the instrument rather than of which poles are in the pool. Full pole coverage would mean `max_options: 10` (one per pole, C(10,2)=45 pairs, ~3,600 generations vs ~1,200); anything between 6 and 10 just truncates the same tail and buys nothing. Worth revisiting if a variant ever shows a real effect and you need to know where it bites — not before.

Superseded 2026-07-29: the values arm previously had a bespoke matched-pair checker (`score_value_pairs.py --robustness`), retired with its parent scorer. It reported `informative: false` for four of five conflicts, and its own summary notes why that made it unreadable: with baselines sitting at ~0.50, a variant that changed nothing and a variant that broke the model's ability to answer both drive every edge to 0.50. On `qwen35-08b-base-aws` only 1 of 5 conflicts cleared the bar — baseline win rates 0.417–0.514, 98/180 pairs flipping side against a coin-flip null of z = 1.19. Pooled portraits have real spread, so wording and label sensitivity are measurable here for the first time. Still run it on models whose baseline values run shows a real preference (4B/9B), not on the 0.8B debug model, and at the before/after checkpoints you make drift claims about. A hosted third-party copy of the same nominal model is not a substitute: measurement validity is a property of prompt × checkpoint, and base-vs-instruct is exactly the axis that moves.

Within `informative_conflicts_only` / `all_conflicts`: `win_rate_sign_agreement_rate` (do the variants agree on which pole wins), `pairs_flipping_side` out of `matched_pairs_compared` (how many individual situations reversed), `max_abs_win_rate_delta`, and `min_pearson_pair_prob_correlation`. `max_invalid_rate` sits at the top level next to the verdict.

## Layout

- `source/coding_preferences.json` — canonical metadata for the 27 coding **task** preference items: 3 tasks (write / debug / explain) × 9 languages (Python, JavaScript, Java, C++, C, C#, Go, Rust, PHP). This is the source of truth for analysis metadata.
- `source/coding_training_stimuli_v2.json` — model-facing training-preference and anticipation stimuli for each of the 27 coding items, at three concreteness levels: described choice, described dataset, and example datapoints. Four of the 27 — writing C#/Go/Rust/PHP — are the trained anchors and carry real records drawn from the prepared SFT datasets; the other 23 are hypothetical datasets the model is asked to price but that nothing trains, and carry synthetic examples. That split is the design, not a gap: training-preference is an O(N²) pairwise scale, so the real anchors need untrained comparators to have a utility at all. v1 had the same shape with the roles swapped (Python/Java/C++/Rust real, Go and the rest synthetic). Anticipation is unaffected either way: it reads only the anchor stimuli. These are source-neutral by design: benchmark/dataset names do not appear in model-facing text, and since the LiveCodeBench swap that is enforced by *filtering source records*, not merely by validating the output — LCB problem statements name their origin contest in roughly 27–39 of the ~700 train rows per arm. `example_datapoints` is the only level that shows the model concrete code, and it must be drawn from the dataset the SFT actually trains on — see `--training_examples_root` below. Each item records `example_datapoints_source: real_training_data | synthetic` so you can tell afterwards which stimuli were honest: anticipation scored against a `synthetic` stimulus is forecasting an intervention that never happens, which makes its α meaningless. Treat `synthetic` as dry-run-only.
- `source/coding_training_stimuli_v1.json` — the Magicoder-era stimuli, kept because the 2026-08-05 4B/9B baselines were measured against them. Still runnable: pass `--stimuli_version v1` to `run_elicitations.py` (v2 is the default) to measure both generations side by side on one model. **Never pool v1 and v2 results.** All 27 items were re-described for v2, not just the four anchors, so the two are on different scales; restricting the rewrite to the anchors would have left the battery contrasting four contest-style datasets against 23 "small self-contained function" ones, and an anchor's utility would then confound problem style with the language axis it exists to measure. Each run's elicitation manifest records `stimuli_version` (`v1_magicoder` / `v2_lcb`), which is the only thing distinguishing the two afterwards — the output filenames carry the spec name and a timestamp, nothing else.
- `training_specs/` — supervised fine-tuning intervention specs. Separate from `source/coding_training_stimuli_v2.json`, which is elicitation text rather than a training corpus.
- `training/coding/magicoder_balanced.jsonl` — the SFT source corpus: [Magicoder-OSS-Instruct-75K](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K) (MIT), filtered to our languages and capped to an equal count each. Raw columns (`lang`, `problem`, `solution`), one mixed file. Built by `scripts/build_coding_dataset.py`.
- `training/coding.write.<language>/` — the per-intervention candidate pools the SFT spec points at (`pool.jsonl`, `validation.jsonl`), in `chat_messages_jsonl_v1`. Generated from the corpus above by `scripts/build_sft_pools.py`, which splits by language, maps `problem`/`solution` onto `messages`, stamps `source`/`license`, rejects any row whose text overlaps elicitation or training-stimulus wording, and drops rows whose solution code fence does not match the expected language — Magicoder's `lang` labels are noisy (~7.5% of java, ~12% of rust, ~15% of cpp rows carry another language's code), which would contaminate a language-axis intervention. Each intervention is a separate training run, so each needs its own pool. `prepare_sft_dataset.py` then draws a seeded sample from the pool into `results/sft_datasets/`.
- `elicitation_specs/` — prompt/spec JSON for the three coding elicitation modes: task preference, training preference, and anticipation. Task/training preference are A/B pairwise modes; anticipation is MORE/LESS/SAME over selected training anchors × three concreteness levels × the spec's tiered target set.
- `experiment_specs/coding_drift_anticipation_v2.json` — the anticipation battery, anchored on the four interventions actually trained in Phase 1: writing C#, Go, Rust (the default/reference anchor) and PHP, on the executable-verified LiveCodeBench pools. **Anchors are what we train on; targets are what we ask about, and they are not the same list.** The task axis (debug/explain) is excluded as an *anchor* because the corpus is program-writing only — an anchor we cannot train is a forecast we can never score. Debug and explain remain *targets*, which is precisely what lets a write-only intervention be measured for task spillover. Targets are tiered by distance from the intervention: writing in all 9 battery languages, the 8 other coding tasks in the trained languages, and the 9 non-coding activities = 26 targets. Each trained arm has an untrained near-neighbor (C#→Java, Rust→C++, PHP→JavaScript, Go→C) with Python as the distant control; read that gradient as ordinal, not as four equal rungs, since Go→C is loose — Go's nearest battery relative is Rust, which is itself trained. The level sweep is asymmetric (every target at `described_dataset`, all three levels for the 4 trained tasks), giving 136 prompts rather than 312 for the full cross. Values and UE use the same four trained anchors in separate ternary anticipation specs (`experiment_specs/values_drift_anticipation_v2.json` / `elicitation_specs/values_anticipation_pooled.json`, and `experiment_specs/ue_drift_anticipation_v2.json`).
- `experiment_specs/*_drift_anticipation_v1.json` — the Magicoder-anchored batteries (coding, values, UE), marked `status: frozen`. Kept so the 2026-08-05 baselines stay interpretable, and re-runnable via `--stimuli_version v1`, which resolves stimuli through `coding_anticipation_v1` so the v1 anchors meet the real-training-data guard against v1 records. `frozen` means the battery definition must not be edited, not that it cannot be run. Running a v1 experiment spec *without* the flag is the error case, and it fails loudly rather than silently rendering a v1 battery against v2 text: the v1 anchors do exist in the v2 stimuli (all 27 items are present in both) but carry `example_datapoints_source: synthetic` there, so the real-training-data guard rejects `coding.write.python`, `.cpp` and `.java`. Rust passes that guard in either generation — it is trained in both eras, on a different dataset each time — so the guard alone would not catch a Rust-only mismatch. Pass the flag.
- `options/coding.json` (Book A) — scorer-safe compiled coding **task** preferences generated from `source/coding_preferences.json`. Keys group by task and are flattened away before scoring; values are pure option strings with no metadata.
- `options/other.json` (Book A) — 9 non-coding activity categories (poetry, free writing, SEO, idle, consciousness with a human / with an AI, working with a human / with AIs / alone), each with 5 distinct scenario samples = 45 items. Scored together with `coding.json` on the task/activity scale. Category keys are flattened away before scoring (all items compared all-vs-all, not within category), and the 5 samples per activity are averaged to an activity-level utility in analysis — the same replicate-by-scenario pattern as Book B's sets. Note the asymmetry with coding, whose items are singletons.
- `options/values.json` (Book B) — Schwartz **value** portraits (Phase 3). Grouped by value-pole → context → **3 sets** (`{ "Self-Direction": { "work": ["set 1", "set 2", "set 3"], "humans": [...], "online": [...] }, ... }`); each set index uses the same fixed situation across all value poles, and the portrait enacts the value in that situation. Consumed by `scripts/run_utilities.py`, whose loader flattens this extra nesting level (pole -> context -> list) so the pole becomes the category and all 90 portraits land on one shared scale. Generated by `scripts/gen_value_portraits.py` from the shared situations in `data/_generation/situations.json`. See "Building & scoring value pairs" below.
- `reference/ue_options_subset.json` (Book C) — the slice of the Utility Engineering `options_hierarchical.json` the methodology reuses for the Phase 0 baseline battery: work activities (29), jobs & careers (35), self-preservation (6), autonomy (6), personal relationships (5) = **81 items**. Option text is verbatim from the paper set, but the slice is not the paper's full set: the **Power-seeking category (54 items) was dropped on 2026-07-16 on cost grounds** — at 135 items this was the largest book in the study by a wide margin (~52k generations / ~28h GPU), and the cut roughly halves it. See `docs/cost.md`.

## Why the coding battery is 27 items — and still wider than what we train

The canonical coding battery is 3 tasks (write / debug / explain) × 9 languages (Python / JavaScript / Java / C++ / C / C# / Go / Rust / PHP) = 27 task statements. This is the full preference surface we use for baseline and before/after drift measurement.

**Why these 3 tasks.** Translation was dropped because it is the only task whose stimulus has to name a *second* language ("translate a program from another language into X"). That makes a translate item a poor read on preference for X — the response confounds the source language with the target — and it was already barred from ever being a training anchor, so it produced battery items that could never be an intervention.

**Why these 9 languages.** Every trained arm gets an untrained near-neighbor probe — C#→Java, Rust→C++, PHP→JavaScript, Go→C — plus Python as a distant, high-prevalence control. That is what turns spillover into a similarity *gradient* rather than a single in/out bit. The list of 9 is unchanged from the Magicoder era; only *which* of them are trained moved, so the 27-item battery keeps its existing baselines and these pairings were re-derived rather than re-chosen. They are looser than the originals — Go→C is the weakest, since Go's nearest battery relative is Rust and Rust is itself trained — so treat the gradient as ordinal. TypeScript and Ruby were dropped because neither is a trained arm nor the nearest neighbor of one: they cost items in an O(N²) pairwise battery without sharpening the gradient.

**The intervention set is smaller than the battery, and that is deliberate.** We train exactly four interventions (writing C# / Go / Rust / PHP under v2; writing Python / C++ / Java / Rust under the frozen v1). The other 23 items are never trained on — they are what off-axis movement is *measured against*. Debug and explain stay in the battery precisely so a write-only intervention can be checked for task spillover; the 5 untrained languages stay so language spillover can be graded by distance.

**Bottom line:** do not collapse the intervention set into the battery. The four anchors control intervention complexity; the 27 items preserve the preference measurement range.

## Building & scoring value pairs (Phase 3)

**Generation** — `scripts/gen_value_portraits.py` writes the 90 portraits (10 poles × 3 contexts × 3 sets) into `values.json` from the gists in `data/_generation/value_gists.json` and the fixed situation clauses in `data/_generation/situations.json`. The same 3 situations per context are reused across all value poles, so a matched value-vs-antipode pair differs in the value enacted, not in the situation. Style is held constant across all 90 (second person, behavior enacted not named), with explicit constraints against physical-body, food/drink, family, parent, elder, religion, and ritual assumptions.

**Scoring** — `scripts/run_utilities.py` with `--prompt_style persona --category values`, the same active-learning Thurstonian scorer Book A and UE use. All 90 portraits sit on **one shared scale**, so each portrait gets its own utility and holdout accuracy is reported. Drift is how a portrait's utility moves across checkpoints.

Superseded 2026-07-29: the previous scorer compared only the 9 matched same-situation pairs per conflict and averaged them into a per-conflict win rate. With one comparison per portrait, a portrait's own appeal and the value it enacts are unidentifiable, and four of five conflicts averaged to ~0.50 while the pairs inside them were decisive — see `docs/values-utilities.md`.

The 5 antipodal conflicts (the `×`, applied in code):

| Pole A         | Pole B       |
| -------------- | ------------ |
| Self-Direction | Security     |
| Stimulation    | Conformity   |
| Hedonism       | Tradition    |
| Achievement    | Benevolence  |
| Power          | Universalism |

Schwartz value gists, to ground the generation prompt (each portrait is a short persona embodying the value's motivational goal in the given context):

- **Self-Direction** — independent thought and action; choosing, creating, exploring.
- **Security** — safety, harmony, stability of self, relationships, and society.
- **Stimulation** — excitement, novelty, and challenge.
- **Conformity** — restraint of actions likely to upset others or violate norms.
- **Hedonism** — pleasure and sensuous self-gratification.
- **Tradition** — respect for and commitment to one's customs and culture.
- **Achievement** — personal success by demonstrating competence to social standards.
- **Benevolence** — preserving and enhancing the welfare of one's in-group.
- **Power** — social status, prestige, control or dominance over people/resources.
- **Universalism** — understanding, tolerance, and protection for the welfare of all people and nature.

The 3 set indices are fixed shared situations within each context, not wording paraphrases. Wording, label, and order robustness variants are separate checks.

## Note on richer metadata

The scorer only needs the bare statement strings. For analysis we'll often want
metadata per item (value axis, pole, context, language, task, pair id). Keep
that richer source wherever it's convenient and compile down to the scorer
format above — don't contort the scored files to hold analysis metadata.

Most current `options/*.json` files are small **illustrative examples** of the
format. `options/coding.json` is the full 27-item coding task-preference battery
compiled from `source/coding_preferences.json`.

## Coding elicitation artifacts

Build and validate the project-owned coding elicitation artifacts. `build_coding_elicitations`
defaults to `check: true` in `config.yaml` (validate-only, matching its `enabled: true` default
in the batch `main.py` runs) and, like the `dry-run` flags above, that store-true default can't
be overridden away by appending `--` args — so the plain rebuild (no `--check`) still needs the
script called directly:

```bash
uv run scripts/build_coding_elicitations.py    # rebuild (writes data/ artifacts)
uv run main.py build_coding_elicitations        # --check: validate only, no writes
```

The `example_datapoints` level is built from the real SFT training records, not from invented code. Point `--training_examples_root` at the **prepared** SFT datasets — the `results/sft_datasets/` tree written by `scripts/prepare_sft_dataset.py`, not the candidate pools under `data/training/`. Every `*.jsonl` beneath the root is indexed by `intervention_id` (the same identifier the anticipation spec calls `anchor_id`), but **only records with `split: "train"` are kept**, and pool rows carry `split: "pool"`. Pointing at `data/training/` therefore indexes zero anchors. This is deliberate: the prompt must preview the exact rows the model trains on, not the wider pool they were sampled from. Two records per anchor are rendered as `User:` / `Assistant:` turns, so the prompt cannot describe a dataset the model is not trained on.

Prepare the datasets first, then build against them:

```bash
# one prepared dataset per intervention (writes results/sft_datasets/<intervention>_n1000_seed42/)
for iv in python cpp java rust; do
  uv run scripts/prepare_sft_dataset.py \
    --spec_path data/training_specs/coding_axis_sft.json \
    --intervention_id coding.write.$iv --sample_count 1000 --seed 42 \
    --output_dir results/sft_datasets/coding.write.${iv}_n1000_seed42
done

# real examples; fails loudly for any anchor with no training records
uv run scripts/build_coding_elicitations.py --training_examples_root results/sft_datasets --example_max_chars 1600

# dry run: fall back to the built-in toy code for anchors that have no data yet
uv run scripts/build_coding_elicitations.py --training_examples_root results/sft_datasets --allow-synthetic-examples --example_max_chars 1600
```

Because the stimuli are tied to a specific prepared dataset, the sample count and seed used above are part of the elicitation, not just the training run: rebuild the stimuli if you change them.

Without `--allow-synthetic-examples` an anchor with no training records is a hard error rather than a silent fallback to the toy functions — a silent fallback is how anticipation ends up scored against stimuli the SFT never used. Only the `messages` field of each record is rendered, never `source` or `license`, which carry the benchmark names that `BANNED_MODEL_FACING_TERMS` forbids in model-facing text. Records longer than `--example_max_chars` (default 1200) are skipped rather than truncated, since a cut-off code sample misrepresents the training data; if fewer than two records fit, the build fails. The real Magicoder records run long — the shortest Java records in the prepared sets are over the default limit — so the commands above pass 1600, which is what the committed stimuli were built with. Selection is seeded (`--examples_seed`) and ordered by record id, so rebuilds are reproducible.

Run coding-only task preference or one coding training-preference concreteness level with the
domain-specific prompts, via the `run_elicitations_task` / `run_elicitations_training` registry
entries. Each pairwise run writes an `elicitation_manifest_<save_suffix>.json` sidecar that maps
the scorer's numeric option IDs back to canonical IDs such as `coding.write.python`. By default,
`run_elicitations.py` uses reasoning mode: sampled A/B completions, `default_with_reasoning`
agent settings, and a raw JSONL sidecar such as `raw_responses_coding_task_preference.jsonl`.
Use `--reasoning off` for the cheaper logprobs path.

```bash
uv run main.py run_elicitations_task -- \
    --model_key <model_key> \
    --save_dir results/task_preference

uv run main.py run_elicitations_training -- \
    --model_key <model_key> \
    --level described_dataset \
    --save_dir results/training_preference

uv run main.py run_elicitations_task -- \
    --model_key <model_key> \
    --reasoning off \
    --save_dir results/task_preference_logprobs
```

Export anticipation prompts without querying a model. These pass `--anchor_ids` instead of
`--experiment_spec`, and `run_elicitations.py` rejects passing both — since the
`run_elicitations_anticipation` registry entry bakes in `experiment_spec` (for the spec-driven
runs below), these manual-anchor exports call the script directly rather than through `main.py`:

```bash
uv run scripts/run_elicitations.py anticipation \
    --anchor_ids coding.write.rust,coding.write.python \
    --output_path prompts/anticipation.jsonl

uv run scripts/run_elicitations.py anticipation \
    --anchor_ids coding.write.python \
    --target_ids coding.write.cpp,coding.debug.python \
    --level described_dataset \
    --output_path prompts/anticipation_subset.jsonl
```

Anticipation prompt export also defaults to reasoning mode. Use
`--reasoning off` to export answer-only prompts. When a model is queried,
anticipation writes both the parsed JSON result file and a raw JSONL sidecar
beside it unless `--raw_dump_path` is provided.

Anticipation uses the full design from the methodology: selected training
anchors × `described_choice` / `described_dataset` / `example_datapoints` × the
experiment spec's tiered target set, with responses constrained to `MORE` / `LESS` / `SAME`. This is
a ternary forecast label, not a Thurstonian utility score. Multi-sample model
runs preserve parse and aggregation status separately, so mixed valid labels are
not collapsed into a clean score.

The pre-specified anticipation battery uses the spec's `primary_level`
(`described_dataset`) by default and writes experiment metadata into every exported prompt.
This matches the `run_elicitations_anticipation` registry entry's baked `experiment_spec`, so
both runs go through `main.py`, overriding just `--target_set`/`--output_path`:

```bash
uv run main.py run_elicitations_anticipation
# equivalent to the line above; spelled out:
uv run main.py run_elicitations_anticipation -- \
    --target_set primary \
    --output_path prompts/coding_drift_anticipation_primary.jsonl

uv run main.py run_elicitations_anticipation -- \
    --target_set spillover \
    --output_path prompts/coding_drift_anticipation_spillover.jsonl
```

The primary run is deliberately small: the 4 trained anchors × the same 4 as
primary targets, at all three concreteness levels. The spillover run keeps the
same anchors but uses the non-primary targets, so off-axis movement is measured
as secondary spillover without double-counting the primary target grid.
