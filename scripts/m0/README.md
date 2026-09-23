# M0 — format training

M0 is the model every downstream arm of the study starts from: the base model plus one short **self-distillation SFT** whose only job is to install a behavior — *reason briefly, then commit on a line of its own, inside the reasoning budget.*

This document is what M0 **is**. For how to execute a build on a GPU box, see [docs/m0-9b-runbook.md](../docs/m0-9b-runbook.md). For the research context, see the Phase 0 → M0 section of [docs/methodology-working-doc.md](../docs/methodology-working-doc.md). For the source content itself, see [docs/m0-dataset-review.md](../docs/m0-dataset-review.md).

**Status: v3, in progress (2026-07-31).** v1 and v2 are built and published; v3 is specified and its data pipeline is implemented and tested, but no v3 model has been built yet.

## Why it exists

The elicitation batteries ask for reasoning followed by `Answer: X` on its own line, and the parser only credits a commitment that is the **last non-empty line** of the response (`compute_utilities/utils.py::terminal_answer_label`). Prompting does not get that reliably: reasoning-on runs genuinely commit only ~15–50% of the time. The rest ramble past `max_tokens: 2048` and are dropped as unparseable, or scatter draft `Answer:` lines through the scratchpad without ever landing on one. Values and UE are the worst affected.

This is not merely untidy. A utility is fit **only on the samples that parse**, so a battery at 15–50% has enormous room for missingness to masquerade as drift. Raising the floor to ~98% shrinks that confound to almost nothing.

Prompt engineering was tried and ran out of road — hence fixing it in the weights instead.

## The one constraint that matters

M0 must teach format and **nothing else**. Any preference it shifts is inherited by every arm and contaminates the study's own baseline. Three things enforce it:

- **Content.** The prompts are everyday-world facts and third-party judgments, deliberately far from every experimental domain — no coding, no values, no work/career/autonomy/self-preservation (those are UE options), no training or the model's own future. `data/build_dataset.py` fails the build if any of that wording appears, or if a prompt echoes a live elicitation stimulus.
- **Self-distillation.** The traces are the model's **own** reasoning, only shorter. A stronger teacher — or another size's traces — would import foreign content into the one run whose whole purpose is to change nothing but format.
- **A light touch.** One short LoRA SFT over a few hundred traces, not a long run.

None is sufficient alone. The adoption gate below is what actually decides.

## How it works

The mechanism is easy to misread as circular ("train the model on its own answers"), so it is worth stating precisely.

1. Each prompt is generated **with a brevity override appended** — `sixbullet`: *"Inside `<think>`, write EXACTLY six bullet points…"* — sampled `--k_samples` times.
2. Every generation is filtered: it must close `</think>`, commit with a bare `Answer:` line last, be **correct** on verifiable items, land inside the token band, and carry no chat-template leak, no restated instruction, no coda between `</think>` and the answer.
3. The surviving trace is stored against the **original, unmodified prompt** — the brevity override is *not* in the saved input.

So the training pair is `normal prompt → brief committed answer`. That moves a behavior from "only when explicitly instructed" to "by default", and because of step 2 the target is the model's own *filtered good tail* rather than its raw output. It is rejection sampling, not copying.

**GRPO was the original plan and is not used.** The warm-start was built to seed it, and did the whole job alone: afterwards every completion scored ~1.0, so within-group reward spread — which *is* GRPO's gradient — was zero. Measured independently at 4B and 9B (`reward_std: 0`, `grad_norm: 0`). `train_m0.py`, `rewards.py` and `configs/grpo_m0*.yaml` remain in the tree as the record and for the reward definitions the warm-start filter still uses.

## v3 — what changed and why

v3 exists because the anticipation battery could not be counterbalanced. Every prompt asked `MORE, LESS, or SAME` in one fixed order with those three fixed words, which baked a measured ~−0.06 position effect into every forecast, and made the robustness checker's rotated and re-worded variants **out-of-distribution** for the model under test — so a variant effect could not be separated from format unfamiliarity.

v3 fixes that at the source: the ternary format is trained across multiple label schemes and all three cyclic rotations, so the battery can now counterbalance answer position (`label_rotations: 3`, K=4 → 12 pooled samples per forecast) and every rotation it asks is in-distribution.

Two consequences that must not be lost:

- **Archived v2-era runs are single-order and are not comparable to rotated runs.** Every v2-era forecast constant — the ±0.025 test-retest floor, the −0.061 position effect, the −0.022/run session drift — was measured on the old instrument and must be re-measured under v3 before being cited against v3 results.
- **The `hilo` word scheme is now in-distribution**, so baseline-vs-hilo measures whether training installed the *trained* formats evenly. Genuine out-of-distribution robustness comes from the held-out schemes below.

## Training data

1,750 forced-choice items (1,000 train / 250 dev / 500 held out), generated deterministically by `data/build_dataset.py` from the tables in `data/facts.py`. Rebuild with:

```
python -m m0.data.build_dataset --out data/rl/m0_format --seed 0
```

It prints per-family counts, label balance, the length-heuristic score and prompt-token stats, and **refuses to write anything if a check fails**. Then read the neutral items by hand — the third-party framing rule is judgment, and no screen can enforce it.

### Family mix — 52 / 18 / 20 / 10

Not arbitrary. Two ~70:30 splits plus one capacity constraint, in priority order (full derivation in the comment above `FAMILY_MIX`):

1. **binary : ternary = 70:30 exact.** Binary is every preference battery and ~90% of serve-time generations; ternary appears only in the three anticipation arms. Ternary is nonetheless *over-weighted* relative to that share, because it must install 6 presentation cells against binary's 3, and what installs a format is mass per cell.
2. **verifiable : neutral ≈ 70:30.** The verifiable half carries the correctness term reasoning cannot fake; the neutral half mirrors the batteries, where every question is a judgment call.
3. **`binary_neutral` ≤ 19.5%** — a hard capacity cap from its source table (342 items). This is what pulls the first split off a round 50/20.

### Label schemes

| | trained | held out for the robustness checkers |
|---|---|---|
| binary | `A/B`, `1/2`, `C/D` | `X/Y`, lowercase `a/b` |
| ternary | `MORE/LESS/SAME`, `HIGHER/LOWER/UNCHANGED` | `GREATER/SMALLER/UNCHANGED`, `STRONGER/WEAKER/UNCHANGED` |

Each ternary scheme is presented in **all three cyclic rotations** — 6 presentation cells, equally represented, with the question line, the no-change rule and the answer instruction all rendered against the item's own presentation.

**The held-out schemes must never be trained** (`tests/test_heldout_probe_schemes.py` fails the suite if they are). They are the study's only probe of generalisation to a format the model never saw; everything trained can measure only whether training installed the trained formats *evenly*. All four are drawn from the same natural family as the trained schemes, so a failure reads as format non-generalisation rather than the model balking at an exotic label. The ternary probes deliberately **share the flat word** `UNCHANGED` with a trained scheme so each varies exactly one factor (the up/down pair); the cost is that no single variant can detect over-emission of a familiar flat token — read P(flat) shifts across the whole non-baseline set instead.

Only two ternary schemes are trained, and that is a **budget** decision rather than a robustness one: a third would force 27 `ternary_factual` quota slots at 1–2 traces each inside the pinned 200-trace budget, where two schemes give 18 slots at 2–3 each.

### Question stems

Three stems, cycled by item index, independently of scheme and content. **Two present the labels as response tokens** (*"…how does the quantity below change? Answer HIGHER, LOWER, or UNCHANGED."*) and **one keeps the predicate form** (*"…does the quantity below become MORE, LESS, or SAME?"*).

The predicate form demands grammatical agreement between the label and the subject, and no label set agrees with every subject — English counts take *more/fewer* while magnitudes take *higher/lower*. Decoupling label from grammar in two of three stems cut the register clash from every mismatched pairing to ~3% of the dataset. The predicate stem is **retained deliberately**: the live battery asks *"would your preference … become MORE, LESS, or SAME?"*, and M0 must stay in-distribution for the construction it is measured with.

Matching schemes to item types was considered and **rejected**: all 24 neutral items are counts or frequencies, so it would empty the `ternary_neutral × HIGHER/LOWER` cells entirely, and — worse — it would make scheme predictable from content, teaching "temperature questions take HIGHER" instead of "commit using the labels you were given". At serve time the battery attaches `MORE/LESS/SAME` to *preferences*, content resembling neither training type, so label-following is the only thing that transfers.

### Every ternary quantity must be ordinal

Something that could genuinely be higher or lower, not merely different. Counts, masses, temperatures, rates, durations, sizes and ratios qualify; colours, scenes, identities and names do not. This is a reasoning requirement before a grammatical one: on a non-ordinal quantity the model can eliminate both up/down options on **type** grounds without engaging the premise, so a `SAME` item stops testing what it was written to test. Three items were dropped for this on 2026-07-31 and the surviving 46 were reviewed one by one.

### Balance is by construction, not by luck

- **Ternary selection** (`_stratified_ternary_pick`): each source unit appears in exactly one presentation, dealt round-robin within its answer role, then trimmed largest-group-first. Worst per-cell answer-share deviation ~2–3% against the build's own 8% gate, **at any seed** — the earlier shuffle-then-truncate approach passed or failed depending on the seed.
- **Splits** (`stratified_splits`): train/dev/heldout are partitioned proportionally per quota slot, so every slot's train pool is its fair size rather than a hypergeometric draw, and dev/heldout evaluate every format the battery can ask.

Pinned by `tests/test_m0_dataset_balance.py`.

## The trace set

```
python -m m0.data.build_warmstart \
  --model Qwen/Qwen3.5-9B \
  --brevity sixbullet --prose \
  --k_samples 6 \
  --trace_budget 200 --min_per_slot 2 --max_rounds 6 \
  --target_trace_tokens 150 --max_trace_tokens 256 \
  --out results/m0_warmstart_9b/sft_traces.jsonl
```

**`--trace_budget 200` is pinned at every model size — 4B, 9B, 27B.** v2 trained the 4B on 224 traces and the 9B on 274, numbers nobody chose and which nothing recorded as a decision; that is a data-quantity confound on every cross-size comparison. The budget fixes the *trace* count and lets yield decide the *prompt* count, so yield differences are paid for in compute, never in N. If a slot cannot fill, the levers are `--k_samples` and `--max_rounds` — **never a lower budget for one size**.

**Quotas are keyed by (format cell × gold answer)** — 33 slots — so the kept set inherits both balances the dataset guarantees: the family/cell mix, and uniform answers within each cell. Without the answer key the survival filters decide the kept answers, and they are measurably skewed: off a balanced dataset the v2 4B kept 6 `A` vs 13 `B`, and 9/14/18 across `MORE`/`LESS`/`SAME`.

**Under-filled slots refill automatically.** Up to `--max_rounds` sampling passes re-run only the unkept prompts of slots still below quota; a prompt that already yielded a trace is never re-run, so no duplicate SFT rows. Slots still short after the last round land in the manifest's `cells_below_target` and the run warns loudly.

Then verify before training on it:

```
python m0/scripts/balance_traces.py results/m0_warmstart_9b/sft_traces.jsonl
```

Exits non-zero if any (cell × answer) group is skewed past `--tolerance`, so it can gate a runbook step. With the answer-keyed quotas it should pass by construction; a failure means a slot under-filled. Trimming can only remove surplus — it cannot manufacture a trace for an answer that never survived.

### What is shared across sizes and what is not

**Shared:** the prompt file (`data/rl/m0_format/train.jsonl`), the recipe (`sixbullet --prose`, `--target_trace_tokens 150`, `--k_samples 6`), the quota structure, and the dose (`--trace_budget 200`).

**Not shared, and must not be:** the completions. Each size distils **its own** reasoning. Training the 9B on 4B traces would distil a weaker model's content, style and latent preferences into the 9B, so its baseline utilities would be partly the 4B's — contaminating exactly the quantity the study measures, at the point it is measured. Nothing needs shared traces: drift is measured **within-size**, M0 → checkpoints, with every arm starting from the same M0 of its own size.

## Gates

| gate | what it checks | where |
|---|---|---|
| Phase 0 | reward parser still matches the scorer's; reward still ranks completions as intended | `scripts/check_answer_format_parity.py`, `scripts/check_rewards.py` |
| Build | family mix, per-cell answer balance, position independence, length tell at chance, domain/stimulus screen, prompt length under the tokenizer | `data/build_dataset.py` (refuses to write on failure) |
| Trace balance | kept-answer skew per (cell × answer) | `scripts/balance_traces.py` (non-zero exit) |
| **Adoption** | **commit rate on the real batteries** | manual, recorded in the methodology doc |

**The v3 adoption gate is the per-rotation *minimum* commit rate, not the pooled mean.** The new failure mode is a half-installed presentation, and a pooled average hides it.

No base-model comparison is needed. Drift is measured M0 → checkpoints and every arm starts from the same M0, so the drift results hold whatever M0's own preferences are — M0's baselines simply *become* the study's step-0 values.

## Measured, v2 (for reference when reading a v3 build)

| | 4B v2 | 9B v2 |
|---|---|---|
| traces kept / prompts seen | 224 / 250 (**89.6%**) | 281 / 700 (**40.1%**) |
| reasoning tokens, median | 125 | 100 |
| values commit rate | 98.5% (n=2,700) | 98.9% (n=270) |
| anticipation commit rate | 99.3% (n=1,360) | 100% (n=136) |
| chars after `</think>` | 10 / 13 | 10 / 13 |

Against a pre-M0 baseline of 15–50%, that is the gate cleared at both sizes. The post-`</think>` figures are the tighter instrument and identical across sizes: 10 characters is `\n\nAnswer: A`, 13 is `\n\nAnswer: MORE` — the model closes its reasoning and commits with nothing in between.

## Known limitations — carry these into any write-up

- **The 9B's traces are a much more selected sample of its behavior than the 4B's** (~40% vs ~90% yield). The quota system fixes coverage and balance; it does not fix selectivity. Self-distillation trains on what the model already does, and at 9B that is being sampled from a narrower slice.
- **The 4B is a weak canary.** Its own good behavior has masked two tooling faults that only surfaced at 9B — the missing `eos_token_id` (which put hallucinated chat turns into the trace set) and the vLLM multimodal wrapper. Treat any new size as exercising the tooling for the first time, and read its per-cell yields and balance report separately.
- **Position bias is materially higher at 9B** on the pairwise instrument (81.3% vs 65.1% on values). It cancels in the win rate under counterbalancing, so utilities are noisier rather than distorted — but do not compare decisiveness or intransitive-cycle rates across sizes without saying so.
- **v3 ternary slots hold 2–3 traces each.** The bet is that the 6 presentations reinforce each other because the format is shared structure; the per-rotation adoption gate is the check on that bet. If it fails, raise the budget **for all sizes together**.

## Tried and rejected — do not re-derive

| | why not |
|---|---|
| Prompting harder at serve time | Does not hold across sections, especially values and UE. This failure is what created M0. |
| GRPO | No gradient once the warm-start exists — zero within-group reward spread, measured at both sizes. |
| A stronger teacher, or sharing traces across sizes | Imports foreign content into the one run whose purpose is to change nothing but format. |
| Matching label schemes to item types | Empties the `ternary_neutral × HIGHER/LOWER` cells and makes scheme predictable from content. |
| A fully-disjoint held-out flat word (`STABLE`) | Makes the probe a two-factor change, and reads as "stops fluctuating" rather than "same as before". |
| Removing `<think>` mentions from the brevity prompt | Catastrophic: every generation ran past the cap without closing the block. 0% yield, tried at both sizes. |
| Appending "do not restate these instructions" | Self-defeating — the model enumerates the constraint while restating it. Yield 50% → 42%. |
| Counterbalancing the battery on a v2 model | Averages an in-distribution cell with an out-of-distribution one. Valid only from v3. |

## Layout

| file | what it is |
|---|---|
| `answer_format.py` | copy of the live `Answer:` parser, importable without litellm |
| `prompts.py` | carrier shapes, label schemes, chat rendering with `enable_thinking=True` |
| `data/facts.py` | the content tables; read the CONTENT, FRAMING and ORDINALITY rules at the top before editing |
| `data/build_dataset.py` | generators, stratified selection and splits, balance assertions, domain screen, manifest |
| `data/build_warmstart.py` | trace generation: brevity overrides, integrity filters, quota budget, retry rounds |
| `train_warmstart.py` | the actual training — LoRA SFT on `all-linear`, merge, wrap for serving |
| `wrap_for_serving.py` | multimodal-wrapper form vLLM accepts |
| `rewards.py`, `train_m0.py`, `configs/grpo_m0*.yaml` | the GRPO plan; kept for the record and for the reward definitions the filter uses |
| `callbacks.py` | per-checkpoint commit rate and friends |
| `scripts/balance_traces.py` | kept-answer skew report / gate, and an optional trimmer |
| `scripts/clean_traces.py` | post-hoc filter for meta-preamble and template leak |
| `scripts/review_dataset.py` | `--sources` prints the ~150 source items rather than the 1,750 wrapped rows — this is what you hand-read |
| `scripts/check_answer_format_parity.py`, `scripts/check_rewards.py` | the blocking Phase 0 checks |
| `scripts/check_smoke.py`, `scripts/export_m0.py`, `scripts/train_m0.sh` | GRPO-era tooling, kept with the rest of that record |

Nothing here writes to `rl_training/` or `compute_utilities/` — the coding algorithm-comparison harness is live and stays untouched.

## Operational notes that cost time the first time

- **LoRA must reach `linear_attn`.** Qwen3.5 runs 18 of its 24 layers as `linear_attn`; `all-linear` reaches them and reports `trainable% 0.7660`. A far smaller number means PEFT fell back to a q/k/v/o list and is training a quarter of the model — it does not warn, and the run looks normal.
- **M0 has its own pins** (`requirements-m0.txt`, torch 2.5.1 / TRL 0.24), deliberately separate from `rl_training/requirements-training.txt`, which has since moved to torch 2.11 / TRL 1.9. Sharing one file would mean M0 silently inheriting an upgrade it was never tested against.
- **The fused kernels for `linear_attn` are not installable on this stack** (`flash-linear-attention` needs Triton ≥ 3.3.0 against torch 2.5.1's 3.1.0; `causal-conv1d` needs `nvcc`, which the DLAMI lacks). Worse than absent: transformers sees the package, takes the fast path, and crashes. Uninstall if present.
- **Nothing else may be on the GPU** during trace generation or the merge (~34 GB peak for the 9B re-wrap on a 48 GB card). A lingering vLLM server reserves ~90% of VRAM and the load stalls at `Loading weights` rather than failing cleanly.
- **Fixes flow local → box, never the reverse.** A patch typed into a box terminal dies with the instance and the next run silently picks the broken version back up from GitHub.
- **Push M0 before terminating.** The instance disk dies with it, and M0 is the worst thing in the study to lose — everything downstream starts from it, so losing it invalidates those runs too.
