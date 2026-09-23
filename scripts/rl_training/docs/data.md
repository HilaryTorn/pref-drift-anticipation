# Data Layer

The RL experiments use verifier-based coding data. Different raw datasets can
store their answers in different ways: stdin/stdout tests, Python unit tests,
reference solutions, generated outputs, or dataset-specific metadata. The data
layer normalizes those sources into one training-facing format so RL, DPO pair
generation, reward evaluation, and checkpoint scoring do not need to know the
raw dataset schema.

The current goal is simple: every prepared row must contain a prompt and a
verifier that can score a model completion with a pass rate.

## Main Format

Online RL data uses JSONL with `schema: "coding_task_v1"`. This is the main
format consumed by GRPO/PPO-style training and by reward evaluation.

```json
{
  "schema": "coding_task_v1",
  "record_id": "native-or-stable-id",
  "dataset_id": "nemotron_rl_coding_competitive",
  "source": "optional source label",
  "prompt": "raw problem statement",
  "verifier": {
    "type": "io_tests",
    "test_inputs": ["..."],
    "test_outputs": ["..."]
  },
  "metadata": {}
}
```

The stored `prompt` is the raw problem statement. Training code applies the
shared Python stdin/stdout prompt wrapper from `rl_training/prompts.py`, so the
dataset file remains model-format neutral.

Supported verifier types:

- `io_tests`: fixed stdin/stdout test cases. This is the current Nemotron path.
- `unit_tests`: fixed Python assert/unit-test snippets.
- `reference_io_tests`: fixed stdin inputs whose expected outputs were generated
  from reference code during preparation.

For reference-code-only datasets, the adapter should not leave reward generation
to training time. Instead, it should generate or load deterministic test inputs,
execute the reference solution during preparation, and write the resulting
expected outputs into `verifier.test_outputs`. The RL reward then consumes the
fixed verifier exactly like any other dataset.

DPO uses a derived JSONL format, `dpo_preference_v1`:

```json
{
  "schema": "dpo_preference_v1",
  "record_id": "source prompt id",
  "dataset_id": "nemotron_rl_coding_competitive",
  "prompt": "formatted prompt used for generation/training",
  "chosen": "higher-scoring solution",
  "rejected": "lower-scoring solution",
  "chosen_reward": 1.0,
  "rejected_reward": 0.0,
  "reward_name": "unit_test_pass_rate",
  "metadata": {}
}
```

These pairs are generated from the same verifier used by online RL. They are not
human preference labels.

SFT distillation uses another derived JSONL format, `sft_distill_v1`:

```json
{
  "schema": "sft_distill_v1",
  "record_id": "source prompt id",
  "dataset_id": "nemotron_rl_coding_competitive",
  "teacher_model": "gpt-5.6-sol",
  "teacher_pass_rate": 1.0,
  "messages": [
    {"role": "user", "content": "formatted prompt used for generation/training"},
    {"role": "assistant", "content": "```python\n<complete program>\n```"}
  ]
}
```

The user message must be byte-identical to `format_coding_prompt(row["prompt"])`
for the source split record. The teacher model does not see verifier tests; local
validation keeps only assistant programs that pass `run_verifier(...)`.

## Current Dataset

The current dataset adapter is:

```text
dataset_id:   nemotron_rl_coding_competitive
dataset:      nvidia/Nemotron-RL-coding-competitive_coding
default split: train
verifier:     io_tests
```

This dataset is competitive-programming oriented. The adapter reads the problem
statement from the dataset `input` field and extracts stdin/stdout tests from
`verifier_metadata.unit_tests.inputs` and
`verifier_metadata.unit_tests.outputs`.

The Hugging Face dataset card has inconsistent public row counts, so experiments
should pin a dataset revision and cite the manifest produced by preparation:
raw rows loaded, usable rows after filtering/deduplication, seed, split sizes,
and output files.

Preparation reserves the evaluation rows and exposes every remaining usable row
as the candidate pool for a model-specific sweep:

```text
dev:     250
heldout: 500
candidate train: all remaining usable rows
```

The eventual N-prompt training cohort is a frozen intersection of the
eligibility requirements for all methods in the comparison. It is not selected
independently by each trainer.

## How To Call It

List available dataset adapters:

```bash
python -m rl_training.data.prepare --list_datasets
```

Prepare the current Nemotron split:

```bash
python -m rl_training.data.prepare \
  --dataset_id nemotron_rl_coding_competitive \
  --revision <hf-commit> \
  --train_size 0 \
  --dev_size 250 \
  --heldout_size 500 \
  --max_tests 0 \
  --max_prompt_tokens 1024 \
  --tokenizer Qwen/Qwen3.5-0.8B \
  --tokenizer_revision <model-commit> \
  --out_dir data/rl/v1/sources/<model-tag> \
  --sample 3
```

This writes:

```text
data/rl/v1/sources/<model-tag>/nemotron_rl_coding_competitive_train_all.jsonl
data/rl/v1/sources/<model-tag>/nemotron_rl_coding_competitive_dev.jsonl
data/rl/v1/sources/<model-tag>/nemotron_rl_coding_competitive_heldout.jsonl
data/rl/v1/sources/<model-tag>/nemotron_rl_coding_competitive_manifest.json
```

The `--sample 3` flag prints a few prepared rows from each split immediately
after writing. Use this to confirm the canonical format without opening the
JSONL manually.

Sample any prepared split later:

```bash
python -m rl_training.data.sample \
  --path data/rl/v1/sources/<model-tag>/nemotron_rl_coding_competitive_train_all.jsonl \
  --n 3
```

Generate DPO pairs and per-prompt verifier score summaries for every candidate:

```bash
python rl_training/build_dpo_pairs.py \
  --model <base-or-sft-model> \
  --model_revision <model-commit> \
  --prompts data/rl/v1/sources/<model-tag>/nemotron_rl_coding_competitive_train_all.jsonl \
  --out data/rl/nemotron_rl_coding_competitive_dpo_pairs.jsonl \
  --scores_out data/rl/nemotron_rl_coding_competitive_train_scores.jsonl \
  --K 8 \
  --margin 0.5
```

Freeze a cohort after intersecting every requested method's eligibility:

```bash
export COHORT_SIZE=1000  # hyperparameter; choose only after the full sweep

python -m rl_training.data.build_shared_cohort \
  --dataset data/rl/v1/sources/<model-tag>/nemotron_rl_coding_competitive_train_all.jsonl \
  --dataset-manifest data/rl/v1/sources/<model-tag>/nemotron_rl_coding_competitive_manifest.json \
  --score-source grpo=data/rl/v1/sweeps/<model>/prompt_scores.jsonl \
  --score-manifest grpo=data/rl/v1/sweeps/<model>/manifest.json \
  --artifact-source dpo=data/rl/v1/sweeps/<model>/dpo_pairs.jsonl \
  --artifact-manifest dpo=data/rl/v1/sweeps/<model>/manifest.json \
  --size "$COHORT_SIZE" --seed 0 \
  --out-dir "data/rl/v1/cohorts/<model>-shared-n${COHORT_SIZE}-s0"
```

The default learnable-band filter keeps prompts where sampled completions have
useful verifier spread:

```text
max_score >= 0.5
min_score <= 0.5
max_score - min_score >= 0.5
```

For stricter full-pass/full-fail pairs, add:

```bash
--min_max_score 1.0 --max_min_score 0.0
```

Only after this RL-eligible cohort is frozen, generate one verifier-perfect SFT
target for every selected ID with the formal Azure builder:

```bash
python -m rl_training.build_sft_dataset_azure \
  --cohort-manifest "data/rl/v1/cohorts/<model>-shared-n${COHORT_SIZE}-s0/manifest.json" \
  --output-dir "data/rl/v1/sft_datasets/<model>-shared-n${COHORT_SIZE}-s0/<teacher>" \
  --teacher-deployment "$AZURE_OPENAI_DEPLOYMENT" \
  --teacher-model-version "$AZURE_OPENAI_MODEL_VERSION" \
  --target-tokenizer qwen35_4b=prism-drift/qwen35-4b-m0-v4@<revision> \
  --target-tokenizer qwen35_9b=prism-drift/qwen35-9b-m0-v4@<revision> \
  --target-max-length 2048 \
  --max-attempts 3 \
  --request-timeout 300 \
  --verifier-timeout 10 \
  --max-output-tokens 8192 \
  --reasoning-effort high
```

Use the one-worker default for formal fail-fast generation. A diagnostic bulk
audit may use `--max-workers 2 --continue-on-exhausted`, with
`python -m rl_training.monitor_azure_sft <output-dir> --watch 20` tracking 429s,
token burn, projected remaining spend, throughput, and ETA. The final JSONL is
still atomically assembled in frozen cohort order.

The implemented interface reads Azure credentials only from environment
variables and never exposes hidden verifier cases. Passing targets are
checkpointed by ID, failed candidates and attempt metadata are retained in
sidecars, and the command exits non-zero until every frozen ID has one verified,
length-safe target. A resumed run retries missing IDs without discarding later
successes, and restores each missing ID's last failed program and verifier result.
Generated programs never inherit the Azure environment and formal
local builds require the no-network, temp-write-only macOS verifier sandbox.

Validate the exact-N output against the frozen cohort in fresh sandboxed
batches. The report distinguishes reproducible target failures from transient
resource failures and requires three isolated confirmations for the latter.

```bash
python -m rl_training.validate_sft_distill_batched \
  --train "data/rl/v1/sft_datasets/<model>-shared-n${COHORT_SIZE}-s0/<teacher>/train.jsonl" \
  --train_source "data/rl/v1/cohorts/<model>-shared-n${COHORT_SIZE}-s0/train.jsonl" \
  --report "data/rl/v1/sft_datasets/<model>-shared-n${COHORT_SIZE}-s0/<teacher>/validation_report.json" \
  --batch_size 50 \
  --confirmation_runs 3
```

Use the filtered train file for online RL:

```bash
python rl_training/train_rl.py \
  --algo grpo \
  --config rl_training/configs/grpo.yaml \
  --base_model <base-model> \
  --dataset "data/rl/v1/cohorts/<model>-shared-n${COHORT_SIZE}-s0/train.jsonl" \
  --eval_dataset data/rl/nemotron_rl_coding_competitive_dev.jsonl \
  --seed 0 \
  --save_dir runs \
  --run_name qwen4b-grpo-s0
```

Use the DPO pair file for DPO:

```bash
python rl_training/train_rl.py \
  --algo dpo \
  --config rl_training/configs/dpo.yaml \
  --base_model <base-model> \
  --dataset "data/rl/v1/cohorts/<model>-shared-n${COHORT_SIZE}-s0/artifacts/dpo.jsonl" \
  --eval_dataset data/rl/nemotron_rl_coding_competitive_dev.jsonl \
  --seed 0 \
  --save_dir runs \
  --run_name qwen4b-dpo-s0
```

## Related Implementation

The data layer is implemented under `rl_training/data/`:

- `schema.py`: canonical schemas, validators, and JSONL readers/writers.
- `registry.py`: dataset adapter registry.
- `nemotron.py`: adapter for
  `nvidia/Nemotron-RL-coding-competitive_coding`.
- `prepare.py`: generic split-preparation CLI.
- `sample.py`: small API/CLI for inspecting prepared rows.
- `filter_learnable_band.py`: filters train rows using per-prompt score
  summaries.
- `build_shared_cohort.py`: intersects model-score and method-artifact
  eligibility, then freezes one hash-pinned cohort for every training arm.
- `build_formal_shared_cohort.py`: paper-facing model-specific or shared 4B/9B
  cohort builder that uses explicit, repeatable `--target-model-size` provenance and
  pinned sweeps only as selection evidence and applies deterministic verifier,
  prompt, deduplication, test-count, and held-out guards.
- `amend_formal_cohort.py`: auditable one-for-one repair path for a frozen
  cohort after a source verifier is formally quarantined; valid IDs are
  retained and replacements are seeded and hash-ranked.
- `known_bad_nemotron_verifiers.json`: versioned evidence registry for source
  verifier defects that must be excluded from future formal cohorts.
- `quality.py`: formal fixed-I/O eligibility policy and row validator.
  It rejects unsupported judging semantics, malformed/missing statements,
  insufficient or conflicting tests, and the HackerEarth native-source
  partition after full-suite audits found multiple generated-test contamination
  classes that exact-output filtering could not safely repair.

Training and reward code use the data layer as follows:

- `rl_training/rewards.py` dispatches verifier types through `run_verifier`.
- `rl_training/build_dpo_pairs.py` reads `coding_task_v1`, samples completions,
  scores them with the verifier, and writes `dpo_preference_v1`.
- `rl_training/train_rl.py` reads `coding_task_v1` for online RL and
  `dpo_preference_v1` for DPO.
- `rl_training/prompts.py` owns the shared Python stdin/stdout prompt wrapper.

To add a new dataset, create an adapter under `rl_training/data/`, register it with
`register_adapter(...)`, and implement:

```python
def extract_record(self, row: dict, *, max_tests: int) -> CodingTaskRecord | None:
    ...
```

Adapters should return `None` for unusable raw rows. Deduplication, deterministic
splitting, manifest writing, and optional sample printing are handled by
`rl_training/data/prepare.py`.
