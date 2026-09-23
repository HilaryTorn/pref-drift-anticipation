# End-to-End Data and Training Runbook

This is the single operational runbook for the shared-data comparison across
SFT, GRPO, DPO, and PPO. Run commands from the repository root.

The invariant is:

> For one exact base-model revision, score the full candidate pool, intersect
> RL-method eligibility once, freeze the configured N prompt IDs, and make
> every training method consume that frozen cohort. Only after the IDs are
> frozen, generate exactly one verifier-perfect SFT target for every frozen ID.

Do not filter independently inside individual training jobs. SFT generation
may retry a frozen ID, but may not drop, replace, or re-filter it. Do not reuse
a sweep after changing the base model or its revision.

## 0. One-time environment

The current smoke-tested machine workflow uses the Conda base environment:

```bash
conda activate base
python --version
```

For a fresh machine, install the pinned project environment with:

```bash
uv sync --locked
```

Without `uv`, install the checked-in training requirements into the activated
Conda environment:

```bash
conda activate base
python -m pip install -r rl_training/requirements-training.txt
```

Set the run variables. `MODEL_REVISION` must be an exact Hugging Face commit,
not `main`.

```bash
export PROJECT_ROOT="$(git rev-parse --show-toplevel)"
export DATA_ROOT="$PROJECT_ROOT"
export RL_DATA_DIR="$PROJECT_ROOT/data/rl/v1"
export RUN_ROOT="$PROJECT_ROOT/runs"
export FIGURES_ROOT="$PROJECT_ROOT/results"

export BASE_MODEL=Qwen/Qwen3.5-0.8B
export MODEL_REVISION="2fc06364715b967f1860aea9cf38778875588b17"
export DATA_RECIPE_TAG="fulltests-v2"
export COHORT_SIZE=1000
export SEED=0

MODEL_ID_TAG=$(printf '%s' "$BASE_MODEL" | tr '[:upper:]./' '[:lower:]--')
REVISION_TAG=$(printf '%s' "$MODEL_REVISION" | tr '[:upper:]/.' '[:lower:]--' | cut -c1-12)
export MODEL_TAG="${MODEL_ID_TAG}--rev-${REVISION_TAG}--data-${DATA_RECIPE_TAG}"

export SOURCE_DIR="$RL_DATA_DIR/sources/$MODEL_TAG"
export SWEEP_DIR="$RL_DATA_DIR/sweeps/$MODEL_TAG"
export TEACHER_DEPLOYMENT="REPLACE_WITH_AZURE_DEPLOYMENT"
TEACHER_TAG=$(printf '%s' "$TEACHER_DEPLOYMENT" | tr '[:upper:]/.' '[:lower:]--')
export COHORT_DIR="$RL_DATA_DIR/cohorts/${MODEL_TAG}-shared-n${COHORT_SIZE}-s${SEED}"
export SFT_DATA_DIR="$RL_DATA_DIR/sft_datasets/${MODEL_TAG}-shared-n${COHORT_SIZE}-s${SEED}/$TEACHER_TAG"

export SOURCE_TRAIN="$SOURCE_DIR/nemotron_rl_coding_competitive_train_all.jsonl"
export DEV_DATA="$SOURCE_DIR/nemotron_rl_coding_competitive_dev.jsonl"
export HELDOUT_DATA="$SOURCE_DIR/nemotron_rl_coding_competitive_heldout.jsonl"
export SOURCE_MANIFEST="$SOURCE_DIR/nemotron_rl_coding_competitive_manifest.json"
export SCORE_DATA="$SWEEP_DIR/prompt_scores.jsonl"
export ALL_DPO_PAIRS="$SWEEP_DIR/dpo_pairs.jsonl"

export SHARED_TRAIN="$COHORT_DIR/train.jsonl"
export SHARED_DPO="$COHORT_DIR/artifacts/dpo.jsonl"
export SHARED_SFT="$SFT_DATA_DIR/train.jsonl"

# Optional upload destination; set only when uploading:
# export HF_DATASET_REPO="your-org/your-rl-dataset-repo"
# Formal policy checkpoints (private unless --hub_public is passed):
export HF_MODEL_REPO="your-org/your-formal-policy-checkpoints"
# Export HUGGINGFACE_HUB_TOKEN from your secret manager before upload/training commands.
export CUDA_VISIBLE_DEVICES=2
```

For exact training weights, resolve the same pinned revision to a local
snapshot and use that path in every trainer:

```bash
export BASE_MODEL_PATH="$(python -c 'import os; from huggingface_hub import snapshot_download; print(snapshot_download(os.environ["BASE_MODEL"], revision=os.environ["MODEL_REVISION"]))')"
```

Never put OpenAI or Hugging Face tokens in repository files or command-line
arguments. Export them from a secret manager or the shell environment.

## Full default data run

For the default 0.8B experiment, all formal data-generation parameters are
defined at the top of `scripts/run_full_data_pipeline.sh`. This one command
prepares every usable candidate and then runs the full resumable K=8 score/DPO
sweep:

```bash
./scripts/run_full_data_pipeline.sh
```

It defaults to the pinned `Qwen/Qwen3.5-0.8B` revision, local
`data/rl/v1/`, `TRAIN_SIZE=0`, complete verifier suites, `LIMIT=0`, vLLM,
K=8, margin 0.5, GPU 2, and vLLM memory utilization 0.6.
Set `HF_DATASET_REPO` only when the completed source and sweep should also be
uploaded. Sections 1 and 3 below document the two stages individually.

## 1. Prepare every usable candidate

This reserves 250 dev and 500 held-out rows, then writes every remaining row
that passes schema, deduplication, and the 1,024-token prompt limit. It retains
every verifier test; the training reward later selects its deterministic 12-test
subset. The source
pool is namespaced by full model ID and revision because tokenization can change
between models.

```bash
BASE_MODEL="$BASE_MODEL" \
MODEL_REVISION="$MODEL_REVISION" \
MODEL_TAG="$MODEL_TAG" \
RL_DATA_DIR="$RL_DATA_DIR" \
HF_DATASET_REPO="$HF_DATASET_REPO" \
./scripts/prepare_rl_dataset.sh
```

Expected outputs:

```text
$SOURCE_DIR/nemotron_rl_coding_competitive_train_all.jsonl
$SOURCE_DIR/nemotron_rl_coding_competitive_dev.jsonl
$SOURCE_DIR/nemotron_rl_coding_competitive_heldout.jsonl
$SOURCE_DIR/nemotron_rl_coding_competitive_manifest.json
```

The script uploads these files to:

```text
hf://datasets/$HF_DATASET_REPO/sources/$MODEL_TAG/
```

## 2. Smoke-test the model sweep

Before paying for the full sweep, score a small prefix. Use a separate smoke
directory so it cannot be resumed into the formal artifacts.

```bash
GEN_MODEL="$BASE_MODEL" \
MODEL_REVISION="$MODEL_REVISION" \
MODEL_TAG="$MODEL_TAG" \
PROMPTS="$SOURCE_TRAIN" \
SWEEP_DIR="$RL_DATA_DIR/smoke_sweeps/$MODEL_TAG" \
LIMIT=200 \
BACKEND=vllm \
HF_DATASET_REPO= \
./scripts/build_dpo_data.sh
```

Inspect the reported `pairs` and `solved` rates. A nearly all-zero sweep means
the base model has a flat verifier reward surface on this dataset.

## 3. Sweep the full pool for scores and DPO pairs

This samples `K=8` completions for every candidate, grades them with the shared
verifier, writes one score row for every prompt, and writes a DPO pair whenever
the best/worst reward margin is at least 0.5.

```bash
GEN_MODEL="$BASE_MODEL" \
MODEL_REVISION="$MODEL_REVISION" \
MODEL_TAG="$MODEL_TAG" \
RL_DATA_DIR="$RL_DATA_DIR" \
PROMPTS="$SOURCE_TRAIN" \
BACKEND=vllm \
K=8 \
MARGIN=0.5 \
RESUME=1 \
HF_DATASET_REPO="$HF_DATASET_REPO" \
./scripts/build_dpo_data.sh
```

Expected outputs:

```text
$SWEEP_DIR/prompt_scores.jsonl
$SWEEP_DIR/dpo_pairs.jsonl
$SWEEP_DIR/manifest.json
```

Every score and pair records `generator_model`, `generator_model_revision`, and
the source-prompt SHA-256. Resume refuses to mix a different model, revision,
prompt file, K, temperature, reward budget, or backend into an existing sweep.

The script uploads the sweep to:

```text
hf://datasets/$HF_DATASET_REPO/sweeps/$MODEL_TAG/
```

## 4. Freeze the shared N-prompt cohort

`COHORT_SIZE` is a selection hyperparameter, not a generation limit. Source
preparation and every model sweep must finish over the entire candidate pool
first; only this step chooses how many eligible prompts to train on.

GRPO eligibility comes from the learnable score band. DPO requires a derived
pair. PPO uses the same prompt cohort and a reward model trained from the same
filtered DPO pairs, so DPO pair eligibility also covers PPO. SFT is generated
after this selection and therefore must not participate in this intersection.

```bash
python -m rl_training.data.build_shared_cohort \
  --dataset "$SOURCE_TRAIN" \
  --dataset-manifest "$SOURCE_MANIFEST" \
  --score-source "grpo=$SCORE_DATA" \
  --score-manifest "grpo=$SWEEP_DIR/manifest.json" \
  --artifact-source "dpo=$ALL_DPO_PAIRS" \
  --artifact-manifest "dpo=$SWEEP_DIR/manifest.json" \
  --size "$COHORT_SIZE" \
  --seed "$SEED" \
  --out-dir "$COHORT_DIR"
```

The command hard-fails if the intersection contains fewer than `COHORT_SIZE`
prompts. It also refuses partial sweeps, provenance mismatches, and an existing
output directory; it never silently trains on a smaller or replaced cohort.

Expected frozen outputs:

```text
$COHORT_DIR/train.jsonl
$COHORT_DIR/selected_ids.json
$COHORT_DIR/artifacts/dpo.jsonl
$COHORT_DIR/manifest.json
```

Archive the frozen cohort:

```bash
python scripts/upload_rl_data.py \
  --local-dir "$COHORT_DIR" \
  --repo-id "$HF_DATASET_REPO" \
  --path-in-repo "cohorts/${MODEL_TAG}-shared-n${COHORT_SIZE}-s${SEED}" \
  --private
```

From this point onward, GRPO/DPO/PPO must use only files under `COHORT_DIR`;
SFT must cover these same ordered IDs exactly.

## 5. Generate verifier-perfect SFT targets for the frozen cohort

The formal Azure builder is implemented in
`rl_training/build_sft_dataset_azure.py`. Its exact-N, hash/provenance, strict
ordered finalization, gap-tolerant resume, secret-handling, target-tokenizer,
and local verifier invariants have local regression coverage. Validate live
Azure transport with the intended deployment before the first formal dataset.

Export secrets from a secret manager. Never put the key in this file or a CLI
argument:

```bash
export AZURE_OPENAI_API_KEY="REPLACE_FROM_SECRET_MANAGER"
export AZURE_OPENAI_ENDPOINT="REPLACE_WITH_ENDPOINT"
export AZURE_OPENAI_API_VERSION="REPLACE_WITH_API_VERSION"
export AZURE_OPENAI_DEPLOYMENT="$TEACHER_DEPLOYMENT"
export AZURE_OPENAI_MODEL_VERSION="REPLACE_WITH_IMMUTABLE_MODEL_VERSION"
```

After the live transport preflight, generate one target per frozen ID:

```bash
python -m rl_training.build_sft_dataset_azure \
  --cohort-manifest "$COHORT_DIR/manifest.json" \
  --output-dir "$SFT_DATA_DIR" \
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

The formal fail-fast run is sequential by default. To complete a diagnostic
audit of every remaining cohort row before fixing failures in one batch, add
`--max-workers 2 --continue-on-exhausted`. The builder limits concurrency to
four and requires the continue flag for more than one worker so in-flight calls
cannot silently weaken fail-fast semantics. Start at two workers and monitor:

```bash
python -m rl_training.monitor_azure_sft "$SFT_DATA_DIR" --watch 20
```

The checkpoint reports accepted rows, API and 429 errors, tokens, estimated
spend, hourly burn, projected remaining spend, throughput, and ETA. Reduce to
one worker if rate-limit errors persist. Parallelism changes wall-clock rate,
not the expected per-record token cost or deterministic final cohort order.

Validate every answer again against the frozen source rows and full verifier.
Use fresh sandboxed batches to prevent resource accumulation across a 1,000-row
sweep; an apparent failed row must pass three isolated confirmations and remains
a hard failure if any confirmation fails.

```bash
python -m rl_training.validate_sft_distill_batched \
  --train "$SHARED_SFT" \
  --train_source "$SHARED_TRAIN" \
  --report "$SFT_DATA_DIR/validation_report.json" \
  --batch_size 50 \
  --confirmation_runs 3 \
  --verifier_timeout 10
```

Only a complete, exact-N dataset may be archived:

```bash
python scripts/upload_rl_data.py \
  --local-dir "$SFT_DATA_DIR" \
  --repo-id "$HF_DATASET_REPO" \
  --path-in-repo "sft_datasets/${MODEL_TAG}-shared-n${COHORT_SIZE}-s${SEED}/${TEACHER_TAG}" \
  --private
```

If a live audit establishes that a source verifier violates its own problem
contract, stop the run, mark its manifest `quarantined`, add the evidence to
`known_bad_nemotron_verifiers.json`, and create a new cohort version with
`python -m rl_training.data.amend_formal_cohort`. Reuse overlapping teacher
targets only through `--import-accepted-from`, which performs full sandboxed
revalidation and records the imported file hash. Do not mutate or resume the
quarantined artifact.

If the failure reveals a systematic source artifact rather than one bad ID,
promote the check into `rl_training/data/quality.py`, audit the entire frozen
cohort, and replace every affected row in one versioned amendment. For the
observed HackerEarth generated-test contamination, use the current formal
quality policy, which excludes that native-source partition, and Codeforces-only
replacements; do not spend teacher calls auditing its malformed cases one by
one.

## 6. Shared formal training settings

All policy methods use:

```text
optimizer steps:       125 (one epoch at N=1000)
checkpoint cadence:     4 steps
learning rate:          1e-6
effective prompt batch: 8 unique prompts
LoRA targets:           all-linear
seed:                   identical across methods
```

For SFT, DPO, and PPO this is `per_device_train_batch_size=1` times eight
gradient-accumulation steps. GRPO counts completion sequences internally: with
K=8 rollouts per prompt it uses `1 x 64 / 8 = 8` unique prompts per optimizer
step. Thus 125 GRPO steps consume all 1,000 frozen prompt IDs exactly once and
produce 8,000 sampled completions. K must never be counted as extra training
prompts. `train_rl.py` rejects a GRPO config or dataset that violates this
prompt-budget invariant.

Checkpoints are saved densely. The primary elicitation trajectory is step 0,
4, 12, 40, and 125; other saved adapters are retained for optional densification.

Formal runs must enable the live Hub archive. At each save, the trainer writes
and uploads a model-only checkpoint; optimizer, scheduler, scaler, and RNG
state are intentionally omitted. The uploader records file hashes, verifies
all paths at the returned immutable Hub commit, and only then removes older
local checkpoints. One newest checkpoint remains locally while training is
active; after the final step-125 adapter is verified, no checkpoint directory
is left locally. A failed upload is retained locally. Transient failures retry
at the next save; a persistent Hub quota failure switches to explicit local
fallback until the marker is removed. Smoke or `/tmp` runs are rejected.

The shared arguments for all four arms are:

```bash
HUB_ARCHIVE_ARGS=(
  --hub_repo_id "$HF_MODEL_REPO"
  --hub_base_model_id "$BASE_MODEL"
  --hub_base_revision "$MODEL_REVISION"
  --hub_keep_local_checkpoints 1
)
```

Do not change one algorithm config without updating and documenting the matched
fields in the other configs.

Create output directories:

```bash
mkdir -p "$RUN_ROOT" "$FIGURES_ROOT"
```

## 7. Train GRPO

```bash
python -m rl_training.train_rl \
  --algo grpo \
  --config rl_training/configs/grpo.yaml \
  --base_model "$BASE_MODEL_PATH" \
  --dataset "$SHARED_TRAIN" \
  --eval_dataset "$DEV_DATA" \
  --seed "$SEED" \
  --save_dir "$RUN_ROOT" \
  --figures_dir "$FIGURES_ROOT" \
  --run_name "${MODEL_TAG}-grpo-s${SEED}" \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target_modules all-linear \
  --cohort_manifest "$COHORT_DIR/manifest.json" \
  "${HUB_ARCHIVE_ARGS[@]}"
```

Add `--use_vllm --vllm_mode colocate` only after its smoke configuration has
passed on the target GPU.

Before model optimization starts, require the log line
`GRPO prompt-budget audit` to report `dataset_unique_prompts=1000`,
`unique_prompts_per_step=8`, and `scheduled_unique_prompts=1000`. A run with
1,000 rollout rows but only 125 distinct `record_id` values is invalid and must
not enter the DPO/GRPO drift comparison.

## 8. Train DPO

```bash
python -m rl_training.train_rl \
  --algo dpo \
  --config rl_training/configs/dpo.yaml \
  --base_model "$BASE_MODEL_PATH" \
  --dataset "$SHARED_DPO" \
  --eval_dataset "$DEV_DATA" \
  --seed "$SEED" \
  --save_dir "$RUN_ROOT" \
  --figures_dir "$FIGURES_ROOT" \
  --run_name "${MODEL_TAG}-dpo-s${SEED}" \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target_modules all-linear \
  --cohort_manifest "$COHORT_DIR/manifest.json" \
  "${HUB_ARCHIVE_ARGS[@]}"
```

## 9. Train the PPO reward model

The reward model must train from the exact filtered DPO file used in section 8.

```bash
export RM_DIR="$RUN_ROOT/${MODEL_TAG}-reward-model-s${SEED}"

python -m rl_training.train_reward_model \
  --config rl_training/configs/reward_model.yaml \
  --base_model "$BASE_MODEL_PATH" \
  --pairs "$SHARED_DPO" \
  --output_dir "$RM_DIR" \
  --seed "$SEED"
```

Inspect `$RM_DIR/reward_model_manifest.json`. Do not run PPO unless
`held_out_pairwise_accuracy >= 0.60`; accuracy near 0.50 means PPO would optimize
noise.

## 10. Train PPO

The TRL 1.9 experimental PPO path has passed a one-update plumbing smoke. That
does not waive the reward-model quality gate above; run this formal command
only when the frozen-cohort reward model reaches the required accuracy.

```bash
python -m rl_training.train_rl \
  --algo ppo \
  --config rl_training/configs/ppo.yaml \
  --base_model "$BASE_MODEL_PATH" \
  --reward_model "$RM_DIR" \
  --dataset "$SHARED_TRAIN" \
  --eval_dataset "$DEV_DATA" \
  --seed "$SEED" \
  --save_dir "$RUN_ROOT" \
  --figures_dir "$FIGURES_ROOT" \
  --run_name "${MODEL_TAG}-ppo-s${SEED}" \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target_modules all-linear \
  --cohort_manifest "$COHORT_DIR/manifest.json" \
  "${HUB_ARCHIVE_ARGS[@]}"
```

## 11. Train verifier-filtered SFT

This trainer enforces the frozen ordered IDs, complete dataset/cohort
manifests, verifier-perfect rows, assistant-only labels, and matched LoRA
settings. It writes periodic `checkpoint-N` adapters and a final adapter at the
run root.

```bash
python -m rl_training.train_sft \
  --config rl_training/configs/sft.yaml \
  --base_model "$BASE_MODEL_PATH" \
  --dataset "$SHARED_SFT" \
  --dataset_manifest "$SFT_DATA_DIR/manifest.json" \
  --cohort_manifest "$COHORT_DIR/manifest.json" \
  --seed "$SEED" \
  --save_dir "$RUN_ROOT" \
  --run_name "${MODEL_TAG}-sft-s${SEED}" \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target_modules all-linear \
  "${HUB_ARCHIVE_ARGS[@]}"
```

## 12. Restart interrupted policy training

Model-only checkpoints cannot restore optimizer, scheduler, or RNG state, so
none of the formal arms may silently resume from them. If a run is interrupted,
diagnose the failure, preserve or archive the incomplete run directory for
forensics, then restart that arm cleanly with the original model snapshot,
cohort, seed, config, and run name. Completed upstream data and training arms
remain reusable. The launch scripts fail closed when they find a non-empty,
incomplete model-only run.

## 13. Register checkpoints for drift scoring

The live archive intentionally prunes completed local checkpoint directories.
Before local drift scoring, download only the headline checkpoint folders
(4/12/40/125) from the Hub into a staging directory, then register that staging
directory. Do not download optimizer state when only inference is needed.

Run once per staged comparison arm:

```bash
python rl_training/register_checkpoints.py \
  --run_dir "$RUN_ROOT/${MODEL_TAG}-grpo-s${SEED}" \
  --base_model "$BASE_MODEL_PATH" \
  --algo grpo \
  --seed "$SEED"

python rl_training/register_checkpoints.py \
  --run_dir "$RUN_ROOT/${MODEL_TAG}-dpo-s${SEED}" \
  --base_model "$BASE_MODEL_PATH" \
  --algo dpo \
  --seed "$SEED"

python rl_training/register_checkpoints.py \
  --run_dir "$RUN_ROOT/${MODEL_TAG}-ppo-s${SEED}" \
  --base_model "$BASE_MODEL_PATH" \
  --algo ppo \
  --seed "$SEED"

python rl_training/register_checkpoints.py \
  --run_dir "$RUN_ROOT/${MODEL_TAG}-sft-s${SEED}" \
  --base_model "$BASE_MODEL_PATH" \
  --algo sft \
  --seed "$SEED"
```

This writes model keys into the generated checkpoint block in `config.yaml`.

## 14. Build real RL-method stimuli — PPO logging blocked

The stimuli builder requires real SFT, GRPO, DPO, PPO, and reward-model
artifacts. GRPO currently writes `rollout_samples.rank0.jsonl`, but the current
PPO trainer does **not** yet persist the required `ppo_rollout_v1` rows with
`record_id`, completion, and reward-model score. Implement that PPO logger
before running this section; do not substitute GRPO rows or synthetic data.

After the PPO logger exists and `PPO_ROLLOUTS` points to its real output, run:

```bash
export PPO_ROLLOUTS="$RUN_ROOT/${MODEL_TAG}-ppo-s${SEED}/ppo_rollout_samples.jsonl"

python scripts/build_rl_stimuli.py \
  --cohort_manifest "$COHORT_DIR/manifest.json" \
  --source_train "$SHARED_TRAIN" \
  --sft_train "$SHARED_SFT" \
  --sft_manifest "$SFT_DATA_DIR/manifest.json" \
  --dpo_pairs "$SHARED_DPO" \
  --grpo_rollouts "$RUN_ROOT/${MODEL_TAG}-grpo-s${SEED}/rollout_samples.rank0.jsonl" \
  --ppo_rollouts "$PPO_ROLLOUTS" \
  --ppo_rm_manifest "$RM_DIR/reward_model_manifest.json"
```

The builder verifies that all examples belong to the frozen cohort and that the
PPO reward model was trained from the supplied DPO pairs.

### Matched DPO-versus-GRPO analysis

The current two-method headline does not wait for SFT/PPO and must not pretend
that those missing arms exist. After the corrected GRPO run is complete, build
its own real stimulus file:

```bash
python scripts/build_dpo_grpo_rl_stimuli.py \
  --cohort_manifest "$COHORT_DIR/manifest.json" \
  --dpo_pairs "$COHORT_DIR/artifacts/dpo.jsonl" \
  --grpo_rollouts "$RUN_ROOT/${MODEL_TAG}-grpo-s${SEED}/rollout_samples.rank0.jsonl" \
  --dpo_run_metadata "$RUN_ROOT/${MODEL_TAG}-dpo-s${SEED}/run_metadata.json" \
  --grpo_run_metadata "$RUN_ROOT/${MODEL_TAG}-grpo-s${SEED}/run_metadata.json"
```

This builder requires exactly 1,000 DPO pair IDs and exactly 1,000 GRPO prompt
IDs with K=8 real rollout rows per ID. If an append-only GRPO log contains
complete groups duplicated by a restart, pass `--deduplicate-resumed-rollouts`;
the builder selects the last complete group per prompt and embeds the complete
recovery audit in provenance. It also verifies base revision, seed, LoRA
settings, optimizer cadence, artifact hashes, and completed step 125.

Serve M0-v4 plus all 32 DPO, 32 GRPO, and 32 PPO adapters (steps 4 through 124
every four steps, plus final step 125) with aliases registered in a model
config, then run the dedicated trajectory:

```bash
python scripts/run_dpo_grpo_rl_trajectory.py
```

The superseded plotting artifacts and generators were removed with the V2/V3
results. Add a V4-provenanced plotter only after the new trajectory schema has
been validated; do not reuse the old figures.

The trajectory asks both `rl_dpo_grpo_preferences` and
`rl_dpo_grpo_anticipation` at M0 and every saved DPO, GRPO, and PPO checkpoint,
in thinking and non-thinking modes and at all three concreteness levels. Only
M0 anticipation is scored as a prospective prediction of realized drift.
Checkpoint anticipation is reported separately as self-forecast drift and is
never pooled into that forecast-accuracy score.

Repeat the data sweep independently for 9B M0-v4. Do not reuse 4B pairs: pair
eligibility and the chosen/rejected completions are generated on-policy from
the model being trained. After freezing the 9B exact-N cohort and completing
both 9B runs, write its separate stimulus file and trajectory:

```bash
python scripts/build_dpo_grpo_rl_stimuli.py \
  --cohort_manifest "$COHORT_9B/manifest.json" \
  --dpo_pairs "$COHORT_9B/artifacts/dpo.jsonl" \
  --grpo_rollouts "$RUN_9B_GRPO/rollout_samples.rank0.jsonl" \
  --dpo_run_metadata "$RUN_9B_DPO/run_metadata.json" \
  --grpo_run_metadata "$RUN_9B_GRPO/run_metadata.json" \
  --output data/source/rl_dpo_grpo_training_stimuli_9b.json
python scripts/run_dpo_grpo_rl_trajectory.py --size 9b
```

Future 9B figures must use a separate V4-provenanced namespace and must not
pool estimates with 4B.

## 15. Score checkpoint drift

After checkpoint registration, score a registered model key:

```bash
export CHECKPOINT_MODEL_KEY="REPLACE_WITH_REGISTERED_CHECKPOINT_MODEL_KEY"
export DRIFT_RESULT_DIR="results/${MODEL_TAG}-grpo-s${SEED}/step40"

python main.py run_elicitations_task -- \
  --model_key "$CHECKPOINT_MODEL_KEY" \
  --save_dir "$DRIFT_RESULT_DIR/task_preference"

python main.py run_utilities_values -- \
  --model_key "$CHECKPOINT_MODEL_KEY" \
  --save_dir "$DRIFT_RESULT_DIR/values"
```

Dense trajectory scoring uses coding task preference. Values, Book A, and UE
remain sparse endpoint/midpoint checks according to the methodology.

## 16. Disposable 20-row GRPO/DPO/PPO smoke

Run all three RL plumbing paths on the same 20 rows in the Conda base
environment. The script creates a guarded `/tmp` tree and deletes its datasets,
checkpoints, reward model, and figures automatically on exit. The weak-reward-
model override exists only inside this smoke path.

```bash
conda run -n base env \
  CUDA_VISIBLE_DEVICES=3 \
  SMOKE_SIZE=20 \
  RUN_PPO_STAGE=1 \
  ./scripts/run_training_smoke.sh
```

For GRPO and DPO only, omit `RUN_PPO_STAGE=1`. The script always creates its
own temporary root; never upload smoke outputs.

## 17. Disposable 20-row SFT checkpoint/elicitation smoke

This smoke uses existing DPO chosen completions only after re-running their
full verifier. It validates plumbing, not Azure transport or experiment
quality. Everything lives under a guarded `/tmp` directory, is never uploaded,
and is deleted automatically when the shell exits.

```bash
export SFT_SMOKE_ROOT="$(mktemp -d /tmp/ai-pref-drift-sft-smoke.XXXXXX)"

cleanup_sft_smoke() {
  case "$SFT_SMOKE_ROOT" in
    /tmp/ai-pref-drift-sft-smoke.*) rm -rf -- "$SFT_SMOKE_ROOT" ;;
    *) echo "Refusing unsafe smoke cleanup path: $SFT_SMOKE_ROOT" >&2 ;;
  esac
}
trap cleanup_sft_smoke EXIT

python -m rl_training.data.build_sft_smoke_fixture \
  --prompts "$SOURCE_TRAIN" \
  --pairs "$ALL_DPO_PAIRS" \
  --size 20 \
  --output-dir "$SFT_SMOKE_ROOT"

python -m rl_training.train_sft \
  --config rl_training/configs/sft_smoke.yaml \
  --base_model "$BASE_MODEL_PATH" \
  --dataset "$SFT_SMOKE_ROOT/sft/train.jsonl" \
  --dataset_manifest "$SFT_SMOKE_ROOT/sft/manifest.json" \
  --cohort_manifest "$SFT_SMOKE_ROOT/cohort/manifest.json" \
  --seed 0 \
  --save_dir "$SFT_SMOKE_ROOT/runs" \
  --run_name qwen08b-sft-smoke \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target_modules all-linear

python rl_training/register_checkpoints.py \
  --run_dir "$SFT_SMOKE_ROOT/runs/qwen08b-sft-smoke" \
  --base_model "$BASE_MODEL_PATH" \
  --algo sft \
  --seed 0 \
  --key_prefix qwen08b-sft-smoke \
  --config_yaml "$SFT_SMOKE_ROOT/models.yaml"

python scripts/run_elicitations.py anticipation \
  --anchor_ids coding.write.python \
  --target_ids coding.write.python \
  --level described_choice \
  --model_key qwen08b-sft-smoke-step1 \
  --models_config_path "$SFT_SMOKE_ROOT/models.yaml" \
  --create_agent_config_key default \
  --K 1 \
  --timeout 120 \
  --reasoning off \
  --output_path "$SFT_SMOKE_ROOT/elicitation/prompts.jsonl" \
  --results_path "$SFT_SMOKE_ROOT/elicitation/results.json" \
  --raw_dump_path "$SFT_SMOKE_ROOT/elicitation/raw.jsonl" \
  --no-timestamp

python -m rl_training.validate_checkpoint_elicitation \
  --prompts "$SFT_SMOKE_ROOT/elicitation/prompts.jsonl" \
  --results "$SFT_SMOKE_ROOT/elicitation/results.json" \
  --raw "$SFT_SMOKE_ROOT/elicitation/raw.jsonl" \
  --model-key qwen08b-sft-smoke-step1 \
  --expected-records 1
```

Set `CUDA_VISIBLE_DEVICES` before the two model-loading commands when a
specific GPU is required. Do not add an upload command to this smoke section.

## 18. Switching base models

When moving from 0.8B to 4B or 9B:

1. change both `BASE_MODEL` and `MODEL_REVISION`;
2. recompute `MODEL_TAG` and all derived directories;
3. rerun source preparation because tokenizer eligibility may change;
4. rerun the full score/DPO sweep;
5. freeze a new shared cohort;
6. generate exact-N SFT targets for that frozen cohort if SFT is included;
7. train every compared method from that model's exact snapshot.

Never reuse the 0.8B score, pair, SFT, or frozen-cohort artifacts for 4B/9B.
