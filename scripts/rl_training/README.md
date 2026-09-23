# Training

This folder contains the verifier-based coding RL pipeline for the preference
drift study.

Research question: **does the RL algorithm change how model preferences drift?**
We fine-tune the same base model on the same coding dataset with different RL
algorithms, save checkpoints on a fixed cadence, and run the same preference
elicitation battery on each checkpoint.

This is verifier-based coding RL, not human-preference RLHF. The reward signal
comes from executable coding tests.

## Documents

- `rl_training/docs/data.md`: data schemas, current dataset, preparation commands,
  sampling, DPO pair generation, and learnable-band filtering.
- `rl_training/docs/rl_tutorial.md`: PPO, GRPO, and DPO components and objectives.
- `rl_training/docs/end_to_end_runbook.md`: the single copy-paste workflow from
  full data generation and Hugging Face archival through shared-cohort freezing,
  training, checkpoint registration, and drift scoring.
- `scripts/run_full_data_pipeline.sh`: one-command full source preparation and
  score/DPO sweep with the complete 0.8B parameters defined as script defaults.

## What Varies

Held fixed across algorithms:

- base model;
- dataset and split;
- verifier reward source;
- prompt format;
- optimizer-step budget;
- checkpoint cadence;
- reference regularization strength (`beta` / `init_kl_coef`);
- seed set.

Varies:

- RL algorithm: `grpo`, `dpo`, or `ppo`.

The primary matching axis is optimizer step. We also log verifier reward at each
checkpoint so runs can be compared post hoc at approximately equal reward.

## Objective

The shared reward is unit-test pass rate:

```text
r(x, y) = pass_rate(y; verifier_x)
```

For online RL, the high-level objective is:

```text
maximize E[ r(x, y) ] - beta * KL(pi_theta(. | x) || pi_ref(. | x))
```

Implementation notes:

- GRPO calls the Python verifier reward directly through `coding_reward`.
- DPO trains on `(chosen, rejected)` pairs generated from the same verifier.
- PPO in TRL expects a reward model, not a Python reward function. To keep PPO
  comparable, `--reward_model` must point to a reward model trained from the
  same verifier signal — built by `train_reward_model.py` from the *same*
  `dpo_preference_v1` pairs DPO trains on (a Bradley-Terry pairwise RM), so all
  three arms share one objective. See "Reward Model (PPO)" below.
- LoRA is supported with `--use_lora`; for the main study, keep the LoRA config
  fixed across algorithms and vary only the RL algorithm.
- Prompts are rendered through the base model's chat template
  (`rl_training/prompts.render_coding_prompt`, thinking mode pinned off). All
  consumers — GRPO rollouts, the reward callback, DPO pair generation, and
  prompt-length filtering — share this one renderer so the token stream is
  identical everywhere. Feeding a chat-tuned Qwen raw text puts it in
  continuation mode and it never answers.
- Models load in bf16 when the GPU supports it (fp32 fallback); GRPO wall time
  is generation-dominated, so this roughly halves it.
- Verifier test cases run in parallel subprocesses; tune with
  `VERIFIER_MAX_WORKERS` (default: min(32, cpu count)).

Security note: the verifier executes model-generated code. The current
subprocess timeout is not a real sandbox. Run on disposable infrastructure or
wrap execution in a proper container/sandbox before large runs.

## Data

The canonical online RL row format is `coding_task_v1`:

```json
{
  "schema": "coding_task_v1",
  "record_id": "...",
  "dataset_id": "nemotron_rl_coding_competitive",
  "prompt": "raw problem statement",
  "verifier": {
    "type": "io_tests",
    "test_inputs": ["..."],
    "test_outputs": ["..."]
  },
  "metadata": {}
}
```

The current dataset is:

```text
dataset_id: nemotron_rl_coding_competitive
dataset:    nvidia/Nemotron-RL-coding-competitive_coding
verifier:   io_tests
```

Data preparation first reserves the fixed evaluation splits, then writes every
remaining usable row as the model-sweep candidate pool:

```text
dev:     250
heldout: 500
candidate train: all remaining usable rows
```

The formal training cohort is selected later: run a full base-model sweep,
intersect the eligibility requirements for every requested method, and freeze a
seeded N-prompt cohort. N is a selection hyperparameter chosen only after the
full model sweep finishes. Changing the base model requires a new sweep and a
new cohort; different methods using the same base model may reuse the same
completed sweep.

Prepare the full candidate pool and print sample rows:

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

This writes train/dev/heldout JSONL files and a manifest with the exact
revision, row counts, split sizes, seed, and output file names. The split
shuffle is seeded, so the same seed/revision/sizes always reproduce identical
splits.

Add `--max_prompt_tokens 1024 --tokenizer <base-model>` to drop rows whose
*rendered* prompt (chat template included) exceeds the trainer's
`max_prompt_length`. Truncated prompts lose part of the problem statement, so
filter at prep time instead. On the Nemotron set, ~89% of rows fit in 1024
tokens. `scripts/prepare_rl_dataset.sh` exposes all of this via env vars
(`TRAIN_SIZE`, `SEED`, `MAX_PROMPT_TOKENS`, `PROMPT_TOKENIZER`, ...).

Sample an existing split:

```bash
python -m rl_training.data.sample \
  --path data/rl/v1/sources/<model-tag>/nemotron_rl_coding_competitive_train_all.jsonl \
  --n 3
```

## DPO Pairs And Learnable Band

DPO does not consume rewards directly. Generate verifier-derived pairs from the
same prompt pool:

```bash
python rl_training/build_dpo_pairs.py \
  --model <base-or-sft-model> \
  --model_revision <model-commit> \
  --prompts data/rl/v1/sources/<model-tag>/nemotron_rl_coding_competitive_train_all.jsonl \
  --out data/rl/v1/sweeps/qwen35-08b/dpo_pairs.jsonl \
  --scores_out data/rl/v1/sweeps/qwen35-08b/prompt_scores.jsonl \
  --manifest_out data/rl/v1/sweeps/qwen35-08b/manifest.json \
  --K 8 \
  --margin 0.5
```

One command runs the full model-specific sweep (pairs + scores + manifest):

```bash
BACKEND=hf ./scripts/build_dpo_data.sh     # Transformers generate, training env
./scripts/build_dpo_data.sh                # vLLM engine (default), ~10x faster sweep
LIMIT=200 BACKEND=vllm ./scripts/build_dpo_data.sh  # pilot; watch `solved=`
```

`BACKEND=vllm` requires a Python environment with the pinned vLLM dependencies
(see `rl_training/requirements-training.txt`). Both backends sample with the
same rendered chat prompt and the same
TRL-matched sampling (temperature-only, top_p=1), so scores are comparable.

The output pair format is `dpo_preference_v1`. A pair is kept only if:

```text
pass_rate(chosen) - pass_rate(rejected) >= margin
```

Do not let each training method filter independently. Once the full RL score
and pair sweep is complete, intersect RL eligibility and freeze one cohort:

```bash
export COHORT_SIZE=1000  # selection hyperparameter, not a sweep limit

python -m rl_training.data.build_shared_cohort \
  --dataset data/rl/v1/sources/<model-tag>/nemotron_rl_coding_competitive_train_all.jsonl \
  --dataset-manifest data/rl/v1/sources/<model-tag>/nemotron_rl_coding_competitive_manifest.json \
  --score-source grpo=data/rl/v1/sweeps/qwen35-08b/prompt_scores.jsonl \
  --score-manifest grpo=data/rl/v1/sweeps/qwen35-08b/manifest.json \
  --artifact-source dpo=data/rl/v1/sweeps/qwen35-08b/dpo_pairs.jsonl \
  --artifact-manifest dpo=data/rl/v1/sweeps/qwen35-08b/manifest.json \
  --size "$COHORT_SIZE" --seed 0 \
  --out-dir "data/rl/v1/cohorts/qwen35-08b-shared-n${COHORT_SIZE}-s0"
```

Add another `--score-source NAME=PATH` for an RL method with score-based
eligibility. `--artifact-source` currently carries the verifier-derived DPO
pairs. SFT must not be included in this intersection: generate its targets only
after the cohort is frozen. The command writes `train.jsonl`,
`selected_ids.json`, filtered RL artifacts, and a hash-pinned manifest.
GRPO/PPO consume the frozen `train.jsonl`; DPO consumes `artifacts/dpo.jsonl`.

Optionally archive each complete sweep in a model-specific Hugging Face path:

```bash
HF_DATASET_REPO=<org>/<rl-data-repo> \
GEN_MODEL=Qwen/Qwen3.5-0.8B \
./scripts/build_dpo_data.sh
```

No upload occurs unless `HF_DATASET_REPO` is set. Set `HUGGINGFACE_HUB_TOKEN`
or `HF_TOKEN`; `HF_PATH_IN_REPO` can override the default
`sweeps/<model-tag>` namespace. The tag is derived from the full model ID plus
the first 12 characters of its pinned revision, for example
`qwen-qwen3-5-0-8b--rev-abc123...`.

Archive the frozen cohort after selection with the same generic uploader:

```bash
python scripts/upload_rl_data.py \
  --local-dir "data/rl/v1/cohorts/<model>-shared-n${COHORT_SIZE}-s0" \
  --repo-id <org>/<rl-data-repo> \
  --path-in-repo "cohorts/<model>-shared-n${COHORT_SIZE}-s0" \
  --private
```

The default filter keeps prompts with:

```text
max_score >= 0.5
min_score <= 0.5
max_score - min_score >= 0.5
```

Competitive programming is hard. If the base model scores zero on almost every
sample, online RL has a flat reward surface and there may be no measurable drift.

## Phase 1.2 model-specific cohorts and post-freeze SFT data

Phase 1.2 uses two model-specific, N=1000 frozen cohorts, archived per cohort at
their own pinned revision of `prism-drift/qwen35-m0-v4-postfix-rl-data`:

| cohort | path | revision |
|---|---|---|
| Qwen3.5 4B M0-v4 | `4b/cohort` | `13ed502d069d17d0a668bc3ecc32b6235aa26281` |
| Qwen3.5 9B M0-v4 | `9b/cohort` | `06f490cdd3f62ffe9103bc89075307e1227f2176` |

The two revisions differ, so `--cohort-hf-revision` is per cohort, not per run set.
Each cohort is shared across SFT, DPO, GRPO, and PPO for that model size. Do not
combine the cohorts, replace records, or use the discarded local cross-model
cohort iterations. Note that `rl_training/data/build_formal_shared_cohort.py`
and `scripts/run_formal_after_dpo.sh` still default to the older
`prism-drift/ai-pref-drift-rl-v4` archive; pass the pins above explicitly rather
than relying on those defaults.

Use the hosted teacher to generate exactly one verifier-perfect target for every
frozen ID in the same order. Phase 1.2 pins `reasoning_effort=high` for every
teacher request. The first attempt is a normal Responses API request; after a
local verifier failure, `--code-interpreter-on-retry` exposes Azure's
sandboxed Code Interpreter so the teacher can run self-created tests and debug
its candidate. Retry requests set `tool_choice=required`, so merely receiving the
tool without executing it is not treated as the sandbox-assisted condition.
Hidden verifier inputs and expected outputs are never sent to the
teacher or the remote sandbox. A worker reuses its active interpreter container
to avoid creating a paid session for every retry.

Verification is two-stage for throughput. Every candidate first runs against a
deterministic 12-test hidden prefix; obvious failures go directly to repair.
Candidates that pass this screen must then pass the complete hidden verifier
suite (`teacher_pass_rate == 1.0`) before acceptance. The screen only saves local
execution time and never relaxes the final correctness criterion. Both outcomes
are retained in the append-only attempt log.

For `io_tests`, the source judge compares stdout to stored reference outputs.
Some constructive problems allow multiple semantically valid answers, so repair
feedback reports only the aggregate number of matched screening references and
asks the teacher to reproduce public-example tie-breaking. Hidden inputs and
outputs remain unavailable. Attempt logs and accepted-row metadata record the
repair protocol version so this exact-match limitation remains auditable.

Generation may retry an ID but may not drop it, replace it, or shrink N.
Accepted rows are checkpointed by ID so later failures do not discard them; the
final `train.jsonl` is atomically assembled in frozen order only when all IDs
are present. Medium-reasoning outputs are not inputs to the final dataset. This
stronger-teacher arm varies both training procedure and supervision source, so
describe it as a training-procedure comparison rather than an algorithm-only
comparison.

The formal executable interface and exact prompt/schema/security contract are
in `rl_training/build_sft_dataset_azure.py`. The runbook has the full command,
validation, upload, and SFT training sequence.

**OpenRouter is the default provider.** One key, any frontier model, and the same key file as the Go arm: put `OPENROUTER_API_KEY=sk-or-...` in the repo's gitignored `.env` (or `api_keys/api_key_openrouter.txt`). An exported environment variable wins over `.env`. The model id is checked against OpenRouter's live catalogue before a single request is sent, so a mistyped slug fails immediately with near-matches instead of 1000 times, and the catalogue entry — exact id, `created`, context length, and the rate card in force — is recorded in the run manifest under `teacher.openrouter_catalogue_entry`.

```bash
python -m rl_training.build_sft_dataset_azure \
  --cohort-manifest "$COHORT_DIR/manifest.json" \
  --output-dir "$SFT_DATA_DIR" \
  --provider openrouter \
  --teacher-model openai/gpt-5.6-sol \
  --cohort-validation-mode frozen_exact \
  --cohort-hf-repo-id prism-drift/qwen35-m0-v4-postfix-rl-data \
  --cohort-hf-revision "$COHORT_HF_REVISION" \
  --cohort-hf-path "$COHORT_HF_PATH" \
  --target-tokenizer qwen35_4b=prism-drift/qwen35-4b-m0-v4@<revision> \
  --target-tokenizer qwen35_9b=prism-drift/qwen35-9b-m0-v4@<revision> \
  --target-max-length 2048 \
  --max-attempts 3 \
  --request-timeout 300 \
  --verifier-timeout 10 \
  --screening-max-tests 12 \
  --max-output-tokens 8192 \
  --reasoning-effort high \
  --require-high-reasoning
```

`--teacher-model-version` is optional under OpenRouter: the catalogue id is the pin OpenRouter exposes, so it is filled in automatically when omitted. Note that `--max-output-tokens` becomes the chat-completions `max_tokens`, which on most OpenRouter models covers reasoning tokens as well as the visible program — at `--reasoning-effort high` a hard problem can spend most of the budget thinking, which surfaces as `finish_reason: length` and is logged as an `incomplete` attempt with `incomplete_details.reason = max_output_tokens`, then retried. Raise the budget rather than lowering effort if those dominate the attempt log.

`--code-interpreter-on-retry` is Azure-only — OpenRouter does not proxy the hosted `code_interpreter` tool — and is rejected at argument validation under `--provider openrouter` rather than failing 1000 requests in.

<details>
<summary>Azure OpenAI instead (funded by the Azure credit)</summary>

```bash
export AZURE_OPENAI_API_KEY='<set locally, never commit>'
export AZURE_OPENAI_ENDPOINT='https://<resource>.openai.azure.com/openai/v1'
export AZURE_OPENAI_API_VERSION='<api version used for provenance>'
export AZURE_OPENAI_DEPLOYMENT='<azure-deployment-name>'
export AZURE_OPENAI_MODEL_VERSION='<immutable deployed model version>'

python -m rl_training.build_sft_dataset_azure \
  ... \
  --provider azure \
  --teacher-deployment "$AZURE_OPENAI_DEPLOYMENT" \
  --teacher-model-version "$AZURE_OPENAI_MODEL_VERSION" \
  --code-interpreter-on-retry
```

</details>

The formal fail-fast default uses one worker. Bounded two-record parallelism is supported with
`--max-workers 2 --continue-on-exhausted`. This reduces wall time without
changing the frozen cohort or final row order. Monitor token burn, 429s, and ETA
with `python -m rl_training.monitor_azure_sft "$SFT_DATA_DIR" --watch 20`; return
to one worker if Azure reports sustained rate limits. Do not use parallel mode
as a substitute for reviewing exhausted records before freezing the dataset.

The target-tokenizer gates use `enable_thinking=False`, matching formal Qwen3.5
training, and reject otherwise valid teacher outputs that would be truncated.
Both providers post through the same `rl_training/openai_responses_worker.py` subprocess and
differ only at the provider seam: OpenRouter uses `https://openrouter.ai/api/v1/chat/completions`
with a `Bearer` header, Azure uses `/openai/v1/responses` with `api-key`. OpenRouter chat
completions are normalized onto the Responses shape inside `normalize_openrouter_response`, so
retry, repair feedback, verification, provenance, and the completeness gate are one shared code
path. An HTTP 200 carrying an `{"error": ...}` envelope and a null `message.content` are both
treated as failed or truncated attempts, never as empty completions. Reasoning text arrives in
`message.reasoning` and is never read into the SFT target. Both write formal
`azure_sft_dataset_manifest_v1` provenance (the schema name predates the provider split; the
provider actually used is in `teacher.provider`).
The API key is read from the environment only — `.env` is loaded into the environment first;
the manifest records the environment variable name and endpoint identifier, never the secret value.
Generated programs receive a sanitized environment. Formal local builds also
require the macOS `sandbox-exec` verifier sandbox, which denies network access,
home-directory reads, writes outside the verifier temp directory, and process
forking. `--allow-unsafe-verifier` is for isolated tests only.

For an amended cohort, `--import-accepted-from` may reuse exact-ID-overlapping
targets from a quarantined predecessor. Every imported row is re-provenanced to
the new cohort and rerun through the full sandboxed verifier before any new
teacher call; non-overlapping rows are ignored.

For Qwen3.5 served through an OpenAI-compatible vLLM/SGLang-style endpoint, set
the model's native thinking mode per request:

```bash
python -m rl_training.build_sft_dataset_azure \
  --cohort-manifest "$COHORT_DIR/manifest.json" \
  --output-dir "$SFT_DATA_DIR" \
  --teacher-deployment "$AZURE_OPENAI_DEPLOYMENT" \
  --chat-template-enable-thinking false
```

This adds `chat_template_kwargs: {"enable_thinking": false}` to the HTTP body.
Use `true` for a native-thinking distillation run. Omit the flag for hosted Azure
teacher deployments that do not support this vLLM/Qwen extension.

Validate before training or publishing. The batched runner performs a full
structural preflight, then executes bounded batches in fresh sandboxed processes
so a long sweep cannot accumulate enough resource pressure to create timeout
false negatives. Any apparent batch failure must pass three isolated fresh-process
confirmations; a genuinely flaky target still fails the gate.

```bash
python -m rl_training.validate_sft_distill_batched \
  --train "$SFT_DATA_DIR/train.jsonl" \
  --train_source "$COHORT_DIR/train.jsonl" \
  --report "$SFT_DATA_DIR/validation_report.json" \
  --batch_size 50 \
  --confirmation_runs 3 \
  --verifier_workers 8 \
  --verifier_timeout 10
```

`rl_training/build_sft_distill.py` is retained only as a legacy, drop-allowed
OpenAI full-pool utility. It requires `--legacy_full_pool` and its outputs must
not be used in the formal SFT/GRPO/DPO/PPO comparison.

## RL Anticipation and Preferences

The SFT/GRPO/DPO/PPO `rl-preferences` and `rl-anticipation` arms are separate
from the existing training-data preference and coding anticipation commands.
The RL anticipation entry delegates to the existing shared `run_anticipation`
render/query/parse/save pipeline with an RL spec and native-thinking agent
factory. RL preferences likewise reuse the existing pairwise scorer and result
schema. Build the shared three-level stimulus file only after real artifacts
exist for every method:

```bash
python3 scripts/build_rl_stimuli.py \
  --cohort_manifest "$SHARED_COHORT_DIR/manifest.json" \
  --source_train "$SHARED_COHORT_DIR/train.jsonl" \
  --sft_train "$SFT_DATA_DIR/train.jsonl" \
  --sft_manifest "$SFT_DATA_DIR/manifest.json" \
  --dpo_pairs "$SHARED_COHORT_DIR/artifacts/dpo.jsonl" \
  --grpo_rollouts "$RUN_ROOT/runs/<grpo-run>/rollout_samples.rank0.jsonl" \
  --ppo_rollouts "$RUN_ROOT/runs/<ppo-run>/ppo_rollout_samples.jsonl" \
  --ppo_rm_manifest "$RUN_ROOT/runs/<reward-model>/reward_model_manifest.json"
```

The builder writes `data/source/rl_training_stimuli.json`. It samples real
artifacts deterministically by `record_id`, records source/config hashes, checks
that every selected ID belongs to the frozen N-prompt cohort, and refuses
synthetic examples. It also requires the PPO reward-model manifest to reference
the same DPO pair file. Missing or stale manifests are hard errors.

Serve one Qwen checkpoint through the existing OpenAI-compatible vLLM endpoint.
For Qwen's native thinking/non-thinking switch, enable its reasoning parser on
the server; the runner sends `chat_template_kwargs.enable_thinking` per request,
so one endpoint supports both variants:

```bash
vllm serve <model-or-checkpoint> \
  --served-model-name <served-name> \
  --reasoning-parser qwen3 \
  --dtype bfloat16 --max-model-len 4096 \
  --api-key dummy-key --host 0.0.0.0 --port 8000
```

Run both RL-method preference and RL drift elicitation in both modes:

```bash
python3 main.py run_rl_preferences_thinking -- --model_key <vllm-model-key>
python3 main.py run_rl_preferences_non_thinking -- --model_key <vllm-model-key>
python3 main.py run_rl_anticipation_thinking -- --model_key <vllm-model-key>
python3 main.py run_rl_anticipation_non_thinking -- --model_key <vllm-model-key>
```

The dedicated RL CLI uses `--thinking on|off`; it deliberately does not expose
the legacy `--reasoning` switch, whose off path may also change prompting and
inference strategy in the older batteries.

All four entries use sampled vLLM chat completions. The two variants keep the
answer-only prompt, bare-label parser, temperature, K, maximum output tokens,
concurrency, and timeout fixed; only
`chat_template_kwargs.enable_thinking` changes. In particular,
`non_thinking` does not use the older reasoning-off logprobs shortcut. Raw
metadata records `prompt_reasoning=false`, `native_thinking`, and the fixed
answer format. Results use the existing `results/<model_key>/pairs/` and
`results/<model_key>/anticipation/` schemas.

For a matched DPO-versus-GRPO-only analysis, use the separate real-artifact
builder and comparison flag:

```bash
python3 scripts/build_dpo_grpo_rl_stimuli.py \
  --cohort_manifest <cohort>/manifest.json \
  --dpo_pairs <cohort>/artifacts/dpo.jsonl \
  --grpo_rollouts <corrected-grpo-run>/rollout_samples.rank0.jsonl \
  --dpo_run_metadata <dpo-run>/run_metadata.json \
  --grpo_run_metadata <corrected-grpo-run>/run_metadata.json

python3 scripts/run_rl_anticipation_preferences.py rl-preferences \
  --comparison dpo-grpo --model_key <vllm-model-key> --level all --thinking on
```

`scripts/run_dpo_grpo_rl_trajectory.py` expands both dedicated instruments
across M0 and all 32 saved checkpoints for DPO, GRPO, and PPO (steps 4 through
124 every four steps, plus final step 125), in both native-thinking modes. M0
anticipation is the prospective forecast; checkpoint anticipation is labeled
separately as self-forecast drift. The superseded V2/V3 figures and plotters
were removed; add V4-provenanced plotting only after the new result schema is
validated.

For 9B, first generate a new model-specific full sweep and exact 1,000-record
cohort; 4B DPO pairs must not be reused. Build the stimuli with
`--output data/source/rl_dpo_grpo_training_stimuli_9b.json`, then run
`scripts/run_dpo_grpo_rl_trajectory.py --size 9b`. Keep future 9B figures in a
separate V4-provenanced namespace rather than pooling them with 4B.

## Train

Install training dependencies separately from the scoring/serving environment.
The canonical project environment comes from the lockfile:

```bash
uv sync --locked
```

On an NVIDIA machine without `uv`, the installer creates the same dedicated
CUDA 12.4 training environment in `.venv` and validates CUDA access:

```bash
./scripts/install_rl_env.sh
source .venv/bin/activate
```

The installer requires Python 3.12 and an NVIDIA driver that supports CUDA
12.4. Set `RL_ENV_PATH=/some/path` to place the environment elsewhere. To
install on a login node where GPUs are intentionally hidden, use
`REQUIRE_CUDA=0 ./scripts/install_rl_env.sh` and validate CUDA on the compute
node before training.

The standard workflow keeps data under this repository's `data/` directory:

```bash
export PROJECT_ROOT="$(git rev-parse --show-toplevel)"
export DATA_ROOT="$PROJECT_ROOT"
export RL_DATA_DIR="$PROJECT_ROOT/data/rl/v1"
export CUDA_VISIBLE_DEVICES=0
./scripts/prepare_rl_dataset.sh
./scripts/train_rl_grpo.sh
```

Prepare the RL smoke dataset:

```bash
./scripts/prepare_rl_dataset.sh
```

Then run the one-step GRPO smoke test:

```bash
./scripts/train_rl_grpo.sh
```

The scripts already use the repository root by default. Set only the Python
environment when needed:

```sh
export PYTHON_BIN="$PWD/.venv/bin/python"
```

`prepare_rl_dataset.sh` only writes train/dev/heldout JSONL files and then
exits. `train_rl_grpo.sh` requires those files and only performs training.
Extra trainer arguments can still be appended to the training command line.

Run a full GRPO experiment:

```sh
export GRPO_CONFIG="$PWD/rl_training/configs/grpo.yaml"
export GRPO_DATASET="$RL_DATA_DIR/cohorts/<model>-shared-n${COHORT_SIZE}-s0/train.jsonl"
export GRPO_EVAL_DATASET="$RL_DATA_DIR/sources/<model>/nemotron_rl_coding_competitive_dev.jsonl"
export GRPO_RUN_NAME="qwen4b-grpo-s0"
```

Then run `./scripts/train_rl_grpo.sh`.

The formal GRPO configuration is matched to DPO by **unique prompt IDs**, not
by rollout rows. At K=8 it trains on the same 1,000 frozen prompts as DPO:
64 completion sequences per optimizer update / 8 generations per prompt =
8 unique prompts per update, for 125 updates and 8,000 total completions. The
trainer performs this audit before training and refuses the old 125-prompt
interpretation.

On a machine with a working vLLM install, set `GRPO_USE_VLLM=1` to switch GRPO
rollouts to vLLM (colocate mode); the default path uses Transformers `generate`
and needs no vLLM. Note: reruns with the same run name append to
`reward_log.jsonl` / `rollout_samples.*.jsonl` — use a fresh `GRPO_RUN_NAME`
per attempt.

Run DPO:

```bash
python -m rl_training.train_rl \
  --algo dpo \
  --config rl_training/configs/dpo.yaml \
  --base_model <base-model> \
  --dataset "data/rl/v1/cohorts/<model>-shared-n${COHORT_SIZE}-s0/artifacts/dpo.jsonl" \
  --eval_dataset data/rl/nemotron_rl_coding_competitive_dev.jsonl \
  --seed 0 \
  --save_dir runs \
  --run_name qwen4b-dpo-s0 \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05
```

For the resumable 4B/9B raw-base data-and-training pilot on a multi-GPU host,
use `scripts/launch_dpo_4b_9b.sh`. It pins the official model revisions, gives
each size its own GPU/source/sweep/cohort/run namespace, freezes N=1000 only
after the complete K=8 sweep, and writes logs/PIDs below
`STORAGE_ROOT` (default `/ndata/xianglin/ai_drift`). Override `BASE_MODEL` and
`MODEL_REVISION` when the adopted M0 checkpoints are available; never reuse a
raw-base sweep for an M0 run.

### Reward Model (PPO)

PPO needs a reward model first. Train one from the SAME DPO pair file (so PPO's
objective matches GRPO/DPO), then point `--reward_model` at the saved directory:

```bash
python -m rl_training.train_reward_model \
  --config rl_training/configs/reward_model.yaml \
  --base_model <base-model> \
  --pairs "data/rl/v1/cohorts/<model>-shared-n${COHORT_SIZE}-s0/artifacts/dpo.jsonl" \
  --output_dir runs/rm-qwen4b-s0
```

This trains a Bradley-Terry pairwise RM (`AutoModelForSequenceClassification`,
`num_labels=1`) and reports **held-out pairwise accuracy** — the quality gate. An
RM near 0.5 means PPO would optimize noise; inspect the pairs before proceeding.
Train one RM per base-model size on that size's own pairs. The saved dir is a
plain seq-classification model (LoRA is merged in on save) that PPO loads as both
reward and value model.

PPO holds four models at once (policy+LoRA, frozen reference, reward, value), so
its train dataset is pre-tokenized prompts only — the `verifier` column that GRPO
carries is dropped (`load_ppo_prompt_dataset`), since PPO's reward comes from the
RM and per-checkpoint verifier eval reads its own `--eval_dataset`. The PPO path
has not yet run end-to-end; validate it on a `*_smoke` config at 0.8B before
booking a real GPU. Three things to confirm on that first run: TRL 0.24's
`RewardTrainer` dataset format (text vs pre-tokenized), the PEFT policy + separate
value model composing, and the LoRA target modules on Qwen3.5's hybrid attention
(18/24 layers are `linear_attn`, which the default `q_proj,...` list skips — use
`--lora_target_modules all-linear` if so, held identical across all arms).

Run PPO with the trained reward model:

```bash
python -m rl_training.train_rl \
  --algo ppo \
  --config rl_training/configs/ppo.yaml \
  --base_model <base-model> \
  --reward_model runs/rm-qwen4b-s0 \
  --dataset "data/rl/v1/cohorts/<model>-shared-n${COHORT_SIZE}-s0/train.jsonl" \
  --eval_dataset data/rl/nemotron_rl_coding_competitive_dev.jsonl \
  --seed 0 \
  --save_dir runs \
  --run_name qwen4b-ppo-s0 \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05
```

Checkpoints are written under:

```text
runs/<run-name>/checkpoint-<step>/
```

Each run also writes `reward_log.jsonl` with verifier reward evaluated at
checkpoint cadence and `run_metadata.json` with the base model, dataset, trainer
config, and LoRA settings.

LoRA defaults:

```text
--lora_r 16
--lora_alpha 32
--lora_dropout 0.05
--lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

Use the same LoRA settings for GRPO/DPO/PPO so LoRA is not an extra experimental
condition.

LoRA checkpoints are PEFT adapter checkpoints. The local HuggingFace scorer can
load them as long as `peft` is installed and checkpoint registration records the
base model as `tokenizer_path`.

## Register Checkpoints

Register saved checkpoints into `config.yaml` so the existing elicitation/scorer
can load them by `model_key`:

```bash
python rl_training/register_checkpoints.py \
  --run_dir runs/qwen4b-grpo-s0 \
  --base_model <base-model> \
  --algo grpo \
  --seed 0
```

This creates model keys such as:

```text
qwen4b-grpo-s0-step4
qwen4b-grpo-s0-step12
qwen4b-grpo-s0-step40
qwen4b-grpo-s0-step125
```

For LoRA runs, the registered `path` points at the adapter checkpoint and
`tokenizer_path` points at the base model. `compute_utilities/llm_agent.py`
detects `adapter_config.json` and loads base model + adapter for local
HuggingFace evaluation.

## Score Drift

After registration, run the existing elicitation pipeline for each checkpoint.
Examples:

```bash
uv run main.py run_elicitations_task -- \
  --model_key qwen4b-grpo-s0-step40 \
  --save_dir results/qwen4b-grpo-s0/step40/task_preference

uv run main.py run_utilities_values -- \
  --model_key qwen4b-grpo-s0-step40 \
  --save_dir results/qwen4b-grpo-s0/step40/values
```

The key readouts are:

- coding task preference drift from `run_elicitations_task`;
- values preference drift from pairwise A/B `run_utilities_values`;
- values anticipation from ternary `MORE` / `LESS` / `SAME` `run_values_anticipation`;
- held-out coding reward/capability movement.

Book A (`run_utilities_book_a`) is available for explicit Book A ablations or
baseline reconstruction, but it is not part of the current live rerun scope.

## Implementation Map

- `rl_training/data/`: dataset adapters, canonical schemas, split preparation,
  sample inspection, and learnable-band filtering.
- `rl_training/rewards.py`: verifier execution and reward dispatch.
- `rl_training/prompts.py`: shared coding prompt format.
- `rl_training/build_dpo_pairs.py`: verifier-derived DPO pair generation
  (local or `--endpoint` vLLM sampling).
- `rl_training/train_rl.py`: unified `grpo` / `dpo` / `ppo` training entry point.
- `rl_training/train_reward_model.py`: Bradley-Terry reward model for PPO,
  trained on the DPO pairs; reports held-out pairwise accuracy.
- `rl_training/configs/`: algorithm-specific matched configs
  (`grpo`/`dpo`/`ppo`/`reward_model`, plus `*_smoke` for shakedowns).
- `rl_training/register_checkpoints.py`: exports checkpoints into `config.yaml`.
- `compute_utilities/llm_agent.py`: local HuggingFace scorer; detects PEFT
  adapter checkpoints for LoRA evaluation.

## Not Built Yet

- `orchestrate_drift.py`: iterate a run's checkpoints and call the three
  elicitation books automatically.
- `analysis/drift.py`: align checkpoint utility scales on linking references and
  emit signed per-book drift trajectories.
