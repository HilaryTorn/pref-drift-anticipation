#!/bin/sh

# One-command full data build for the default 0.8B experiment.
#
# This runs both stages over the complete usable candidate pool:
#   1. prepare the full source pool after fixed dev/held-out reservation;
#   2. sample K completions for every candidate and build score/DPO artifacts.
#
# The defaults below are the complete 0.8B full-run settings. Environment overrides
# remain available for a different model or machine. When BASE_MODEL changes,
# MODEL_REVISION must also be supplied so outputs cannot be attributed to the
# wrong model revision.

set -eu

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

# Model identity. The revision is the exact Hugging Face commit resolved for
# Qwen/Qwen3.5-0.8B when this full-run entry point was created.
DEFAULT_BASE_MODEL="Qwen/Qwen3.5-0.8B"
DEFAULT_MODEL_REVISION="2fc06364715b967f1860aea9cf38778875588b17"
BASE_MODEL=${BASE_MODEL:-"$DEFAULT_BASE_MODEL"}
if [ "$BASE_MODEL" = "$DEFAULT_BASE_MODEL" ]; then
    MODEL_REVISION=${MODEL_REVISION:-"$DEFAULT_MODEL_REVISION"}
else
    MODEL_REVISION=${MODEL_REVISION:-}
fi
if [ -z "$MODEL_REVISION" ]; then
    echo "MODEL_REVISION is required when BASE_MODEL differs from $DEFAULT_BASE_MODEL." >&2
    exit 1
fi

# Local layout.
DATA_ROOT=${DATA_ROOT:-"$REPO_ROOT"}
RL_DATA_DIR=${RL_DATA_DIR:-"$DATA_ROOT/data/rl/v1"}
DATA_RECIPE_TAG=${DATA_RECIPE_TAG:-"fulltests-v2"}
PYTHON_BIN=${PYTHON_BIN:-"python3"}

# Full source-pool preparation.
DATASET_ID=${DATASET_ID:-"nemotron_rl_coding_competitive"}
DATASET_REVISION=${DATASET_REVISION:-"ae1f446f299823ea3c4c00217942b53787278b31"}
TRAIN_SIZE=${TRAIN_SIZE:-0}              # 0 = every usable candidate
DEV_SIZE=${DEV_SIZE:-250}
HELDOUT_SIZE=${HELDOUT_SIZE:-500}
MAX_TESTS=${MAX_TESTS:-0}               # 0 = preserve every verifier test
MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS:-1024}
SAMPLE_COUNT=${SAMPLE_COUNT:-3}
SEED=${SEED:-0}

# Full score/DPO sweep.
BACKEND=${BACKEND:-"vllm"}
K=${K:-8}
MARGIN=${MARGIN:-0.5}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
TEMPERATURE=${TEMPERATURE:-1.0}
REWARD_TIMEOUT=${REWARD_TIMEOUT:-5}
LIMIT=${LIMIT:-0}                         # 0 = every prepared prompt
RESUME=${RESUME:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.6}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
VERIFIER_MAX_WORKERS=${VERIFIER_MAX_WORKERS:-32}

# Optional Hugging Face archive. Leave empty for local-only generation.
HF_DATASET_REPO=${HF_DATASET_REPO:-}
HF_PRIVATE=${HF_PRIVATE:-1}

echo "[full-data] repository=$REPO_ROOT"
echo "[full-data] data=$RL_DATA_DIR"
echo "[full-data] model=$BASE_MODEL@$MODEL_REVISION"
echo "[full-data] recipe=$DATA_RECIPE_TAG prepare=train:$TRAIN_SIZE dev:$DEV_SIZE heldout:$HELDOUT_SIZE tests:$MAX_TESTS"
echo "[full-data] sweep=backend:$BACKEND K:$K margin:$MARGIN limit:$LIMIT resume:$RESUME"

DATA_ROOT="$DATA_ROOT" \
RL_DATA_DIR="$RL_DATA_DIR" \
BASE_MODEL="$BASE_MODEL" \
MODEL_REVISION="$MODEL_REVISION" \
DATA_RECIPE_TAG="$DATA_RECIPE_TAG" \
PYTHON_BIN="$PYTHON_BIN" \
DATASET_ID="$DATASET_ID" \
DATASET_REVISION="$DATASET_REVISION" \
TRAIN_SIZE="$TRAIN_SIZE" \
DEV_SIZE="$DEV_SIZE" \
HELDOUT_SIZE="$HELDOUT_SIZE" \
MAX_TESTS="$MAX_TESTS" \
MAX_PROMPT_TOKENS="$MAX_PROMPT_TOKENS" \
SAMPLE_COUNT="$SAMPLE_COUNT" \
SEED="$SEED" \
HF_DATASET_REPO="$HF_DATASET_REPO" \
HF_PRIVATE="$HF_PRIVATE" \
"$REPO_ROOT/scripts/prepare_rl_dataset.sh"

RL_DATA_DIR="$RL_DATA_DIR" \
GEN_MODEL="$BASE_MODEL" \
MODEL_REVISION="$MODEL_REVISION" \
DATA_RECIPE_TAG="$DATA_RECIPE_TAG" \
PYTHON_BIN="$PYTHON_BIN" \
BACKEND="$BACKEND" \
K="$K" \
MARGIN="$MARGIN" \
MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
TEMPERATURE="$TEMPERATURE" \
REWARD_TIMEOUT="$REWARD_TIMEOUT" \
LIMIT="$LIMIT" \
RESUME="$RESUME" \
GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
MAX_MODEL_LEN="$MAX_MODEL_LEN" \
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
VERIFIER_MAX_WORKERS="$VERIFIER_MAX_WORKERS" \
HF_DATASET_REPO="$HF_DATASET_REPO" \
HF_PRIVATE="$HF_PRIVATE" \
"$REPO_ROOT/scripts/build_dpo_data.sh"

echo "[full-data] source preparation and full score/DPO sweep complete."
