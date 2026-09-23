#!/bin/sh

set -eu

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
DATA_ROOT=${DATA_ROOT:-"$REPO_ROOT"}
RL_DATA_DIR=${RL_DATA_DIR:-"$DATA_ROOT/data/rl/v1"}
BASE_MODEL=${BASE_MODEL:-"Qwen/Qwen3.5-0.8B"}
MODEL_REVISION=${MODEL_REVISION:-}
DATA_RECIPE_TAG=${DATA_RECIPE_TAG:-"fulltests-v2"}
PYTHON_BIN=${PYTHON_BIN:-python3}
mkdir -p "$DATA_ROOT"

# Prompt-length eligibility can change with the tokenizer, so source pools are
# namespaced by the base model just like generation sweeps.
MODEL_TAG_BASE=$(printf '%s' "$BASE_MODEL" | tr '[:upper:]./' '[:lower:]--')
if [ -n "$MODEL_REVISION" ]; then
    REVISION_TAG=$(printf '%s' "$MODEL_REVISION" | tr '[:upper:]/.' '[:lower:]--' | cut -c1-12)
    MODEL_TAG_BASE="$MODEL_TAG_BASE--rev-$REVISION_TAG"
fi
MODEL_TAG_BASE="$MODEL_TAG_BASE--data-$DATA_RECIPE_TAG"
MODEL_TAG=${MODEL_TAG:-"$MODEL_TAG_BASE"}
SOURCE_DIR=${SOURCE_DIR:-"$RL_DATA_DIR/sources/$MODEL_TAG"}

# All knobs are env-overridable. By default preparation writes the full usable
# candidate pool. A model-specific sweep then scores every candidate, and the
# final shared formal cohort is frozen later by build_shared_cohort.py.
# For a smoke split, override TRAIN_SIZE/DEV_SIZE/HELDOUT_SIZE and OUTPUT_DIR,
# e.g. TRAIN_SIZE=20 DEV_SIZE=5 HELDOUT_SIZE=5
# OUTPUT_DIR="$DATA_ROOT/data/rl/smoke" ./scripts/prepare_rl_dataset.sh.
# SEED drives the (seeded, manifest-recorded) split shuffle in prepare.py —
# reruns with the same seed/revision/sizes reproduce byte-identical splits.
DATASET_ID=${DATASET_ID:-"nemotron_rl_coding_competitive"}
DATASET_REVISION=${DATASET_REVISION:-"ae1f446f299823ea3c4c00217942b53787278b31"}
TRAIN_SIZE=${TRAIN_SIZE:-0} # 0 = all usable rows remaining after dev/heldout
REQUIRE_MODEL_REVISION=${REQUIRE_MODEL_REVISION:-1}
DEV_SIZE=${DEV_SIZE:-250}
HELDOUT_SIZE=${HELDOUT_SIZE:-500}
MAX_TESTS=${MAX_TESTS:-0} # 0 = retain every verifier test; reward sampling happens later
SEED=${SEED:-0}
SAMPLE_COUNT=${SAMPLE_COUNT:-3}
# Rows whose rendered prompt exceeds this are dropped so training never sees a
# truncated problem. Keep equal to max_prompt_length in rl_training/configs/*.yaml,
# and measure with the tokenizer of the model actually being trained.
MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS:-1024}
PROMPT_TOKENIZER=${PROMPT_TOKENIZER:-"$BASE_MODEL"}
OUTPUT_DIR=${OUTPUT_DIR:-"$SOURCE_DIR"}

# Optional source-pool archive. No external write occurs unless set.
HF_DATASET_REPO=${HF_DATASET_REPO:-}
HF_SOURCE_PATH_IN_REPO=${HF_SOURCE_PATH_IN_REPO:-"sources/$MODEL_TAG"}
HF_PRIVATE=${HF_PRIVATE:-1}

if [ "$REQUIRE_MODEL_REVISION" = "1" ] && [ -z "$MODEL_REVISION" ]; then
    echo "MODEL_REVISION must pin the exact Hugging Face model commit for a formal source pool." >&2
    echo "Set REQUIRE_MODEL_REVISION=0 only for a local immutable model or smoke run." >&2
    exit 1
fi

echo "[rl-data] dataset=$DATASET_ID@$DATASET_REVISION"
echo "[rl-data] sizes=train:$TRAIN_SIZE dev:$DEV_SIZE heldout:$HELDOUT_SIZE seed:$SEED"
echo "[rl-data] max_prompt_tokens=$MAX_PROMPT_TOKENS tokenizer=$PROMPT_TOKENIZER"
echo "[rl-data] model_tag=$MODEL_TAG model_revision=${MODEL_REVISION:-default}"
echo "[rl-data] output=$OUTPUT_DIR"
echo "[rl-data] python=$PYTHON_BIN"

cd "$REPO_ROOT"
TOKENIZER_REVISION_ARGS=""
if [ -n "$MODEL_REVISION" ]; then
    TOKENIZER_REVISION_ARGS="--tokenizer_revision $MODEL_REVISION"
fi
"$PYTHON_BIN" -m rl_training.data.prepare \
    --dataset_id "$DATASET_ID" \
    --revision "$DATASET_REVISION" \
    --train_size "$TRAIN_SIZE" \
    --dev_size "$DEV_SIZE" \
    --heldout_size "$HELDOUT_SIZE" \
    --max_tests "$MAX_TESTS" \
    --max_prompt_tokens "$MAX_PROMPT_TOKENS" \
    --tokenizer "$PROMPT_TOKENIZER" \
    $TOKENIZER_REVISION_ARGS \
    --seed "$SEED" \
    --out_dir "$OUTPUT_DIR" \
    --sample "$SAMPLE_COUNT"

if [ -n "$HF_DATASET_REPO" ]; then
    HF_PRIVATE_ARG=""
    if [ "$HF_PRIVATE" = "1" ]; then
        HF_PRIVATE_ARG="--private"
    fi
    "$PYTHON_BIN" scripts/upload_rl_data.py \
        --local-dir "$OUTPUT_DIR" \
        --repo-id "$HF_DATASET_REPO" \
        --path-in-repo "$HF_SOURCE_PATH_IN_REPO" \
        --allow-pattern "${DATASET_ID}_train_*.jsonl" \
        --allow-pattern "${DATASET_ID}_dev.jsonl" \
        --allow-pattern "${DATASET_ID}_heldout.jsonl" \
        --allow-pattern "${DATASET_ID}_manifest.json" \
        $HF_PRIVATE_ARG
fi
