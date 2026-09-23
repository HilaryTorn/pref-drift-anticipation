#!/bin/sh

# Build a model-specific score/DPO sweep from a prepared coding_task_v1
# candidate pool:
#
#   1. sample K completions per prompt from the base model, score each with the
#      I/O verifier (rl_training/build_dpo_pairs.py); writes the DPO pairs file
#      and a per-prompt score file;
#   2. archive the model-specific sweep locally and, optionally, on Hugging
#      Face. Final filtering is deliberately separate: once the experiment's
#      methods are chosen, build_shared_cohort.py intersects every method's
#      eligibility and freezes one cohort for all training arms.
#
# The generator model MUST be the same base model the online arms train, so the
# DPO pairs are on-policy for the study. If the base model changes (e.g. 0.8B
# -> 4B), this whole script must be re-run with GEN_MODEL set accordingly.
#
# Run scripts/prepare_rl_dataset.sh first to create the split. For a quick
# pilot, set LIMIT=200 — the progress bar's `solved=` ratio tells you early
# whether the model is strong enough before you pay for the full sweep.

set -eu

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

RL_DATA_DIR=${RL_DATA_DIR:-"$REPO_ROOT/data/rl/v1"}
GEN_MODEL=${GEN_MODEL:-"Qwen/Qwen3.5-0.8B"}
MODEL_REVISION=${MODEL_REVISION:-}
DATA_RECIPE_TAG=${DATA_RECIPE_TAG:-"fulltests-v2"}
BACKEND=${BACKEND:-"vllm"}
PYTHON_BIN=${PYTHON_BIN:-"python3"}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}

# Each full model ID + revision gets its own namespace. Pair/score eligibility
# is model-specific and must never be reused after changing either one.
MODEL_TAG_BASE=$(printf '%s' "$GEN_MODEL" | tr '[:upper:]./' '[:lower:]--')
if [ -n "$MODEL_REVISION" ]; then
    REVISION_TAG=$(printf '%s' "$MODEL_REVISION" | tr '[:upper:]/.' '[:lower:]--' | cut -c1-12)
    MODEL_TAG_BASE="$MODEL_TAG_BASE--rev-$REVISION_TAG"
fi
MODEL_TAG_BASE="$MODEL_TAG_BASE--data-$DATA_RECIPE_TAG"
MODEL_TAG=${MODEL_TAG:-"$MODEL_TAG_BASE"}
SOURCE_DIR=${SOURCE_DIR:-"$RL_DATA_DIR/sources/$MODEL_TAG"}
SWEEP_DIR=${SWEEP_DIR:-"$RL_DATA_DIR/sweeps/$MODEL_TAG"}

PROMPTS=${PROMPTS:-"$SOURCE_DIR/nemotron_rl_coding_competitive_train_all.jsonl"}
OUT_PAIRS=${OUT_PAIRS:-"$SWEEP_DIR/dpo_pairs.jsonl"}
OUT_SCORES=${OUT_SCORES:-"$SWEEP_DIR/prompt_scores.jsonl"}
OUT_MANIFEST=${OUT_MANIFEST:-"$SWEEP_DIR/manifest.json"}

K=${K:-8}
MARGIN=${MARGIN:-0.5}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
TEMPERATURE=${TEMPERATURE:-1.0}
REWARD_TIMEOUT=${REWARD_TIMEOUT:-5}
LIMIT=${LIMIT:-0}   # 0 = all prompts; set e.g. 200 for a pilot sweep
RESUME=${RESUME:-1}
REQUIRE_MODEL_REVISION=${REQUIRE_MODEL_REVISION:-1}

# BACKEND=vllm runs generation on the dedicated vLLM env (~10x faster sweep).
# vLLM pins its own torch/CUDA so it lives in its own env, NOT the training
# env — see rl_training/requirements-training.txt for the version constraints.

GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.6}   # vllm backend; GPU may be shared
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}                     # vllm backend

# Optional archive. No external write occurs unless HF_DATASET_REPO is set.
HF_DATASET_REPO=${HF_DATASET_REPO:-}
HF_PATH_IN_REPO=${HF_PATH_IN_REPO:-"sweeps/$MODEL_TAG"}
HF_PRIVATE=${HF_PRIVATE:-1}

if [ "$REQUIRE_MODEL_REVISION" = "1" ] && [ -z "$MODEL_REVISION" ]; then
    echo "MODEL_REVISION must pin the exact Hugging Face model commit for a formal sweep." >&2
    echo "Set REQUIRE_MODEL_REVISION=0 only for a local immutable model or smoke run." >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
VERIFIER_MAX_WORKERS=${VERIFIER_MAX_WORKERS:-32}
export CUDA_VISIBLE_DEVICES VERIFIER_MAX_WORKERS

if [ ! -f "$PROMPTS" ]; then
    echo "Prompts split not found: $PROMPTS" >&2
    echo "Run scripts/prepare_rl_dataset.sh first, e.g.:" >&2
    echo "  RL_DATA_DIR=$RL_DATA_DIR ./scripts/prepare_rl_dataset.sh" >&2
    exit 1
fi

LIMIT_ARGS=""
if [ "$LIMIT" -gt 0 ]; then
    LIMIT_ARGS="--limit $LIMIT"
    echo "[dpo-data] PILOT MODE: only the first $LIMIT prompts are scored;"
    echo "[dpo-data] the score/pair outputs will cover just that subset."
fi

MODEL_REVISION_ARGS=""
if [ -n "$MODEL_REVISION" ]; then
    MODEL_REVISION_ARGS="--model_revision $MODEL_REVISION"
fi

RESUME_ARGS=""
if [ "$RESUME" = "1" ]; then
    RESUME_ARGS="--resume"
fi

echo "[dpo-data] generator model=$GEN_MODEL backend=$BACKEND"
echo "[dpo-data] model_tag=$MODEL_TAG model_revision=${MODEL_REVISION:-default}"
echo "[dpo-data] python=$PYTHON_BIN"
echo "[dpo-data] prompts=$PROMPTS"
echo "[dpo-data] K=$K margin=$MARGIN max_new_tokens=$MAX_NEW_TOKENS timeout=${REWARD_TIMEOUT}s"
echo "[dpo-data] pairs -> $OUT_PAIRS"
echo "[dpo-data] scores -> $OUT_SCORES"
echo "[dpo-data] manifest -> $OUT_MANIFEST"
echo "[dpo-data] sweep dir=$SWEEP_DIR"

cd "$REPO_ROOT"

"$PYTHON_BIN" -m rl_training.build_dpo_pairs \
    --model "$GEN_MODEL" \
    --prompts "$PROMPTS" \
    --out "$OUT_PAIRS" \
    --scores_out "$OUT_SCORES" \
    --manifest_out "$OUT_MANIFEST" \
    --backend "$BACKEND" \
    --K "$K" \
    --margin "$MARGIN" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --reward_timeout "$REWARD_TIMEOUT" \
    --device cuda \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    --max_model_len "$MAX_MODEL_LEN" \
    $MODEL_REVISION_ARGS \
    $RESUME_ARGS \
    $LIMIT_ARGS

echo "[dpo-data] done."
echo "[dpo-data] DPO-eligible pairs: $(wc -l < "$OUT_PAIRS")"
echo "[dpo-data] scored prompts:     $(wc -l < "$OUT_SCORES")"
echo "[dpo-data] Full sweep complete. Choose COHORT_SIZE only after every requested method artifact exists:"
echo "  $PYTHON_BIN -m rl_training.data.build_shared_cohort --dataset $PROMPTS --score-source grpo=$OUT_SCORES --score-manifest grpo=$OUT_MANIFEST --artifact-source dpo=$OUT_PAIRS --artifact-manifest dpo=$OUT_MANIFEST --size \$COHORT_SIZE --out-dir $RL_DATA_DIR/cohorts/$MODEL_TAG-shared-n\$COHORT_SIZE-s0"

if [ -n "$HF_DATASET_REPO" ]; then
    HF_PRIVATE_ARG=""
    if [ "$HF_PRIVATE" = "1" ]; then
        HF_PRIVATE_ARG="--private"
    fi
    "$PYTHON_BIN" scripts/upload_rl_data.py \
        --local-dir "$SWEEP_DIR" \
        --repo-id "$HF_DATASET_REPO" \
        --path-in-repo "$HF_PATH_IN_REPO" \
        $HF_PRIVATE_ARG
fi
