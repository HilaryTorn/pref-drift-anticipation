#!/usr/bin/env bash

# End-to-end, resumable DPO pipeline for one Qwen3.5 model size:
# model-specific source preparation -> full K=8 sweep -> frozen N=1000 cohort
# -> one-epoch matched LoRA DPO training.  Run from launch_dpo_4b_9b.sh for
# detached 4B/9B jobs, or invoke directly for debugging.

set -euo pipefail

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
MODEL_SIZE=${1:?usage: scripts/run_dpo_model.sh 4B|9B GPU_ID}
GPU_ID=${2:?usage: scripts/run_dpo_model.sh 4B|9B GPU_ID}

case "$MODEL_SIZE" in
    4B|4b)
        MODEL_SIZE=4B
        DEFAULT_MODEL=prism-drift/qwen35-4b-m0-v4
        DEFAULT_REVISION=dca63300371b0265bd7099f87b9eed55937bea22
        DEFAULT_HUB_REPO=prism-drift/qwen35-4b-m0-v4-formal-checkpoints
        ;;
    9B|9b)
        MODEL_SIZE=9B
        DEFAULT_MODEL=prism-drift/qwen35-9b-m0-v4
        DEFAULT_REVISION=8f3d499236d7b9d38a1f616f6b3241b22a381047
        DEFAULT_HUB_REPO=prism-drift/qwen35-9b-m0-v4-formal-checkpoints
        ;;
    *)
        echo "MODEL_SIZE must be 4B or 9B, got: $MODEL_SIZE" >&2
        exit 2
        ;;
esac

STORAGE_ROOT=${STORAGE_ROOT:-/ndata/xianglin/ai_drift}
BASE_MODEL=${BASE_MODEL:-$DEFAULT_MODEL}
MODEL_REVISION=${MODEL_REVISION:-$DEFAULT_REVISION}
DATA_RECIPE_TAG=${DATA_RECIPE_TAG:-fulltests-v2}
COHORT_SIZE=${COHORT_SIZE:-1000}
SEED=${SEED:-0}
K=${K:-8}
MARGIN=${MARGIN:-0.5}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.50}
VERIFIER_MAX_WORKERS=${VERIFIER_MAX_WORKERS:-32}
EXPERIMENT_TAG=${EXPERIMENT_TAG:-m0-v4-formal}
PYTHON_BIN=${PYTHON_BIN:-python}
HF_MODEL_REPO=${HF_MODEL_REPO:-$DEFAULT_HUB_REPO}
HUB_VISIBILITY=${HUB_VISIBILITY:-private}
FORMAL_ARCHIVE_MODE=${FORMAL_ARCHIVE_MODE:-hub}

if [ "$FORMAL_ARCHIVE_MODE" != hub ] && [ "$FORMAL_ARCHIVE_MODE" != local ]; then
    echo "FORMAL_ARCHIVE_MODE must be hub or local, got: $FORMAL_ARCHIVE_MODE" >&2
    exit 2
fi

if [ ! -d "$STORAGE_ROOT" ] || [ ! -w "$STORAGE_ROOT" ]; then
    echo "Storage root is missing or not writable: $STORAGE_ROOT" >&2
    exit 1
fi
if [ -z "$MODEL_REVISION" ] || [ "$MODEL_REVISION" = main ]; then
    echo "MODEL_REVISION must be an exact immutable commit, not empty/main." >&2
    exit 1
fi

MODEL_ID_TAG=$(printf '%s' "$BASE_MODEL" | tr '[:upper:]./' '[:lower:]--')
REVISION_TAG=$(printf '%s' "$MODEL_REVISION" | tr '[:upper:]/.' '[:lower:]--' | cut -c1-12)
MODEL_TAG="${MODEL_ID_TAG}--rev-${REVISION_TAG}--data-${DATA_RECIPE_TAG}"

RL_DATA_DIR="$STORAGE_ROOT/data/rl/v1"
SOURCE_DIR="$RL_DATA_DIR/sources/$MODEL_TAG"
SWEEP_DIR="$RL_DATA_DIR/sweeps/$MODEL_TAG"
COHORT_DIR="$RL_DATA_DIR/cohorts/${MODEL_TAG}-shared-n${COHORT_SIZE}-s${SEED}"
RUN_ROOT="$STORAGE_ROOT/runs"
FIGURES_ROOT="$STORAGE_ROOT/results"
RUN_NAME="qwen35-${MODEL_SIZE,,}-dpo-${EXPERIMENT_TAG}-n${COHORT_SIZE}-s${SEED}"
RUN_DIR="$RUN_ROOT/$RUN_NAME"

SOURCE_TRAIN="$SOURCE_DIR/nemotron_rl_coding_competitive_train_all.jsonl"
DEV_DATA="$SOURCE_DIR/nemotron_rl_coding_competitive_dev.jsonl"
HELDOUT_DATA="$SOURCE_DIR/nemotron_rl_coding_competitive_heldout.jsonl"
SOURCE_MANIFEST="$SOURCE_DIR/nemotron_rl_coding_competitive_manifest.json"
SCORE_DATA="$SWEEP_DIR/prompt_scores.jsonl"
ALL_DPO_PAIRS="$SWEEP_DIR/dpo_pairs.jsonl"
SWEEP_MANIFEST="$SWEEP_DIR/manifest.json"
SHARED_DPO="$COHORT_DIR/artifacts/dpo.jsonl"

mkdir -p "$RL_DATA_DIR" "$RUN_ROOT" "$FIGURES_ROOT" "$STORAGE_ROOT/triton/$RUN_NAME"
export CUDA_VISIBLE_DEVICES=$GPU_ID
export VERIFIER_MAX_WORKERS
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="$STORAGE_ROOT/triton/$RUN_NAME"

is_complete_sweep() {
    [ -s "$SWEEP_MANIFEST" ] && "$PYTHON_BIN" -c \
        'import json,sys; x=json.load(open(sys.argv[1])); raise SystemExit(0 if x.get("schema")=="model_data_sweep_v1" and x.get("status")=="complete" else 1)' \
        "$SWEEP_MANIFEST"
}

is_complete_training() {
    [ -s "$RUN_DIR/run_metadata.json" ] && "$PYTHON_BIN" -c \
        'import json,sys; x=json.load(open(sys.argv[1])); raise SystemExit(0 if x.get("status")=="complete" and x.get("global_step")==125 else 1)' \
        "$RUN_DIR/run_metadata.json"
}

echo "[pipeline] model=$BASE_MODEL@$MODEL_REVISION size=$MODEL_SIZE gpu=$GPU_ID"
echo "[pipeline] experiment=$EXPERIMENT_TAG storage=$STORAGE_ROOT"
echo "[pipeline] cohort=$COHORT_SIZE seed=$SEED K=$K margin=$MARGIN"
if [ "$FORMAL_ARCHIVE_MODE" = hub ]; then
    echo "[pipeline] checkpoint_archive=$HF_MODEL_REPO visibility=$HUB_VISIBILITY"
else
    echo "[pipeline] checkpoint_archive=local-only storage=$RUN_ROOT"
fi

if [ "${PREFLIGHT_ONLY:-0}" = 1 ]; then
    "$PYTHON_BIN" -c \
        'import yaml; from scripts.build_rl_stimuli import validate_config_parity; c={m: yaml.safe_load(open(f"rl_training/configs/{m}.yaml")) for m in ("sft","grpo","dpo","ppo")}; print("[pipeline] config parity", validate_config_parity(c))'
    echo "[pipeline] preflight complete"
    exit 0
fi

if [ ! -s "$SOURCE_TRAIN" ] || [ ! -s "$DEV_DATA" ] || [ ! -s "$HELDOUT_DATA" ] || [ ! -s "$SOURCE_MANIFEST" ]; then
    echo "[pipeline] preparing model-specific source pool"
    BASE_MODEL="$BASE_MODEL" \
    MODEL_REVISION="$MODEL_REVISION" \
    MODEL_TAG="$MODEL_TAG" \
    DATA_ROOT="$STORAGE_ROOT" \
    RL_DATA_DIR="$RL_DATA_DIR" \
    SOURCE_DIR="$SOURCE_DIR" \
    OUTPUT_DIR="$SOURCE_DIR" \
    PYTHON_BIN="$PYTHON_BIN" \
    "$REPO_ROOT/scripts/prepare_rl_dataset.sh"
else
    echo "[pipeline] source pool already present; reusing $SOURCE_DIR"
fi

if ! is_complete_sweep; then
    echo "[pipeline] running/resuming full on-policy score and DPO sweep"
    GEN_MODEL="$BASE_MODEL" \
    MODEL_REVISION="$MODEL_REVISION" \
    MODEL_TAG="$MODEL_TAG" \
    RL_DATA_DIR="$RL_DATA_DIR" \
    SOURCE_DIR="$SOURCE_DIR" \
    SWEEP_DIR="$SWEEP_DIR" \
    PROMPTS="$SOURCE_TRAIN" \
    K="$K" \
    MARGIN="$MARGIN" \
    BACKEND=vllm \
    GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    VERIFIER_MAX_WORKERS="$VERIFIER_MAX_WORKERS" \
    RESUME=1 \
    PYTHON_BIN="$PYTHON_BIN" \
    "$REPO_ROOT/scripts/build_dpo_data.sh"
else
    echo "[pipeline] completed sweep already present; reusing $SWEEP_DIR"
fi

if [ ! -s "$COHORT_DIR/manifest.json" ]; then
    echo "[pipeline] freezing exact shared cohort n=$COHORT_SIZE"
    "$PYTHON_BIN" -m rl_training.data.build_shared_cohort \
        --dataset "$SOURCE_TRAIN" \
        --dataset-manifest "$SOURCE_MANIFEST" \
        --score-source "grpo=$SCORE_DATA" \
        --score-manifest "grpo=$SWEEP_MANIFEST" \
        --artifact-source "dpo=$ALL_DPO_PAIRS" \
        --artifact-manifest "dpo=$SWEEP_MANIFEST" \
        --size "$COHORT_SIZE" \
        --seed "$SEED" \
        --out-dir "$COHORT_DIR"
else
    echo "[pipeline] frozen cohort already present; reusing $COHORT_DIR"
fi

PAIR_ROWS=$(wc -l < "$SHARED_DPO")
if [ "$PAIR_ROWS" -ne "$COHORT_SIZE" ]; then
    echo "Frozen DPO artifact has $PAIR_ROWS rows, expected $COHORT_SIZE: $SHARED_DPO" >&2
    exit 1
fi

if is_complete_training; then
    echo "[pipeline] DPO training already complete: $RUN_DIR"
    exit 0
fi

echo "[pipeline] resolving immutable local snapshot"
BASE_MODEL_PATH=$(BASE_MODEL="$BASE_MODEL" MODEL_REVISION="$MODEL_REVISION" \
    "$PYTHON_BIN" -c \
    'import os; from huggingface_hub import snapshot_download; print(snapshot_download(os.environ["BASE_MODEL"], revision=os.environ["MODEL_REVISION"]))')

if [ -d "$RUN_DIR" ] && [ -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
    echo "Incomplete DPO run cannot resume because checkpoints intentionally omit optimizer state: $RUN_DIR" >&2
    echo "Diagnose/archive the incomplete run, then restart this arm cleanly." >&2
    exit 1
fi

echo "[pipeline] training DPO for 125 optimizer steps (one epoch), save every 4"
HUB_ARGS=()
if [ "$FORMAL_ARCHIVE_MODE" = hub ] && [ -n "$HF_MODEL_REPO" ]; then
    HUB_ARGS=(
        --hub_repo_id "$HF_MODEL_REPO"
        --hub_base_model_id "$BASE_MODEL"
        --hub_base_revision "$MODEL_REVISION"
        --cohort_manifest "$COHORT_DIR/manifest.json"
        --hub_keep_local_checkpoints 1
    )
    if [ "$HUB_VISIBILITY" = public ]; then
        HUB_ARGS+=(--hub_public)
    elif [ "$HUB_VISIBILITY" != private ]; then
        echo "HUB_VISIBILITY must be private or public, got: $HUB_VISIBILITY" >&2
        exit 2
    fi
fi

TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=0 "$PYTHON_BIN" -m rl_training.train_rl \
    --algo dpo \
    --config "$REPO_ROOT/rl_training/configs/dpo.yaml" \
    --base_model "$BASE_MODEL_PATH" \
    --dataset "$SHARED_DPO" \
    --eval_dataset "$DEV_DATA" \
    --seed "$SEED" \
    --save_dir "$RUN_ROOT" \
    --figures_dir "$FIGURES_ROOT" \
    --run_name "$RUN_NAME" \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules all-linear \
    "${HUB_ARGS[@]}"

echo "[pipeline] complete model=$MODEL_SIZE run=$RUN_DIR"
