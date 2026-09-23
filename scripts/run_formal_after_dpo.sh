#!/usr/bin/env bash

# Continue one model's formal comparison after run_dpo_model.sh has frozen the
# shared N=1000 cohort and completed DPO. The arms run strictly sequentially on
# one GPU. Any failure stops the sequence. Completed artifacts are reused, but
# an incomplete training arm must restart cleanly because model-only checkpoints
# intentionally omit optimizer/scheduler/RNG state.

set -euo pipefail

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
MODEL_SIZE=${1:?usage: scripts/run_formal_after_dpo.sh 4B|9B GPU_ID}
GPU_ID=${2:?usage: scripts/run_formal_after_dpo.sh 4B|9B GPU_ID}

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
EXPERIMENT_TAG=${EXPERIMENT_TAG:-m0-v4-formal}
HF_MODEL_REPO=${HF_MODEL_REPO:-$DEFAULT_HUB_REPO}
HF_DATASET_REPO=${HF_DATASET_REPO:-prism-drift/ai-pref-drift-rl-v4}
HUB_VISIBILITY=${HUB_VISIBILITY:-private}
FORMAL_ARCHIVE_MODE=${FORMAL_ARCHIVE_MODE:-hub}
PYTHON_BIN=${PYTHON_BIN:-python}
DPO_WAIT_SECONDS=${DPO_WAIT_SECONDS:-60}
RUN_SFT=${RUN_SFT:-0}
HUB_UPLOADS_START_DISABLED=${HUB_UPLOADS_START_DISABLED:-0}

if [ "$RUN_SFT" != 0 ] && [ "$RUN_SFT" != 1 ]; then
    echo "RUN_SFT must be 0 or 1, got: $RUN_SFT" >&2
    exit 2
fi
if [ "$HUB_UPLOADS_START_DISABLED" != 0 ] && [ "$HUB_UPLOADS_START_DISABLED" != 1 ]; then
    echo "HUB_UPLOADS_START_DISABLED must be 0 or 1, got: $HUB_UPLOADS_START_DISABLED" >&2
    exit 2
fi
if [ "$FORMAL_ARCHIVE_MODE" != hub ] && [ "$FORMAL_ARCHIVE_MODE" != local ]; then
    echo "FORMAL_ARCHIVE_MODE must be hub or local, got: $FORMAL_ARCHIVE_MODE" >&2
    exit 2
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
DEV_DATA="$SOURCE_DIR/nemotron_rl_coding_competitive_dev.jsonl"
SHARED_TRAIN="$COHORT_DIR/train.jsonl"
SHARED_DPO="$COHORT_DIR/artifacts/dpo.jsonl"
COHORT_MANIFEST="$COHORT_DIR/manifest.json"

DPO_RUN_NAME="qwen35-${MODEL_SIZE,,}-dpo-${EXPERIMENT_TAG}-n${COHORT_SIZE}-s${SEED}"
GRPO_RUN_NAME="qwen35-${MODEL_SIZE,,}-grpo-${EXPERIMENT_TAG}-n${COHORT_SIZE}-s${SEED}"
PPO_RUN_NAME="qwen35-${MODEL_SIZE,,}-ppo-${EXPERIMENT_TAG}-n${COHORT_SIZE}-s${SEED}"
SFT_RUN_NAME="qwen35-${MODEL_SIZE,,}-sft-${EXPERIMENT_TAG}-n${COHORT_SIZE}-s${SEED}"
RM_DIR="$RUN_ROOT/qwen35-${MODEL_SIZE,,}-reward-model-${EXPERIMENT_TAG}-n${COHORT_SIZE}-s${SEED}"

mkdir -p "$RUN_ROOT" "$FIGURES_ROOT" "$STORAGE_ROOT/triton"
export CUDA_VISIBLE_DEVICES=$GPU_ID
export VERIFIER_MAX_WORKERS=${VERIFIER_MAX_WORKERS:-32}
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

run_complete() {
    local manifest=$1
    local algo=$2
    [ -s "$manifest" ] && "$PYTHON_BIN" -c \
        'import json,sys; x=json.load(open(sys.argv[1])); raise SystemExit(0 if x.get("status")=="complete" and x.get("algo")==sys.argv[2] and x.get("global_step")==125 else 1)' \
        "$manifest" "$algo"
}

refuse_incomplete_model_only_run() {
    local run_dir=$1
    if [ -d "$run_dir" ] && [ -n "$(find "$run_dir" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
        echo "Incomplete run cannot resume because checkpoints intentionally omit optimizer state: $run_dir" >&2
        echo "Diagnose/archive the incomplete run, then restart this arm cleanly." >&2
        exit 1
    fi
}

HUB_ARGS=()
if [ "$FORMAL_ARCHIVE_MODE" = hub ]; then
    HUB_ARGS=(
        --hub_repo_id "$HF_MODEL_REPO"
        --hub_base_model_id "$BASE_MODEL"
        --hub_base_revision "$MODEL_REVISION"
        --cohort_manifest "$COHORT_MANIFEST"
        --hub_keep_local_checkpoints 1
    )
    if [ "$HUB_VISIBILITY" = public ]; then
        HUB_ARGS+=(--hub_public)
    elif [ "$HUB_VISIBILITY" != private ]; then
        echo "HUB_VISIBILITY must be private or public, got: $HUB_VISIBILITY" >&2
        exit 2
    fi
fi

echo "[sequence] waiting for completed DPO: $RUN_ROOT/$DPO_RUN_NAME"
while ! run_complete "$RUN_ROOT/$DPO_RUN_NAME/run_metadata.json" dpo; do
    sleep "$DPO_WAIT_SECONDS"
done

for required in "$DEV_DATA" "$SHARED_TRAIN" "$SHARED_DPO" "$COHORT_MANIFEST"; do
    if [ ! -s "$required" ]; then
        echo "Required formal artifact is missing: $required" >&2
        exit 1
    fi
done

DATA_ARCHIVE_MARKER="$COHORT_DIR/.formal_hub_archive_complete"
data_archive_complete() {
    [ -s "$DATA_ARCHIVE_MARKER" ] &&
        grep -Fqx "repo=$HF_DATASET_REPO" "$DATA_ARCHIVE_MARKER" &&
        grep -Fqx "model_tag=$MODEL_TAG" "$DATA_ARCHIVE_MARKER"
}
if [ "$FORMAL_ARCHIVE_MODE" = local ]; then
    echo "[sequence] formal data/checkpoints remain local under $STORAGE_ROOT"
elif ! data_archive_complete; then
    DATA_VISIBILITY_ARGS=()
    if [ "$HUB_VISIBILITY" = private ]; then
        DATA_VISIBILITY_ARGS=(--private)
    fi
    echo "[sequence] archiving formal source, sweep, and frozen cohort"
    if "$PYTHON_BIN" "$REPO_ROOT/scripts/upload_rl_data.py" \
            --local-dir "$SOURCE_DIR" \
            --repo-id "$HF_DATASET_REPO" \
            --path-in-repo "sources/$MODEL_TAG" \
            "${DATA_VISIBILITY_ARGS[@]}" && \
        "$PYTHON_BIN" "$REPO_ROOT/scripts/upload_rl_data.py" \
            --local-dir "$SWEEP_DIR" \
            --repo-id "$HF_DATASET_REPO" \
            --path-in-repo "sweeps/$MODEL_TAG" \
            "${DATA_VISIBILITY_ARGS[@]}" && \
        "$PYTHON_BIN" "$REPO_ROOT/scripts/upload_rl_data.py" \
            --local-dir "$COHORT_DIR" \
            --repo-id "$HF_DATASET_REPO" \
            --path-in-repo "cohorts/${MODEL_TAG}-shared-n${COHORT_SIZE}-s${SEED}" \
            "${DATA_VISIBILITY_ARGS[@]}"; then
        MARKER_TMP="$DATA_ARCHIVE_MARKER.tmp.$$"
        printf 'repo=%s\nmodel_tag=%s\n' "$HF_DATASET_REPO" "$MODEL_TAG" > "$MARKER_TMP"
        mv "$MARKER_TMP" "$DATA_ARCHIVE_MARKER"
    else
        echo "WARNING: formal dataset upload failed; retaining all data locally and continuing training" >&2
        echo "[sequence] dataset upload will be retried on the next sequence invocation"
    fi
fi

echo "[sequence] resolving immutable local snapshot"
BASE_MODEL_PATH=$(BASE_MODEL="$BASE_MODEL" MODEL_REVISION="$MODEL_REVISION" \
    "$PYTHON_BIN" -c \
    'import os; from huggingface_hub import snapshot_download; print(snapshot_download(os.environ["BASE_MODEL"], revision=os.environ["MODEL_REVISION"]))')

GRPO_RUN_DIR="$RUN_ROOT/$GRPO_RUN_NAME"
if ! run_complete "$GRPO_RUN_DIR/run_metadata.json" grpo; then
    refuse_incomplete_model_only_run "$GRPO_RUN_DIR"
    echo "[sequence] training GRPO"
    TRITON_CACHE_DIR="$STORAGE_ROOT/triton/$GRPO_RUN_NAME" \
    TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=0 "$PYTHON_BIN" -m rl_training.train_rl \
        --algo grpo \
        --config "$REPO_ROOT/rl_training/configs/grpo.yaml" \
        --base_model "$BASE_MODEL_PATH" \
        --dataset "$SHARED_TRAIN" \
        --eval_dataset "$DEV_DATA" \
        --seed "$SEED" \
        --save_dir "$RUN_ROOT" \
        --figures_dir "$FIGURES_ROOT" \
        --run_name "$GRPO_RUN_NAME" \
        --use_lora \
        --lora_r 16 \
        --lora_alpha 32 \
        --lora_dropout 0.05 \
        --lora_target_modules all-linear \
        "${HUB_ARGS[@]}"
fi

if [ ! -s "$RM_DIR/reward_model_manifest.json" ]; then
    if [ -d "$RM_DIR" ] && [ -n "$(find "$RM_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
        echo "Incomplete reward-model directory is not automatically overwritten: $RM_DIR" >&2
        exit 1
    fi
    echo "[sequence] training PPO reward model"
    TRANSFORMERS_OFFLINE=1 "$PYTHON_BIN" -m rl_training.train_reward_model \
        --config "$REPO_ROOT/rl_training/configs/reward_model.yaml" \
        --base_model "$BASE_MODEL_PATH" \
        --pairs "$SHARED_DPO" \
        --output_dir "$RM_DIR" \
        --seed "$SEED"
fi

RM_ACCURACY=$("$PYTHON_BIN" -c \
    'import json,sys; print(float(json.load(open(sys.argv[1]))["held_out_pairwise_accuracy"]))' \
    "$RM_DIR/reward_model_manifest.json")
"$PYTHON_BIN" -c \
    'import sys; x=float(sys.argv[1]); raise SystemExit(0 if x >= 0.60 else 1)' \
    "$RM_ACCURACY" || {
        echo "Reward model failed the formal >=0.60 accuracy gate: $RM_ACCURACY" >&2
        exit 1
    }
echo "[sequence] reward-model gate passed: $RM_ACCURACY"

PPO_RUN_DIR="$RUN_ROOT/$PPO_RUN_NAME"
if ! run_complete "$PPO_RUN_DIR/run_metadata.json" ppo; then
    if [ -d "$PPO_RUN_DIR" ] && [ -n "$(find "$PPO_RUN_DIR" -mindepth 1 -maxdepth 1 ! -name '.hub_uploads_disabled' -print -quit 2>/dev/null)" ]; then
        echo "TRL PPO cannot resume; incomplete PPO run needs manual diagnosis: $PPO_RUN_DIR" >&2
        exit 1
    fi
    if [ "$HUB_UPLOADS_START_DISABLED" = 1 ]; then
        mkdir -p "$PPO_RUN_DIR"
        touch "$PPO_RUN_DIR/.hub_uploads_disabled"
        echo "[sequence] PPO Hub uploads start disabled; checkpoints remain local until marker removal"
    fi
    echo "[sequence] training PPO"
    TRITON_CACHE_DIR="$STORAGE_ROOT/triton/$PPO_RUN_NAME" \
    TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=0 "$PYTHON_BIN" -m rl_training.train_rl \
        --algo ppo \
        --config "$REPO_ROOT/rl_training/configs/ppo.yaml" \
        --base_model "$BASE_MODEL_PATH" \
        --reward_model "$RM_DIR" \
        --dataset "$SHARED_TRAIN" \
        --eval_dataset "$DEV_DATA" \
        --seed "$SEED" \
        --save_dir "$RUN_ROOT" \
        --figures_dir "$FIGURES_ROOT" \
        --run_name "$PPO_RUN_NAME" \
        --use_lora \
        --lora_r 16 \
        --lora_alpha 32 \
        --lora_dropout 0.05 \
        --lora_target_modules all-linear \
        "${HUB_ARGS[@]}"
fi

if [ "$RUN_SFT" != 1 ]; then
    echo "[sequence] complete: $MODEL_SIZE DPO -> GRPO -> PPO (SFT intentionally skipped)"
    exit 0
fi

if [ -z "${AZURE_OPENAI_API_KEY:-}" ] || [ -z "${AZURE_OPENAI_ENDPOINT:-}" ] || \
   [ -z "${AZURE_OPENAI_API_VERSION:-}" ] || [ -z "${AZURE_OPENAI_DEPLOYMENT:-}" ]; then
    echo "[sequence] $MODEL_SIZE PPO complete; SFT is waiting for Azure credentials." >&2
    echo "Set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, and AZURE_OPENAI_DEPLOYMENT, then rerun this script." >&2
    exit 20
fi

TEACHER_TAG=$(printf '%s' "$AZURE_OPENAI_DEPLOYMENT" | tr '[:upper:]/.' '[:lower:]--')
SFT_DATA_DIR="$RL_DATA_DIR/sft_datasets/${MODEL_TAG}-shared-n${COHORT_SIZE}-s${SEED}/$TEACHER_TAG"
if [ ! -s "$SFT_DATA_DIR/manifest.json" ]; then
    echo "[sequence] generating exact-N Azure SFT targets"
    SFT_BUILD_RESUME_ARGS=()
    if [ -d "$SFT_DATA_DIR" ] && [ -n "$(find "$SFT_DATA_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
        SFT_BUILD_RESUME_ARGS=(--resume)
    fi
    "$PYTHON_BIN" -m rl_training.build_sft_dataset_azure \
        --cohort-manifest "$COHORT_MANIFEST" \
        --output-dir "$SFT_DATA_DIR" \
        --max-attempts 3 \
        "${SFT_BUILD_RESUME_ARGS[@]}"
fi

SFT_RUN_DIR="$RUN_ROOT/$SFT_RUN_NAME"
if ! run_complete "$SFT_RUN_DIR/training_manifest.json" sft; then
    refuse_incomplete_model_only_run "$SFT_RUN_DIR"
    echo "[sequence] training SFT"
    TRITON_CACHE_DIR="$STORAGE_ROOT/triton/$SFT_RUN_NAME" \
    TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=0 "$PYTHON_BIN" -m rl_training.train_sft \
        --config "$REPO_ROOT/rl_training/configs/sft.yaml" \
        --base_model "$BASE_MODEL_PATH" \
        --dataset "$SFT_DATA_DIR/train.jsonl" \
        --dataset_manifest "$SFT_DATA_DIR/manifest.json" \
        --cohort_manifest "$COHORT_MANIFEST" \
        --seed "$SEED" \
        --save_dir "$RUN_ROOT" \
        --run_name "$SFT_RUN_NAME" \
        --lora_r 16 \
        --lora_alpha 32 \
        --lora_dropout 0.05 \
        --lora_target_modules all-linear \
        "${HUB_ARGS[@]}"
fi

echo "[sequence] complete: $MODEL_SIZE DPO -> GRPO -> PPO -> SFT"
