#!/bin/sh

set -eu

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
DATA_ROOT=${DATA_ROOT:-"$REPO_ROOT"}

GRPO_CONFIG=${GRPO_CONFIG:-"$REPO_ROOT/rl_training/configs/grpo_smoke.yaml"}
GRPO_BASE_MODEL=${GRPO_BASE_MODEL:-"Qwen/Qwen3.5-0.8B"}
GRPO_DATASET=${GRPO_DATASET:-"$DATA_ROOT/data/rl/smoke/nemotron_rl_coding_competitive_train_20.jsonl"}
GRPO_EVAL_DATASET=${GRPO_EVAL_DATASET:-"$DATA_ROOT/data/rl/smoke/nemotron_rl_coding_competitive_dev.jsonl"}
GRPO_SAVE_DIR=${GRPO_SAVE_DIR:-"$DATA_ROOT/runs"}
# Live reward-curve PNG: <GRPO_FIGURES_DIR>/<run_name>/reward_curve.png,
# refreshed every logging_steps and after each checkpoint eval.
GRPO_FIGURES_DIR=${GRPO_FIGURES_DIR:-"$REPO_ROOT/results"}
GRPO_RUN_NAME=${GRPO_RUN_NAME:-"qwen08b-grpo-smoke-s0-v2"}
GRPO_SEED=${GRPO_SEED:-0}
# GRPO_USE_VLLM=1 switches rollouts to vLLM (colocate mode) on machines that
# have it installed; the default Transformers-generate path needs no vLLM.
GRPO_USE_VLLM=${GRPO_USE_VLLM:-0}
# Parallel verifier subprocesses per reward call (see rl_training/rewards.py).
VERIFIER_MAX_WORKERS=${VERIFIER_MAX_WORKERS:-32}
export VERIFIER_MAX_WORKERS

USE_VLLM_FLAG=""
if [ "$GRPO_USE_VLLM" = "1" ]; then
    USE_VLLM_FLAG="--use_vllm"
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton-cache-grpo}
export CUDA_VISIBLE_DEVICES TRITON_CACHE_DIR


if [ ! -f "$GRPO_CONFIG" ]; then
    echo "GRPO config not found: $GRPO_CONFIG" >&2
    exit 1
fi

echo "[grpo] config=$GRPO_CONFIG"
echo "[grpo] model=$GRPO_BASE_MODEL"
echo "[grpo] data_root=$DATA_ROOT"
echo "[grpo] dataset=$GRPO_DATASET"
echo "[grpo] run=$GRPO_SAVE_DIR/$GRPO_RUN_NAME"

cd "$REPO_ROOT"

if [ ! -f "$GRPO_DATASET" ]; then
    echo "GRPO dataset not found: $GRPO_DATASET" >&2
    exit 1
fi

if [ ! -f "$GRPO_EVAL_DATASET" ]; then
    echo "GRPO eval dataset not found: $GRPO_EVAL_DATASET" >&2
    exit 1
fi

python -m rl_training.train_rl \
    --algo grpo \
    --config "$GRPO_CONFIG" \
    --base_model "$GRPO_BASE_MODEL" \
    --dataset "$GRPO_DATASET" \
    --eval_dataset "$GRPO_EVAL_DATASET" \
    --seed "$GRPO_SEED" \
    --save_dir "$GRPO_SAVE_DIR" \
    --run_name "$GRPO_RUN_NAME" \
    --figures_dir "$GRPO_FIGURES_DIR" \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    $USE_VLLM_FLAG \
    "$@"
