#!/bin/sh
#
# Launch the M0 format-training run. Modeled on scripts/train_rl_grpo.sh, with M0 defaults.
#
# Smoke (2 steps, same 4B model -- see the note in grpo_m0_smoke.yaml about why not the 0.8B):
#   M0_CONFIG=m0/configs/grpo_m0_smoke.yaml M0_RUN_NAME=qwen4b-m0-smoke ./m0/scripts/train_m0.sh
#
# Real run:
#   ./m0/scripts/train_m0.sh
#
# 9B, after 4B has passed the adoption gate:
#   M0_BASE_MODEL=Qwen/Qwen3.5-9B M0_RUN_NAME=qwen9b-m0-s0 ./m0/scripts/train_m0.sh

set -eu

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
DATA_ROOT=${DATA_ROOT:-"$REPO_ROOT"}

M0_CONFIG=${M0_CONFIG:-"$REPO_ROOT/m0/configs/grpo_m0.yaml"}
# 4B is the debug AND headline rung for M0. The 0.8B is deliberately not a default anywhere in
# this package: it reasons in visible content, so it cannot exercise the native-thinking path.
M0_BASE_MODEL=${M0_BASE_MODEL:-"Qwen/Qwen3.5-4B"}
M0_DATA_DIR=${M0_DATA_DIR:-"$DATA_ROOT/data/rl/m0_format"}
M0_DATASET=${M0_DATASET:-"$M0_DATA_DIR/train.jsonl"}
M0_EVAL_DATASET=${M0_EVAL_DATASET:-"$M0_DATA_DIR/dev.jsonl"}
M0_SAVE_DIR=${M0_SAVE_DIR:-"$DATA_ROOT/runs"}
M0_RUN_NAME=${M0_RUN_NAME:-"qwen4b-m0-s0"}
M0_SEED=${M0_SEED:-0}
M0_USE_VLLM=${M0_USE_VLLM:-0}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton-cache-m0}
# The L4 runs close to full on this job, so allocator fragmentation is the difference between
# fitting and not: the first OOM here died needing 16 MiB with 168 MiB reserved-but-unallocated.
# expandable_segments lets the allocator grow segments instead of stranding those blocks.
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_VISIBLE_DEVICES TRITON_CACHE_DIR PYTORCH_CUDA_ALLOC_CONF

USE_VLLM_FLAG=""
if [ "$M0_USE_VLLM" = "1" ]; then
    USE_VLLM_FLAG="--use_vllm"
fi

for path in "$M0_CONFIG" "$M0_DATASET" "$M0_EVAL_DATASET"; do
    if [ ! -f "$path" ]; then
        echo "missing: $path" >&2
        echo "build the prompt set first: python -m m0.data.build_dataset --out $M0_DATA_DIR" >&2
        exit 1
    fi
done

echo "[m0] config=$M0_CONFIG"
echo "[m0] model=$M0_BASE_MODEL"
echo "[m0] data=$M0_DATA_DIR"
echo "[m0] run=$M0_SAVE_DIR/$M0_RUN_NAME"

cd "$REPO_ROOT"

python -m m0.train_m0 \
    --config "$M0_CONFIG" \
    --base_model "$M0_BASE_MODEL" \
    --dataset "$M0_DATASET" \
    --eval_dataset "$M0_EVAL_DATASET" \
    --seed "$M0_SEED" \
    --save_dir "$M0_SAVE_DIR" \
    --run_name "$M0_RUN_NAME" \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules all-linear \
    $USE_VLLM_FLAG \
    "$@"
