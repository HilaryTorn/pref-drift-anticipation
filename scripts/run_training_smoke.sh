#!/bin/sh

set -eu

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
BASE_MODEL=${BASE_MODEL:-"Qwen/Qwen3.5-0.8B"}
SMOKE_SIZE=${SMOKE_SIZE:-20}
SEED=${SEED:-0}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
SMOKE_ROOT=$(mktemp -d /tmp/ai-pref-drift-training-smoke.XXXXXX)
case "$SMOKE_ROOT" in
    /tmp/ai-pref-drift-training-smoke.*) ;;
    *)
        echo "SMOKE_ROOT must match /tmp/ai-pref-drift-training-smoke.*" >&2
        exit 1
        ;;
esac
SMOKE_DATA_DIR="$SMOKE_ROOT/data"
RUN_ROOT="$SMOKE_ROOT/runs"
FIGURES_ROOT="$SMOKE_ROOT/results"
# The default milestone is GRPO + DPO. Set RUN_PPO_STAGE=1 to additionally
# train the reward model and exercise experimental TRL PPO.
RUN_PPO_STAGE=${RUN_PPO_STAGE:-0}

PILOT_PROMPTS=${PILOT_PROMPTS:-"$REPO_ROOT/data/rl/v1/nemotron_rl_coding_competitive_train_1000_learnable.jsonl"}
PILOT_PAIRS=${PILOT_PAIRS:-"$REPO_ROOT/data/rl/v1/dpo_pairs.jsonl"}
PILOT_EVAL=${PILOT_EVAL:-"$REPO_ROOT/data/rl/v1/nemotron_rl_coding_competitive_dev.jsonl"}

export CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-"$SMOKE_ROOT/triton-cache"}

cleanup_smoke() {
    case "$SMOKE_ROOT" in
        /tmp/ai-pref-drift-training-smoke.*) rm -rf -- "$SMOKE_ROOT" ;;
        *) echo "Refusing unsafe smoke cleanup path: $SMOKE_ROOT" >&2 ;;
    esac
}
trap cleanup_smoke EXIT HUP INT TERM

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Training Python not found: $PYTHON_BIN" >&2
    echo "Create the pinned environment before running this script." >&2
    exit 1
fi

cd "$REPO_ROOT"
mkdir -p "$RUN_ROOT" "$FIGURES_ROOT"

"$PYTHON_BIN" -m rl_training.data.build_training_smoke \
    --prompts "$PILOT_PROMPTS" \
    --pairs "$PILOT_PAIRS" \
    --eval-prompts "$PILOT_EVAL" \
    --size "$SMOKE_SIZE" \
    --eval-size 1 \
    --output-dir "$SMOKE_DATA_DIR"

COMMON_ARGS="--base_model $BASE_MODEL --seed $SEED --save_dir $RUN_ROOT --figures_dir $FIGURES_ROOT --use_lora --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --lora_target_modules all-linear"

echo "[training-smoke] GRPO"
# shellcheck disable=SC2086
"$PYTHON_BIN" -m rl_training.train_rl \
    --algo grpo \
    --config rl_training/configs/grpo_smoke.yaml \
    --dataset "$SMOKE_DATA_DIR/train.jsonl" \
    --eval_dataset "$SMOKE_DATA_DIR/eval.jsonl" \
    --run_name qwen08b-grpo-smoke-n${SMOKE_SIZE}-s${SEED} \
    $COMMON_ARGS

echo "[training-smoke] DPO"
# shellcheck disable=SC2086
"$PYTHON_BIN" -m rl_training.train_rl \
    --algo dpo \
    --config rl_training/configs/dpo_smoke.yaml \
    --dataset "$SMOKE_DATA_DIR/dpo.jsonl" \
    --eval_dataset "$SMOKE_DATA_DIR/eval.jsonl" \
    --run_name qwen08b-dpo-smoke-n${SMOKE_SIZE}-s${SEED} \
    $COMMON_ARGS

if [ "$RUN_PPO_STAGE" != "1" ]; then
    echo "[training-smoke] GRPO and DPO completed and validated"
    echo "[training-smoke] set RUN_PPO_STAGE=1 when ready to continue with RM/PPO"
    echo "[training-smoke] deleting transient outputs under $SMOKE_ROOT"
    exit 0
fi

RM_DIR="$RUN_ROOT/qwen08b-rm-smoke-n${SMOKE_SIZE}-s${SEED}"
echo "[training-smoke] reward model for PPO"
"$PYTHON_BIN" -m rl_training.train_reward_model \
    --config rl_training/configs/reward_model_smoke.yaml \
    --base_model "$BASE_MODEL" \
    --pairs "$SMOKE_DATA_DIR/dpo.jsonl" \
    --output_dir "$RM_DIR" \
    --seed "$SEED"

echo "[training-smoke] PPO"
# shellcheck disable=SC2086
"$PYTHON_BIN" -m rl_training.train_rl \
    --algo ppo \
    --config rl_training/configs/ppo_smoke.yaml \
    --reward_model "$RM_DIR" \
    --allow_weak_reward_model \
    --dataset "$SMOKE_DATA_DIR/train.jsonl" \
    --eval_dataset "$SMOKE_DATA_DIR/eval.jsonl" \
    --run_name qwen08b-ppo-smoke-n${SMOKE_SIZE}-s${SEED} \
    $COMMON_ARGS

echo "[training-smoke] all training paths completed and validated"
echo "[training-smoke] deleting transient outputs under $SMOKE_ROOT"
