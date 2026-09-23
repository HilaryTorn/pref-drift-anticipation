#!/bin/sh
# Smoke test for the Big5 trait-axis SFT arms (data/training_specs/big5_trait_sft.json).
# Stage 1 (always): prepare + validate an n=8 subset for every arm — exercises the full data path (spec resolution, schema/duplicate/elicitation-overlap validation, train/validation split) with no GPU or training deps.
# Stage 2 (RUN_TRAIN=1): one tiny LoRA training run on the first arm with the 0.8B pilot model — requires the pinned SFT environment (requirements-sft.txt) and, per docs, scripts/preflight_qwen35.py should pass before spending real GPU time.
# Usage: sh scripts/run_big5_sft_smoke.sh            # data path only
#        RUN_TRAIN=1 sh scripts/run_big5_sft_smoke.sh  # also train (add --no-bf16 via NO_BF16=1 on hardware without bfloat16)

set -eu

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-"$REPO_ROOT/.venv/bin/python"}
SPEC="$REPO_ROOT/data/training_specs/big5_trait_sft.json"
SMOKE_SIZE=${SMOKE_SIZE:-8}
SEED=${SEED:-42}
RUN_TRAIN=${RUN_TRAIN:-0}
NO_BF16=${NO_BF16:-0}
SMOKE_ROOT=$(mktemp -d /tmp/ai-pref-drift-big5-sft-smoke.XXXXXX)

cleanup_smoke() {
    case "$SMOKE_ROOT" in
        /tmp/ai-pref-drift-big5-sft-smoke.*) rm -rf -- "$SMOKE_ROOT" ;;
        *) echo "Refusing unsafe smoke cleanup path: $SMOKE_ROOT" >&2 ;;
    esac
}
trap cleanup_smoke EXIT HUP INT TERM

cd "$REPO_ROOT"

ARMS="big5.openness.high big5.openness.low big5.extraversion.high big5.extraversion.low"

echo "[big5-smoke] stage 1: prepare + validate n=$SMOKE_SIZE for every arm"
for ARM in $ARMS; do
    "$PYTHON_BIN" scripts/prepare_sft_dataset.py \
        --spec_path "$SPEC" \
        --intervention_id "$ARM" \
        --sample_count "$SMOKE_SIZE" \
        --seed "$SEED" \
        --output_dir "$SMOKE_ROOT/$ARM" >/dev/null
    echo "[big5-smoke]   $ARM OK"
done

if [ "$RUN_TRAIN" != "1" ]; then
    echo "[big5-smoke] data path validated for all arms; set RUN_TRAIN=1 to also run a tiny 0.8B LoRA training pass"
    exit 0
fi

TRAIN_ARM=big5.openness.high
TRAIN_ARGS=""
if [ "$NO_BF16" = "1" ]; then
    TRAIN_ARGS="--no-bf16"
fi
echo "[big5-smoke] stage 2: tiny LoRA train on $TRAIN_ARM (0.8B pilot default)"
# shellcheck disable=SC2086
"$PYTHON_BIN" scripts/train_sft_lora.py \
    --spec_path "$SPEC" \
    --intervention_id "$TRAIN_ARM" \
    --dataset_dir "$SMOKE_ROOT/$TRAIN_ARM" \
    --output_dir "$SMOKE_ROOT/run" \
    $TRAIN_ARGS
echo "[big5-smoke] training pass completed; transient outputs under $SMOKE_ROOT are deleted on exit"
