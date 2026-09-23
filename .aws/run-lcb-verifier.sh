#!/usr/bin/env bash
set -euo pipefail

if env | grep -q '^AWS_.*\(CREDENTIAL\|ACCESS_KEY\|SECRET\|SESSION_TOKEN\)'; then
  echo "AWS credentials unexpectedly entered the verifier sandbox" >&2
  exit 1
fi

case "${LCB_EVAL_WORKERS:-}" in
  ''|*[!0-9]*|0)
    echo "LCB_EVAL_WORKERS must be a positive integer" >&2
    exit 1
    ;;
esac

cd multi-lcb
case "${LCB_LANGUAGE:-}" in
  php)
    python -m pytest -q tests/tests_plangs/test_php.py
    ;;
  csharp)
    python -m pytest -q tests/tests_plangs/test_csharp.py
    ;;
  *)
    echo "LCB_LANGUAGE must be php or csharp" >&2
    exit 1
    ;;
esac
cd ..

python scripts/score_multilcb.py \
  --model_key "lcb-${LCB_LANGUAGE}-teacher" \
  --languages "$LCB_LANGUAGE" \
  --release_version release_v5 \
  --n 1 \
  --evaluate_only \
  --diagnose \
  --num_process_evaluate "$LCB_EVAL_WORKERS"
