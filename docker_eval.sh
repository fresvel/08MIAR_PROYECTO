#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/dqn_base.yaml}"
WEIGHTS="${2:-outputs/weights/dqn_base_01_weights.h5f}"
EPISODES="${3:-100}"
RUN_ID="${4:-}"

SAVE_ARGS=()
if [ -n "$RUN_ID" ]; then
  SAVE_ARGS=(--save "outputs/runs/${RUN_ID}/metrics/eval.json")
fi

docker compose run --rm rl \
  python scripts/eval.py --config "$CONFIG" --weights "$WEIGHTS" --episodes "$EPISODES" "${SAVE_ARGS[@]}"
