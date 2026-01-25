#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/dqn_base.yaml}"
RUN_ID="${2:-dqn_base_01}"
EPISODES="${3:-100}"

./docker_train.sh "$CONFIG" "$RUN_ID"

WEIGHTS="outputs/weights/${RUN_ID}_weights.h5f"
./docker_eval.sh "$CONFIG" "$WEIGHTS" "$EPISODES" "$RUN_ID"
