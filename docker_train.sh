#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/dqn_base.yaml}"
RUN_ID="${2:-dqn_base_01}"

docker compose run --rm rl \
  python scripts/train.py --config "$CONFIG" --run-id "$RUN_ID"
