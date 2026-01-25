#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/dqn_base.yaml}"
RUN_ID="${2:-dqn_base_01}"

docker compose run --rm rl \
  python scripts/train.py --config "$CONFIG" --run-id "$RUN_ID"

# Genera figuras de entrenamiento desde el log del run
docker compose run --rm -e PYTHONPATH=/workspace/src -e RUN_ID="$RUN_ID" rl python - <<'PY'
import os
from pathlib import Path
from acrobot_dqn.plots import plot_training_logs

run_id = os.environ.get("RUN_ID")
run_dir = Path("outputs") / "runs" / run_id
logs_dir = run_dir / "logs"
log_files = sorted(logs_dir.glob("*.json"))
if log_files:
    plot_training_logs(log_files[0], run_dir / "figures")
else:
    print(f"No se encontro log en {logs_dir}")
PY
