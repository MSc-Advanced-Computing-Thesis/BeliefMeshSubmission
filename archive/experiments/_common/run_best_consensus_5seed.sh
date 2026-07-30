#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
source .venv/Scripts/activate

for seed in 42 1042 2042 3042 4042; do
  echo "=================================================="
  echo "SEED $seed :: best_consensus_config"
  echo "=================================================="
  EXP_SEED=$seed python -u experiments/stage6_spatial_mesh/run_best_consensus_config_5seed.py
done

echo ""
echo "=== BEST CONSENSUS CONFIG 5-SEED DONE ==="
