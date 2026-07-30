#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
source .venv/Scripts/activate

LAM="$1"
RHO="$2"
TAG="$3"

for seed in 42 1042 2042 3042 4042; do
  echo "=================================================="
  echo "SEED $seed :: $TAG (lam=$LAM rho=$RHO)"
  echo "=================================================="
  EXP_SEED=$seed python -u experiments/stage6_spatial_mesh/run_5seed_config.py --lam "$LAM" --rho "$RHO" --tag "$TAG"
done

echo ""
echo "=== $TAG 5-SEED DONE ==="
