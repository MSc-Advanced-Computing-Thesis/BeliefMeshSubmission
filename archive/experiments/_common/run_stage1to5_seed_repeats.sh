#!/usr/bin/env bash
# Repeats Stages 1-5 (core claims) across 4 additional seeds, on top of the
# existing canonical seed-42 run (left untouched -- these write to
# runs/stageN/.../*_seed{N}/ siblings, never overwriting the original).
# 5 seeds total per experiment: 42 (already exists) + these 4.
set -uo pipefail
cd "$(dirname "$0")/../.."
source .venv/Scripts/activate

SEEDS="1042 2042 3042 4042"
SCRIPTS=(
  "experiments/stage1_shift/run.py"
  "experiments/stage2_peer_supervision/run.py"
  "experiments/stage3_naive_aggregation/run.py"
  "experiments/stage4_weighting_and_fusion/run_4a_scalar_weighting.py"
  "experiments/stage4_weighting_and_fusion/run_4b_fusion_vs_scalar.py"
  "experiments/stage4_weighting_and_fusion/run_4c_pairwise_control.py"
  "experiments/stage5_variable_environments/run.py"
)

FAILED=()
for seed in $SEEDS; do
  for script in "${SCRIPTS[@]}"; do
    echo "=================================================="
    echo "SEED $seed :: $script"
    echo "=================================================="
    EXP_SEED=$seed python -u "$script"
    if [ $? -ne 0 ]; then
      echo "!!! FAILED: seed=$seed script=$script"
      FAILED+=("seed=$seed script=$script")
    fi
  done
done

echo ""
echo "=== STAGE 1-5 SEED REPEATS DONE ==="
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "All runs succeeded."
else
  echo "FAILURES:"
  printf '  %s\n' "${FAILED[@]}"
fi
