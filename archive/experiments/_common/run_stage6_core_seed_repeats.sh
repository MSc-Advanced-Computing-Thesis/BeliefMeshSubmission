#!/usr/bin/env bash
# Repeats the Stage 6 CORE experiments across 4 additional seeds, on top of
# the existing canonical seed-42 runs (untouched -- these write to
# runs/stage6/.../*_seed{N}/ siblings). 5 seeds total per experiment: 42
# (already exists) + these 4. Same seed set as the Stage 1-5 batch for
# consistency across the replication register.
#
# Core = the results that directly anchor the thesis's central claims:
#   - offset_world/static/main_arms   (Stage 1 offset table, 5 cells)
#   - colour_world/dynamic/v2_fast_drift  (6D verdict table, v2, 6 arms)
#   - colour_world/dynamic/v3_whiteout    (6D verdict table, v3, 6 arms)
#   - offset_world/dynamic            (Stage 4 headline result, 2 arms --
#     SUPERSEDED 2026-08: this pre-CoordConv-fix result was reversed by
#     run_dynamic_v2_test.py/run_dynamic_v2_5seed.py on the same field/task
#     with the Stage 6 fixes applied; run_stage4_dynamic_world_final.py
#     moved to archive/ but kept runnable here as a historical record)
#
# Each seed takes a substantial amount of GPU time (390-step mesh training
# per arm, ~19 mesh trainings per seed) -- expect this to run for hours.
set -uo pipefail
cd "$(dirname "$0")/../.."
source .venv/Scripts/activate

SEEDS="1042 2042 3042 4042"
FAILED=()

run() {
  echo "=================================================="
  echo "SEED $1 :: ${@:2}"
  echo "=================================================="
  EXP_SEED=$1 python -u "${@:2}"
  if [ $? -ne 0 ]; then
    echo "!!! FAILED: seed=$1 cmd=${@:2}"
    FAILED+=("seed=$1 cmd=${@:2}")
  fi
}

for seed in $SEEDS; do
  run "$seed" experiments/stage6_spatial_mesh/run_offset_experiments.py
  run "$seed" experiments/stage6_spatial_mesh/run_6d_three_way_comparison.py --env v2
  run "$seed" experiments/stage6_spatial_mesh/run_6d_three_way_comparison.py --env v3
  run "$seed" experiments/stage6_spatial_mesh/archive/run_stage4_dynamic_world_final.py
done

echo ""
echo "=== STAGE 6 CORE SEED REPEATS DONE ==="
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "All runs succeeded."
else
  echo "FAILURES:"
  printf '  %s\n' "${FAILED[@]}"
fi
