#!/usr/bin/env bash
# Section 5.9 ablation batch. Sequential by design: one GPU, and concurrent
# batches are what corrupted nine run directories previously.
#
# Every family gets its OWN output root and its OWN log, per the protocol.
# Nothing writes into the legacy runs/stage6/offset_world/* roots.
#
# Run:  bash experiments/section5_2/run_ablations_batch.sh
# Stop: kill the bash wrapper FIRST, then the python child, then verify by
#       process listing -- a killed wrapper alone lets the loop respawn.

set -u
PY=.venv/Scripts/python.exe
S=experiments/stage6_spatial_mesh
SEEDS=42,1042,2042,3042,4042
OUT=runs/chapter5_v2
LOGS=logs/ablations_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOGS"
echo "batch logs -> $LOGS"

run () {                       # run <family> <root> <script> [args...]
  local fam=$1 root=$2 script=$3; shift 3
  echo "=== [$(date +%H:%M:%S)] $fam -> $root"
  "$PY" -u "$S/$script.py" --seeds "$SEEDS" --root "$root" "$@" \
      >> "$LOGS/$fam.log" 2>&1
  echo "    exit=$? log=$LOGS/$fam.log"
}

for m in epistemic aleatoric total; do
  run "uncert_$m" "$OUT/s5_9_uncertainty_$m" run_uncertainty_measure_ablation --measure "$m"
done

for a in het_temper het_notemper homog_temper; do
  run "temper_$a" "$OUT/s5_9_temper_$a" run_temper_ablation --arm "$a"
done

run "nig_weighted" "$OUT/s5_9_nig_weighted" run_nig_product_consensus_comparator

for m in epistemic aleatoric total; do
  run "colour_uncert_$m" "$OUT/s5_9_colour_uncertainty_$m" \
      run_colour_world_uncertainty_measure_ablation --measure "$m"
done

echo "=== [$(date +%H:%M:%S)] BATCH COMPLETE"
