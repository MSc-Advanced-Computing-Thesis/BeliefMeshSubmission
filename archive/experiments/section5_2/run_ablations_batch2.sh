#!/usr/bin/env bash
# Section 5.9 ablation batch 2: matched routing, lambda sweep, consensus rho.
#
# Runs concurrently with batch 1 -- the roots are disjoint and the GPU has
# headroom. Own log directory, own output roots, nothing shared with batch 1.
#
# Routing runs FIRST: it is the only place the chapter shows that directing
# wearables by uncertainty improves the regions they reach.
#
# Run:  bash experiments/section5_2/run_ablations_batch2.sh
# Stop: kill the bash wrapper FIRST, then the python child, then verify by
#       process listing -- a killed wrapper alone lets the loop respawn.

set -u
PY=.venv/Scripts/python.exe
S=experiments/stage6_spatial_mesh
S2=experiments/section5_2
SEEDS=42,1042,2042,3042,4042
OUT=runs/chapter5_v2
LOGS=logs/ablations2_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOGS"
echo "batch2 logs -> $LOGS"

# ---- MATCHED ROUTING: 2 policies x 5 seeds = 10 runs ----
echo "=== [$(date +%H:%M:%S)] routing_matched -> $OUT/s5_9_routing_matched"
"$PY" -u "$S2/run_routing_matched.py" --seeds "$SEEDS" \
    --root "$OUT/s5_9_routing_matched" >> "$LOGS/routing_matched.log" 2>&1
echo "    exit=$?"

# ---- LAMBDA SWEEP: lr and rho fixed, lam alone. 5 values x 5 seeds = 25 ----
# Run through run_mesh_sampled.py (the 5.3.1 generator) so the sweep is
# matched to 5.3.1 by construction: lr=3e-5, rho=0.2 from config, sampled
# targets, 36 nodes stride 3, dynamic offset world.
for lam in 1.0 2.5 5.0 10.0 25.0; do
  tag=$(echo "$lam" | tr '.' 'p')
  echo "=== [$(date +%H:%M:%S)] lam=$lam -> $OUT/s5_9_lam_$tag"
  "$PY" -u "$S2/run_mesh_sampled.py" --mode nig_product --lam "$lam" \
      --seeds "$SEEDS" --root "$OUT/s5_9_lam_$tag" \
      >> "$LOGS/lam_$tag.log" 2>&1
  echo "    exit=$?"
done

# ---- CONSENSUS WEIGHTING: rho sweep, 7 values x 5 seeds = 35 runs ----
echo "=== [$(date +%H:%M:%S)] consensus rho -> $OUT/s5_9_consensus_rho"
"$PY" -u "$S/run_dynamic_v2_consensus.py" --part rho --seeds "$SEEDS" \
    --root "$OUT/s5_9_consensus_rho" >> "$LOGS/consensus_rho.log" 2>&1
echo "    exit=$?"

echo "=== [$(date +%H:%M:%S)] BATCH 2 COMPLETE"
