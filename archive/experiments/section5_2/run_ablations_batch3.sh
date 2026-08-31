#!/usr/bin/env bash
# RESUME batch: only the work that had not completed when batches 1 and 2 were
# killed, plus the all-wide homogeneous arm for Section 5.6.
#
# Completed and NOT re-run: routing_matched (10), uncertainty x3 (15),
# temper x3 (15), lam 1.0/2.5/5.0/10.0 (20), lam 25.0 seed 42 (1),
# nig_weighted seeds 42 and 1042 (2).
#
# Sequential, own log dir, own output roots.
#
# Run:  bash experiments/section5_2/run_ablations_batch3.sh
# Stop: use experiments/section5_2/stop_batch.sh -- do NOT match on a script
#       name with a broad process filter, which also matches the shell that
#       is running the filter.

set -u
PY=.venv/Scripts/python.exe
S=experiments/stage6_spatial_mesh
S2=experiments/section5_2
OUT=runs/chapter5_v2
LOGS=logs/ablations3_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOGS"
echo "$$" > "$LOGS/batch.pid"
echo "batch3 logs -> $LOGS  (wrapper pid $$)"

# ---- lambda 25.0: seed 42 already done, 4 remain ----
echo "=== [$(date +%H:%M:%S)] lam=25.0 (4 remaining seeds)"
"$PY" -u "$S2/run_mesh_sampled.py" --mode nig_product --lam 25.0 \
    --seeds 1042,2042,3042,4042 --root "$OUT/s5_9_lam_25p0" \
    >> "$LOGS/lam_25p0.log" 2>&1
echo "    exit=$?"

# ---- nig_product_weighted: seeds 42 and 1042 done, 3 remain ----
echo "=== [$(date +%H:%M:%S)] nig_weighted (3 remaining seeds)"
"$PY" -u "$S/run_nig_product_consensus_comparator.py" \
    --seeds 2042,3042,4042 --root "$OUT/s5_9_nig_weighted" \
    >> "$LOGS/nig_weighted.log" 2>&1
echo "    exit=$?"

# ---- Section 5.6 all-wide homogeneous arm: 5 runs ----
# node_variants=["wide"]*36 AND the checkpoint dict, so the models are actually
# width 64 and the wide checkpoint loads. Verified before launch.
echo "=== [$(date +%H:%M:%S)] 5.6 all-wide homogeneous (5 runs)"
"$PY" -u "$S/run_heterogeneous_comparator.py" --mode nig_product \
    --homogeneous --homog-variant wide --seeds 42,1042,2042,3042,4042 \
    --root "$OUT/s5_6_homog_wide" >> "$LOGS/homog_wide.log" 2>&1
echo "    exit=$?"

# ---- consensus rho sweep: 7 rho x 5 seeds = 35 runs, none done ----
echo "=== [$(date +%H:%M:%S)] consensus rho (35 runs)"
"$PY" -u "$S/run_dynamic_v2_consensus.py" --part rho \
    --seeds 42,1042,2042,3042,4042 --root "$OUT/s5_9_consensus_rho" \
    >> "$LOGS/consensus_rho.log" 2>&1
echo "    exit=$?"

echo "=== [$(date +%H:%M:%S)] BATCH 3 COMPLETE"
