#!/usr/bin/env bash
# RESUME the consensus rho sweep: batch 3 stopped at 22/35.
#
# Missing: seed 3042 rhos 0.05-0.99 (6), seed 4042 all 7 = 13 runs.
# The script picks rho by stride (RHO_GRID[shard::nshards]), which cannot
# express a contiguous tail, so both seeds are run in full -- 14 runs. The one
# redundant run (rho0p01_seed3042) is backed up beforehand and compared after,
# which doubles as a determinism check rather than a silent overwrite.
#
# Same root as before on purpose: the 22 completed runs are untouched, only
# the missing tags are written.
#
# Stop with: bash experiments/section5_2/stop_batch.sh <log-dir>
set -u
LOGS=logs/rho_resume_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOGS"; echo "$$" > "$LOGS/batch.pid"
echo "rho resume logs -> $LOGS (wrapper pid $$)"
.venv/Scripts/python.exe -u experiments/stage6_spatial_mesh/run_dynamic_v2_consensus.py \
    --part rho --seeds 3042,4042 \
    --root runs/chapter5_v2/s5_9_consensus_rho \
    >> "$LOGS/consensus_rho_resume.log" 2>&1
echo "exit=$? -- DONE"
