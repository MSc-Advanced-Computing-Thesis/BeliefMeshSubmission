#!/usr/bin/env bash
# Section 5.6 all-narrow homogeneous arm. Own root, own log, wrapper pid
# recorded so stop_batch.sh can stop it safely.
set -u
LOGS=logs/homog_narrow_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOGS"; echo "$$" > "$LOGS/batch.pid"
echo "logs -> $LOGS (wrapper pid $$)"
.venv/Scripts/python.exe -u experiments/stage6_spatial_mesh/run_heterogeneous_comparator.py \
    --mode nig_product --homogeneous --homog-variant narrow \
    --seeds 42,1042,2042,3042,4042 --root runs/chapter5_v2/s5_6_homog_narrow \
    >> "$LOGS/homog_narrow.log" 2>&1
echo "exit=$? -- DONE"
