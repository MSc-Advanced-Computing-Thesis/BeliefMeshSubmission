#!/usr/bin/env bash
set -u
PY=./.venv/Scripts/python.exe
echo "=== A) manifest-field inertness: het:naive seed 42 vs the regenerated arm ==="
$PY -u -c "
import sys; sys.path.insert(0,'experiments/section5_2')
import run_avgfusion_het_multifield as A
A.run_het_mode('naive', 42, 'runs/scratch/manifest_inert', patched=False)
print('MANIFEST INERT RUN DONE')"
echo "=== B) filter-off homogeneous Average Fusion, 5 seeds ==="
$PY -u experiments/section5_2/run_avgfusion_het_multifield.py --which homog_avg
echo "=== STEP 1 RUNS COMPLETE ==="
