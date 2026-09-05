#!/usr/bin/env bash
# Queued work, gated on the gossip crossed batch finishing CLEANLY.
# Refuses to start on a partial prior batch, per the standing rule.
set -u
PY=./.venv/Scripts/python.exe
LOG=logs/gossip_multifield_crossed.log
while tasklist //FI "PID eq 33752" //NH 2>/dev/null | grep -qi python; do sleep 300; done
if ! grep -q "GOSSIP MULTIFIELD COMPLETE" "$LOG" 2>/dev/null; then
  echo "QUEUE ABORTED: gossip batch did not print its completion sentinel"; exit 1
fi
if ! grep -q "COMPLETE: 80 ok, 0 failed" "$LOG" 2>/dev/null; then
  echo "QUEUE ABORTED: gossip batch was partial --"; grep "COMPLETE:" "$LOG"; exit 1
fi
echo "=== gossip complete and clean; starting queued work ==="
# 1) manifest-field inertness check: one het:naive seed 42 vs the regenerated arm
$PY -u -c "
import sys; sys.path.insert(0,'experiments/section5_2')
import run_avgfusion_het_multifield as A
A.run_het_mode('naive', 42, 'runs/scratch/manifest_inert', patched=False)
print('MANIFEST INERT RUN DONE')"
# 2) filter-off homogeneous Average Fusion, 5 seeds
$PY -u experiments/section5_2/run_avgfusion_het_multifield.py --which homog_avg
echo "=== QUEUE COMPLETE ==="
