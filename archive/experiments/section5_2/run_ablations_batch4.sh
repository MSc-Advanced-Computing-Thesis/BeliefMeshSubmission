#!/usr/bin/env bash
# Batch 4 -- regenerate the two arm sets that lack cell_beliefs_steps, so the
# whole chapter can sit on ONE readout convention.
#
#  1. gossip_uniform / gossip_weighted / fedavg_global (5.5 parameter exchange).
#     Only the fusion arm had beliefs; a figure mixing conventions across bars
#     is not reportable.
#  2. agg_cmp_frozen (the 0.09697 reference). Frozen nodes hold the same
#     checkpoint but NOT the same input -- CoordConv local-FOV coordinate
#     channels differ per node -- so their covering beliefs need not be
#     identical and the reference may move under averaging.
#
# Fresh roots: the originals stay intact so the regenerated argmax numbers can
# be checked against them before anything is adopted.
set -u
PY=.venv/Scripts/python.exe
S=experiments/stage6_spatial_mesh
SEEDS=42,1042,2042,3042,4042
OUT=runs/chapter5_v2
LOGS=logs/ablations4_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOGS"; echo "$$" > "$LOGS/batch.pid"
echo "batch4 logs -> $LOGS (wrapper pid $$)"

for m in gossip_uniform gossip_weighted fedavg_global; do
  echo "=== [$(date +%H:%M:%S)] $m"
  "$PY" -u "$S/run_gossip_comparator.py" --mode "$m" --seeds "$SEEDS" \
      --root "$OUT/s5_5_gossip_$m" >> "$LOGS/gossip_$m.log" 2>&1
  echo "    exit=$?"
done

echo "=== [$(date +%H:%M:%S)] frozen reference"
"$PY" -u "$S/run_aggregation_comparator.py" --mode frozen --seeds "$SEEDS" \
    --root "$OUT/s5_3_1_frozen" >> "$LOGS/frozen.log" 2>&1
echo "    exit=$?"

echo "=== [$(date +%H:%M:%S)] BATCH 4 COMPLETE"
