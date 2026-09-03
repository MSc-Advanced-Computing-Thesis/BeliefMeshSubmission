#!/usr/bin/env bash
# Section 5.3 three-node colour-world experiment: 5 arms x 5 seeds = 25 runs.
# Own root, own log, wrapper pid recorded for stop_batch.sh.
set -u
LOGS=logs/colour3_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOGS"; echo "$$" > "$LOGS/batch.pid"
echo "logs -> $LOGS (wrapper pid $$)"
.venv/Scripts/python.exe -u experiments/section5_2/run_colour_three_node.py \
    --seeds 42,1042,2042,3042,4042 \
    --root runs/chapter5_v2/s5_3_colour_three_node >> "$LOGS/colour3.log" 2>&1
echo "exit=$? -- DONE"
