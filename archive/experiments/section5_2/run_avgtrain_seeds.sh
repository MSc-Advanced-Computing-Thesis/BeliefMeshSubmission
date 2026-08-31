#!/usr/bin/env bash
# Averaged-training diagnostic, remaining 4 seeds. Seed 42 already done and
# already used wearable_seed_base=200, which matches the baseline convention,
# so it is NOT rerun.
set -u
LOGS=logs/avgtrain_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOGS"; echo "$$" > "$LOGS/batch.pid"
echo "logs -> $LOGS (wrapper pid $$)"
.venv/Scripts/python.exe -u experiments/section5_2/run_averaged_training_diagnostic.py \
    --seeds 1042,2042,3042,4042 >> "$LOGS/avgtrain.log" 2>&1
echo "exit=$? -- DONE"
