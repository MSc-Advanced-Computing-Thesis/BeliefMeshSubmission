# Section 5.6 -- Uncertainty-Guided Measurement

Regenerates, from artefacts/5.6_uncertainty_guided_measurement/
{routing_product_fusion, routing_average_fusion}/ (random and
uncertainty-guided wearable routing under each fusion rule, 5 seeds each):

    Figure 5.9  fig_5_9_routing.png
    and the numbers quoted in the prose: table_5_9_routing.csv (+ .txt)

--rerun re-runs the 20 experiments (GPU, ~4 h) through
experiment_routing_product_fusion.py and experiment_routing_average_fusion.py.

## Run

```
python reproduce.py            # regenerate from the stored artefacts
python reproduce.py --help     # options (output directories, artefact root, seeds)
```

`--rerun` re-runs the section's experiments before regenerating (GPU; hours). Never run by default.

## Files

- `experiment_routing_average_fusion.py`
- `experiment_routing_product_fusion.py`
