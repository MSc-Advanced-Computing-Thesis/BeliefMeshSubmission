# Section 5.3.2 -- Aggregation at Mesh Scale

Regenerates, from artefacts/5.3_belief_aggregation/5.3.2_mesh_scale/:

    Figure 5.6  fig_5_6_mesh_snapshots.png     three snapshots of one Product
                                               fusion run (snapshot_run/)
    Table 5.2   table_5_2_aggregation_mesh_scale.csv (+ .txt)
                Frozen, Naive, Certainty, Product fusion (aggregation_arms/,
                frozen/) and Average fusion (average_fusion/), 5 seeds each,
                averaged-NIG readout over the last-50 window.

--rerun re-runs the 25 mesh experiments behind the table and the snapshot run
(GPU, ~4 h): experiment_mesh_aggregation.py for the three sampled-target arms,
experiment_frozen_reference.py for Frozen, experiment_average_fusion.py for
Average fusion, experiment_snapshot_run.py for the figure's run.

## Run

```
python reproduce.py            # regenerate from the stored artefacts
python reproduce.py --help     # options (output directories, artefact root, seeds)
```

`--rerun` re-runs the section's experiments before regenerating (GPU; hours). Never run by default.

## Files

- `experiment_average_fusion.py`
- `experiment_frozen_reference.py`
- `experiment_mesh_aggregation.py`
- `experiment_snapshot_run.py`
- `figure_mesh_snapshots.py`
