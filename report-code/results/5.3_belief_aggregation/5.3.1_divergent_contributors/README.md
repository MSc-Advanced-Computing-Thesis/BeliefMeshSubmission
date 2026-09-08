# Section 5.3.1 -- Aggregation Under Divergent Contributor Conditions

Regenerates, from artefacts/5.3_belief_aggregation/5.3.1_divergent_contributors/
three_node_colour_world/ (five arms x five seeds of the three-node colour
world):

    Figure 5.5  fig_5_5_three_node_geometry.png   (pure geometry: no artefacts)
    Table 5.1   table_5_1_three_node_aggregation.csv (+ .txt with the full
                                                      printed analysis)

--rerun re-runs the 25 three-node experiments (GPU, ~2 h) via
experiment_three_node_colour_world.py before regenerating.

## Run

```
python reproduce.py            # regenerate from the stored artefacts
python reproduce.py --help     # options (output directories, artefact root, seeds)
```

`--rerun` re-runs the section's experiments before regenerating (GPU; hours). Never run by default.

## Files

- `experiment_three_node_colour_world.py`
- `figure_three_node_geometry.py`
- `table_three_node_results.py`
