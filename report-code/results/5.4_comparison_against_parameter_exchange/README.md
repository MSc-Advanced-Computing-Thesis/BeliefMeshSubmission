# Section 5.4 -- Comparison Against Parameter Exchange

Regenerates, from artefacts/5.4_comparison_against_parameter_exchange/
five_environments/ (5 environments x 4 mechanisms x 5 seeds = 100 runs):

    Table 5.3   table_5_3_parameter_exchange.csv (+ .txt)
                accuracy and per-node payload per timestep, pooled over the
                25 runs of each mechanism
    Figure 5.7  fig_5_7_exchange_trajectories.png
                cell-space MSE per timestep in field0 and field3
    (Appendix D's five-environment figure reuses this section's readout; see
    appendix_D_parameter_exchange_trajectories/reproduce.py)

Each output is written twice. The plain file scores every run against the
offset field it was generated with. The *_as_published file scores every run
against field0, which is how the report's Table 5.3 and Figure 5.7 were
produced; it is retained so the published numbers can be reproduced and the
difference inspected (see exchange_readout.py).

--rerun re-runs the 100 mesh experiments (GPU, ~16 h) through
experiment_parameter_exchange.py.

## Run

```
python reproduce.py            # regenerate from the stored artefacts
python reproduce.py --help     # options (output directories, artefact root, seeds)
```

`--rerun` re-runs the section's experiments before regenerating (GPU; hours). Never run by default.

## Files

- `exchange_readout.py`
- `experiment_gossip_comparator.py`
- `experiment_parameter_exchange.py`
