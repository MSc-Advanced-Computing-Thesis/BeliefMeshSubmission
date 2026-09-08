# Section 5.8.1 -- Node Failure

Regenerates, from artefacts/5.8_system_robustness/5.8.1_node_failure/node_loss/
(random and clustered node loss at 10-60 % of the mesh plus the 0 % control,
5 seeds each, 55 runs):

    Figure 5.11 fig_5_11_coverage_maps.png   surviving coverage at 40 % loss,
                                             seed 42, random beside clustered
    Figure 5.12 fig_5_12_node_loss.png       dead-zone fraction, surviving-region
                                             MSE difference, coverage and hw:RMS
                                             against the fraction of nodes failed

Figure 5.12's second to fourth panels read two derived CSVs (node_loss_paired_
averaged.csv, node_loss_calibration.csv) that analysis_node_loss_paired.py and
analysis_node_loss_calibration.py compute from the runs; --recompute-analysis
rebuilds them (a few minutes), otherwise the stored CSVs are used.

--rerun re-runs the 55 node-failure experiments (GPU, ~9 h) through
experiment_node_loss.py, then recomputes the analysis CSVs.

## Run

```
python reproduce.py            # regenerate from the stored artefacts
python reproduce.py --help     # options (output directories, artefact root, seeds)
```

`--rerun` re-runs the section's experiments before regenerating (GPU; hours). Never run by default.

## Files

- `analysis_node_loss_calibration.py`
- `analysis_node_loss_paired.py`
- `experiment_node_failure.py`
- `experiment_node_loss.py`
