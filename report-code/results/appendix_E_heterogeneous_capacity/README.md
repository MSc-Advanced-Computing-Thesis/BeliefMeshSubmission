# Appendix E -- Heterogeneous Capacity Across Environments

Regenerates Table E.1 (table_E_1_heterogeneous_capacity.csv + .txt) from
artefacts/appendix_E_heterogeneous_capacity/mixed_capacity_five_environments/
{naive,certainty,nig_product,avgfusion}/het/: the four aggregation rules on the
mixed-capacity mesh, one run per offset environment with each environment
paired with its own seed. Each run is scored against its own environment.

--rerun re-runs the 20 experiments (GPU, ~4 h) through
experiment_mixed_capacity_five_environments.py.

## Run

```
python reproduce.py            # regenerate from the stored artefacts
python reproduce.py --help     # options (output directories, artefact root, seeds)
```

`--rerun` re-runs the section's experiments before regenerating (GPU; hours). Never run by default.

## Files

- `experiment_heterogeneous_mesh.py`
- `experiment_mixed_capacity_five_environments.py`
- `experiment_multifield_heterogeneous.py`
