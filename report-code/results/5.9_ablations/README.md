# Section 5.9 -- Ablations (and Appendix F's regularisation weight sweep)

Regenerates, from artefacts/5.9_ablations/:

    Table 5.8   table_5_8_ablation_summary.csv (+ .txt)
                five ablation families, each recorded as null where the
                cross-variant spread of whole-run MSE is below twice the
                within-variant seed sd:
                  lambda_{1p0,2p5,5p0,10p0,25p0}/   regularisation weight
                  consensus_rho/                    consensus weighting rho
                  weighted_fusion/ (+ 5.3.2 arm)    weighted fusion off / on
                  uncertainty_{epistemic,aleatoric,total}/
                  tempering_{off,on}/               gradient tempering
    Table F.1   table_F_1_lambda_sweep.csv (+ .txt), the lambda family in full
                (also written by appendix_F_regularisation_weight_sweep/reproduce.py)

--rerun re-runs the 105 ablation experiments (GPU, ~20 h) through the
experiment_*.py scripts in this directory.

## Run

```
python reproduce.py            # regenerate from the stored artefacts
python reproduce.py --help     # options (output directories, artefact root, seeds)
```

`--rerun` re-runs the section's experiments before regenerating (GPU; hours). Never run by default.

## Files

- `experiment_consensus_rho.py`
- `experiment_heterogeneous_mesh.py`
- `experiment_mesh_aggregation.py`
- `experiment_multifield_heterogeneous.py`
- `experiment_tempering.py`
- `experiment_uncertainty_measure.py`
- `experiment_weighted_fusion.py`
