# Appendix F -- Regularisation Weight Sweep

Regenerates Table F.1 (table_F_1_lambda_sweep.csv + .txt) from
artefacts/5.9_ablations/lambda_{1p0,2p5,5p0,10p0,25p0}/ (Product fusion on the
reference mesh at five values of the regularisation weight, 5 seeds each). The
computation is Section 5.9's; this script exists so the appendix has its own
entry point.

--rerun re-runs the 25 lambda experiments (GPU, ~5 h) through the Section 5.9
experiment_mesh_aggregation.py script.

## Run

```
python reproduce.py            # regenerate from the stored artefacts
python reproduce.py --help     # options (output directories, artefact root, seeds)
```

## Files

