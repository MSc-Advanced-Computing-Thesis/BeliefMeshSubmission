# Appendix C -- Operation at Deployment Granularity

Regenerates, from artefacts/appendix_C_deployment_granularity/deployment_66x66/
deployment_scale_seed42/ (36 nodes at stride 9, 21x21 fields of view, 66x66
cells, 27 wearables, 390 steps, one seed):

    Table C.1   table_C_1_deployment_configuration.csv (+ .txt)
                geometry and cost of the deployment-granularity run against the
                reference 22x22 configuration (wall time from the stored
                manifests; array sizes from the stored arrays)
    Table C.2   table_C_2_deployment_performance.csv (+ .txt)
                accuracy and calibration of the run against the reference
                configuration's five-seed mean and range (the Section 5.3.2
                Product fusion arm)

--rerun re-runs the deployment-granularity experiment (GPU, ~70 min) through
experiment_deployment_66x66.py.

## Run

```
python reproduce.py            # regenerate from the stored artefacts
python reproduce.py --help     # options (output directories, artefact root, seeds)
```

`--rerun` re-runs the section's experiments before regenerating (GPU; hours). Never run by default.

## Files

- `experiment_deployment_66x66.py`
