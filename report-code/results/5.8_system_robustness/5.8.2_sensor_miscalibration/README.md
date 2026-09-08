# Section 5.8.2 -- Sensor Miscalibration

Regenerates, from artefacts/5.8_system_robustness/5.8.2_sensor_miscalibration/
miscalibrated_wearable/ (one of three wearables biased by +20 degrees, 5 seeds)
against the matched baseline (the Section 5.3.2 Product fusion arm):

    Table 5.6   table_5_6_miscalibration.csv (+ .txt)
    Table 5.7   table_5_7_miscalibration_by_distance.csv (+ .txt)
                MSE difference, hw:RMS and half-width change by distance from
                the miscalibrated wearable's path

--rerun re-runs the five miscalibrated experiments (GPU, ~1 h) through
experiment_miscalibrated_wearable.py.

## Run

```
python reproduce.py            # regenerate from the stored artefacts
python reproduce.py --help     # options (output directories, artefact root, seeds)
```

`--rerun` re-runs the section's experiments before regenerating (GPU; hours). Never run by default.

## Files

- `experiment_miscalibrated_wearable.py`
- `table_miscalibration.py`
- `table_miscalibration_decomposition.py`
