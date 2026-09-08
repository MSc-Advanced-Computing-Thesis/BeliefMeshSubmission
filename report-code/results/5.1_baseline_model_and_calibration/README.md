# Section 5.1 -- Baseline Model and Calibration

Regenerates, from the stored evaluation dumps in
artefacts/5.1_baseline_model_and_calibration/:

    Figure 5.1  fig_5_1_angle_response.png    (held-out population beside the
                                               mesh digit instance, error and
                                               epistemic uncertainty by angle)
    Figure 5.2  fig_5_2_calibration.png       (error distribution, interval
                                               coverage against nominal)
    Figure 5.3  fig_5_3a_colour_sweep.png, fig_5_3b_offset_sweep.png

and prints the quantities quoted in the section's prose (table_5_1_prose.txt).

--rerun re-evaluates the pretrained checkpoints (GPU, ~1 h) via the experiment
scripts in this directory, in the order r1a -> angle bins -> single instance
-> r1b -> r1c, writing fresh dumps into the artefact directory.

## Run

```
python reproduce.py            # regenerate from the stored artefacts
python reproduce.py --help     # options (output directories, artefact root, seeds)
```

`--rerun` re-runs the section's experiments before regenerating (GPU; hours). Never run by default.

## Files

- `analyse.py`
- `analysis_angle_bins.py`
- `dump.py`
- `experiment_r1a_held_out.py`
- `experiment_r1b_colour_sweep.py`
- `experiment_r1c_offset_sweep.py`
- `experiment_single_instance.py`
- `figure_5_1_angle_response.py`
- `figure_5_2_calibration.py`
- `figure_angle_response_stacked.py`
- `figure_sweeps.py`
- `protocol.py`
