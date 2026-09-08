# Section 5.5 -- Mesh Geometry

Regenerates, from the stored artefacts:

    Figure 5.8  fig_5_8_mesh_density.png
                accuracy, coverage and hw:RMS against mean per-cell coverage
                for the 4x4, 5x5, 6x6 and 10x10 meshes
                (artefacts/5.5_mesh_geometry/mesh_density_sweep/, 5 seeds each)
    Table 5.4   table_5_4_hop_calibration.csv (+ .txt)
                hw:RMS, half-width and RMS error by hop distance from the
                wearables, on the reference 6x6 mesh (the Section 5.3.2 Product
                fusion runs, artefacts/5.3_belief_aggregation/5.3.2_mesh_scale/
                aggregation_arms/nig_product_sampled_seed*)

--rerun re-runs the 20 mesh-density experiments (GPU, ~4 h) through
experiment_mesh_density.py --profile chapter5. The hop table's runs belong to
Section 5.3.2 and are re-run from there.

## Run

```
python reproduce.py            # regenerate from the stored artefacts
python reproduce.py --help     # options (output directories, artefact root, seeds)
```

`--rerun` re-runs the section's experiments before regenerating (GPU; hours). Never run by default.

## Files

- `experiment_mesh_density.py`
