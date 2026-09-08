# Section 5.7 -- Heterogeneous Devices

Regenerates, from artefacts/5.7_heterogeneous_devices/:

    Table 5.5   table_5_5_mixed_vs_uniform.csv (+ .txt)
                the mixed-capacity mesh (mixed_capacity/nig_product/) against
                uniform narrow / baseline / wide meshes, Product fusion, 5 seeds.
                The uniform baseline mesh is the Section 5.3.2 Product fusion
                arm (artefacts/5.3_belief_aggregation/5.3.2_mesh_scale/
                aggregation_arms/nig_product_sampled_seed*).
    Figure 5.10 fig_5_10_heterogeneity.png
                Naive, Certainty, Product fusion and Average fusion on the
                homogeneous mesh (Section 5.3.2 arms) and on the mixed mesh
                (mixed_capacity/{naive,certainty,nig_product,avgfusion}/).

--rerun re-runs the mixed-capacity arms and the uniform narrow / wide meshes
(GPU, ~6 h) through experiment_mixed_capacity.py; the homogeneous baseline
arms belong to Section 5.3.2.

## Run

```
python reproduce.py            # regenerate from the stored artefacts
python reproduce.py --help     # options (output directories, artefact root, seeds)
```

`--rerun` re-runs the section's experiments before regenerating (GPU; hours). Never run by default.

## Files

- `experiment_heterogeneous_mesh.py`
- `experiment_mixed_capacity.py`
