# Chapters 3 and 4 -- design figures (no stored artefacts; pure computation)

Figure 3.2  fig_3_2_fusion_rules.png      Product and Average fusion on three
                                              contributor configurations
    Figure 4.2  fig_4_2_environments.png      colour world and offset world
                                              rendered from the environment code
    Figure 4.3  fig_4_3_mesh_geometry.png     node placement and cell coverage of
                                              the reference 36-node mesh
                (fig_4_3_mesh_geometry_annotated.png carries per-cell counts)

Figure 4.1 (example rotations of the digit seven) is not produced by any
script in the original tree and is not regenerated here.

There is no --rerun: nothing here depends on an experiment.

## Run

```
python reproduce.py            # regenerate from the stored artefacts
python reproduce.py --help     # options (output directories, artefact root, seeds)
```

## Files

- `figure_3_2_fusion_rules.py`
- `figure_4_2_environments.py`
- `figure_4_3_mesh_geometry.py`
