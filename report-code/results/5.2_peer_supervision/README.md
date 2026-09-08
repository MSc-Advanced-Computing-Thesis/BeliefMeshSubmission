# Section 5.2 -- Peer Supervision as a Training Signal

Regenerates, from artefacts/5.2_peer_supervision/two_node_offset_world/
(frozen, direct, student with mode target, student with sampled target;
5 seeds each on the two-node offset world):

    Figure 5.4  fig_5_4_peer_supervision.png
    and the numbers quoted in the prose: table_5_4_peer_supervision.csv (+ .txt)

--rerun re-runs the 20 two-node experiments (GPU, ~1 h) through
experiment_peer_supervision.py --world offset.

## Run

```
python reproduce.py            # regenerate from the stored artefacts
python reproduce.py --help     # options (output directories, artefact root, seeds)
```

`--rerun` re-runs the section's experiments before regenerating (GPU; hours). Never run by default.

## Files

- `experiment_peer_supervision.py`
