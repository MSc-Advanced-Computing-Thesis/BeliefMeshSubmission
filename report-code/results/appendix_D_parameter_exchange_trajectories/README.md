# Appendix D -- Parameter Exchange Trajectories by Environment

Regenerates Figure D.1 (fig_D_1_exchange_trajectories_all.png): cell-space MSE
per timestep for the four exchange mechanisms in all five offset environments,
from the Section 5.4 artefacts. The readout is Section 5.4's
(5.4_comparison_against_parameter_exchange/exchange_readout.py); as there,
the figure is written twice, once scoring each run against its own field and
once (_as_published) scoring every run against field0, as the report did.

--rerun re-runs the 100 Section 5.4 experiments (GPU, ~16 h).

## Run

```
python reproduce.py            # regenerate from the stored artefacts
python reproduce.py --help     # options (output directories, artefact root, seeds)
```

## Files

