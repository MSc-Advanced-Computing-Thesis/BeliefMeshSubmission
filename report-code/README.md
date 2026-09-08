# BeliefMesh

Code, stored experiment outputs and figure/table generators for the MSc report
*Belief Exchange for Deployable Decentralised Continual Learning under Model
Heterogeneity and Environmental Dynamism* (Christian Wood, Imperial College
London, 2026).

The repository has two parts:

| Part | Where | What |
|---|---|---|
| The system | `beliefmesh/` | The BeliefMesh implementation as a library: evidential (NIG) models, closed-form belief fusion, the node and mesh, the proxy-task environments, and the deployment simulation loop. See [beliefmesh/README.md](beliefmesh/README.md). |
| The results | `results/` and `artefacts/` | One directory per report section, each with a `reproduce.py` that regenerates that section's figures and tables from the stored artefacts, plus the experiment scripts that produced the artefacts. See [results/README.md](results/README.md). |

Supporting material:

- `checkpoints/` the three pretrained models (baseline, narrow, wide widths) with their training manifests
- `config/base.yaml` the shared configuration
- `artefacts/` the stored runs behind every reported figure and table, organised by report section (arrays are compressed `.npz`)
- `BUILD_LOG.md` how this repository was assembled from the working tree, with the numerical checks made against the report

## Quick start

```
python -m venv .venv && .venv/Scripts/activate      # or source .venv/bin/activate
pip install -e .
python results/make_all.py                            # every figure and table, from artefacts (~15 min)
```

Outputs land in `results/figures/` and `results/tables/`. MNIST is downloaded
on first use into `data/mnist/`.

Re-running the experiments is deliberate and slow (about forty GPU-hours in
total) and is never done by default; see `results/README.md`.
