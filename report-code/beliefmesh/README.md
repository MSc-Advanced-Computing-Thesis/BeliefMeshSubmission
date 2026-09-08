# beliefmesh: the system

BeliefMesh is a mesh of fixed nodes that each watch part of a gridded
deployment area, learn continually from sparse ground truth supplied by mobile
wearables, and collaborate by exchanging compact *beliefs* about shared cells
rather than model parameters (report Chapter 3). A belief is the four
parameters of a Normal-Inverse-Gamma distribution over a cell's value; beliefs
from several nodes combine in closed form, and the fused belief is what a node
trains on where it has no wearable of its own.

## How the pieces fit together

```
beliefmesh/
  models/      evidential.py       NIG head, Student-t marginal, evidential loss (Eq. 3.4)
               evidential_cnn.py   the CNN with the four-parameter NIG output head
               variants.py         narrow / baseline / wide convolutional widths (Sec 4.4)
  fusion/      nig_product.py      closed-form NIG fusion: Product and Average fusion (Eq. 3.2, App A)
               consensus.py        consensus (agreement) weighting used by the 5.9 ablation
               grid.py             circular candidate grid for mode finding
               product_of_experts.py  earlier product-of-experts fusion, kept for the tests
  metrics/     circular.py         circular MSE / difference on the [-1, 1) angle domain (Eq. 4.1)
  data/        digits.py           MNIST digit-7 dataset with rotation labels (Sec 4.1)
               filters.py          colour filters of the colour world (Sec 4.2.1)
               grid_environment.py the gridded environment: per-cell rotation, colour and offset
  node/        node.py             a single learner (used by the two-node experiments)
               mesh.py             MeshNode and Mesh: overlap graph, BFS propagation, fusion,
                                   node update rule, node failure, comparators (gossip, FedAvg)
  simulation/  runner.py           run_mesh_experiment(): one full deployment, writes the artefacts
               offset_fields.py    the offset-world field (Sec 4.2.2) and the random wearable walk
               field_variants.py   the five offset environments of Sec 5.4 (field0 ... field4)
               assets.py           paths to checkpoints and the reference 36-node environment
               cert_mse_metrics.py certainty/error correlation helpers
               environments/       the reference environment: node centres, colour field, path
```

Per timestep (Sections 3.3 to 3.5) `Mesh.run_timestep` finds the nodes whose
field of view contains a wearable, trains them on the measurement, and then
propagates outward breadth-first: each untrained node fuses the beliefs its
trained neighbours hold on their shared cells (`fusion.nig_product`), samples a
training target from the fused Student-t, weights each cell by the fused
certainty, and takes one optimiser step. Every node then predicts fresh beliefs
over its whole field of view, which is what the runner records.

The `mode` argument selects the aggregation or the comparator:

| mode | meaning |
|---|---|
| `nig_product` | Product fusion (the system as reported) |
| `naive`, `certainty` | mode averaging, unweighted / certainty-weighted (Sec 5.3) |
| `nig_product_weighted`, `nig_product_consensus` | weighted-fusion and consensus ablations (Sec 5.9) |
| `frozen` | no training |
| `gossip_uniform`, `gossip_weighted`, `fedavg_global` | parameter-exchange comparators (Sec 5.4) |

Average fusion is Product fusion with per-contributor weights 1/N; the
experiments that report it wrap `fuse_nig_product` accordingly (see
`results/5.3_belief_aggregation/5.3.2_mesh_scale/experiment_average_fusion.py`).

## Running a single mesh experiment

```python
import numpy as np, random, torch
from pathlib import Path
from beliefmesh.config import load_config
from beliefmesh.simulation.assets import BASELINE_CHECKPOINT, ENVIRONMENT_DIR, EXCLUDED_ROTATION_RANGES
from beliefmesh.simulation.offset_fields import build_dynamic_offset_field, build_random_wander_path
from beliefmesh.simulation.runner import run_mesh_experiment

cfg = load_config()
cfg.model.lr = 3e-5
G, T, seed = 22, 390, 42
centres = np.load(ENVIRONMENT_DIR / "node_centres.npy")           # the 36-node reference mesh
paths = [build_random_wander_path(T, G, seed=200 + i) for i in range(3)]   # three wearables
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

result = run_mesh_experiment(
    cfg, condition="demo", run_dir=Path("runs/demo"),
    all_grids=np.full((T, G, G), 0.5),          # uniform colour (the offset world)
    wearable_paths=paths, node_centres=centres, fov_size=7,
    mode="nig_product", baseline_checkpoint=BASELINE_CHECKPOINT,
    n_wearable_samples=1, n_train_repeats=1,
    offset_field=build_dynamic_offset_field(G, T), env_seed=seed,
    excluded_rotation_ranges=EXCLUDED_ROTATION_RANGES, lam=5.0,
    sample_target=True, draws_per_target=1)
print(result["mean_mse_last_50"])
```

This is the configuration used throughout Chapter 5 (36 nodes, 7x7 fields of
view, stride 3, three randomly wandering wearables, lambda 5, lr 3e-5, 390
steps). A run takes about ten minutes on a GPU and writes `manifest.yaml`
plus the per-cell arrays (`cell_beliefs_steps.npy`, `cell_ncov_steps.npy`,
`cell_nig_steps.npy`, `cell_mse_steps.npy`, ...) that the results scripts read.
Every experiment script under `results/` is a variation of this call.

## Checkpoints and data

`checkpoints/{baseline,narrow,wide}/pretrained_digit7.pth` are the pretrained
models of Section 4.4, with the manifest of the pretraining run alongside.
MNIST is downloaded on first use into `data/mnist/`.
