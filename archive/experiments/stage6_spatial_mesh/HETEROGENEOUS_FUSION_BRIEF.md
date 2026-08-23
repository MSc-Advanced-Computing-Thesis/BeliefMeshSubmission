# Heterogeneous-device fusion: experiment and result

Self-contained brief for discussion. Written 2026-08-13. All numbers are
measured, not estimated; where a claim is weak or contested that is stated
explicitly.

## 1. System

A decentralised mesh of 36 nodes on a 22x22 spatial grid. Each node has a
7x7 field of view, so neighbouring nodes' FOVs overlap and multiple nodes
"cover" the same cell. Each cell contains an image (an MNIST digit 7 rotated
to a per-cell angle) and the task is to regress that rotation angle.

Every node runs an evidential CNN that outputs a Normal-Inverse-Gamma belief
per cell: `(gamma, nu, alpha, beta)`, where `gamma` is the angle estimate and
`nu` is the accumulated evidence count backing it (epistemic uncertainty is
`1/nu`). Nodes exchange these 4-float beliefs with overlapping neighbours and
train on the aggregate. Runs are 390 timesteps; 3 mobile "wearables" supply
sparse ground-truth labels along random-wander paths.

Two environments are used across the project. This experiment uses the
**offset world**: the image appearance is uniform everywhere, but each cell
adds a location-dependent offset to the *correct answer*
(`label = rotation + offset(cell)`). The offset field is smooth and
time-varying, max 60 degrees. So identical-looking inputs demand different
answers at different locations, which no single global function can fit --
nodes must specialise locally, which is what makes collaboration non-trivial.

## 2. The prior null, and its explanation

Three aggregation modes are compared:

- **naive** -- plain unweighted mean of contributors' point estimates.
- **certainty** -- weights contributors by predictive certainty
  (`beta/(nu(alpha-1))`, mixing aleatoric and epistemic).
- **nig_product** -- closed-form product of the contributors' NIG densities.

For the product of NIGs, the fused estimate is

```
gamma* = sum_i (nu_i * gamma_i) / sum_i (nu_i)
```

i.e. a **nu-weighted average**. Naive is the **plain average**,
`sum_i gamma_i / N`. These are algebraically identical when all `nu_i` are
equal. Fusion can therefore only differ from naive to the extent contributors
to the *same cell in the same event* disagree about how much evidence they
hold.

Two consequences worth noting:

- `gamma*` is **scale-invariant in nu**: multiply every `nu_i` by a constant
  and it cancels. Anything that moves all nodes' confidence together (e.g.
  the evidence-regulariser coefficient `lambda`) cannot change the
  fusion-vs-naive gap at all.
- The natural diagnostic is **nu-CV**: within one fusion event, `std(nu_i) /
  mean(nu_i)`, averaged over all events (~278k per run). It is itself
  scale-invariant by construction.

A prior arc of six environment constructions (boundary offsets, combined
colour+offset, noise boundaries, per-cell digit instances, per-node digit
instances, large offsets) all tried to induce nu asymmetry **through the
environment** and all produced population-level nulls. Measured in the
homogeneous mesh:

| quantity | value | meaning |
|---|---|---|
| nu-CV | 0.030 | contributors' evidence counts differ by ~3% |
| max weight minus uniform | 0.011 | fusion weights ~1% from uniform |
| `gamma*` vs unweighted estimate | 0.00054 | **fusion's answer is 0.19 deg from naive's** |

Against a run RMSE of roughly 33 degrees. Fusion was not losing to naive; it
was computing almost exactly the same number.

The diagnosis: the environment is *shared* by all contributors to a cell, so
environmental manipulations hit them equally. The nodes were identical
clones (same architecture, LR, data budget), so they accumulated identical
evidence.

## 3. Hypothesis

Make the **devices** unequal rather than the environment. If nodes differ in
capacity, a weaker node genuinely knows less, the evidential regulariser
(`|error| * (2 nu + alpha)`) should drive its `nu` down, nu-CV should widen,
and fusion should down-weight it automatically while naive averages it in at
full strength.

This matters because the thesis framing is heterogeneous devices. If fusion
merely *matches* naive on homogeneous hardware but *wins* on heterogeneous
hardware, that motivates its use without needing a homogeneous-case win.

## 4. Design

Width variants (channel counts per conv block):

| variant | widths | pretraining holdout MSE |
|---|---|---|
| narrow | (16, 32, 64) | 0.0127 |
| baseline | (32, 64, 128) | 0.0154 |
| wide | (64, 128, 256) | 0.0120 |

Each variant loads its **own independently pretrained Stage-0 checkpoint**,
so capacity is not confounded with transfer damage from slicing a baseline
checkpoint, nor with a cold random init.

**Important:** the three checkpoints start at comparable quality -- all
within the +/-0.002-0.003 eval noise the Stage-0 manifest documents, and
`baseline` actually starts *worst* of the three. So any in-mesh performance
ordering is **not** inherited from unequal pretraining; it emerges during
online adaptation, where capacity limits how well a node fits the
location-dependent offset mapping.

36 nodes are assigned equal thirds narrow/baseline/wide, **shuffled** so
capacities are spatially interleaved -- every fusion event then mixes
capacities. (A clustered layout would leave most neighbourhoods internally
homogeneous and dilute the effect.)

Everything else is held identical to the matched homogeneous control:
lr=3e-5, lambda=5.0, 60-degree offset field, no colour filter, same excluded
rotation ranges, same wearable paths, same seeds. The **only** difference
between the heterogeneous and homogeneous runs is `node_variants`.

## 5. Result A -- the gating worked (solid)

Homogeneous vs heterogeneous, seed 42, `nig_product`:

| quantity | homogeneous | heterogeneous | change |
|---|---|---|---|
| nu-CV | 0.0300 | **0.1396** | 4.7x |
| max weight minus uniform | 0.0112 | **0.0511** | 4.6x |
| `gamma*` vs unweighted | 0.00054 | **0.00512** | 9.5x |
| contributor estimate spread (`spread_std`) | 0.0226 | 0.0306 | 1.35x |

The first construction in the whole arc to move nu-CV materially. Note the
last row: contributors' *estimates* barely spread further apart, so this is
not simply "everything got noisier" -- it is specifically the **evidence
counts** that separated, which is the quantity the algebra says matters.
Fusion's answer now sits 0.92 degrees from naive's rather than 0.19.

This is a direct measurement and is not in question.

## 6. Result B -- accuracy at seed 42 (strong, but single-seed)

Whole-mesh MSE, paired by timestep (n=390):

| mesh | naive | fusion | effect |
|---|---|---|---|
| homogeneous | 0.01006 | 0.01010 | -0.4% (dz=-0.31) |
| heterogeneous | 0.01301 | 0.01007 | **+22.6%** (dz=+1.03) |

Framed the striking way: adding weak devices cost naive +29% error
(0.01006 -> 0.01301) and cost fusion nothing (0.01010 -> 0.01007).

Per-variant node-level MSE at seed 42 (node-level, so absolute values are not
directly comparable to the cell-level table above; the within-table
comparison is the valid one):

| variant | naive | certainty | fusion |
|---|---|---|---|
| narrow | 0.0178 | 0.0178 | 0.0166 |
| baseline | 0.0201 | 0.0181 | **0.0090** |
| wide | 0.0176 | 0.0169 | **0.0095** |

Under naive, the strong nodes are dragged down to roughly the weak nodes'
level -- baseline nodes reach 0.0201, *worse* than narrow nodes achieve under
fusion. Plain averaging propagates the narrow nodes' errors into every
neighbour overlapping them. Under fusion the strong nodes retain ~0.009.

**Specificity:** `certainty` scored 0.01255, closing only 3.6% of the 22.6%
gap (about one sixth), and shows naive's contamination profile rather than
fusion's. So the effect is specific to the closed-form NIG product's
nu-weighting, not to confidence-aware aggregation generally. (`certainty` has
only been run at seed 42.)

## 7. Result C -- five-seed replication (this is the problem)

Seeds 42, 1042, 2042, 3042, 4042, with the **device layout reseeded per seed**
as well as the environment, so replication varies mesh composition too.

| seed | whole-run | last50 |
|---|---|---|
| 42 | +22.6% | +26.3% |
| 1042 | +20.3% | +33.3% |
| 2042 | +48.1% | +67.8% |
| 3042 | +8.1% | +10.2% |
| 4042 | **-5.1%** | +9.7% |

- Whole-run: mean **+18.8% +/- 19.7pp**, paired t p=0.18, fusion ahead 4/5.
- Last50: mean **+29.5% +/- 23.8pp**, paired t p=0.165, fusion ahead 5/5
  (Wilcoxon p=0.0625, which is the smallest two-sided value attainable at
  n=5).

**The standard deviation is about the size of the mean.** Seed 42's +22.6%
is not representative; the range runs -5% to +48%. This is the same failure
mode as an earlier finding in the arc (a seed-42 spatial correlation of
+0.34 that fell to a 5-seed mean of +0.105 +/- 0.135).

Much of the mean rides on seed 2042, where naive degenerated (0.0247 whole /
0.0452 last50) while fusion held (0.0129 / 0.0145). Excluding seed 2042:
+11.5% whole-run (3/4), +19.9% last50 (4/4).

### Variance, and a caveat about how it was tested

Fusion's outcome is markedly more stable across seeds:

| | naive range | spread | fusion range | spread |
|---|---|---|---|---|
| whole-run | 0.0113-0.0247 | 2.20x | 0.0101-0.0133 | 1.32x |
| last50 | 0.0152-0.0452 | 2.97x | 0.0124-0.0167 | 1.34x |

Variance ratios are 13.8x and 68.5x. **But the two variance tests disagree**:
the F-test gives p=0.026 and p=0.0012, while Levene gives p=0.45 and p=0.23.
The F-test assumes normality and is unreliable at n=5 with an outlier;
Levene is robust and more conservative. The descriptive range comparison is
the honest version -- the F-test p-values should not be quoted.

## 8. Where this leaves the claim

Defensible now:

1. Device heterogeneity is the **only** manipulation tried across seven
   constructions that materially widened within-event evidence asymmetry
   (nu-CV 4.7x). Direct measurement, replicated mechanism.
2. The mechanism is not a pretraining artifact -- the three checkpoints start
   at equivalent quality, with baseline actually worst.
3. Under heterogeneity, fusion is directionally better in 5/5 seeds on last50
   and 4/5 on whole-run, and its across-seed outcome is far more tightly
   bounded than naive's.
4. The benefit is specific to NIG-product fusion, not confidence weighting in
   general (seed 42 only).

**Not** defensible: that fusion improves accuracy by ~22% under
heterogeneity. Mean improvement does not reach significance at n=5, and one
seed reverses on whole-run. The narrower honest reading is that fusion
**avoids the bad outcomes naive is prone to** rather than reliably improving
the typical run -- worst-case robustness on mixed hardware.

## 9. Open questions for discussion

1. **Power.** n=5 has no power for an effect this variable. Would 15-20 seeds
   settle it, or is the across-seed variance intrinsic to the setup such that
   more seeds just estimate a wide distribution more precisely?
2. **Which headline?** "Reduces worst-case degradation" (well-supported,
   less punchy) vs "improves mean accuracy" (punchier, currently
   unsupported). Is variance reduction a legitimate primary claim?
3. **Seed 2042.** Naive degenerated there and fusion did not. Is that the
   effect working as intended on a hard draw -- i.e. the most informative
   seed -- or an unstable-training artifact that should be investigated
   before being counted as evidence?
4. **Is the comparison fair?** A skeptic will say a heterogeneous mesh is a
   setup engineered to favour fusion. Counter-argument: heterogeneous
   hardware is the realistic deployment condition, and the homogeneous case
   is reported as a null rather than hidden.
5. **Design sensitivity.** Would the effect survive a clustered rather than
   interleaved layout, or a different mix than 1/3-1/3-1/3? Interleaving was
   chosen deliberately to maximise within-event capacity mixing.
6. **Mechanism completeness.** nu-CV widened 4.7x and accuracy moved, but the
   causal chain (capacity -> lower nu -> lower fusion weight -> less
   contamination) has not been verified end-to-end; nu has not been broken
   down by variant. That would be a direct check.

## 10. Reproduction

```
python -u experiments/stage6_spatial_mesh/run_heterogeneous_comparator.py --mode nig_product --seeds 42,1042,2042,3042,4042
python -u experiments/stage6_spatial_mesh/run_heterogeneous_comparator.py --mode naive       --seeds 42,1042,2042,3042,4042
python -u experiments/stage6_spatial_mesh/run_heterogeneous_comparator.py --mode nig_product --homogeneous   # matched control
```

Outputs in `runs/stage6/offset_world/heterogeneous_comparator/`. The
homogeneous control at 60 degrees with no colour filter is in
`runs/stage6/offset_world/large_offset_comparator_deg60/`.
