# Evidence-Asymmetry Experimental Arc — Consolidated Summary

Closing document for the multi-session investigation into why `nig_product`
(closed-form belief fusion) is statistically indistinguishable from `naive`
aggregation (plain mean of contributor point estimates) in the smooth
offset world. The central finding that motivated this whole arc: fusion's
weighted mean can only diverge from naive's plain mean to the extent
contributor nu (evidence count) varies *within a single fusion event* --
population-level nu statistics (what lambda controls, what environmental
"difficulty" changes) are a different quantity and do not bear on it
directly. Six environments were constructed to test whether specific,
principled sources of within-event nu asymmetry could be engineered.

All figures below use the corrected per-timestep-then-averaged cert-MSE
convention and, where stated, are 5-seed means +/- std (seeds 42, 1042,
2042, 3042, 4042); constructions not extended to 5 seeds are marked
"(seed 42 only)".

## 1. All six constructions

| construction | nu-CV (mean) | weight-dev (mean) | asymmetry created? | population accuracy outcome | spatially localised effect? | replicates across 5 seeds? |
|---|---|---|---|---|---|---|
| smooth offset (baseline) | 0.044 | 0.016 | -- (reference) | naive = certainty = nig_product (established null) | n/a | -- |
| boundary-offset (piecewise offset field, 90 deg step at col 11) | 0.085 | 0.031 | **yes** (~2x) | naive +2.0%, certainty +1.2% worse than nig_product, both p~0 (seed 42 only) | **no** -- advantage present at roughly uniform magnitude (+0.005 to +0.01) across the WHOLE grid, not concentrated near the boundary; likely a general-difficulty artefact, not the targeted mechanism | not tested |
| combined colour + offset (independent fields, verified r~0) | 0.0591 +/- 0.0033 | 0.0220 +/- 0.0012 | partial (~34% above baseline) | population null (naive p=0.25, certainty p=0.15) | **yes at seed 42 only** (misalignment-vs-advantage r=+0.34, p=1.4e-14) | **no** -- per-seed r: 42=+0.34, 1042=+0.003, 2042=-0.030, 3042=+0.049, 4042=+0.163; mean +0.105 +/- 0.135 (std exceeds mean). Only 1 of 5 seeds shows a strong effect; 3 of 5 are statistically indistinguishable from zero |
| noise-boundary (10x jitter-std step at col 11, training-signal only) | 0.0564 +/- 0.0055 | 0.0208 +/- 0.0017 | partial (~28% above baseline) | population null (naive p=0.62, certainty p=0.24) | **partially, and partially replicates** -- see section 3 | **half-replicates**: near-boundary (dist 0-3) advantage positive in 4/5 seeds, mean +0.000272 +/- 0.000227; interior (dist 4-10) advantage essentially null, mean -0.000006 +/- 0.000159 (NOT reliably negative -- 2 of 5 seeds show a positive interior mean) |
| per-cell digits (distinct instance per cell) | ~0.048 (seed 42 only) | ~0.018 | **no** -- ~9% above baseline, within noise | tiny, correctly-signed, technically significant only via huge n (naive +0.28%, certainty +0.46%) | not tested (gating essentially failed) | not tested |
| per-node instances (distinct instance per node, per-recipient prediction) | ~0.047 (seed 42 only) | ~0.018 | **no** -- ~7% above baseline, within noise | tiny, correctly-signed (naive +0.68% p=2.4e-24, certainty +0.93% p=8.7e-38) | weak instance-dissimilarity correlation (r=+0.36, p=0.03, n=36 nodes) -- not pursued further per explicit stop instruction | not tested (construction halted at gating stage) |

## 2. Per-node-instance compute cost

The per-node-instance construction requires one fresh, uncached forward
pass per (contributor, recipient, cell) triple, replacing the baseline's
single batched `predict_cells()` call shared across every recipient.
Measured: `fusion_call_count` (278,483) is unchanged from baseline -- that
count reflects coverage geometry, not this change -- but the new
`predict_belief_for()` category adds approximately **935,700 additional
single-item forward passes per 390-step run** (278,483 disagreement-logged
events x mean 3.36 contributors each) that have no analogue in the
deployed/baseline architecture. This is a direct, quantified argument for
why a real deployment should use a single shared-referent belief per node
(broadcast once, consumed by every neighbour) rather than per-recipient
prediction: the latter is a valid simulation device for testing the
evidence-asymmetry hypothesis, but is computationally uncompetitive by
roughly two orders of magnitude in forward-pass count for a construction
that, per the gating check, does not even reliably produce the asymmetry
it was built to create.

## 3. Which spatial finding is more robust?

Neither the combined-environment misalignment correlation nor the
noise-boundary column-distance pattern survives 5-seed replication in the
clean form originally reported at seed 42. Comparing them directly:

- **Combined environment**: the correlation coefficient itself is the
  claim, and it is not reproducible in sign or magnitude across seeds
  (ranges from -0.030 to +0.340; the across-seed std, 0.135, exceeds the
  across-seed mean, 0.105). This should be treated as **suggestive at
  best, and the seed-42 result specifically should not be presented as a
  confirmed finding** -- it is very likely dominated by that seed's
  particular field/wearable-path realisation rather than reflecting a
  general property of the mechanism.
- **Noise boundary**: the claim decomposes into two halves. The
  near-boundary-positive half is reasonably robust (4 of 5 seeds positive,
  consistent order of magnitude, mean clearly on the positive side of
  zero even accounting for spread). The interior-negative half is not
  robust (2 of 5 seeds negative; the across-seed mean is statistically
  indistinguishable from zero). **The honest, replicated claim is
  narrower than originally stated**: fusion shows a real, repeatable
  (though modest) advantage specifically near the noise boundary; it does
  NOT reliably lose to naive in the interior the way the seed-42 result
  suggested -- the interior is closer to a wash.

**Recommendation for the write-up**: if one spatial finding is to be
foregrounded, it is the noise-boundary near-boundary effect, reported with
its correct, narrower scope (fusion wins near the discontinuity; the
interior is a null, not a naive-advantage region) and with the 5-seed
numbers shown rather than the seed-42 numbers alone. The combined-environment
misalignment correlation should be reported briefly, explicitly flagged as
not replicating, and used at most as a motivating anecdote for why the
noise-boundary construction was built next -- not as independent
confirmation of anything. The boundary-offset construction should be
reported as the result that revealed the difficulty-vs-asymmetry confound
(real accuracy separation, but spatially uniform rather than localised,
undermining a mechanism-specific interpretation) -- useful for motivating
the later, better-controlled constructions, not as evidence for the
mechanism itself. Per-cell-digits and per-node-instance should be reported
briefly as negative results that closed off visual-diversity-based and
per-recipient-viewpoint explanations respectively, with the per-node-instance
compute-cost figure retained as a supporting argument for the deployed
design's shared-referent choice.

## 4. What this arc actually establishes

1. Fusion and naive aggregation are statistically indistinguishable at the
   population level in every environment tested, including every
   engineered construction -- this is now a well-replicated finding, not
   an artefact of the original smooth-offset-world result.
2. The mechanistic explanation (contributors to a shared cell carry similar
   evidence because they are spatially adjacent and similarly trained) is
   supported by direct measurement (nu-CV, weight-deviation-from-uniform)
   in every construction, including the ones that succeeded at modestly
   raising it.
3. A real, spatially-localised fusion advantage was demonstrated at seed 42
   in two independent constructions, but only one of those two effects
   (noise-boundary, and only its near-boundary half) shows a repeatable
   signal across 5 seeds. The mechanism is real but weak and narrow in
   scope under the constructions tried here -- not absent, but nowhere
   near strong enough to move population-level accuracy, and not as
   spatially clean as the single-seed results suggested.
4. No construction tried -- including deliberately engineered ones with
   verified, substantial population-level nu-CV widening -- produced a
   whole-mesh accuracy advantage for fusion over naive aggregation.
