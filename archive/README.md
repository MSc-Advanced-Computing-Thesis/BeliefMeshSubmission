# Experiment Specification V2

## Uncertainty-Driven Decentralised Continual Learning via Bayesian Belief Fusion

Christian Wood · MSc Advanced Computing · Imperial College London · 2025–2026

This document supersedes the original Node Learning UQ Experiment Proposal. It records the experimental design as implemented, the corrections applied following the codebase audit, and the reasoning behind each design decision including the alternatives rejected. It is the reference from which the final report's experimental sections are written.

---

## 1. Purpose and Status

The system exchanges calibrated beliefs rather than model parameters. Each node runs a frozen pretrained backbone and a trainable Normal-Inverse-Gamma evidential regression head. Nodes emit evidential predictions, which are combined by Bayesian belief fusion propagated breadth-first through an overlap graph from wearable ground-truth anchors. The claim under test is that belief-level exchange preserves performance under non-stationarity where parameter-level averaging collapses.

Two defects were identified in the previous implementation. Both are confined to the N-node mesh regime. Stages 0 to 5 are single-node or two-expert and are unaffected.

**Defect 1: N-way fusion was never Bayesian.** The validated `bayesian_fusion_grid` accepts exactly two NIG belief tuples. No N-way variant existed. In the mesh regime the code fell back to certainty-weighted averaging of collapsed point estimates, `(preds * certs).sum() / certs.sum()`, with fused certainty `min(certs)`. The pairwise Bayesian function was dead code, because `SpatialGridNode.predict_cells` collapses each prediction to `(pred, uncertainty, certainty)` and discards `γ, ν, α, β` before fusion is reached.

**Defect 2: training certainty was never wired into any loss.** The quantity was computed and stored per node but never multiplied into a gradient update. Certainty-weighted training was described but not implemented. The instantaneous form is additionally unsound, as established in Section 6.

Every Stage 6 figure describing Bayesian fusion is therefore superseded. Those results were produced by scalar weighted averaging and cannot appear in the report labelled as Bayesian fusion. The nine-way overlap threshold, measured under scalar averaging, is provisional pending regeneration.

---

## 2. Proxy Task and Setup

The task is rotation angle regression on MNIST digit seven. The dataset is restricted to the digit seven because it is not rotationally symmetric. The task is selected for its regressive nature; rotation angle prediction is analogous to prediction of a continuous environmental property such as water velocity.

A circular MSE formulation is used throughout. Standard MSE treats -179° and 179° as maximally different when they differ by two degrees. The circular formulation wraps the prediction error before squaring.

The dataset is split into a fixed training set and a held-out evaluation set under seed 42, with 20% of samples reserved for evaluation. All evaluation metrics are computed exclusively on the held-out set.

**Decision: held-out evaluation split.** The original experiments drew evaluation instances from the training distribution, producing an uncertainty-error correlation of 0.72. That figure reflected memorisation of seen instances rather than generalisation. The corrected split reduces the correlation to 0.5649. The lower figure is the honest one. This is a finding about the original protocol, not a degradation of the model.

**Decision: Stage 6 retains its single-digit-per-cell design.** Applying the held-out evaluation approach to the mesh regime caused Bayesian fusion to degrade toward naive aggregation levels through data sparsity. The design difference between Stages 0 to 5 and Stage 6 is acknowledged rather than concealed. The interaction is noted: sparsity was already pushing fused performance toward naive under the scalar rule, so whether true Bayesian fusion is more robust to that sparsity is an open question the rerun answers.

Distributional shift is induced by red and blue colour filters applied to the rotated digit images. The background and digit colours are inverted, and the digit is recoloured green before filter application. Green shares no dominant channel with either red or blue in RGB, so both filters suppress the digit colour equally. Red and blue filters of equal strength therefore produce symmetric distributional shifts with no directional bias.

---

## 3. Uncertainty Quantification

**Decision: NIG evidential regression, replacing MC Dropout.** The original proposal specified MC Dropout with T=20 stochastic forward passes. MC Dropout is rejected because inference cost scales linearly with T. On MCU-class hardware a single forward pass already consumes a significant fraction of the compute budget, so a twentyfold multiplier is prohibitive. The NIG head produces four parameters in a single forward pass at negligible overhead, and additionally decomposes predictive uncertainty into epistemic and aleatoric components, which MC Dropout does not.

The head outputs `(γ, ν, α, β)`. The epistemic component `1/ν` captures reducible model ignorance. The aleatoric component `β/(α−1)` captures irreducible data noise. The marginal predictive over the target is a Student-t with location `γ`, degrees of freedom `2α`, and scale `√(β(1+ν)/(να))`.

The backbone is a small CNN with three convolutional blocks and two fully connected layers, trained with Adam at `lr = 3 × 10⁻⁴` for 30 epochs. Architecture and hyperparameters are not optimised; they exist solely to support analysis of the belief-exchange mechanism.

---

## 4. Fusion Rule

### 4.1 Specification

Fusion treats each contributing node's belief as an independent likelihood factor concerning the same latent angle. Under a flat prior the combined posterior is proportional to their product. Products of densities are computed as sums of log-densities:

```
total(y) = Σᵢ wᵢ · log pᵢ(y)
fused_mode = grid[argmax(total)]
```

where `pᵢ` is contributor `i`'s Student-t marginal predictive, `wᵢ` is its running consensus, and the grid spans the circular angle domain at `G = 360` uniformly spaced candidate values.

**Decision: the N-way rule is the pairwise rule with more terms.** Nothing in the product-of-experts derivation is pairwise. Summing the logs of two densities is the same operation as summing the logs of N. The previous implementation hardcoded two terms, and the mesh regime silently fell back to scalar averaging rather than extending the sum. There is no separate pairwise code path in the corrected implementation; `N = 2` is the special case validated against Stage 4c.

**Decision: the log transform is exact; the grid is the only approximation.** `log(p_A · p_B) = log p_A + log p_B` identically, and `argmax` is invariant under the monotonic log. The product of Student-t densities has no closed form, so the maximisation is performed numerically.

**Decision: `G = 360`, giving one candidate per degree.** Grid error is quantisation error, bounded by half the spacing. At `G = 360` the spacing is 1.0° and the mode is located to within 0.5°, so the fusion primitive is accurate to the degree and grid resolution cannot be mistaken for a source of error in any result. The previous implementation used `G = 200`, a spacing of 1.8° and a bound of 0.9°, which is coarser than the quantity being measured in the calibration experiments. The cost is linear in `G`, so `G = 360` is 1.8 times the previous cost of an operation that is already negligible relative to the CNN forward pass producing the NIG parameters. Should profiling show the cost material on the microcontroller, `G` is reduced and the resulting resolution loss is reported as a deployment cost rather than concealed. `G = 180`, giving a 1.0° bound, is the fallback.

**Decision: grid search over gradient descent.** Both were evaluated. Grid search and optimisation fusion produce comparable performance, with optimisation marginally ahead at high filter strengths, confirming that principled mode estimation rather than the specific estimator is the source of gain. Grid search is preferred on deployment grounds: it is computationally predictable, and in the real scenario where inference operates on a single datapoint, `G` candidate evaluations are trivially cheap relative to the CNN forward pass that produced the NIG parameters. Gradient descent is additionally sensitive to initialisation when the joint distribution is broad and flat, which is precisely the low-consensus regime the system must handle honestly.

### 4.2 Confidence enters through density shape, not external weighting

A confident node's Student-t is tall and narrow. Its log-density changes steeply as `y` departs from its location, so it pulls the argmax toward itself. A diffuse node's density is flat and barely perturbs the sum. Every density integrates to unity; no external certainty scalar is applied inside fusion.

This is the structural difference from the defective implementation. Scalar averaging collapsed each belief to `(pred, certainty)` and applied certainty as an explicit multiplicative weight, discarding `α` and `β`. Product-of-experts consumes `α` and `β` directly, because they determine peakedness and tail behaviour. The weighting is implicit and principled rather than imposed.

**Rejected: fusion weighted by epistemic uncertainty alone.** The Student-t marginal mixes epistemic and aleatoric uncertainty, so fusion weights each contributor by total predictive uncertainty. A node observing a genuinely ambiguous cell is downweighted as much as an undertrained node. This is correct. For the purpose of locating the latent angle, a belief is uninformative whether its width arises from noise or from ignorance. The epistemic and aleatoric distinction governs propagation rather than fusion, as established in Section 5.

### 4.3 Circular wraparound

The grid and the difference `y − γ` must wrap on the circle, using the same wrapping as the circular MSE. A belief at +179° and one at −179° describe nearly the same angle. A flat difference makes them maximally distant, and the product then exhibits two spurious disjoint modes rather than one sharp shared mode. Rotation is the task, so this condition arises constantly rather than rarely.

---

## 5. What Propagates

The fused mode is used as a scalar pseudo-label. The receiving node trains against it with the ordinary NIG loss and derives its own `(γ, ν, α, β)` locally.

**Decision: mode-matching, not distribution-matching.** Training a receiver to reproduce the full fused distribution would inject its neighbours' aleatoric uncertainty into a node that has no reason to hold it. Aleatoric uncertainty is a property of local observation conditions. A node's predictive uncertainty must reflect its own model's competence on its own inputs. What propagates is the improved estimate; uncertainty is regenerated locally at every hop.

Consensus governs propagation. It is not itself the thing propagated.

---

## 6. Reliability Weighting

### 6.1 Consensus replaces training certainty

Fusion already consumes belief sharpness through the density product. It does not capture disagreement. Three sharply peaked beliefs centred on different angles produce a confident-looking fused mode supported by none of them, and nothing downstream distinguishes this from three concordant sharp beliefs. Disagreement has no other home in the pipeline, so it is the quantity that propagates.

**Rejected: instantaneous training certainty.** The previous design weighted a node by the certainty of the pseudo-label it received that turn, multiplied by its predictive certainty. A single uncertain training example then suppresses an otherwise well-calibrated and confident node. Model competence is the accumulation of all prior calibration, not one training instance. This is a parameterisation error rather than a design error; the compounding structure is sound because the system is continual and training-signal quality is a real reliability property.

**Rejected: belief sharpness as a separate propagated term.** Sharpness is consumed at fusion time by the density product. Carrying it forward as an additional multiplicative weight decays certainty twice per hop for no clear gain.

### 6.2 Consensus definition

Disagreement among the `N` contributors is measured by the Jensen-Shannon divergence, evaluated on the same grid already constructed for the mode, using the mixture `m = (1/N) Σᵢ pᵢ`.

**Decision: Jensen-Shannon over Kullback-Leibler.** KL is asymmetric, which forces a choice of direction that has no principled answer. In fusion no belief is privileged over the fused result, since the fused density is a construct of the contributors rather than a reference truth. JS is symmetric and treats all densities as peers, matching the structure of product-of-experts. JS is additionally bounded above by `log 2`, which yields a consensus score in `[0,1]` with no free sensitivity parameter. KL, unbounded above, would require `exp(−λ · D_kl)` and `λ` would demand separate justification.

```
agreement  = 1 − D_js / log 2
inherited  = Σᵢ cᵢ · cᵢ / Σᵢ cᵢ
c_new      = agreement × inherited
c_receiver ← (1 − ρ) · c_receiver + ρ · c_new
```

**Decision: consensus compounds through the chain.** The inherited term is required. Without it, a node at hop five whose contributors agreed perfectly scores identically to an anchor. Consensus is not correctness: nodes trained on a uniformly stale distribution agree strongly with each other while being uniformly wrong together, and under domain shift this is precisely the failure case. Multiplying by the contributors' own trust caps a node's score at their reliability regardless of how well they agreed. Consensus then decays monotonically from unity at the wearable, and the certainty gradient across the mesh is a derived consequence of this compounding rather than an imposed decay.

**Decision: consensus-weighted inheritance.** The inherited term weights each contributor's consensus by `cᵢ`, its own tempering exponent, because that is the power to which its density was raised in the product and therefore how much the fusion listened to it. A distrusted contributor is doubly discounted, once at fusion and once in inheritance. An arithmetic mean would allow a barely-influential flat density to drag down the inherited trust of a mode it did not shape. The approximation is noted: `cᵢ` is the tempering exponent rather than true influence, since a sharp density with low `cᵢ` may still dominate a diffuse one with high `cᵢ`. True influence would require the curvature contribution at the mode.

**Decision: tempering is applied post-log.** Scaling the log-density raises the density to the power `wᵢ`, flattening it for `wᵢ < 1` and reducing its influence on the argmax. Scaling the density before the log contributes `Σᵢ log wᵢ`, a constant in `y`, which shifts the summed log-density uniformly and does not move the argmax at all. Pre-log scaling is therefore a no-op on the mode. This is verified empirically rather than asserted.

### 6.3 Anchors

Anchor status is a transient property of the timestep, not a node attribute. A node is an anchor at timesteps when the wearable lies within its coverage, and it trains on ground truth then. There is no fusion at such a timestep, so no disagreement exists and consensus updates toward unity. When the wearable departs, the node resumes training on fused pseudo-labels and its consensus decays according to the disagreement it observes.

This removes the undefined-consensus edge case entirely, since every node follows one rule. The certainty gradient becomes spatiotemporal rather than purely spatial, which is the correct behaviour for a moving anchor. Sensor error rates, which would set an anchor's consensus below unity, are outside scope.

### 6.4 Decay

**Decision: a single global `ρ`.** The decay rate governs the horizon beyond which stale consensus ceases to be informative about current conditions. This is a property of environmental dynamism, which is a property of the environment rather than of any node.

**Rejected: per-node decay driven by each node's aleatoric uncertainty.** Nodes in noisy cells would forget faster than nodes in calm cells, so their consensus values would be running averages over different effective time windows. Multiplying non-comparable quantities through the propagation chain is incoherent. A mesh-level dynamism-adaptive `ρ`, driven by mean aleatoric uncertainty across recently in-coverage nodes, is coherent and is retained as a separate later condition. It is not the default, because a term whose first-order value is unproven does not warrant a second-order refinement.

**Rejected: decay modulated by epistemic uncertainty.** A node's epistemic uncertainty depends on its own training state, which depends on the weights it receives, which depends on its consensus. Tying decay to it produces a coupled dynamical system with no obvious fixed point.

### 6.5 Cost

The `(N, G)` log-density array is constructed once for the mode. Jensen-Shannon reuses it with one additional `logsumexp` for the mixture. At `N = 9` and `G = 360` this is approximately 3240 operations per cell per round. The divergence requires no closed form.

---

## 7. Spatial Configuration

Square fields of view are used for simplicity. Circular fields would be more natural for drone cameras but are not expected to affect results.

An ablation across three configurations established a threshold effect. The 7×7 field of view at stride 3 (36 nodes, maximum cell coverage 9) achieved strong learning. Both 5×5 configurations, at stride 3 (maximum coverage 4) and stride 2 (maximum coverage 9), failed to learn meaningfully. The differentiating factor is maximum cell coverage. Nine-way overlap provides sufficient belief diversity for effective fusion; four-way does not.

**Status: provisional.** This threshold was measured under scalar averaging. Under true product-of-experts fusion it may move, sharpen, or dissolve. It requires regeneration and cannot be reported as a fusion result until then.

A single digit-seven image serves as the visual input for all cells. The mechanism under test is rotation angle prediction under colour filtering, not visual diversity between cells. In deployment each physical location would have its own visual character, but visual variation between adjacent cells in a flood environment is far smaller than would result from assigning distinct digit images. At each timestep the image is rotated by a freshly sampled angle specific to each cell, and that cell's current filter value is applied.

---

## 8. Experiment Stages

Each stage adds exactly one element to the previous, so the contribution of each component is isolable. Every stage is validated before the next is trusted.

### Stage 0: Baseline model and calibration

Train the NIG head on unfiltered rotated digits with true angle labels. Establishes the pretrained model used by all subsequent stages.

**Evaluation.** Held-out MSE, and the Pearson correlation between predicted uncertainty and absolute error.

**Reference result.** MSE 0.0106 in normalised angle space, approximately 18.5° mean error. Uncertainty-error Pearson correlation 0.5649.

**Purpose.** Two gates, not one. Accuracy establishes that the task is learnable and the architecture adequate. Calibration establishes that the head's uncertainty is honest. Both are required: a calibrated but inaccurate model propagates confident ignorance, and an accurate but uncalibrated model cannot be weighted by fusion.

### Stage 1: Degradation under distributional shift

Evaluate the frozen pretrained model across increasing red filter strength.

**Reference result.** MSE degrades monotonically with filter strength.

**Purpose.** Establishes that adaptation under dynamic conditions is necessary rather than optional.

### Stage 2: Peer supervision sufficiency

Node A fine-tunes on red-filtered inputs against true angles. Node B receives only Node A's predictions as its training signal and sees red-filtered inputs. This is the minimal peer supervision setup: one well-matched anchor supervising an out-of-coverage learner.

**Reference result.** Node B tracks Node A across all filter strengths and outperforms the frozen baseline throughout. Node B's MSE curve exhibits lower variance than Node A's, consistent with the knowledge distillation literature, where training on a teacher's smooth predictive distributions acts as implicit regularisation relative to noisy ground truth labels.

A simultaneous-training variant confirms that peer supervision remains effective when the anchor is itself still adapting. Node B's loss lags briefly in early epochs, reflecting the time required for Node A to develop an informative predictive signal.

### Stage 3: Naive aggregation baseline

Node A fine-tunes on red, Node C on blue, both against true angles. Node B trains exclusively on the naive average of A's and C's predictions on their shared cell, with no ground truth. Node B is evaluated on red-filtered inputs, placing A in-distribution and C out-of-distribution.

**Reference result.** Node C maintains elevated uncertainty around 0.035 to 0.038 across all filter strengths, reflecting its out-of-distribution status. The frozen baseline reaches approximately 0.07 at full red strength. Node B sits between A and C in MSE, as naive averaging predicts.

**Purpose.** Establishes the cost of averaging anchors of unequal reliability, and confirms that the evidential head detects distributional mismatch without explicit out-of-distribution signalling. The asymmetry between A and C in both MSE and uncertainty is what uncertainty-aware aggregation exists to exploit.

### Stage 4: Weighting and fusion techniques

Scalar weighting strategies (naive, strict-weighted, square-weighted, gated, softmax-weighted, winner-only) are compared against Bayesian fusion.

**Reference result.** Uncertainty-aware weighting consistently improves on naive averaging. Winner-takes-all achieves the strongest absolute performance but is a degenerate case specific to this configuration: since Node B's filter matches Node A's training distribution, winner-takes-all reduces to always selecting Node A, and would fail wherever A is not the dominant anchor. Excluding it, certainty-weighted and softmax-weighted perform best. Squared-weighted exhibits higher variance, consistent with squaring amplifying any miscalibration. Gated thresholding performs worst; the hard threshold is too blunt, falling back to naive averaging when both nodes fall below it and discarding C's signal entirely when only C does.

Bayesian fusion substantially outperforms all scalar strategies at high filter strengths, where uncertainty asymmetry between anchors is most pronounced.

**Purpose.** All scalar weighting approaches share a fundamental limitation: they treat the evidential output as a point estimate with a reliability score, discarding the distributional information encoded in the NIG parameters. This stage establishes that using that information directly produces further gains, and it is the pairwise control against which the corrected N-way implementation is validated.

**Correction.** The `N = 2` case of the new fusion function must reproduce `bayesian_fusion_grid` exactly on identical inputs. This is the hard gate. Divergence here means the N-way mathematics is wrong and no downstream result is interpretable.

### Stage 5: Variable environments

Node A remains fixed on red, Node C on blue, both with true labels. Node B receives no ground truth and trains on either the naive average or the Bayesian fusion of A and C's predictions. Node B's input filter interpolates between red and blue under three regimes: linear transition across 30 epochs, sinusoidal oscillation, and uniform random resampling each epoch.

**Reference result.** Fusion maintains lower MSE than naive averaging throughout the linear and oscillatory regimes, with the gap widening as distributional mismatch grows. At the linear transition midpoint, fusion achieves lower MSE than either Node A or Node C individually. This is only possible because fusion exploits complementary partial knowledge from both anchors: at the midpoint both hold partial but useful representations of Node B's input space, and fusion finds the region of highest joint agreement rather than selecting one anchor or averaging equally.

Under oscillatory conditions, fusion's MSE is slightly elevated during the second red period relative to the first, indicating partial forgetting of the red distribution after adapting toward blue. Under random conditions fusion continues to outperform naive averaging, indicating that the evidential head produces sufficient uncertainty differentiation between anchors even under high-frequency change.

### Stage 6: Spatially structured deployment

A 7×7 field of view at stride 3 places 36 nodes over a 2D grid with an evolving colour-filter condition map and semi-random wearable trajectories, across 390 timesteps.

At each timestep, nodes whose field of view contains a wearable are in-coverage and receive one ground-truth training sample at the wearable's cell. Breadth-first propagation then proceeds outward through the overlap graph. Each out-of-coverage node adjacent to an already-trained node collects that neighbour's beliefs on their shared cells, fuses them, trains on the fused mode, and predicts across its own field of view for downstream neighbours. Propagation completes within two to three hops.

Three sub-conditions of increasing environmental dynamism are run: static, lightly dynamic, and highly dynamic.

**Evaluation.** Mean MSE per hop distance over time; spatial MSE and certainty maps; per-cell certainty against MSE.

**Reference results, static and dynamic.** MSE stabilises around 0.020 against a pretrained baseline of 0.075 rising to 0.100 as conditions shift. Certainty gradients disseminate outward from wearable positions. Under dynamic conditions the certainty map is the superposition of two effects: the wearable coverage gradient, and the distributional distance gradient produced by the filter shift. Nodes in heavily filtered regions are further from the pretrained distribution and correctly express lower certainty.

**The cone-shaped certainty-MSE relationship is the expected signature, not a linear correlation.** High-certainty cells show consistently low MSE with low variance. Low-certainty cells show higher MSE variance: some are predicted well by chance, others poorly, but the system correctly communicates that predictions there are less trustworthy. Certainty functions as an honest upper bound on reliability rather than a point estimate of error, which is the behaviour required for operationally useful uncertainty communication.

**Hop distance is a noisy label.** It reflects current wearable position rather than cumulative training history. A node labelled hop 2 at a given timestep may have received extensive direct calibration earlier. Cumulative in-coverage time per node is the more meaningful metric.

**Self-healing.** When a node's local conditions shift, it receives high-certainty predictions from already-adapted neighbours observing those conditions, driving re-adaptation faster than wearable ground truth alone could achieve. This property emerges from the overlap graph structure: the mesh collectively contains knowledge of all current conditions distributed across its nodes, and belief exchange routes that knowledge to wherever it is needed.

**Collective memory covers the spatial regime only.** A condition currently held by some node can be supplied to others through fusion. A condition that has decayed out of every node cannot. That temporal regime is what EWC addresses, and it is separate. Both are bounded by the same overlap dependence, so collective memory and the fusion threshold are the same constraint viewed twice. The frozen backbone confines all forgetting to the head.

### Stage 6D: Three-way comparison

The central empirical result. Three systems run in lockstep on the highly dynamic environment with identical wearable positions and identical fresh pretrained weight initialisation: frozen pretrained baseline, naive aggregation, and Bayesian fusion belief exchange. The naive system is identical to the Bayesian system in every respect except the aggregation mechanism, isolating the contribution of fusion.

**Prior result, superseded.** Naive aggregation degraded to mean MSE 0.30 to 0.35, three to four times worse than the frozen baseline, while the fusion arm maintained stable low MSE. The naive failure is a systematic consequence of aggregating conflicting updates from nodes observing heterogeneous conditions without uncertainty weighting; the model's weights are pulled simultaneously toward red-condition and blue-condition representations. This is directly analogous to FedAvg's documented failure under non-IID data. Naive averaging without uncertainty awareness is not just suboptimal, it is actively harmful.

**Correction.** The fusion arm ran certainty-weighted scalar averaging, not Bayesian fusion. The measured gap is between two averaging rules. The result must be regenerated under true product-of-experts fusion before it can be reported as a fusion result. The direction of the corrected result is not assumed: Stage 4 validates fusion over averaging at two experts, which supports but does not guarantee the same at nine, where many-expert reinforcement behaviour differs.

---

## 9. Validation Order

Each condition is a separate design decision and must earn its place independently.

1. **Fusion primitive.** Unit-test circular wraparound with beliefs at +179° and −179°. Verify empirically that pre-log scaling leaves the argmax unmoved.
2. **Two-expert control.** Unweighted product-of-experts at `N = 2` reproduces Stage 4c's `bayesian_fusion_grid` output exactly. Hard gate.
3. **N-way unweighted fusion.** The corrected mechanism, no consensus term. Rerun Stage 6D under it. This alone is what Stage 6D should have run.
4. **Overlap threshold.** Regenerate the configuration ablation under true fusion.
5. **Consensus tempering.** Ablate against unweighted fusion. If consensus does not improve fused MSE, drop it and report that contributor disagreement carried no usable signal beyond what the density product already consumes. That is a finding.
6. **Decay.** Sweep global `ρ`. Dynamism-adaptive `ρ` only if constant `ρ` is shown to matter.

Data, wearable paths, node configuration, seed 42, and Stage 6's single-digit-per-cell design are held fixed throughout. The fusion rule and the consensus term are the only changed variables. Any result difference must be attributable to them.

---

## 10. Scope

Excluded deliberately, and recorded as such:

**Parameter exchange.** Nodes share beliefs, not weights. This is the contribution, not an omission.

**Forgetting evaluation and EWC.** The forgetting question and the MCU memory budget are separate determinations. EWC storage scales with trainable parameter count, and the backbone is frozen, so the cost is head-scoped and small. The binding constraint is estimability: the diagonal Fisher is an empirical estimate, and under the single-digit-per-cell design the per-condition sample counts are small. The sparsity that threatens the Fisher estimate is the same sparsity that collapses fusion below the overlap threshold.

**Water velocity deployment.** No accessible labelled dataset exists within the project window. Rotated colour-filtered digits serve as the controlled proxy, justified as a continuous regression task under induced distributional shift. The disaster response scenario motivates the design constraints; it is not itself demonstrated.

**Heterogeneous sensor modalities and multi-platform deployment.** Descoped in favour of depth on the exchange mechanism.

**Information-theoretic framework.** The cascade channel model, data processing inequality bound, and feasibility boundary remain unvalidated and are not claimed as contributions.

**Multi-anchor validation.** Multi-anchor operation is architecturally motivated but the experiments run a single wearable path per configuration.
