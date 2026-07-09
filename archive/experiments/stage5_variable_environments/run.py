# Stage 5: Variable environments. Experiment Specification Sec 8.
#
# Node A fixed red, Node C fixed blue, both true labels. Node B trains on
# naive average or Bayesian fusion of A/C, no ground truth. Node B's input
# filter interpolates red<->blue under three regimes: linear (30 epochs),
# sinusoidal, uniform random resampling per epoch.
