# Stage 0: Baseline model and calibration. Experiment Specification Sec 8.
#
# Train the NIG head on unfiltered rotated digits with true angle labels.
# Establishes the pretrained model used by every later stage.
#
# Evaluation: held-out circular MSE, and Pearson correlation between
# predicted uncertainty and absolute error.
# Reference result: MSE 0.0106 (normalised angle space, ~18.5 deg), Pearson 0.5649.
