# Stage 3: Naive aggregation baseline. Experiment Specification Sec 8.
#
# Node A on red, Node C on blue, both against true angles. Node B trains on
# the naive average of A and C's predictions, no ground truth, evaluated on
# red-filtered inputs (A in-distribution, C out-of-distribution).
