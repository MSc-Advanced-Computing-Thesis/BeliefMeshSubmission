# Stage 6a-c: Spatially structured deployment + overlap threshold ablation.
# Experiment Specification Sec 7-8 & Sec 9 step 4.
#
# 7x7 FOV stride 3 (36 nodes, max coverage 9) vs both 5x5 configurations
# (max coverage 4 and 9), under static / lightly dynamic / highly dynamic
# filter conditions. Must be regenerated under true fuse() -- the prior
# 9-way overlap threshold was measured under scalar averaging and is
# provisional (Spec Sec 7, "Status: provisional").
