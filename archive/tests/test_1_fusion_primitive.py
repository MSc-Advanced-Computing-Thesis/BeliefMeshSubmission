# Experiment Specification Sec 9, validation order step 1: fusion primitive.
#
# - Unit-test circular wraparound with beliefs at +179 deg and -179 deg
#   (must fuse to a single sharp mode near +/-180, not two disjoint modes).
# - Verify empirically that pre-log scaling of a density leaves the argmax
#   of the fused log-density sum unmoved (Spec Sec 6.2).
