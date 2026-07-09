# Experiment Specification Sec 9, validation order step 5: consensus tempering.
#
# Ablate consensus-weighted fusion against unweighted fusion. If consensus
# does not improve fused MSE, that is a reportable finding (Spec Sec 9) --
# not a bug to chase -- so this test should compare, not just assert "better".
