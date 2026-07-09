# Experiment Specification Sec 9, validation order step 2: two-expert control.
# HARD GATE.
#
# Unweighted product-of-experts at N=2 must reproduce Stage 4c's prior
# bayesian_fusion_grid output exactly on identical inputs. If this fails,
# the N-way fusion mathematics is wrong and no downstream result is
# interpretable -- do not proceed to step 3 until this passes.
