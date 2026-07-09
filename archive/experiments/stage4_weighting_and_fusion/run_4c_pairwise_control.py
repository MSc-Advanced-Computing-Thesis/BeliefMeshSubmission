# Stage 4c: Pairwise control. Experiment Specification Sec 8 & Sec 9 step 2.
#
# HARD GATE: fuse() at N=2 unweighted must reproduce the prior
# bayesian_fusion_grid output exactly on identical inputs. Divergence here
# means the N-way fusion math is wrong and no downstream Stage 4-6 result is
# interpretable. See tests/test_2_pairwise_control.py for the actual assertion --
# this script is for generating the comparison result/plot, not the gate itself.
