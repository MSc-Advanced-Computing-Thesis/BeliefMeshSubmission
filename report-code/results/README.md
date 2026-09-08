# results: every figure and table of the report

One directory per report section, numbered and titled as in the report. Each
holds a `reproduce.py` that regenerates that section's figures and tables from
the stored artefacts, the experiment scripts that produced those artefacts, and
a short README. Shared code lives in `_shared/`.

## Two modes

```
python results/make_all.py            # DEFAULT: regenerate every figure and table from artefacts/  (~15 min, CPU)
python results/make_all.py --only 5.4 D
python results/make_all.py --list
```

The default mode never trains anything. It reads the stored runs under
`artefacts/` and writes `results/figures/` and `results/tables/`.

Re-running the experiments is separate and explicit:

```
python results/make_all.py --rerun-experiments --i-understand-this-takes-days
python results/5.6_uncertainty_guided_measurement/reproduce.py --rerun         # one section
```

This overwrites the artefacts of the sections it runs and needs roughly forty
GPU-hours across all sections (the per-section estimate is in each
`reproduce.py --help`). It is guarded by the second flag so it cannot happen by
accident.

Run order matters only in that some sections read another section's runs (they
are listed under "reads" below); in default mode nothing needs to be run first.

## Figure and table map

| Report item | Regenerated file | Script (under `results/`) | Artefacts (under `artefacts/`) |
|---|---|---|---|
| Fig 3.2 fusion rules | `fig_3_2_fusion_rules.png` | `chapter_3_4_design_figures/figure_3_2_fusion_rules.py` | none (computed) |
| Fig 4.1 digit rotations | not regenerated | no script in the working tree produced it | none |
| Fig 4.2 colour / offset worlds | `fig_4_2_environments.png` | `chapter_3_4_design_figures/figure_4_2_environments.py` | none (computed) |
| Fig 4.3 mesh geometry | `fig_4_3_mesh_geometry.png` (+ `_annotated`) | `chapter_3_4_design_figures/figure_4_3_mesh_geometry.py` | none (computed) |
| Fig 5.1 angle response | `fig_5_1_angle_response.png` | `5.1_.../figure_5_1_angle_response.py` | `5.1_.../{r1a_held_out_evaluation,f_angle_bins,j_single_instance}` |
| Fig 5.2 calibration | `fig_5_2_calibration.png` | `5.1_.../figure_5_2_calibration.py` | `5.1_.../r1a_held_out_evaluation` |
| Fig 5.3 colour and offset sweeps | `fig_5_3a_colour_sweep.png`, `fig_5_3b_offset_sweep.png` | `5.1_.../figure_sweeps.py` | `5.1_.../{r1b_colour_sweep,r1c_offset_sweep}` |
| Section 5.1 prose numbers | `section_5_1_prose_numbers.txt` | `5.1_.../reproduce.py` | as above |
| Fig 5.4 peer supervision | `fig_5_4_peer_supervision.png`, `table_5_4_peer_supervision.csv` | `5.2_peer_supervision/reproduce.py` | `5.2_.../two_node_offset_world` |
| Fig 5.5 three-node geometry | `fig_5_5_three_node_geometry.png` | `5.3_.../5.3.1_.../figure_three_node_geometry.py` | none (geometry) |
| Table 5.1 | `table_5_1_three_node_aggregation.csv` | `5.3_.../5.3.1_.../table_three_node_results.py` | `5.3_.../5.3.1_.../three_node_colour_world` |
| Fig 5.6 mesh snapshots | `fig_5_6_mesh_snapshots.png` | `5.3_.../5.3.2_mesh_scale/figure_mesh_snapshots.py` | `5.3_.../5.3.2_mesh_scale/snapshot_run` |
| Table 5.2 | `table_5_2_aggregation_mesh_scale.csv` | `5.3_.../5.3.2_mesh_scale/reproduce.py` | `5.3_.../5.3.2_mesh_scale/{aggregation_arms,frozen,average_fusion}` |
| Table 5.3 | `table_5_3_parameter_exchange.csv` (+ `_as_published`) | `5.4_.../reproduce.py` | `5.4_.../five_environments` |
| Fig 5.7 exchange trajectories (field2, field3) | `fig_5_7_exchange_trajectories.png` (+ `_as_published`) | `5.4_.../reproduce.py` | `5.4_.../five_environments` |
| Fig 5.8 mesh density | `fig_5_8_mesh_density.png` | `5.5_mesh_geometry/reproduce.py` | `5.5_.../mesh_density_sweep` |
| Table 5.4 hop calibration | `table_5_4_hop_calibration.csv` | `5.5_mesh_geometry/reproduce.py` | reads `5.3_.../5.3.2_mesh_scale/aggregation_arms` |
| Fig 5.9 routing | `fig_5_9_routing.png`, `table_5_9_routing.csv` | `5.6_.../reproduce.py` | `5.6_.../{routing_product_fusion,routing_average_fusion}` |
| Table 5.5 | `table_5_5_mixed_vs_uniform.csv` | `5.7_heterogeneous_devices/reproduce.py` | `5.7_.../{mixed_capacity/nig_product,uniform_narrow,uniform_baseline,uniform_wide}` |
| Fig 5.10 heterogeneity | `fig_5_10_heterogeneity.png`, `figure_5_10_numbers.txt` | `5.7_heterogeneous_devices/reproduce.py` | `5.7_.../mixed_capacity`; reads 5.3.2 for the homogeneous arms |
| Fig 5.11 coverage maps | `fig_5_11_coverage_maps.png` | `5.8_.../5.8.1_node_failure/reproduce.py` | `5.8_.../5.8.1_.../node_loss` |
| Fig 5.12 node loss | `fig_5_12_node_loss.png` | `5.8_.../5.8.1_node_failure/reproduce.py` | `node_loss` + the two derived CSVs from `analysis_node_loss_*.py` |
| Table 5.6, Table 5.7 | `table_5_6_miscalibration.csv`, `table_5_7_miscalibration_by_distance.csv` | `5.8_.../5.8.2_.../reproduce.py` | `5.8_.../5.8.2_.../miscalibrated_wearable`; reads 5.3.2 for the baseline |
| Table 5.8 ablations | `table_5_8_ablation_summary.csv` (+ `_variants.txt`) | `5.9_ablations/reproduce.py` | `5.9_ablations/*`; reads 5.3.2 for the weighted-fusion "off" arm |
| Table C.1, Table C.2 | `table_C_1_deployment_configuration.csv`, `table_C_2_deployment_performance.csv` | `appendix_C_.../reproduce.py` | `appendix_C_.../deployment_66x66`; reads 5.3.2 for the reference |
| Fig D.1 | `fig_D_1_exchange_trajectories_all.png` (+ `_as_published`) | `appendix_D_.../reproduce.py` | `5.4_.../five_environments` |
| Table E.1 | `table_E_1_heterogeneous_capacity.csv` | `appendix_E_.../reproduce.py` | `appendix_E_.../mixed_capacity_five_environments` |
| Table F.1 | `table_F_1_lambda_sweep.csv` | `appendix_F_.../reproduce.py` (via 5.9) | `5.9_ablations/lambda_*` |

Every table is written as CSV plus a plain-text rendering with the numbers as
printed. Figures are PNG only, 6.3 in wide at 200 dpi (Figures 5.10 and 5.12
are 7.6 in, as in the report).

## Readout convention

All cell-space numbers use the averaged-NIG readout of `_shared/averaged_readout.py`:
every belief covering a cell is fused under Average fusion (weights 1/N) and
the cell's error is that of the fused mode against the reconstructed ground
truth (`_shared/truth.py`), over the last-50 window unless stated. Ground truth
is reconstructed from the run's seed and its own offset environment.

Two outputs are also written in an `_as_published` variant (Table 5.3, Figure
5.7, Figure D.1). The report's versions of these scored every run of the five
environments against the field0 environment; the plain files score each run
against the environment it was generated with. Both are kept so the published
numbers remain reproducible and the difference is inspectable. The same
applies to the gradient-tempering row of Table 5.8, noted in its text file.

## Experiment scripts

`experiment_*.py` in each section are the scripts that generated the stored
runs, ported to this layout; `reproduce.py --rerun` invokes them with the
arguments used. Each takes `--root` (output directory) and, where applicable,
`--seeds`. Several also take `--verify`, which checks their configuration
without training. `analysis_*.py` are intermediate computations whose outputs
are stored alongside the runs.
