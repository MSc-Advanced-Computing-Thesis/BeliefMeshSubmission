# Archive

Superseded scripts, kept for provenance rather than deleted, because each one
documents a step in the Stage 4 / Stage 6 narrative written up in
`THESIS_EXPERIMENTS_SUMMARY.md`. The current, load-bearing scripts for the
dynamic-world result are `../run_dynamic_v2_test.py` and
`../run_dynamic_v2_5seed.py` (Stage 6 fixes applied; see below).

| script | documents | doc section |
|---|---|---|
| `run_priority_batch.py` | 8-run batch on the STATIC "shadow world" (superseded once the temporal-staticness flaw was identified and the dynamic offset world replaced it) | superseded pre-Stage-4 work, not cited in the current doc |
| `run_dynamic_offset_test.py` | the decisive test: fusion vs global on the first dynamic-wake field, single wearable, `epistemic_staleness` routing — reversed Christian's prediction | §4.1 |
| `run_routing_fix_test.py` | discovered the wearable-routing confound (fusion camped 100% in one quadrant) and introduced the anti-camping `policy_cooldown` mechanism | §4.2 |
| `run_cooldown_sweep.py` | swept cooldown 15/30/60/100 — diminishing, non-monotonic returns, motivated abandoning routing tuning | §4.2 |
| `run_multi_wearable_test.py` | first multi-wearable attempt: 3 wearables under the same online routing policy, still clumped onto one location | §4.3 (first half) |
| `run_random_wander_test.py` | the fix: 3 independent random walks, no routing — parity with global on the (then-undiagnosed) flattening-biased field. Also the original home of `build_random_wander_path`, since moved to `run_offset_experiments.py` | §4.3 (second half) / §4.4 (old-field number) |
| `run_static_replay_test.py` | dead end — interrupted mid-run when Christian redirected to the 3-random-wearable design instead; never produced a result | not cited |
| `run_stage4_dynamic_world_final.py` | **SUPERSEDED 2026-08** -- was "THE current Stage 4 result" (fusion whole_run=0.0252 vs global 0.0228, ~10-33% gap depending on seed averaging) on the pre-CoordConv-fix architecture. Same field/task, rerun with the Stage 6 fixes (CoordConv, angle exclusion, calibration-favouring lam=5) in `run_dynamic_v2_test.py` -- the finding REVERSED: fusion now beats fedavg_global, confirmed significant across 5 seeds in `run_dynamic_v2_5seed.py` (p=0.0011 with uncertainty_guided routing, p=0.054 with plain random_wander). `experiments/_common/run_stage6_core_seed_repeats.sh` still points at this archived path and remains runnable as a historical record. | Stage 4 (superseded) / Stage 6 dynamic-world result |

`build_random_wander_path()` itself is NOT archived — it was promoted to
`run_offset_experiments.py` alongside `build_dynamic_offset_field()` since
it's a reusable environment-generation utility, not a one-off experiment.

The RESULTS each of these scripts produced live in
`runs/stage6/offset_world/dynamic/archive/`, grouped into the same four
numbered steps (01-03 + `old_field/`) — see that folder's own README for the
full numbers.
