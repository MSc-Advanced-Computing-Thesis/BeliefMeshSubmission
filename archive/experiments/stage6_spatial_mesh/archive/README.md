# Archive

Superseded scripts, kept for provenance rather than deleted, because each one
documents a step in the Stage 4 narrative written up in
`THESIS_EXPERIMENTS_SUMMARY.md`. The current, load-bearing script is
`../run_stage4_dynamic_world_final.py`.

| script | documents | doc section |
|---|---|---|
| `run_priority_batch.py` | 8-run batch on the STATIC "shadow world" (superseded once the temporal-staticness flaw was identified and the dynamic offset world replaced it) | superseded pre-Stage-4 work, not cited in the current doc |
| `run_dynamic_offset_test.py` | the decisive test: fusion vs global on the first dynamic-wake field, single wearable, `epistemic_staleness` routing — reversed Christian's prediction | §4.1 |
| `run_routing_fix_test.py` | discovered the wearable-routing confound (fusion camped 100% in one quadrant) and introduced the anti-camping `policy_cooldown` mechanism | §4.2 |
| `run_cooldown_sweep.py` | swept cooldown 15/30/60/100 — diminishing, non-monotonic returns, motivated abandoning routing tuning | §4.2 |
| `run_multi_wearable_test.py` | first multi-wearable attempt: 3 wearables under the same online routing policy, still clumped onto one location | §4.3 (first half) |
| `run_random_wander_test.py` | the fix: 3 independent random walks, no routing — parity with global on the (then-undiagnosed) flattening-biased field. Also the original home of `build_random_wander_path`, since moved to `run_offset_experiments.py` so `run_stage4_dynamic_world_final.py` doesn't depend on this archived file | §4.3 (second half) / §4.4 (old-field number) |
| `run_static_replay_test.py` | dead end — interrupted mid-run when Christian redirected to the 3-random-wearable design instead; never produced a result | not cited |

`build_random_wander_path()` itself is NOT archived — it was promoted to
`run_offset_experiments.py` alongside `build_dynamic_offset_field()` since
it's a reusable environment-generation utility, not a one-off experiment.

The RESULTS each of these scripts produced live in
`runs/stage6/offset_world/dynamic/archive/`, grouped into the same four
numbered steps (01-03 + `old_field/`) — see that folder's own README for the
full numbers.
