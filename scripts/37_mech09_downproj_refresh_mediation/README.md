# MECH-09 / MECH-09R: down-projection refresh-timing mediation

## Active protocol: MECH-09R

The active confirmatory protocol is `MECH-09R`, contract version
`2026-07-28.2`. It repairs a pre-intervention equivalence failure in the
original independent-worker design without relaxing its threshold or reading
post-treatment outcomes into the repair design.

Each checkpoint-origin x data-replica unit is now one causal-tree worker:

- all three arms share one computed trajectory through completed step 31;
- internal production branches at step 32 and refreshes `down_proj`;
- delayed and frozen share one no-`down_proj`-refresh trajectory through
  completed step 63;
- delayed branches and refreshes at step 64, while frozen continues without
  any `down_proj` refresh.

The two branch states include model parameters and buffers, both optimizer
states, the next training batch, loader state, RNG states, and matrix global
step. Restores are checked with fixed-sample hashes over every named state
tensor plus exact next-batch and loader-state checks.

The formal matrix is 4 origins x 3 data replicas = 12 causal-tree workers.
Each worker exposes three logical arms: internal production, delayed refresh,
and frozen refresh.

The sole active remote entry point is:

`commands/37_mech09_downproj_refresh_mediation/20260728_mech09r_causal_tree_repair.sh`

Active files use the `mech09r_*` prefix and
`refresh_mediation_repair_contract.json`.

Accepted formal result:

- run `20260728T075907+0000`;
- 12/12 causal-tree workers passed;
- independent audit: 298/298 checks passed;
- scientific classification: `full_support`;
- durable review:
  `docs/reports/20260729_mech09r_result_review.md`.

`audit_mech09r_handoff.py` independently reconstructs the contrasts and AUC
from raw worker CSVs and verifies the archive, manifests, shared prefixes,
branch restores, refresh schedules, and aggregate outputs.

## Legacy invalid protocol

MECH-09 is the single targeted branch authorized by the MECH-08 decision gate.
It asks whether the production `down_proj` full-K refresh causally creates the
post-refresh degradation observed for original Newton-Muon.

It is not a Selective-diag versus Selective-none experiment.

## Frozen design

MECH-09 reuses the 48 hash-frozen MECH-08 control workers and adds only:

- `delayed_down_refresh`: skip the `down_proj` refresh at step 32, then refresh
  at steps 64, 96, and 128;
- `frozen_down_refresh`: keep the freshly built initial `down_proj` K fixed for
  all 128 steps.

Attention-input, attention-output, and gate/up K groups retain the production
refresh schedule at steps 32, 64, 96, and 128. Skipped `down_proj` activation
statistics are zeroed at the event boundary so they cannot leak into a later
refresh.

The formal matrix contains:

- four checkpoint origins;
- three matched data-order replicas;
- two new interventions.

This is exactly 24 new formal workers. No additional intervention may be added
after outcomes are observed.

## Causal prediction

Full support requires all of the following frozen patterns:

1. both interventions match the production control before the first refresh;
2. delayed and frozen refresh both protect against the step-32 production
   refresh by evaluation step 48;
3. delayed and frozen remain matched at step 48;
4. after the delayed arm refreshes at step 64, it becomes worse than the
   frozen arm by evaluation step 80.

The analysis integrity pass is separate from the scientific classification
(`full_support`, `partial_support`, `not_supported`, or `invalid`).

## Files

- `refresh_mediation_contract.json`: frozen question, schedules, endpoints,
  decision rules, stopping rule, and 24-job cap.
- `mech08_control_reference.json`: SHA256 inventory for every reused MECH-08
  control file and checkpoint hash certificate.
- `build_mech08_control_reference.py`: deterministic reference-manifest builder.
- `mech09_worker.py`: one origin/intervention/replica rollout.
- `run_mech09.py`: preflight, checkpoint certificate reuse, smoke gate,
  two-GPU scheduling, resume, and final analysis.
- `analyze_mech09.py`: paired timing-mediation analysis against MECH-08.
- `test_mech09_contract.py`: CPU-only contract and reference tests.
- `test_mech09_worker.py`: tensor-level tests for surgical refresh handling.

The legacy remote entry point was:

`commands/37_mech09_downproj_refresh_mediation/20260728_mech09_downproj_refresh_mediation.sh`

Do not launch that entry point for confirmatory work. Run
`20260728T023217+0000` failed its frozen pre-intervention equivalence gate:
delayed versus frozen already differed at step 16. Its 24 workers are retained
only as `invalid_pre_intervention_equivalence` implementation evidence.

Outputs are written directly under:

`${SNM_RESULTS_ROOT}/37_mech09_downproj_refresh_mediation/<timestamp>`

No archive workflow is part of this experiment. Worker timing and CUDA
allocation remain diagnostic and are not paper efficiency evidence.
