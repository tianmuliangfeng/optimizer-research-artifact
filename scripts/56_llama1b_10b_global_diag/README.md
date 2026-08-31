# Experiment 56: LLaMA-1B 10B-token global diagonal

This is the long-token extension of the fixed `global_diag` route accepted in
Experiment 52.  It applies a coordinate-diagonal activation second moment to
every eligible matrix and matches the LLaMA-1B profile, FineWeb stream,
batching, learning rates, forked 1,800-step cooldowns, validation cadence, and
three formal seeds of Experiment 48.

## Frozen design

- Formal arm: `global_diag` only.
- Formal seeds: 2024, 2025, 2026.
- Preflight regenerates all three initial models and requires their parameter
  hashes to equal the accepted Experiment 48 seed-specific initialization
  hashes before any training is allowed.
- Primary endpoints: 3.2505856B, 6.969360384B, and 9.999745024B tokens.
- Engineering seed: 5601; it tests interruption, exact in-place resume,
  source-checkpoint branching, and no-wrap loader state. It is not used for
  method selection or scientific outcomes.
- Every formal unit starts from initialization. Experiment 52 checkpoints and
  its 50-shard view are explicitly outside this contract.
- The FineWeb stream must match the stable name/size/content projection of the
  103-train-plus-one-validation shard inventory accepted for Experiment 48.
  Paths and mtimes may differ after a byte-for-byte copy; content may not.
- The 36 frozen Experiment 48 endpoint rows are read-only paired controls. The
  original accepted endpoint table is identified in `formal_contract.json`;
  `frozen_ex48_controls.csv` is a path-free compact projection of those rows.
- The authorized run is frozen to physical GPU 3 on the four-GPU long-budget
  host; its three formal seeds run serially as single-GPU processes.  Physical
  GPUs 0, 1, and 2 are reserved for the three Experiment-57 Moonlight seeds. No
  DDP or batch-geometry change is introduced. Timing remains ineligible because
  this orchestration differs from the accepted Experiment 48 run.

The global-diagonal trainer is deterministically derived from the accepted
Experiment 17 trainer by the same hash-pinned builder used in Experiment 52.
The long-token segment worker inherits the tested Experiment 48 checkpoint,
cursor, phase-branching, cooldown, and retirement semantics.

## Stages and recovery

The command wrapper supports `check`, `preflight`, `pilot`, `formal`, `resume`,
`verify`, `upload`, and `all`.  `all` is restart-safe at stage boundaries: once
a formal suite plan exists it invokes `resume`, and passed phases/units are
skipped after their hashes and manifests are checked. A mid-phase interruption
continues from that phase's atomic `checkpoint_latest.pt` and trims metrics to
the checkpoint boundary.  The explicit `resume` mode advances formal work but
does not issue the final acceptance receipt; run `verify` afterwards, or rerun
`all` on the same `EX56_RUN_DIR` to resume and verify in one command.

The formal run retains nine endpoint checkpoints (three budgets by three
seeds). Fork checkpoints are deleted only after all direct children pass, and
each deletion receives a retirement certificate. Do not delete the endpoint
checkpoints before the final `verify --full-checkpoint-hash` receipt is saved.

## Primary outputs

- `analysis/endpoint_results.csv`: nine new global-diagonal endpoints.
- `analysis/unified_endpoint_results.csv`: frozen controls plus the new arm.
- `analysis/paired_contrasts.csv`: seed-paired global-minus-control contrasts.
- `analysis/classification.json`: prespecified 10B directional summary.
- `analysis/analysis_manifest.json` and `handoff_manifest.json`: integrity and
  transfer manifests.

Local CSV/JSON files are primary evidence. W&B is a secondary mirror and an
upload failure does not invalidate a scientifically complete local run.
