# MDP-04 original-host refresh streaming

Status: **archived complete computation / formal `numeric_gate_failed`**.

This directory is retained to reproduce MDP-04, not to launch MDP-05. The
frozen scientific worker, metrics, validator, contract, tests, and pinned
Triton source remain the MDP-04 implementation. `run_stream_replay.py` differs
from the accepted `source_snapshot_v4` only by a post-validation CLI cleanup:
an expected failed formal gate now exits concisely instead of raising a
`CalledProcessError` traceback. The immutable source snapshot inside the result
directory remains authoritative for byte-exact historical reproduction.

This directory implements the matrix-level continuation of accepted experiment
37 without modifying experiment 37, its checkpoints, or `HANDOFF.md`.

The controller reuses the accepted MECH-09R worker for initialization, source
hash checks, fresh preconditioner build, data offsets, RNG state, compilation,
branch snapshot/restore, and optimizer execution.  It computes only the two
pre-registered boundaries:

- production refresh at completed step 32;
- delayed refresh at completed step 64.

Each origin--replica unit therefore computes 65 real optimizer steps.  At each
boundary, all 18 down-projection layers are processed one at a time.  Full K,
runtime inverse, raw gradient, momentum, and update tensors are not saved.
Only scalar metrics, 17-point cross-run drift diagnostics, and six registered
128-coordinate validation slices are persisted.

## Authoritative remote paths

```bash
REPO=${SNM_REPO}
RESULTS=${SNM_RESULTS_ROOT}
EX37=${RESULTS}/37_mech09_downproj_refresh_mediation/20260728T075907+0000
CTRL_PY=${SNM_CONTROLLER_PYTHON}
CHILD_PY=${SNM_TRAINING_PYTHON}
MDP=${REPO}/scripts/mdp_refresh_streaming
```

The accepted official/data root is
`${SNM_OFFICIAL_REPO}`, not the `official-r0` path used
by some later R1 experiments.

## Upload and run

```bash
# Locally, synchronize these two project paths to the same paths on the host:
#   scripts/mdp_refresh_streaming/
#   commands/mdp04_refresh_streaming/20260803_mdp04_refresh_streaming.sh

cd "${REPO}"
bash commands/mdp04_refresh_streaming/20260803_mdp04_refresh_streaming.sh check
bash commands/mdp04_refresh_streaming/20260803_mdp04_refresh_streaming.sh dry-run
MDP04_RUN_DIR=/absolute/result/path \
  bash commands/mdp04_refresh_streaming/20260803_mdp04_refresh_streaming.sh archive-verify
```

`formal` and `resume` are disabled by default because MDP-04 is closed. An
explicitly authorized historical rerun must additionally set
`MDP04_ALLOW_ARCHIVAL_RERUN=1`; such a rerun cannot promote the accepted
evidence or change the frozen threshold.

The command wrapper fixes the accepted experiment-37 input path and runtime,
runs ten tests, checks both H100 lanes, applies physical GPU locks, and prints
the exact formal result directory.  No source archive is required.

## What the three commands do

- `check`: verify synchronized hashes, run the ten frozen framework tests, and
  run three dependency-free archive-inspector tests;
- `dry-run`: validate the accepted sources, runtime, and sealed 12-unit plan;
- `formal`: run the registered canary, the remaining two-GPU replay, and the
  final validator, only after explicit archival-rerun acknowledgement;
- `verify`: require `passed=true` and exit cleanly if the result did not pass;
- `archive-verify`: verify that a complete archived run reproduces the expected
  `numeric_gate_failed` adjudication without treating it as an engineering
  crash.

All experiment-37 artifacts that exist on the original host and feed the replay
require exact SHA-256 matches.  The `independent_review/` directory was created
locally after the experiment-37 handoff, is absent from the original host, and
is therefore recorded only as `local_posthoc_audit_reference` in the contract.
It is not a remote input and is not used by the worker or final validator.

The accepted experiment-37 Triton source is also included byte-for-byte under
`pinned_ex37_runtime/`.  Every formal worker rewrites the mutable upstream
`--triton-kernels` argument to that sealed source and checks its size, SHA-256,
and source-snapshot manifest entry before importing the training runtime.  The
MECH-09R repair contract and MECH-08 checkpoint-certificate reference are also
copied into the same immutable source snapshot and used from there.

### Cross-run replay amendments v3 and v4

The first real H100 canary failed before any MDP-04 layer metric was accepted:
its structure, 697-tensor count, next batches, loader position, matrix step,
source/runtime contracts, and displayed step-16 loss matched experiment 37,
while only the aggregate floating-state SHA differed.  That SHA was originally
an exact within-process branch snapshot/restore audit, not a portable
`torch.compile`/Triton binary certificate.

Contract v3 kept exact within-replay snapshot/restore hashes and
exact cross-run structure/data/step/source checks.  The old aggregate branch
SHA was retained as a diagnostic.  A later `late_newton_full/replica_1`
fresh-process replay then matched the checkpoint, source, loader/data,
structure, and step boundary but exceeded the v3 elementwise 17-point
tolerance before any event or layer metric passed.  A covariance discrepancy
of about `1.48e-5` was amplified by the 5504-dimensional Cholesky inverse to
about `3.38e-2`; this demonstrated that old element values are also not a
portable fresh-process certificate.

Contract v4 does not increase or tune that tolerance.  It records the original
v3 comparison, hashes, absolute error, and relative error as diagnostics, while
hard-gating their metadata, sample count, and finiteness.  Exact
checkpoint/source/data/step fields, exact within-replay snapshot/restore,
event-scoped hooks, current-replay matrix identities and residuals, and
shadow-to-actual gradient/NS5 fingerprints remain hard gates.  Stream hooks
are installed only for the registered step-32 and step-64 refresh events and
must be absent at both branch anchors.

Contract v4 supports one explicit migration from the byte-exact v3 contract.
The original `source_snapshot/` remains immutable, a new
`source_snapshot_v4/` is sealed, and only v3 units with a selected passed
manifest are inherited.  Those units satisfied the stricter v3 numeric gate;
failed or incomplete attempts are never selected and receive a new v4 attempt.
The final validator reports both lineages.  Contracts v1/v2 and any unlisted
contract hash remain non-resumable.

The dry-run terminal suffix must be:

```text
Ran 10 tests ... OK
MDP-04 dry run passed; scheduled jobs: 12
```

## Resume and verify

The formal command prints `MDP04_RUN_DIR=...` before computation starts.  Reuse
that exact path after an interruption:

```bash
MDP04_RUN_DIR=/absolute/result/path \
  bash commands/mdp04_refresh_streaming/20260803_mdp04_refresh_streaming.sh resume
```

Passed units are never overwritten; a retry receives a new `attempt_NNN`
directory.  After completion, verify the manifest read-only with:

```bash
MDP04_RUN_DIR=/absolute/result/path \
  bash commands/mdp04_refresh_streaming/20260803_mdp04_refresh_streaming.sh verify
```

For the registered v3-to-v4 repair, use the same v3 formal result path with
`resume` after synchronizing v4.  The controller writes
`resume_contract_migration_v3_to_v4.json`; do not edit or delete the original
snapshot or passed unit selections.

## Completion gate

The controller automatically runs the final validator.  No separate analysis
command is required.  The final manifest must report 432 layer-event rows, 24
unit-event summaries, 12 accepted unit outcome rows, six validation slices,
and `passed=true`:

The accepted `20260803T063912+0000` run completes all coverage but does not meet
this gate. Its correct reproducibility target is therefore
`archive-verify -> numeric_gate_failed`, not `verify -> passed`.

## Result handoff

After `verify` passes, copy or download the printed formal result directory in
the same way as the earlier experiment directories.  Send that directory to
the local analysis workspace; no `.tar` handoff step is prescribed here.

## Independent local analysis

The completed handoff is analyzed read-only with `analyze_stream_replay.py`.
The output must be a sibling directory, not a child of the immutable handoff:

```powershell
python scripts/mdp_refresh_streaming/analyze_stream_replay.py `
  --run-dir ${SNM_RESULTS_ROOT}/_shared/analysis/method_deepening_mdp04_refresh_replay/20260803T063912+0000 `
  --output-dir ${SNM_RESULTS_ROOT}/_shared/analysis/method_deepening_mdp04_refresh_replay/20260803T063912+0000_local_analysis `
  --handoff-zip ${HANDOFF_ZIP}
```

The analyzer re-hashes all summary artifacts, selected unit manifests, unit
selections, and scientific unit files; checks exact coverage; separates the
frozen formal adjudication from descriptive evidence; and emits deterministic
CSV, JSON, and Markdown audit artifacts. Correlations are computed over the 12
nested origin--replica units per event, with within-origin and
leave-one-origin-out diagnostics. They are never reported as 432 independent
layer observations or as seed-level inference.

For the completed `20260803T063912+0000` run, all computation and scientific
file integrity checks pass, but six late-stage layer-3 rows exceed the frozen
resolvent hard gate. The formal result is therefore `numeric_gate_failed`; the
descriptive matched-G/NS5 alignment does not change that adjudication.

## Clean reproduction sequence

On the original host:

```bash
cd ${SNM_REPO}
bash commands/mdp04_refresh_streaming/20260803_mdp04_refresh_streaming.sh check
MDP04_RUN_DIR=${SNM_RESULTS_ROOT}/_shared/analysis/method_deepening_mdp04_refresh_replay/20260803T063912+0000 \
  bash commands/mdp04_refresh_streaming/20260803_mdp04_refresh_streaming.sh archive-verify
```

For an exact audit of what actually ran, use the files below the result's
`source_snapshot_v4/`; do not reconstruct the old v1--v4 repair sequence from
the mutable working tree. For a local statistical rebuild, run
`analyze_stream_replay.py` into a new sibling output directory. Neither route
modifies the archived result or `HANDOFF.md`.

## Scientific boundary

- 432 layer rows are nested repeated measurements, not 432 independent seeds.
- Full-state Frobenius, matched-gradient, and source NS5 metrics are exact
  reductions of the actual runtime tensors.
- Condition, operator norm, and full resolvent metrics are fixed-probe proxies.
- Exact float64 inverse/resolvent/SVD calculations are confined to registered
  small slices and are implementation calibration only.
- Accepted step-48, step-80, and AUC losses are joined read-only from experiment
  37; the replay does not replace them.
- Cross-run floating hashes and 17-point value differences are reproducibility
  diagnostics, not proof of bitwise identity and not scientific pass/fail
  gates.  Their maxima and original-v3-tolerance status remain in the final
  manifest.
