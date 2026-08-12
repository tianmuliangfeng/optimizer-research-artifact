# Experiment 46 / MDP-05 confirmatory update-shock mediation

MDP-05 is a new, outcome-independent confirmation run. It is not a repair,
resume, or reclassification of MDP-04. MDP-04 remains `numeric_gate_failed`.

The implementation reuses the accepted experiment-37 checkpoint loader,
optimizer-state transfer, causal-tree code, and pinned Triton source, but it
derives a new execution contract before any formal outcome is opened:

- four accepted checkpoint origins;
- three new data replicas (`3/4/5`), with optimizer-step offsets
  `768/1024/1280`;
- new preconditioner-build and evaluation token intervals;
- production refresh at step 32, delayed refresh at step 64, frozen control;
- all branches run through step 80 and generate new endpoint/AUC outcomes;
- all 18 down-projection layers are retained, including layer 3;
- matched-G and the actual source-pinned NS5 update are the primary mediators;
- resolvent probes remain reported diagnostics, not a hard gate for this claim.

## Why this avoids the MDP-04 failure modes

- CUDA/runtime preflight is executed with the training venv, never system
  Python.
- The accepted independent-audit file is not required as a remote input.
- Every run gets an immutable source snapshot and one contract hash; a changed
  contract must use a new run directory. The sole code-repair path is gated to
  the known pre-outcome NS5 step-boundary failure, zero accepted formal units,
  and no opened analysis. It preserves the original snapshot and plan, seals a
  second snapshot/plan plus an activation manifest, and starts new attempts.
- The accepted Triton file is copied into the sealed snapshot.
- Cross-process floating-point hashes and 17-value numerical samples are not
  hard gates. Exact fingerprints are used only inside the same replay process
  for shadow-to-actual gradient and NS5 checks.
- A failed unit is never overwritten. Resume creates `attempt_002`, etc., and
  only skips a selected passed unit from the identical contract.
- `worker.log` is excluded from scientific artifact hashes and is sealed by the
  controller only after the worker process exits.
- Scientific null/partial results are a completed experiment; only integrity
  failures produce a failed formal run.
- Controller failures write `status.json` and print one concise summary instead
  of exposing a Python traceback as the primary terminal result.

## Remote use

Upload/sync this directory to:

`${SNM_REPO}/scripts/46_mdp05_confirmatory_update_shock`

Then sync `commands/46_mdp05_confirmatory_update_shock/20260804_ex46_mdp05_confirmatory_update_shock.sh`. No tar
archive is needed. Run the six command modes in order as documented in the
command file: `check`, `dry-run`, `pilot`, `formal`, `resume`, and `verify`.

The pilot is outcome-blind: it opens neither checkpoint nor dataset. It only
certifies the frozen float64 slice diagnostic on one H100. Formal requires that
pilot certificate explicitly, preventing an accidental outcome-dependent
precision-mode switch.

## Formal analysis

The 432 layer-event rows are nested measurements. The statistical unit is one
origin x held-out data replica (12 units per event). For each of the two events
and two primary mediators, analysis reports:

- Spearman correlation with oriented endpoint loss harm;
- within-origin centred correlation;
- leave-one-origin-out Spearman range;
- an exact one-sided randomization p-value from all `6^4 = 1296` within-origin
  outcome permutations;
- Holm adjustment across the four frozen primary tests.

Supportive event-window AUC is reported but does not enter the success gate.
No added replicas or follow-on MDP variant may be triggered by the result.

## Pre-outcome step-boundary repair (version 2026-08-04.2)

Version `.1` checked the number of actual NS5 calls when
`_apply_preconditioners()` returned. That point is inside the optimizer step;
the source optimizer invokes NS5 later in the same `step()`, so the audit was
premature and failed at `production_refresh_32`. Version `.2` leaves gradient
validation at the apply boundary but performs the NS5 count/fingerprint gate
only after the complete optimizer step returns.

For a `.1` run that has this exact failure, do not delete the run. Sync the
updated scripts and invoke `resume` with the same run directory and pilot
certificate. The controller requires zero selected formal units and no final
analysis, retains every `attempt_001`, creates
`source_snapshot_step_boundary_v2`, and launches `attempt_002`. It refuses this
migration for any other failure or after a formal unit has been accepted.
