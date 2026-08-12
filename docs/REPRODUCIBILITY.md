# Reproducibility contract

The artifact uses four distinct operations. They are deliberately not treated
as synonyms.

## Operations

### `reproduce`

Creates a plan for a fresh run. It must use a new result directory and the
experiment's declared code, seeds, configuration, and data contract. The
dispatcher prints the full source-tree SHA-256 and a plan SHA-256 before any
process is started.

### `resume`

Continues an interrupted run in the exact existing result directory. Resume is
available only when metadata declares a native recovery mode. The late formal
controllers additionally validate checkpoint, loader, RNG, contract, and
source-snapshot lineage.

### `verify`

Performs common read-only artifact validation:

- the run must be a strict child of an explicitly supplied results root;
- every JSON file must parse;
- `source_snapshot_manifest.json` file hashes and byte counts are checked;
- no training, upload, timestamp update, or result mutation is allowed.

This is an archival integrity check, not a scientific pass/fail decision.

### `native-verify`

Runs the experiment-specific validator when one exists. Native validators
check scientific contracts, accepted cell counts, claim boundaries, and other
experiment-specific gates that cannot be inferred generically.

## Evidence tiers

| Tier | Meaning |
|---|---|
| `sealed_source_snapshot` | The run freezes its controller/worker sources and supports strong lineage validation. |
| `partial_or_run_specific_snapshot` | Some sources or generated workers are preserved, but recovery semantics vary by experiment. |
| `legacy_command_or_live_source` | Exact command records or live runners remain, but a complete immutable runtime cannot be reconstructed automatically. |

The metadata reports these tiers rather than claiming that every historical
run is bitwise reproducible. Fresh CUDA/Triton processes can also differ at the
last bits even under an identical contract; numerical tolerances and native
integrity gates remain authoritative.

## Relocatable paths

Repository code discovers its root from `pyproject.toml`, `scripts/`, and the
reference backend layout; it does not depend on the outer directory name.
Generated results default to `runs/`. External upstream code, data, and
archives must be selected with `SNM_OFFICIAL_REPO`, `SNM_RESULTS_ROOT`, or
receipt-bound `--env` arguments. Historical path labels retained in sealed
lineage records are metadata only and are never used as private host paths.

## Guarded execution

`reproduce`, `resume`, and `native-verify` print a deterministic JSON plan by
default. To execute it:

1. inspect the printed command, environment, source-tree hash, and
   `plan_sha256`;
2. repeat the same invocation with `--execute --receipt <plan_sha256>`;
3. preserve the resulting receipt beside the run artifacts.

The receipt is invalidated by any metadata, code, launcher, path, or argument
change.

## Experiment-specific notes

- Experiments 43 and 44 restore through run-local source snapshots and skip
  already accepted cells.
- Experiment 29 provides a one-command fresh rerun and the accepted formal
  analyzer, but its quality runs intentionally disable checkpoints and do not
  claim mid-run resume.
- MDP-04 is an archived `numeric_gate_failed` result. Its public reproduction
  entrypoint validates that negative adjudication; it must not be presented as
  a passed mechanism result.
- Experiments 46, 47, 47B, 48, and 49 have explicit native resume/verify modes.
  Experiment 46 resume also requires the original pilot precision
  certificate.
- Experiment 48's public controller snapshots its organized launcher in
  addition to the contract, workers, analyzers, and inherited LLaMA sources.
  Its accepted replacement protocol uses one host with exactly four H100 GPUs;
  the deleted incomplete two-GPU attempt is neither resumable nor evidence.
- Experiment 49 resume is accepted-unit/upload level. Its `reproduce_full`
  entrypoint explicitly executes `preflight`, `pilot`, `formal`, then
  `verify`; an interrupted non-accepted training unit restarts from the frozen
  initialization rather than an arbitrary optimizer step.
- Generic verification can validate only artifacts that are actually
  available. Raw result archives are intentionally not bundled with source.

## Historical command provenance

The public launchers under `commands/` are organized by experiment and replace
private absolute paths with an explicit environment profile. The original 37
private-era launcher contents are not published; their SHA-256 values and the
sanitized public counterparts are recorded in
`provenance/legacy_command_inventory.json`.

`provenance/command_path_migration.json` records every old flat launcher path
to organized public path. Path-only rewrites of hash-bound contracts retain the
historical hash alongside an explicit public hash. These packaging changes do
not alter optimizer settings, seeds, schedules, data identities, or scientific
decision rules.
