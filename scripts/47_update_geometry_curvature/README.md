# Experiment 47 / GEO-01 actual-update direction–curvature mechanism

GEO-01 is an independently authorized exploratory project. It is not MDP-06,
does not repair MDP-05, and cannot reclassify the accepted MDP-05
`partial_or_null` scalar-mediation result.

The candidate local explanation is

```text
d = parameter_update_refresh - parameter_update_frozen
predicted_delta_loss = <g_val, d> + 0.5 * <d, H_val d>
```

`d` must be reconstructed from the source-pinned optimizer with the identical
raw gradient and historical momentum on both paths. The experiment constructs
no full Hessian and persists no full update direction. It records directional
first order, HVP curvature, exact functional line losses, finite-difference
calibration, hashes and invariance certificates.

## Current implementation boundary

Implemented and locally testable:

- frozen pilot-only contract and CPU-only validation;
- exact source-formula counterfactual update-direction kernel;
- autograd directional HVP, functional line evaluation and adaptive central
  finite difference;
- parameter non-mutation audit;
- pilot analyzer that refuses disabled discovery/confirmation phases;
- sealed dry-run and quadratic toy pilot;
- source-pinned accepted MECH-09R checkpoint loader and causal-tree worker;
- exact shadow-to-actual preconditioned-gradient and NS5 fingerprints;
- immutable remote source snapshot, training-venv runtime preflight, independent
  smoke, failed-attempt retention, same-contract resume, pilot analysis and
  compact handoff.

Still blocked in version `2026-08-04.3`: discovery, confirmation and LLaMA 10B
execution. Passing the H100 pilot certifies engineering/numerical feasibility
only and cannot unlock a scientific claim by itself.

## Local commands

From the live repository root:

```bash
python scripts/47_update_geometry_curvature/run_geo01.py check
python -m unittest scripts/47_update_geometry_curvature/test_geo01.py -v
python scripts/47_update_geometry_curvature/run_geo01.py dry-run \
  --run-dir /tmp/geo01_dryrun
python scripts/47_update_geometry_curvature/run_geo01.py toy-pilot \
  --run-dir /tmp/geo01_toy
```

On Windows use the project `venv/muonTest` Python for the tests because the
system Python does not include PyTorch.

## Remote pilot workflow

Sync this whole directory to

```text
${SNM_REPO}/
  scripts/47_update_geometry_curvature/
```

and sync `commands/47_update_geometry_curvature/20260804_ex47_update_geometry_curvature.sh`. No tar archive
is used. The command modes are:

```bash
bash commands/47_update_geometry_curvature/20260804_ex47_update_geometry_curvature.sh check
bash commands/47_update_geometry_curvature/20260804_ex47_update_geometry_curvature.sh dry-run
bash commands/47_update_geometry_curvature/20260804_ex47_update_geometry_curvature.sh pilot
RUN_DIR=/path/to/existing/run bash commands/47_update_geometry_curvature/20260804_ex47_update_geometry_curvature.sh resume
RUN_DIR=/path/to/completed/run bash commands/47_update_geometry_curvature/20260804_ex47_update_geometry_curvature.sh verify
```

`dry-run` seals the full source and derived execution contract but does not
touch CUDA. `pilot` first checks both H100s with the training venv, runs a new
outcome-blind smoke unit, then launches one `early_muon/replica_7` engineering
unit on GPU 0. A failed worker is never overwritten. `resume` creates the next
attempt and skips only a selected passed unit from the identical contract.
The launcher deliberately ignores the generic `CHILD_PYTHON` variable and
uses `GEO01_TRAINING_PYTHON`, defaulting to the frozen
`venv-r0-torch280-cu126` interpreter; it prints both interpreter paths before
every mode.
The controller converts that path to an absolute spelling without resolving
its symlink: resolving `venv/bin/python` to `/usr/bin/python3.10` would disable
Python's virtualenv discovery. Runtime preflight hard-gates both the exact
requested executable spelling and an active `sys.prefix != sys.base_prefix`.

## Pilot frozen shape

The engineering pilot uses `early_muon`, new data replica `7`, production
refresh at completed step 32, layers `0/8/17`, plus a joint direction over the
same layers. It uses two held-out `1 x 128` microbatches in strict float32 with
TF32 disabled. These choices are frozen before remote execution and do not
authorize scientific metric selection.

After the H100 worker passes, the pilot may calibrate only feasibility and
numerical closure. Discovery requires a new sealed contract. Confirmation
requires a separately frozen predictor and unseen checkpoint/data units.
