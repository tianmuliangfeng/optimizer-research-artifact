# LLaMA/SwiGLU-124M extended baselines

This independent experiment family moves Moonlight Muon and NorMuon through a
progressive architecture gate before either method can enter LLaMA-1B.  It
does not modify or retroactively enlarge the frozen five-method LLaMA-124M
core analysis.

## Why a pilot is required

The original LLaMA structural trio (`newton_full`, `down_none`, `down_diag`)
uses one shared matrix LR (`0.01`) because their causal comparison must change
only the down-projection K representation.  Muon also uses matrix LR `0.01`.
AdamW is a method-level recipe baseline and uses backup LR `0.0036` and hidden
matrix LR `0.000576`; the original experiment never required every optimizer
family to use a numerically identical LR.

Moonlight and NorMuon change update normalization and shape scaling, so their
R1-selected LRs cannot be assumed optimal after moving to separate-QKV,
SwiGLU LLaMA.  A short seed2026 pilot prevents a knowingly weak baseline:

| method | auxiliary LR | matrix LR cells | weight decay |
|---|---:|---:|---:|
| NorMuon | 0.0003 | 0.005, 0.010, 0.020 | 0.01 |
| Moonlight Muon | matched to matrix LR | 0.001, 0.0018, 0.003 | 0.10 |

Both new methods receive exactly three cells and 1000 steps per cell.  The
primary selection endpoint is finite step-1000 validation loss.  If two cells
are within 0.002, the mean of the final three validation points breaks the tie;
normalized AUC is secondary.  One configuration per method is then frozen.
Seeds2024/2025 may not be used to reselect LR.

This tuning budget is disclosed separately from the pre-specified core
recipes.  The paper's causal claim remains the shared-LR Newton K-structure
comparison, not a claim that every optimizer received an identical numerical
LR or an exhaustive global hyperparameter search.  Existing same-seed Muon
and diag curves are the anchors; they are not rerun merely to equalize the
number of pilot cells.

## Implementation boundary

`train_llama_swiglu_extended.py` leaves the validated core trainer unchanged
and replaces only optimizer construction.  It reuses:

- the exact 123,551,232-parameter LLaMA/SwiGLU model and initialization;
- FineWeb10B ordering, global batch 512, context 1024 and validation budget;
- checkpoint/RNG/loader resume, metric, memory and runtime recording;
- the audited R1 Moonlight/NorMuon kernels.

All 84 non-embedding 2D tensors go to the matrix optimizer.  The tied token
embedding and 25 RMSNorm gains go to auxiliary AdamW.  LLaMA has separate Q,
K and V projections, so the GPT packed-QKV split is not activated.  Source
certificates hash the runner, adapter, base trainer and optimizer kernel.

## Commands on the LLaMA H100 host

Run from the project root.  GPU0 is used below; a job on physical GPU1 does
not invalidate loss-vs-step quality evidence, but all timing from concurrent
execution is ineligible for the paper performance table.

```bash
PROJECT_ROOT=${SNM_REPO}
OFFICIAL_REPO=${SNM_OFFICIAL_REPO}
CTRL_PY=${SNM_CONTROLLER_PYTHON}
TRAIN_PY=${SNM_TRAINING_PYTHON}
RUNNER="$PROJECT_ROOT/scripts/23_llama_swiglu_extended_baselines/run_llama_swiglu_extended.py"

export CUDA_VISIBLE_DEVICES=0
cd "$PROJECT_ROOT"
```

Inspect the frozen grid without accessing CUDA or FineWeb:

```bash
"$CTRL_PY" "$RUNNER" \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY" \
  --dry-run
```

Preflight:

```bash
"$CTRL_PY" "$RUNNER" \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY" \
  --preflight
```

Run the two-method 34-step implementation smoke:

```bash
"$CTRL_PY" "$RUNNER" \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY" \
  --numerical-smoke \
  --wandb-mode disabled
```

Copy the printed manifest path exactly, then run the six-cell pilot:

```bash
SMOKE=/absolute/path/from/the/previous/command/llama_extended_manifest.json

"$CTRL_PY" "$RUNNER" \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY" \
  --pilot \
  --smoke-manifest "$SMOKE" \
  --wandb-mode online
```

Stop after the pilot and analyze it before formal training.  Do not select a
cell by looking at seeds2024/2025.

## Frozen pilot decision (2026-07-23)

All six seed2026 cells completed 1000 steps / 524,288,000 tokens.  The frozen
primary endpoint selects:

- `moonlight_high`: final validation loss 3.715696, matrix/auxiliary LR 0.003;
- `normuon_r1scale`: final validation loss 3.823241, matrix LR 0.01 and
  auxiliary LR 0.0003.

Moonlight passes the architecture/quality gate: at the matched seed2026
1000-step prefix it is 0.005432 below Muon and within 0.001848--0.003038 of
the Newton trio at the final endpoint.  Its winning LR is the upper boundary
of the predeclared grid, so no higher post-hoc cell may be added.  Seed2026 is
the tuning seed and must not be described as independent confirmation.

NorMuon is 0.102112 above Muon and roughly 0.109--0.111 above the Newton trio.
It therefore stops after the pilot even though it remains better than AdamW.
The `normuon_r1scale` choice is retained only as an archival/reviewer-request
configuration.

The downloaded local pilot artifacts pass 76/76 checks: manifest completion,
six summary hashes, status files, metric grids, W&B upload identities, and
1,896 local/W&B metric comparisons all agree.  `moonlight_high` has
648,671,336 optimizer-state bytes, zero K state, and 35,421.80 MiB peak
allocated at device batch 64.  Its optimizer-state bytes are exactly equal to
the existing LLaMA-124M Muon run and its peak allocation differs by less than
1 MiB.

## Moonlight-only 6200-step formal

Protocol v2 accepts an explicit subset of pilot-frozen cells, so NorMuon is
not rerun.  Both formal smoke and formal are bound to the completed v1 pilot
manifest.  The old numerical-smoke manifest is not reusable after the
controller change.

```bash
PILOT_MANIFEST=${SNM_RESULTS_ROOT}/23_llama_swiglu_extended_baselines/results/20260722T085408+0000_pilot/llama_extended_manifest.json

"$CTRL_PY" "$RUNNER" \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY" \
  --preflight \
  --cells moonlight_high \
  --wandb-mode disabled

"$CTRL_PY" "$RUNNER" \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY" \
  --formal-smoke \
  --seeds 2026 \
  --cells moonlight_high \
  --pilot-manifest "$PILOT_MANIFEST" \
  --wandb-mode disabled
```

Copy the new formal-smoke manifest path exactly:

```bash
FORMAL_SMOKE=/absolute/path/from/formal-smoke/llama_extended_manifest.json

"$CTRL_PY" "$RUNNER" \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY" \
  --formal \
  --seeds 2026 \
  --cells moonlight_high \
  --pilot-manifest "$PILOT_MANIFEST" \
  --smoke-manifest "$FORMAL_SMOKE" \
  --wandb-mode online
```

The formal controller runs one GPU task at a time and orders them by seed,
then method.  Re-run an interrupted batch with the same mode/options plus
`--resume-batch /absolute/path/to/the/batch`; formal training resumes from its
atomic checkpoint, while a pilot cell restarts from step zero.

Run seed2026 first as a tuned-seed long-range gate.  Only if the 6200-step
result remains competitive should seeds2024/2025 be run with the identical
cell and a new matching formal-smoke certificate.

## Moonlight-only fixed-memory/OOM boundary

This is a separate capacity-only experiment.  It uses 34 updates per cell,
fixed accumulation 8, a predeclared coarse device-batch grid, and an automatic
integer binary refinement between the largest success and first CUDA OOM.
It requires at least 98% free memory on the visible GPU before every cell and
refuses to run on a busy card.  Global batch changes with device batch, so its
loss and timing are not quality/performance evidence.  W&B is not used.

```bash
CAPACITY_RUNNER="$PROJECT_ROOT/scripts/23_llama_swiglu_extended_baselines/run_llama_swiglu_extended_capacity.py"

"$CTRL_PY" "$CAPACITY_RUNNER" \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY" \
  --pilot-manifest "$PILOT_MANIFEST" \
  --dry-run

"$CTRL_PY" "$CAPACITY_RUNNER" \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY" \
  --pilot-manifest "$PILOT_MANIFEST"
```

The exact result is written to `capacity_boundary.json`.  Since device batch
64 already completed 1000 pilot steps at 35,421.80 MiB, this capacity sweep is
supporting evidence and does not block the 6200-step quality run.
