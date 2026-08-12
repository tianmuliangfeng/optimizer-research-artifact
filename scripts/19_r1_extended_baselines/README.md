# R1 extended-baseline pilot and frozen formal experiment

This experiment family adds three baselines without changing the completed R1
scripts or artifacts:

- official AdamW (`train_gpt_adam_1.py` parameter grouping and LR ratio);
- official NorMuon math, adapted only to split GPT-2 packed QKV into three
  logical matrices;
- Moonlight Muon scaling and decoupled weight decay, with the same packed-QKV
  adaptation.

All runs are single-GPU and serial. The pilot is screening evidence, not a new
formal R1 result. Its primary metric is validation loss at matched optimizer
steps/tokens. Timing is diagnostic only.

## What preflight proves

Preflight fails closed unless all of these checks pass:

1. pinned official repository commit and official AdamW source hash;
2. R1 data shards and training runtime compatibility;
3. identical parameter initialization SHA across all three derived sources;
4. exhaustive/non-overlapping routing of 48 hidden matrices and one tied
   embedding/head parameter, including exactly 12 packed-QKV tensors;
5. single-step NorMuon row-normalization/state and Moonlight
   `0.2*sqrt(max(A,B))`/weight-decay equivalence;
6. packed QKV is numerically treated as Q, K, and V separately.

The generated JSON artifact records source hashes, runtime, routing, and
single-step audit results.

## Pilot grid

The default pilot runs nine cells in one serial batch:

| Method | LR cells | Auxiliary LR | Weight decay |
|---|---|---:|---:|
| AdamW | base `0.0027 / 0.0036 / 0.0045`; hidden LR is `0.16x` | same base | `0` |
| NorMuon | matrix `0.01 / 0.02 / 0.03` | `0.0003` | `0.01` |
| Moonlight Muon | matrix `0.001 / 0.0018 / 0.003` | same as matrix | `0.1` |

The center cells include the public-author defaults. The `r1scale` cells cover
the update scale near the completed R1 Muon configuration.

The default is 1000 updates, full R1 validation (`10,485,760` tokens) every
100 steps, and the same plateau LR as the first 1000 steps of formal R1. This
makes step-1000 validation losses directly comparable to the completed
seed-2026 R1 curves. No checkpoints are written.

## Remote commands

Run from the public artifact repository. The controller interpreter owns W&B;
the isolated R0 interpreter performs training.

### 1. Implementation/runtime preflight

```bash
cd ${SNM_REPO}

CUDA_VISIBLE_DEVICES=0 \
${SNM_CONTROLLER_PYTHON} \
scripts/19_r1_extended_baselines/run_r1_extended_baselines.py \
    --official-repo ${SNM_OFFICIAL_REPO} \
    --python-exe ${SNM_TRAINING_PYTHON} \
    --seed 2026 \
    --preflight
```

### 2. Three-cell exact-shape smoke

```bash
CUDA_VISIBLE_DEVICES=0 \
${SNM_CONTROLLER_PYTHON} \
scripts/19_r1_extended_baselines/run_r1_extended_baselines.py \
    --official-repo ${SNM_OFFICIAL_REPO} \
    --python-exe ${SNM_TRAINING_PYTHON} \
    --seed 2026 \
    --numerical-smoke \
    --smoke-steps 10 \
    --wandb-mode disabled
```

### 3. Nine-cell short pilot

```bash
CUDA_VISIBLE_DEVICES=0 \
${SNM_CONTROLLER_PYTHON} \
scripts/19_r1_extended_baselines/run_r1_extended_baselines.py \
    --official-repo ${SNM_OFFICIAL_REPO} \
    --python-exe ${SNM_TRAINING_PYTHON} \
    --seed 2026 \
    --pilot \
    --pilot-steps 1000 \
    --wandb-mode online
```

W&B project: `Selective-Newton-Muon-MainConf-R1-ExtendedPilot-20260721`.

If power is interrupted, rerun the same pilot command and append the full
artifact directory printed as `R1 extended artifacts`:

```bash
    --resume-batch ${SNM_RESULTS_ROOT}/19_r1_extended_baselines/results/PASTE_BATCH_DIRECTORY
```

Completed cells are reused; only the interrupted cell is restarted from step
zero. Each run preserves the derived source, official-to-derived patch,
terminal output, official log-with-source, scalar CSV, summary JSON, state
bytes, peak memory, and W&B upload status.

## Interpretation gate after the pilot

- First discard non-finite or clearly unstable cells.
- Select at most one LR per method using final/best validation loss plus the
  curve shape; do not select on wall-clock time.
- Compare selected step-1000 losses against the existing R1 seed-2026
  step-1000 validation points.
- Only selected configurations advance to full 6200-step seed-2026 runs.
- Multi-seed expansion is decided after those full runs, not from this pilot.

## Pilot result frozen on 2026-07-21

All nine cells completed with exact expected W&B coverage. The preserved
analysis is under:

```text
${SNM_RESULTS_ROOT}/
  19_r1_extended_baselines/analysis/wandb_20260721_pilot/
```

The selected cells are:

| Method | Formal cell | Auxiliary LR | Matrix/hidden LR | Step-1000 val |
|---|---|---:|---:|---:|
| Moonlight Muon | `moonlight_r1scale` | 0.0018 | 0.0018 | 3.7541 |
| NorMuon | `normuon_r1scale` | 0.0003 | 0.0100 | 3.8634 |
| AdamW | `adamw_low` | 0.0027 | 0.000432 | 4.0259 |

Moonlight `r1scale` is the only selected cell close to the existing R1 Muon
step-1000 result (3.7439; gap +0.0102). NorMuon and AdamW remain useful formal
reference baselines. NorMuon `r1scale` and AdamW `low` are lower-bound winners
within the prespecified grids, so they must not be described as proven global
LR optima.

## Frozen formal experiment (implemented 2026-07-21)

The formal profile is a separate, fail-closed execution path. It always runs
exactly these three pilot-selected cells:

- `adamw_low`: base LR `0.0027`, hidden LR `0.000432`, weight decay `0`;
- `normuon_r1scale`: auxiliary LR `0.0003`, matrix LR `0.010`, weight decay `0.01`;
- `moonlight_r1scale`: auxiliary/matrix LR `0.0018`, weight decay `0.1`.

Each run uses 6200 updates, 512x1024 tokens/update (3,250,585,600 total
tokens), validation every 100 steps, the official 1800-step terminal
warmdown, and a required final checkpoint. The prespecified primary endpoint
is validation loss at step 6200. Tail-five validation mean, validation-curve
mean, and best validation loss are secondary endpoints. Timing remains
diagnostic because this is not the isolated performance experiment.

Seed 2026 is labeled the tuned-seed long-horizon screen. Seeds 2024 and 2025
are independent confirmatory seeds. Do not present the three seeds as three
fully independent tuning trials.

Before each seed's formal batch, run its 34-step exact-shape smoke. The formal
runner rejects a certificate if its seed, selected cells, initialization hash,
derived-source hashes, runtime fingerprint, or validity status differs.

### Formal multi-seed commands

Run serially on one otherwise idle H100. The block deliberately runs seed 2026
first, followed by confirmatory seeds 2024 and 2025.

```bash
# Safe for an interactive bash/tmux pane: do not enable `set -e` here.
set +e
set +u
set +o pipefail

ROOT=${SNM_REPO}
CTRL_PY=${SNM_CONTROLLER_PYTHON}
TRAIN_PY=${SNM_TRAINING_PYTHON}
OFFICIAL_REPO=${SNM_OFFICIAL_REPO}
RUNNER="$ROOT/scripts/19_r1_extended_baselines/run_r1_extended_baselines.py"
RESULTS=${SNM_RESULTS_ROOT}/19_r1_extended_baselines/results
GPU_ID=0
LAUNCH_LOGS="$RESULTS/launcher_logs"

cd "$ROOT"
mkdir -p "$LAUNCH_LOGS"

for SEED in 2026 2024 2025; do
  SMOKE_LOG="$LAUNCH_LOGS/formal_smoke_seed${SEED}_$(date +%Y%m%dT%H%M%S).log"
  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU_ID" "$CTRL_PY" -u "$RUNNER" \
    --official-repo "$OFFICIAL_REPO" \
    --python-exe "$TRAIN_PY" \
    --results-dir "$RESULTS" \
    --seed "$SEED" \
    --formal-smoke \
    --wandb-mode disabled 2>&1 | tee "$SMOKE_LOG"
  RUN_STATUS=${PIPESTATUS[0]}

  if [ "$RUN_STATUS" -ne 0 ]; then
    echo "Smoke failed for seed $SEED (exit $RUN_STATUS). Terminal remains open."
    echo "Inspect: $SMOKE_LOG"
    break
  fi

  SMOKE_BATCH=$(sed -n 's/^R1 extended artifacts: //p' "$SMOKE_LOG" | tail -n 1)
  if [ -z "$SMOKE_BATCH" ]; then
    echo "Could not recover the smoke artifact directory. Inspect: $SMOKE_LOG"
    break
  fi

  SMOKE_MANIFEST="$SMOKE_BATCH/formal_smoke_manifest.json"
  if [ ! -s "$SMOKE_MANIFEST" ]; then
    echo "Missing smoke manifest: $SMOKE_MANIFEST"
    break
  fi

  FORMAL_LOG="$LAUNCH_LOGS/formal_seed${SEED}_$(date +%Y%m%dT%H%M%S).log"
  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU_ID" "$CTRL_PY" -u "$RUNNER" \
    --official-repo "$OFFICIAL_REPO" \
    --python-exe "$TRAIN_PY" \
    --results-dir "$RESULTS" \
    --seed "$SEED" \
    --formal \
    --smoke-manifest "$SMOKE_MANIFEST" \
    --wandb-mode online 2>&1 | tee "$FORMAL_LOG"
  RUN_STATUS=${PIPESTATUS[0]}

  if [ "$RUN_STATUS" -ne 0 ]; then
    echo "Formal batch stopped for seed $SEED (exit $RUN_STATUS). Terminal remains open."
    echo "Inspect: $FORMAL_LOG"
    break
  fi
done
```

Formal W&B project:
`Selective-Newton-Muon-MainConf-R1-ExtendedFormal-20260721`.

If a formal batch is interrupted, do not create a replacement batch. Re-run
the same seed with its printed formal batch directory:

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" "$CTRL_PY" "$RUNNER" \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY" \
  --seed 2026 \
  --formal \
  --resume-batch /absolute/path/to/TIMESTAMP_formal_seed2026 \
  --wandb-mode online
```

Completed cells are reused. An interrupted cell restarts from step zero; the
runner does not claim within-cell checkpoint resume. Valid local evidence is
written before W&B upload, so a W&B outage can be repaired by the same resume
command without retraining completed cells.
