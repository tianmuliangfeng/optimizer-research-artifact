# R1 dense-full alpha mechanism pilot

## Latest confirmatory audit

The seed2024/2025 W&B confirmation exports were independently audited together
with the earlier seed2026 pilot on 2026-07-29. The frozen primary contrast
passed in both new seeds. The retained audit is:

```text
${SNM_RESULTS_ROOT}/24_r1_dense_full_alpha/analysis/wandb_20260729_multiseed_confirmation/
```

The final classification is `strong_confirmatory_support`. The seed2024/2025
run/smoke manifests, exact local metric grids, source hashes, and all seven
dense refresh diagnostics are verified; the delivery status is `complete`.
Timing remains ineligible by design. See
`docs/reports/20260729_r1_dense_full_alpha_multiseed_review.md`.

This family tests whether the shallow U-shape in the completed R1 block4-alpha curve is caused by
the block partition.  It reproduces the earlier OpenWebText intervention at the exact R1 shape by
using one complete 3072 x 3072 `c_proj` covariance per layer:

`K_alpha = diag(K_full) + alpha * (K_full - diag(K_full))`.

The five formal cells are `fullalpha0/fullalpha0p25/fullalpha0p50/fullalpha0p75/fullalpha1`, all at
seed2026 and 6200 steps.  They run sequentially on one visible GPU.  Do not mix these runs with the
official block4 quality baseline or with R1-PERF.

## R1 host setup

Use the original R1 host, its `official-r0` checkout and the matching torch 2.8/cu126 environment.
The controller and training interpreters are deliberately explicit.

```bash
PROJECT_ROOT=${SNM_REPO}
OFFICIAL_REPO=${SNM_OFFICIAL_REPO}
CTRL_PY=${SNM_CONTROLLER_PYTHON}
TRAIN_PY=${SNM_TRAINING_PYTHON}
RUNNER="$PROJECT_ROOT/scripts/24_r1_dense_full_alpha/run_r1_dense_full_alpha.py"

export CUDA_VISIBLE_DEVICES=1
cd "$PROJECT_ROOT"
```

First inspect the five audited jobs and validate code/data/runtime without training:

```bash
"$CTRL_PY" "$RUNNER" \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY" \
  --wandb-mode disabled \
  --dry-run

"$CTRL_PY" "$RUNNER" \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY" \
  --wandb-mode disabled \
  --preflight
```

Run all five exact-shape smoke cells.  Step 34 is required because the first full 3072 x 3072
inverse occurs at optimizer step 31:

```bash
"$CTRL_PY" "$RUNNER" \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY" \
  --numerical-smoke \
  --smoke-steps 34 \
  --wandb-mode disabled
```

Set `SMOKE` to the printed `r1_manifest.json`, not its directory.  Then one command runs all five
formal cells sequentially and uploads completed local evidence to W&B:

```bash
SMOKE=${SNM_RESULTS_ROOT}/24_r1_dense_full_alpha/results/REPLACE_smoke_seed2026/r1_manifest.json

"$CTRL_PY" "$RUNNER" \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY" \
  --seed 2026 \
  --smoke-manifest "$SMOKE" \
  --wandb-mode online
```

If GPU0 on the same node is training, append both
`--concurrent-node-training --concurrent-workload SHORT_LABEL`.  Quality and per-process memory
remain usable; timing is ineligible regardless because the mechanism diagnostics synchronize the
GPU at selected refreshes.

Dense-full refresh is materially more expensive than block4.  Use the smoke wall time to refine the
estimate; for the first launch reserve a 24-hour host window.  If interrupted, resume the exact
printed batch directory:

```bash
BATCH=${SNM_RESULTS_ROOT}/24_r1_dense_full_alpha/results/REPLACE_formal_seed2026

"$CTRL_PY" "$RUNNER" \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY" \
  --seed 2026 \
  --resume-batch "$BATCH" \
  --wandb-mode online
```

Each run preserves `dense_full_alpha_diagnostics.csv` alongside its metric and manifest evidence.
The frozen interpretation and seed-expansion rules are in `ANALYSIS_CONTRACT_20260723.md`.

After completion, analyze the five full-alpha cells with the matched efficient-diag run.  Copy the
previous `alpha_run_summary.csv` to the host and pass it as `--block-alpha-summary` to obtain the
matched full-minus-block topology contrasts as well:

```bash
"$CTRL_PY" "$PROJECT_ROOT/scripts/24_r1_dense_full_alpha/analyze_r1_dense_full_alpha.py" \
  --full-alpha-batch "$BATCH" \
  --diag-run /path/to/completed_r1_seed2026_diag_run \
  --block-alpha-summary /path/to/alpha_run_summary.csv
```

## Separately frozen seed-2024/2025 confirmation

The seed-2026 automatic expansion gate did not pass. A later explicit decision
authorized a separate contradiction-resolution study; it must use
`--confirmatory` and must not be described as a passed pilot expansion. The
frozen design is in `CONFIRMATORY_CONTRACT_20260727.md`.

For each of seeds 2024 and 2025, the confirmation runs the complete five-cell
grid `alpha = 0, 0.25, 0.50, 0.75, 1`. Formal evidence is written locally
before W&B upload, and the confirmatory controller accepts a seed only when
both local validation and W&B upload are complete.

The normal repository command is:

```bash
bash commands/24_r1_dense_full_alpha/20260727_r1_dense_full_alpha_confirmatory_multiseed.sh
```

When both R1 H100s are idle, run block-alpha on physical GPU 0 and dense-full
alpha on physical GPU 1 with:

```bash
bash commands/_workflows/22_24_r1_alpha_two_gpu/20260727_r1_alpha_block_and_dense_two_gpu.sh
```

Same-node concurrency makes timing ineligible; quality, diagnostics, and
per-process state evidence remain eligible.
