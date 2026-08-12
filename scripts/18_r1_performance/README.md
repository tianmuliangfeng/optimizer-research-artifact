# R1-PERF: isolated implementation-efficiency study

This suite explains the sub-1% timing reversal observed in R1 without changing
the R1 quality evidence.  It must not compare a method timed on the old R1 host
with another method timed on the newer LLaMA host.  Every performance table is
formed from methods rerun under one recorded runtime on one visible GPU.

## Methods

The short end-to-end benchmark contains six methods:

| Method | Role |
|---|---|
| `diag` | current correct diagonal c_proj implementation |
| `block4` | official optimized four-block Newton-Muon control |
| `none` | no c_proj K; lower-overhead Newton control |
| `dense_full` | full 3072 x 3072 c_proj K; compute/memory upper bound |
| `muon` | official Muon recipe throughput reference |
| `adamw` | fused AdamW on hidden matrices plus the existing head AdamW |

Only `none/diag/block4/dense_full` appear in the K-operator table.  AdamW and
Muon have no K accumulation or inverse and therefore belong only in the
end-to-end table.

`dense_full` changes the c_proj K representation and is not a quality baseline.
The performance source stores one full covariance and one full inverse per
layer, uses the same EWMA/ridge/refresh rule, and applies it by batched matrix
multiplication.

## What the suite measures

1. Operator components at the exact R1 shape:
   activation-statistic collection over eight microbatches and twelve layers,
   inverse refresh, gradient application, and the 1/32-step amortized c_proj
   overhead.
2. Short exact-shape GPT training: the official timer discards the first 32
   updates, then measures 512 updates by default.
3. Repeated rotated order: each repeat rotates the method order to expose
   cache/order sensitivity.

Validation/checkpoint time is outside the official timer.  W&B is deliberately
absent from the timed process.  The controller saves source, diff, stdout,
runtime fingerprint, raw runs, aggregate medians, state bytes, and peak memory.

## Scientific boundary

- Existing R1 remains the source of loss/step-to-loss evidence.
- R1-PERF measures implementation throughput and memory only.
- A restarted or cross-host long run is never paired with an old method time.
- A speed difference below about 1% is not claimed from one run; use repeated
  rotated measurements.
- If only the claim “less state/memory with non-inferior quality” is needed,
  this suite is recommended supporting evidence rather than a hard blocker.

Current seed2026 R1 reaches common loss 3.2771 at approximately 5862 interpolated
steps for diag and 5875 for block4.  Because validation is every 100 steps, this
is best described as similar step-to-loss, slightly favoring diag.  Diag reaches
that target about 21.5 seconds later because its measured step time is slightly
higher.  Step-to-loss and time-to-loss must therefore remain separate metrics.

## New-host commands

Run from `selective-newton-muon`.  Select the idle GPU before
launching Python; it will appear as `cuda:0` inside the process.

```bash
cd ${SNM_REPO}

export CUDA_VISIBLE_DEVICES=1
CTRL_PY=${SNM_CONTROLLER_PYTHON}
TRAIN_PY=${SNM_TRAINING_PYTHON}
OFFICIAL_REPO=${SNM_OFFICIAL_REPO}
RUNNER=scripts/18_r1_performance/run_r1_performance.py
```

Inspect the plan without importing CUDA:

```bash
"$CTRL_PY" "$RUNNER" \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY" \
  --dry-run
```

Validate the runtime, data, all six generated sources, and identical model
initialization:

```bash
"$CTRL_PY" "$RUNNER" \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY" \
  --preflight
```

The component benchmark is the only useful phase while another GPU on the same
node is training LLaMA.  It is exploratory until repeated on an otherwise idle
node.  Running it may still perturb the sibling job's timing through shared
power/CPU resources, although it does not change that job's loss trajectory.

```bash
"$CTRL_PY" "$RUNNER" \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY" \
  --methods diag block4 dense_full none \
  --operator-benchmark \
  --operator-warmup 3 \
  --operator-repeats 10
```

After the LLaMA job finishes, run the required exact-shape 34-step smoke.  It
reaches the first K refresh and exercises AdamW and dense-full as well:

```bash
"$CTRL_PY" "$RUNNER" \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY" \
  --numerical-smoke
```

Substitute the printed smoke manifest.  A rough single-repeat screen is:

```bash
SMOKE_MANIFEST=/absolute/path/to/smoke_seed2026/perf_manifest.json

"$CTRL_PY" "$RUNNER" \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY" \
  --smoke-manifest "$SMOKE_MANIFEST" \
  --training-benchmark \
  --timed-steps 256 \
  --repeats 1
```

Do not use that single repeat for a sub-1% speed claim.  The paper-grade run is:

```bash
"$CTRL_PY" "$RUNNER" \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY" \
  --smoke-manifest "$SMOKE_MANIFEST" \
  --training-benchmark \
  --timed-steps 512 \
  --repeats 3
```

Artifacts are written to:

```text
${SNM_WORKSPACE_ROOT}/experiment_csv/
  selective-newton-muon/18_r1_performance/results/<batch>/
```

The final batch contains `training_benchmark_runs.csv` and
`training_benchmark_summary.csv`.  Report median step time, tokens/s, peak
memory, exact K state, and the percentage difference from block4.  Treat an
interval overlapping zero as throughput parity rather than a speedup.
