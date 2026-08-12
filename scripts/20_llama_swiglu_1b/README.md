# LLaMA / SwiGLU 1.014B scale validation

This directory prepares the scale experiment that follows the 124M
architecture study. It does not change `scripts/17_llama_swiglu_validation`
or any existing 124M artifacts.

## Frozen profile

| Field | Value |
|---|---:|
| Layers | 18 |
| Width | 2048 |
| Attention heads | 16 |
| SwiGLU intermediate width | 5504 |
| Context | 1024 |
| Parameters | 1,013,690,368 |
| Formal updates | 6200 |
| Formal train tokens | 3,250,585,600 |

The five methods retain the 124M definitions: `down_diag`, `down_none`,
`newton_full`, `muon`, and `adamw`. Exact expected persistent K-state is:

| Method | K-state MiB |
|---|---:|
| `down_none` | 1728.0000 |
| `down_diag` | 1728.7559 |
| `newton_full` | 5888.2500 |
| `muon`, `adamw` | 0 |

These are exact tensor-byte predictions, not total-memory predictions. Real
H100 peak memory must be measured by the probe and smoke.

## Source boundary

`train_llama_swiglu_1b.py` loads the audited 124M trainer, copies it into each
run artifact, verifies its SHA-256, and changes only the model shape. Optimizer
math, activation statistics, FineWeb loader, validation, checkpoint, resume,
and finite-value gates remain in the existing trainer. The artifact records
both the wrapper hash and base-trainer hash.

Sync the entire `scripts/20_llama_swiglu_1b/` directory. It also requires the
matching local file:

```text
scripts/17_llama_swiglu_validation/train_llama_swiglu.py
```

Do not modify or replace that base trainer between smoke, medium, and formal
stages. The currently pinned SHA-256 is
`b72eb0d2a1dfa91b61cd49b4784b3e0739ecebc2fd3228b8f719cec125706f2a`;
startup and every later certificate reject source drift.

## Stage contract

The endpoint, practical margin, medium gate, missing-run handling, and
conditional extension rules were frozen before observing any 1B medium/formal
curve in `ANALYSIS_CONTRACT_20260722.md`.

| Stage | Purpose | Evidence status |
|---|---|---|
| `dry-run` | Print the immutable plan without CUDA/data access | none |
| `preflight` | Runtime, data, initialization, routing, shape and exact K-byte audit | certificate only |
| `probe` | One update per method; fail fast on OOM/routing/runtime problems | screening only |
| `smoke` | 34 updates, reaching the first K refresh at step 32 | required certificate |
| `medium` | 1000 or 2000 plateau-LR updates; stability and cost screen | screening only |
| `formal` | 6200 updates with the formal 1800-step warmdown | formal quality evidence |

Formal launch is refused unless both a completed 34-step smoke manifest and a
completed >=1000-step medium manifest cover every requested method under the
same source, runtime, data, initialization, and model profile.

Medium deliberately uses a plateau LR (`warmdown_iters=0`). It estimates the
first portion of the formal trajectory and is not a shortened formal run.
Formal is the only stage labeled `formal_quality`; all timing is recorded with
`timing_eligible=false` because these quality runs may share a node.

## Fixed-memory/OOM capacity track

Capacity evidence is deliberately separated from the quality stages.  The
capacity controller reuses the pinned model, optimizer math, data, and base
trainer, but writes `evidence_class=capacity_only`, records both peak allocated
and peak reserved CUDA memory, and stops increasing a method's device batch
after its first failure.  Capacity results never certify medium/formal runs.

Run the three structural methods sequentially on one otherwise idle H100.  The
two completed batch-8 smoke manifests provide the baseline; the controller
then tests 16, 32, 64, and 128 while holding context 1024 and global batch 512:

```bash
export CUDA_VISIBLE_DEVICES=0

SMOKE_GPU0=/absolute/path/to/gpu0_smoke/llama_manifest.json
SMOKE_GPU1=/absolute/path/to/gpu1_smoke/llama_manifest.json

"$CTRL_PY" scripts/20_llama_swiglu_1b/run_llama_swiglu_1b_capacity.py \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY" \
  --methods down_none down_diag newton_full \
  --device-batches 16 32 64 128 \
  --baseline-smoke-manifests "$SMOKE_GPU0" "$SMOKE_GPU1" \
  --seed 2026
```

The outputs are `capacity_manifest.json`, `capacity_results.csv`, and one
fully logged cell artifact per attempted method/batch pair.  A successful cell
must complete 34 updates, including the first K refresh at step 32.  Do not
change medium/formal from device batch 8 based on this capacity grid; doing so
would change accumulation and invalidate the matched quality protocol.

### Capacity result (2026-07-22, seed 2026)

The certified H100-80GB grid completed with the same boundary for all three
structural methods: device batch 32 completed all 34 steps and device batch 64
failed with a genuine CUDA OOM in the first training forward.  Batch 128 was
correctly skipped by the predeclared stop-after-first-failure rule.  Therefore
the tested capacity interval is `[32, 64)` for `down_none`, `down_diag`, and
`newton_full`; this grid does not establish a larger maximum device batch for
the reduced-state methods.

At device batch 32, `down_none` and `down_diag` saved 6425.11 and 6421.29 MiB
of peak allocated memory relative to `newton_full`, and retained 11507.63 and
11481.63 MiB of reserved headroom versus 5103.63 MiB for `newton_full`.
The defensible claim is lower memory use and substantially greater headroom,
not a demonstrated batch-size tier increase.  The local analysis bundle is in
`${SNM_RESULTS_ROOT}/20_llama_swiglu_1b/analysis/capacity_20260722_seed2026/`;
the archived input SHA-256 is
`bac6fa304e2f9384cca5c0cfe382729a31840ef6b467f98e42db91435aa75e61`.

### Fine OOM-boundary sweep

The coarse grid cannot test device batches between 32 and 64 while preserving
global batch 512, because there are no additional divisors of 512 in that
interval.  The separate `capacity_fine` protocol therefore fixes gradient
accumulation at 8 and defines global batch as `8 * device_batch_size`.  This is
strictly capacity-only evidence: loss, tokens per update, timing, and quality
must not be compared across cells or with medium/formal runs.

The first pass reruns batch 32 as a cross-protocol anchor and scans every two
sequences through batch 44.  Each method stops after its first failure:

```bash
export CUDA_VISIBLE_DEVICES=0

"$CTRL_PY" scripts/20_llama_swiglu_1b/run_llama_swiglu_1b_capacity_fine.py \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY" \
  --methods muon newton_full down_none down_diag \
  --device-batches 32 34 36 38 40 42 44 \
  --seed 2026
```

The controller writes `capacity_fine_manifest.json`,
`capacity_fine_results.csv`, `capacity_fine_boundaries.json`, and a complete
log/artifact directory for every attempted cell under the `capacity_fine/`
output root.  Before every cell, the selected GPU must have at least 98% of
its physical memory free; otherwise the controller aborts rather than
misclassifying resource contention as a method OOM.  A success still requires all 34 steps and therefore covers the
step-32 K refresh.  The primary endpoint is the ordered pair
`(max_tested_success_batch, first_tested_oom_batch)` for each method.  If that
pair differs by two, run the odd batch between the endpoints later as a
separate exact-boundary confirmation; do not choose it until this first pass
has completed.

Muon is included under the identical fine-capacity protocol so that peak
allocated/reserved memory, exact optimizer-state bytes, and maximum feasible
microbatch can be compared directly against the selective Newton variants.
Do not substitute the older batch-8 Muon smoke for this matched comparison.

### Exact odd-batch confirmation

After the even-grid fine sweep has completed, its frozen analysis contract
permits exactly one additional batch per method when the successful and OOM
endpoints differ by two.  The confirmation controller validates the parent
manifest and automatically runs only the unique midpoint; it does not repeat
the batch-32 anchor or perform a result-dependent rescan:

```bash
FINE_MANIFEST=/absolute/path/to/capacity_fine_manifest.json

"$CTRL_PY" scripts/20_llama_swiglu_1b/run_llama_swiglu_1b_capacity_exact.py \
  --fine-manifest "$FINE_MANIFEST" \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY" \
  --methods newton_full down_none down_diag muon \
  --seed 2026
```

For the certified seed-2026 even grid this resolves to batches 35, 37, 37,
and 39 respectively.  The outputs are `capacity_exact_manifest.json`,
`capacity_exact_results.csv`, and `capacity_exact_boundaries.json`.  The same
98%-free target-GPU guard, 34-step requirement, fixed accumulation of eight,
and capacity-only interpretation continue to apply.

The certified seed-2026 confirmation completed on 2026-07-22.  Batch 35 was
a genuine CUDA OOM for `newton_full`; batches 37, 37, and 39 completed all 34
updates for `down_none`, `down_diag`, and `muon`.  The resulting exact integer
boundaries are therefore `34/35`, `37/38`, `37/38`, and `39/40`.  Relative to
`newton_full`, both compressed down-projection variants increase maximum
successful device batch by 3 (8.82%); Muon increases it by 5 (14.71%).  The
archived analysis is under
`${SNM_RESULTS_ROOT}/20_llama_swiglu_1b/analysis/capacity_exact_20260722_seed2026/`.

This closes the core four-method 1B capacity track.  A future LLaMA-1B
optimizer baseline may receive its own matched fine-capacity appendix only
after that optimizer has been explicitly added to the LLaMA experiment; it
does not retroactively alter the core boundary endpoint.

## Remote setup

Run on the same LLaMA host/runtime family used for the 124M experiment:

```bash
cd ${SNM_REPO}

export CUDA_VISIBLE_DEVICES=0
CTRL_PY=${SNM_CONTROLLER_PYTHON}
TRAIN_PY=${SNM_TRAINING_PYTHON}
OFFICIAL_REPO=${SNM_OFFICIAL_REPO}
RUNNER=scripts/20_llama_swiglu_1b/run_llama_swiglu_1b.py
```

The controller may use a different environment for W&B, but CUDA training
must use the validated Torch 2.8/cu126 interpreter. Seeing `cuda:0` after
setting `CUDA_VISIBLE_DEVICES` is normal logical remapping.

## Commands

Plan:

```bash
"$CTRL_PY" "$RUNNER" \
  --stage dry-run \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY"
```

CPU/runtime/data/initialization preflight:

```bash
"$CTRL_PY" "$RUNNER" \
  --stage preflight \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY"
```

One-update memory probe. The conservative default device batch is 8:

```bash
"$CTRL_PY" "$RUNNER" \
  --stage probe \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY"
```

Required 34-step smoke:

```bash
"$CTRL_PY" "$RUNNER" \
  --stage smoke \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY"
```

Use the printed `llama_manifest.json` from that smoke:

```bash
SMOKE_MANIFEST=/absolute/path/to/smoke/llama_manifest.json

"$CTRL_PY" "$RUNNER" \
  --stage medium \
  --medium-steps 1000 \
  --smoke-manifest "$SMOKE_MANIFEST" \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY" \
  --wandb-mode online
```

After auditing the medium batch, launch formal seed2026:

```bash
MEDIUM_MANIFEST=/absolute/path/to/medium/llama_manifest.json

"$CTRL_PY" "$RUNNER" \
  --stage formal \
  --smoke-manifest "$SMOKE_MANIFEST" \
  --medium-manifest "$MEDIUM_MANIFEST" \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY" \
  --seed 2026 \
  --wandb-mode online
```

If a medium or formal batch is interrupted, resume the exact batch without
passing certificates again:

```bash
"$CTRL_PY" "$RUNNER" \
  --stage formal \
  --resume-batch /absolute/path/to/formal_batch \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY" \
  --seed 2026 \
  --wandb-mode online
```

The checkpoint contains model, optimizer/K state, activation statistics,
loader cursor, prefetched batch, and RNG state. A resumed quality trajectory
remains usable, while its timing remains ineligible.

## Four-GPU formal launch

One run remains single-GPU; do not use DDP. To use four physical GPUs, launch
four controllers with disjoint `CUDA_VISIBLE_DEVICES` and one method each.
Each formal invocation must receive the medium manifest that certified that
method. Run the fifth method on the first card that becomes free.

Parallel quality runs can be compared by loss-vs-step/token, final/tail/AUC,
state bytes, and per-process peak allocation. Their wall-clock, step time,
throughput, and energy must not enter the paper performance table.

## Stop conditions

Do not proceed to formal training if any method:

- cannot complete the 34-step refresh smoke without OOM or nonfinite loss;
- changes model/source/runtime/data/initialization fingerprints;
- cannot preserve global batch 512 and the same 3.2506B-token formal budget;
- has unreliable checkpoint/resume or W&B/local evidence;
- is clearly unstable in the 1000–2000-step medium screen.

Changing `--device-batch-size` changes accumulation and requires fresh probe,
smoke, and medium certificates. Never reuse certificates across that change.

## Formal-6200 seed2026 result (2026-07-24)

The seven W&B history exports passed 90/90 curve/data checks and cover all
four core methods through the frozen step-6200 endpoint.  This run consumed
3,250,585,600 training tokens.  `FineWeb10B` is the dataset/cache name; this
batch is not the optional approximately 10B-token extension.

| Method | Step-6200 val loss | Tail-5 mean | Normalized AUC |
|---|---:|---:|---:|
| `muon` | **2.969126** | **2.977270** | **3.369437** |
| `down_none` | 2.974385 | 2.982569 | 3.372874 |
| `down_diag` | 2.975809 | 2.983888 | 3.370659 |
| `newton_full` | 2.977071 | 2.985143 | 3.373821 |

All three Newton paths are ahead of Muon at step 1000, but Muon becomes
persistently better at interpolated steps 1409.9 (`down_none`), 1748.3
(`newton_full`), and 1923.0 (`down_diag`).  The final Newton-minus-Muon gaps
are +0.005260, +0.007945, and +0.006683, respectively; all exceed the
predeclared 0.0020 practical margin.  Tail-5 and AUC preserve the same
direction.

Combined with the exact 1B capacity endpoints (`newton_full` 34/35,
`down_none` 37/38, `down_diag` 37/38, Muon 39/40), seed2026 places Muon ahead
on both quality and memory under this fixed recipe.  Therefore the current
1B evidence does not support a Pareto-improvement claim over Muon.  It still
supports the narrower within-family result that the selective variants reduce
state and improve loss relative to `newton_full`.

This remains a single-seed result until seeds 2024 and 2025 finish.  Each new
seed requires its own 34-step smoke and 1000-step medium certificate; the
seed2026 manifests must not be reused.  Do not change LR or other optimizer
settings in these confirmation runs.  Pause the optional approximately
10B-token extension until the three-seed result is known.

The archived raw exports, SHA-256 manifest, normalized histories, tables,
figures, checks, and reproducible analyzer are under:

```text
${SNM_RESULTS_ROOT}/
  20_llama_swiglu_1b/analysis/formal6200_20260724_seed2026/
```

The W&B exports do not replace the formal local certificate.  Preserve and
later audit `llama_manifest.json`, `llama_swiglu_summary.csv`, every method's
`summary.json`/`metrics.csv`, checkpoint metadata, and any resume record.

## Formal-6200 three-seed result (2026-07-27)

Seeds 2024 and 2025 subsequently completed under the frozen recipe.  The
combined `3 seeds x 4 methods` audit passes 235 curve/data checks with no
failures.  Muon has the lowest fixed-budget final and tail-5 loss in all three
seeds.  Mean paired final deltas (method minus Muon) are:

- `down_none`: +0.004471 +/- 0.000805;
- `down_diag`: +0.005623 +/- 0.000944;
- `newton_full`: +0.006587 +/- 0.001176.

Every seed-level delta exceeds the frozen +0.0020 practical margin.  The
within-Newton-family result also replicates: both selective paths beat
`newton_full` on final, tail-5, and AUC in all three seeds.  The seed2026
section above is retained as historical first-screen documentation; its
single-seed caveat has now been resolved by the three-seed analysis.

Reproduce the combined audit with
`analyze_llama1b_formal_multiseed.py`.  The raw inputs, hashes, normalized
histories, paired tables, figures, checks, and report are under:

```text
${SNM_RESULTS_ROOT}/
  20_llama_swiglu_1b/analysis/formal6200_multiseed_20260727/
```

The remaining caveat is certificate-level rather than curve-level: collect
the seed2024/2025/2026 formal-6200 compact manifests, plans, summaries, metrics, and
checkpoint/resume metadata.  Copying the approximately 10 GB checkpoint
payload itself is not required.
