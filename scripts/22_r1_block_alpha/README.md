# R1-native block-alpha mechanism pilot

This family tests the mechanism that the older 24L dense-full alpha sweep could not establish for
the official R1 architecture.  For each of the four official `mlp.c_proj` blocks it uses

`K_alpha = diag(K) + alpha * (K - diag(K))`.

The raw EMA covariance, ridge rule, inverse refresh schedule, learning rates, data order, model
initialization, and all other R1 controls remain unchanged.  Every new cell retains dense official
block4 covariance/inverse storage.  Therefore this is a quality/mechanism experiment, not a memory
or performance experiment.

## Predeclared seed2026 pilot

New full-length runs: `alpha = 0, 0.25, 0.50, 0.75`.  The matched existing efficient `diag` and
official `block4` seed2026 runs are reused as analysis endpoints; do not rerun them.  Primary endpoint
is step-6200 validation loss.  Tail-five validation mean and normalized validation AUC are secondary.

The pilot expands to seed2024/2025 only if dense alpha=0 is within 0.001 of efficient diag for both
final and tail-five loss, final-loss Spearman rho across alpha is at least 0.5, and block4 is not
better than diag at the primary endpoint.  Strict pointwise monotonicity is descriptive, not a gate.

Smoke must use at least 34 steps so every cell executes the first K inverse refresh at step 32.
Numerical smoke never uploads to W&B.  Formal runs save complete local evidence first and then upload
to the default `Selective-Newton-Muon-MainConf-R1-BlockAlpha-20260722` project.

## Launch order (one command runs all four cells sequentially)

Run from the project root on the R1 H100 host.  Use the same clean pinned official checkout and the
same training Python used by the completed R1 runs.

```bash
PROJECT_ROOT=${SNM_REPO}
OFFICIAL_REPO=${SNM_OFFICIAL_REPO}
CTRL_PY=${SNM_CONTROLLER_PYTHON}
TRAIN_PY=${SNM_TRAINING_PYTHON}
RUNNER="$PROJECT_ROOT/scripts/22_r1_block_alpha/run_r1_block_alpha.py"
export CUDA_VISIBLE_DEVICES=1
cd "$PROJECT_ROOT"
```

This pilot belongs to the original R0/R1 hardware and evidence family, so use the same clean
`Newton-Muon-official-r0` checkout and its FineWeb shards as the completed R1 endpoints.  Do not use
the LLaMA-host `Newton-Muon-official-bridge-clean` worktree here, even when its tracked source commit
is identical: that worktree was created specifically for the cross-hardware bridge and may resolve
its data symlink to a different host-side shard copy.

```bash
"$CTRL_PY" "$RUNNER" --official-repo "$OFFICIAL_REPO" --python-exe "$TRAIN_PY" --dry-run

"$CTRL_PY" "$RUNNER" --official-repo "$OFFICIAL_REPO" --python-exe "$TRAIN_PY" --preflight --wandb-mode disabled

"$CTRL_PY" "$RUNNER" --official-repo "$OFFICIAL_REPO" --python-exe "$TRAIN_PY" --numerical-smoke --smoke-steps 34 --wandb-mode disabled
```

Set `SMOKE` to the final `r1_manifest.json` printed by the smoke batch, not to the batch directory:

```bash
SMOKE=${SNM_RESULTS_ROOT}/22_r1_block_alpha/results/REPLACE_smoke_seed2026/r1_manifest.json

"$CTRL_PY" "$RUNNER" --official-repo "$OFFICIAL_REPO" --python-exe "$TRAIN_PY" --seed 2026 --smoke-manifest "$SMOKE" --wandb-mode online
```

The formal command runs all four cells sequentially on the one visible GPU.  If interrupted, use the
exact batch directory printed as `R1 artifacts`:

Completed R1 runs on this hardware recorded about 7,442--7,481 seconds of official training time per
cell (roughly 2.1 hours).  Budget about 9 hours for four cells plus validation/controller overhead;
select a 12-hour host window rather than an 8-hour window.

If another physical GPU on the same node is training, append
`--concurrent-node-training --concurrent-workload SHORT_LABEL`.  Quality-vs-step and per-process
memory remain eligible, while timing is already ineligible for this experiment family.

```bash
BATCH=${SNM_RESULTS_ROOT}/22_r1_block_alpha/results/REPLACE_formal_seed2026
"$CTRL_PY" "$RUNNER" --official-repo "$OFFICIAL_REPO" --python-exe "$TRAIN_PY" --seed 2026 --resume-batch "$BATCH" --wandb-mode online
```

After completion, join the new pilot with the matched seed2026 R1 endpoints:

```bash
"$CTRL_PY" "$PROJECT_ROOT/scripts/22_r1_block_alpha/analyze_r1_block_alpha.py" \
  --alpha-batch "$BATCH" \
  --diag-run /path/to/completed_seed2026_diag_run_directory \
  --block4-run /path/to/completed_seed2026_block4_run_directory
```

## Separately preregistered multi-seed confirmation

The seed-2026 pilot did not pass its original automatic-expansion gate.  Seeds
2024/2025 must therefore use the explicit `--confirmatory` protocol in
`CONFIRMATORY_CONTRACT_20260724.md`; do not label them as a passed pilot
expansion.

The unattended controller below runs, for each seed, a matching four-cell
exact-shape smoke and then the four formal dense-alpha cells
`0, 0.25, 0.50, 0.75`.  The completed same-seed official block4 run is the
mathematically exact alpha=1 endpoint used later in analysis.

```bash
CONFIRM_RUNNER="$PROJECT_ROOT/scripts/22_r1_block_alpha/run_confirmatory_batch.py"

"$CTRL_PY" "$CONFIRM_RUNNER" \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY" \
  --seeds 2024 2025 \
  --wandb-mode online \
  --concurrent-node-training \
  --concurrent-workload owt_block_alpha_gpu0 \
  --continue-on-error
```

Eight formal cells take about 16.7 hours of recorded training time at the
historical R1 rate.  Allow 24 hours at minimum; when GPU0 is concurrently
training or the host will be unattended over the weekend, select a 30--36 hour
host window.  Timing remains ineligible, while loss-vs-step evidence remains
usable.

## 2026-07-29 multi-seed confirmation result

The seed-2024/2025 W&B confirmation exports passed all 351 data-quality
checks.  The preregistered strong criterion passed: dense-storage
`alpha=0.5` beat both `alpha=0` and the matched official block4 `alpha=1`
endpoint at step 6200 in both new seeds.  The curvature statistic was
`-0.0009` for seed2024 and `-0.0017` for seed2025; the exploratory seed2026
value was `-0.0012`.  Tail-five loss and normalized validation AUC had the
same negative-curvature direction in all three seeds.

This is evidence for an interior block-local mixture, not evidence that
`alpha=0.5` is universally optimal.  The descriptive best alpha was `0.25`
for seed2024 and `0.5` for seeds 2025/2026.  It is also a supporting R1
mechanism/sensitivity result, not a replacement for the primary Selective
versus Muon and Selective versus original Newton--Muon comparisons.

The preserved W&B audit is:

```text
${SNM_RESULTS_ROOT}/22_r1_block_alpha/
analysis/wandb_20260729_multiseed_confirmation/
```

The controller handoff was subsequently audited against the W&B exports:
327/327 local-artifact checks passed in addition to the 351/351 W&B checks.
The final delivery status is `accepted_checkpoint_transfer_excluded`.

The eight remote step-6200 checkpoints are about 1.539 GB each and were
explicitly excluded from transfer by the user.  Non-loadable short fragments
in the handoff ZIP were inventoried and removed from the preserved result
tree; this does not block the scientific result, but no local checkpoint
replay claim is made.  See
`docs/reports/20260729_r1_block_alpha_multiseed_review.md`.
