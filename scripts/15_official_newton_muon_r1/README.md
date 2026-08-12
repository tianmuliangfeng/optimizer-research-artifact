# Official-architecture R1: controlled Muon/block4/none/diag

R1 moves the central `mlp.c_proj` result onto the pinned Newton-Muon-1 H100
setup while preserving a clean boundary from R0:

- R0 executes two unchanged upstream scripts and does not control seed.
- R1 derives auditable scripts from those pinned sources, fixes seed/model
  initialization, and introduces only the declared `mlp.c_proj` K variants.

## Four jobs

| Method | `mlp.c_proj` rule | Base LR | Matrix LR | Interpretation |
|---|---|---:|---:|---|
| `muon` | no K anywhere | 0.0036 | 0.00036 | official Muon recipe baseline |
| `block4` | four dense `768 x 768` K blocks | 0.0040 | 0.00040 | official Newton-Muon control |
| `none` | no K only on `mlp.c_proj` | 0.0040 | 0.00040 | single-module K ablation |
| `diag` | four diagonal length-768 K vectors | 0.0040 | 0.00040 | diagonal `mlp.c_proj` K |

The Newton trio uses one parameterized derived source and the same LR. It
differs only in `R1_CPROJ_K_MODE`. Muon intentionally retains its upstream
method-specific LR. Therefore:

- `block4` vs `none` vs `diag` is the controlled structural comparison;
- Muon vs a Newton variant is an official-recipe comparison, not a shared-LR
  causal claim.

## Mathematical definition of `diag`

For the four contiguous MLP input blocks

\[
z = [z_1,z_2,z_3,z_4], \qquad z_j \in \mathbb{R}^{d},
\]

official block4 tracks

\[
K_j = \operatorname{EMA}\left[\frac{1}{N}Z_j^\top Z_j\right].
\]

The diagonal variant tracks exactly the diagonal of the same matrix:

\[
k_j = \operatorname{EMA}\left[\frac{1}{N}
\sum_{n=1}^{N} z_{n,j}\odot z_{n,j}\right].
\]

With the same official ridge rule

\[
\lambda_j = 0.2\,\operatorname{mean}(k_j)+10^{-8},
\]

the right preconditioner is

\[
P_j = \operatorname{diag}\left((k_j+\lambda_j)^{-1}\right),
\qquad G_j \leftarrow G_jP_j.
\]

`none` performs no activation collection, K allocation, or right
preconditioning for `mlp.c_proj`; all other Newton-Muon matrices remain
unchanged.

## Controls and audit trail

- official commit: `df78af0db523d8bceb25af4919a3e3e7082b80f3`;
- official 50-shard FineWeb10B input and fixed validation shard;
- single H100 80GB, 12L/12H/768D, sequence length 1024;
- global batch 512 sequences, 6200 updates, about 3.25B tokens;
- no warmup and official 1800-step linear warmdown;
- fixed Python, NumPy, PyTorch, and CUDA seed before model construction;
- deterministic data order from sorted shards and loader reset;
- preflight constructs all four models in fresh processes and requires one
  identical SHA-256 over every initialized named parameter;
- generated source, exact official-to-R1 patch, source hashes, environment,
  stdout, full official-style log, checkpoint, metrics CSV, summary, and
  manifest are saved per run;
- W&B upload happens only after training, so network logging cannot perturb
  official training time;
- W&B tables are disabled.

Fixed seed does not imply cross-machine bitwise determinism. The claim is a
same-H100, same-runtime controlled initialization/data-order comparison.

## Commands

Run from `selective-newton-muon`.

The controller and training runtimes are intentionally separate. On the
validated H100 host, reuse the R0 training interpreter:

```bash
CTRL_PY=${SNM_CONTROLLER_PYTHON}
TRAIN_PY=${SNM_TRAINING_PYTHON}
OFFICIAL_REPO=../Newton-Muon-official-r0
```

The controller runtime owns W&B. `TRAIN_PY` only needs the pinned official
training dependencies. The accepted training stack is currently PyTorch
2.8.0+cu126 and Triton 3.4.0 on the H100 80GB host.
With `--wandb-mode online`, preflight also verifies that the controller can
authenticate to the W&B API before any long training begins.

Inspect the exact four-job plan without data/GPU:

```bash
$CTRL_PY scripts/15_official_newton_muon_r1/run_official_newton_muon_r1.py \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY" \
  --wandb-mode disabled \
  --dry-run
```

Full preflight, including four-process initialization fingerprint audit:

```bash
$CTRL_PY scripts/15_official_newton_muon_r1/run_official_newton_muon_r1.py \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY" \
  --wandb-mode online \
  --preflight
```

Required exact-formal-shape 10-step numerical smoke (not formal evidence, no
checkpoint or W&B). It exercises all four methods through the second and later
optimizer updates and terminates immediately on NaN/Inf:

```bash
$CTRL_PY scripts/15_official_newton_muon_r1/run_official_newton_muon_r1.py \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY" \
  --wandb-mode disabled \
  --numerical-smoke \
  --smoke-steps 10
```

Formal seed-2026 batch. Replace `SMOKE_MANIFEST` with the `r1_manifest.json`
printed by the successful smoke. The default order is
`diag -> none -> block4 -> muon`:

```bash
SMOKE_MANIFEST=/absolute/path/to/smoke_batch/r1_manifest.json
$CTRL_PY scripts/15_official_newton_muon_r1/run_official_newton_muon_r1.py \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY" \
  --seed 2026 \
  --smoke-manifest "$SMOKE_MANIFEST" \
  --wandb-mode online
```

If the host powers off, rerun the same batch with its printed artifact
directory. Completed methods are strictly revalidated and skipped; a method
interrupted during training starts a new `_retryNN` attempt from step zero;
valid local evidence whose W&B upload was interrupted retries upload only:

```bash
BATCH_DIR=/absolute/path/to/15_official_newton_muon_r1/results/<formal_batch>
$CTRL_PY scripts/15_official_newton_muon_r1/run_official_newton_muon_r1.py \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY" \
  --seed 2026 \
  --resume-batch "$BATCH_DIR" \
  --wandb-mode online
```

Do not attempt a mid-method checkpoint resume: the official program does not
persist the full loader/RNG/scheduler state required for an equivalent
continuation. Batch-level resume preserves completed evidence without changing
the controlled training trajectory.

Do not use `--continue-on-error` for the first formal batch. A subset via
`--methods` is only for a documented retry.

## W&B metrics to export

Default project:

```text
Selective-Newton-Muon-MainConf-OfficialR1-Controlled-20260717
```

Only useful scalar histories are uploaded:

| W&B key | Use |
|---|---|
| `val/loss` | primary quality curve |
| `train/loss_step` | downsampled divergence/sanity trace; official last microbatch, not accumulation mean |
| `time/train_s` | official cumulative training-only time, excluding validation/checkpoint |
| `performance/step_avg_ms` | steady-state update speed after the official timing reset |
| `lr/adamw` | exact embedding/head LR schedule |
| `lr/matrix` | exact Muon/Newton matrix LR schedule |
| `memory/peak_allocated_mib` | peak CUDA allocated memory |
| `memory/k_state_mib` | exact persistent K covariance + applied inverse storage |
| `memory/optimizer_state_mib` | exact final optimizer tensor-state storage |

The run summary also contains K covariance/inverse bytes separately,
activation-stat bytes, preconditioner workspace bytes, total preconditioner
bytes, model bytes, checkpoint bytes, final/best losses, curve mean,
tokens/second, milestone losses, and step/token/train-time to validation loss
3.6, 3.5, 3.4, and 3.3. Those summaries do not create extra W&B panels.

Local artifacts are under:

```text
${SNM_RESULTS_ROOT}/
  15_official_newton_muon_r1/results/<batch>/
```

`r1_common_target_comparison.csv` additionally computes the first observed
step/token/official training time at the worst final loss shared by the
completed methods. This dynamic comparison is kept local because its target is
defined only after all jobs finish.
