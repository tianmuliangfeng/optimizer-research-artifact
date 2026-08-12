# `mlp.c_proj` K-structure mechanism experiment

This runner changes only the activation K used by `mlp.c_proj`. Every other eligible matrix keeps
the existing full Newton-Muon path.

Modes:

- `none`: Muon on `mlp.c_proj`, with no c_proj K-state;
- `full`: the current full `(4d) x (4d)` c_proj K;
- `block4`: the official four-block `d x d` approximation;
- `diag`: a diagonal activation-second-moment control.
- `scalar`: `mean(diag(K)) I`, retaining only a time-varying global scale;
- `alpha`: dense `K_alpha = D + alpha * (K - D)` with `D = diag(K)`; this scales
  only the non-diagonal entries and leaves the running diagonal unchanged.
- `block_alpha`: block-local `K_alpha = D_block + alpha * O_within`; it scales only
  within-block off-diagonals while all cross-block entries remain zero. Its damping
  stays block-local so alpha=1 exactly matches `block4`; alpha=0 is a block-path
  code/storage control and is not silently treated as the globally damped `diag`.

For the default 24L/D1024 configuration, the expected float32 K-state allocations are:

| Mode | Total | c_proj | Released versus full |
|---|---:|---:|---:|
| `full` | 5472.00 MiB | 4608.00 MiB | 0.00% |
| `block4` | 2016.00 MiB | 1152.00 MiB | 63.16% |
| `diag` | 864.75 MiB | 0.75 MiB | 84.20% |
| `scalar` | 864.00018 MiB | 0.00018 MiB | 84.21% |
| `none` | 864.00 MiB | 0.00 MiB | 84.21% |
| `alpha` | 5472.00 MiB | 4608.00 MiB | 0.00% |
| `block_alpha` | 2016.00 MiB | 1152.00 MiB | 63.16% |

Any `0 <= alpha < 1` run retains the dense c_proj state and therefore has the same expected
K-state allocation as `full`. In particular, `alpha=0` is mathematically diagonal but deliberately
uses dense storage and the dense inversion code path; it is an implementation-equivalence control,
not a memory-saving mode.

Every `block_alpha` run retains the four-block state and therefore has the same expected K-state
allocation as `block4`. The runner disables model checkpoint saving by default: the 12-run
dual-path batch would otherwise leave one large best checkpoint in every run directory.

## Dense-versus-block alpha topology pilot

The seed2026 pilot contains 12 runs:

- scientific anchors: `none`, efficient `diag`, `full`, and `block4`;
- dense path: `alpha={0, 0.25, 0.5, 0.75}`, with `full` reused as alpha=1;
- block path: `block_alpha={0, 0.25, 0.5, 0.75}`, with `block4` reused as alpha=1.

WikiText-103 dry-run:

```bash
python scripts/06_kstate_spectrum/run_cproj_k_structure.py \
  --dry-run \
  --dataset wikitext103_gpt2_50m \
  --seeds 2026 \
  --modes none diag alpha block_alpha full block4 \
  --offdiag-alphas 0 0.25 0.5 0.75 \
  --run-prefix mainconf_wikitext103_24L_dual_alpha_seed2026
```

The formal command is identical without `--dry-run`. For OpenWebText, change only:

```text
--dataset openwebtext_gpt2_50m
--run-prefix mainconf_owt_24L_block_alpha_seed2026
```

For an unattended batch, `--continue-on-error` records the failed run name and continues with
the remaining variants before returning a non-zero batch exit. Do not use that option for the
initial numerical smoke.

## Scalar and dense-alpha0 mechanism controls

These two seed2026 runs use the exact 24L/D1024 OpenWebText configuration from the alpha sweep.
They answer different questions:

- `scalar`: whether the benefit of diag requires coordinate-wise scale heterogeneity rather than
  only the time-varying global mean second moment;
- dense `alpha=0`: whether efficient diag and the mathematically identical dense endpoint agree,
  ruling out storage layout or code-path artifacts.

Two-run dry-run:

```powershell
python scripts/06_kstate_spectrum/run_cproj_k_structure.py --dry-run --python-exe python --modes scalar alpha --offdiag-alphas 0 --seeds 2026
```

Two-run formal experiment:

```powershell
python scripts/06_kstate_spectrum/run_cproj_k_structure.py --python-exe python --modes scalar alpha --offdiag-alphas 0 --seeds 2026
```

The runner validates that exactly two unique commands are generated, uses the compact `paper`
W&B profile with tables and update-similarity probes disabled, and prints the expected K-state
allocation before starting either run.

Nine-run non-diagonal-strength dry-run:

```powershell
python scripts/06_kstate_spectrum/run_cproj_k_structure.py --dry-run --modes alpha --offdiag-alphas 0.25 0.50 0.75 --seeds 2024 2025 2026
```

Nine-run formal experiment:

```powershell
python scripts/06_kstate_spectrum/run_cproj_k_structure.py --modes alpha --offdiag-alphas 0.25 0.50 0.75 --seeds 2024 2025 2026
```

The existing three-seed `diag` and `full` runs are reused as alpha=0 and alpha=1; do not rerun
those endpoints.

## Preregistered targeted diagonal generalization

The alpha gate passed on 2026-07-16. The next tranche is exactly seven `diag` runs and reuses
the existing matched none/full anchors:

| Anchor | Seeds | Fixed configuration | Runs |
|---|---|---|---:|
| OpenWebText 12L/100M | 2024, 2025, 2026 | D768, batch16, 12,208 steps, LR decay 12,208, matrix LR 0.02 | 3 |
| WikiText-103 24L/12k | 2024, 2025, 2026 | D1024, batch2, 12,000 steps, LR decay 3,000, matrix LR 0.01 | 3 |
| WikiText-103 12L/100M probe | 2026 | D768, batch16, 12,208 steps, LR decay 12,208, matrix LR 0.02 | 1 |

All three anchors use input EMA 0.95, ridge 0.2, refresh interval 32, sample cap 2048,
`wandb_log_profile=paper`, and no W&B tables or update-similarity probe metrics.

Seven-run dry-run:

```powershell
python scripts/06_kstate_spectrum/run_targeted_diag_generalization.py --dry-run --python-exe python
```

Seven-run formal experiment:

```powershell
python scripts/06_kstate_spectrum/run_targeted_diag_generalization.py
```

To resume only one anchor or selected failed seeds, for example:

```powershell
python scripts/06_kstate_spectrum/run_targeted_diag_generalization.py --anchors wikitext_24l_12k --wikitext-24l-seeds 2025 2026
```

The runner validates the number of generated commands, unique run names, the `diag` mode,
the four K-estimator hyperparameters, disabled diagnostics/probes, and the compact W&B profile
before launching any training process.

Run from `selective-newton-muon`.

Critical dry-run:

```powershell
python scripts/06_kstate_spectrum/run_cproj_k_structure.py --dry-run --modes block4 --seeds 2026
```

Critical formal run:

```powershell
python scripts/06_kstate_spectrum/run_cproj_k_structure.py --modes block4 --seeds 2026
```

Add the diagonal control:

```powershell
python scripts/06_kstate_spectrum/run_cproj_k_structure.py --modes diag --seeds 2026
```

Generate all four matched controls when an implementation-local comparison is needed:

```powershell
python scripts/06_kstate_spectrum/run_cproj_k_structure.py --modes none full block4 diag --seeds 2026
```

If the runner Python is not the PyTorch training Python, add:

```text
--python-exe /absolute/path/to/python
```

The formal default is OpenWebText, 24L/D1024/H16, batch 2, block 512, 12k iterations, LR decay
3k, matrix LR 0.01, beta 0.95, ridge 0.2, refresh 32, and seed2026. The runner records generated
commands under `${SNM_RESULTS_ROOT}/06_kstate_spectrum/commands/`.

## M1-P0 local quadratic-score probe

`run_quadratic_score_probe.py` starts one OpenWebText 24L none/release84 trajectory and runs a
shadow-only diagnostic at step 10,000. At the same model state and on the same diagnostic batch,
it constructs the instantaneous `none`, `scalar`, `diag`, `block4`, and `full` directions for
`mlp.c_proj` layers h0, h11, and h23.

The probe records:

- normalized and raw gradient alignment;
- exact parameter-Hessian curvature from autograd HVP;
- the scale-invariant quadratic score;
- right-projector drift and row-orthogonality residual;
- same-batch and held-out one-dimensional line searches;
- fresh-K diagonal/off-diagonal statistics;
- probe runtime and isolated peak memory.

The main optimizer state and training parameters are not changed by any candidate. Detailed
diagnostics are written to local CSV/JSON artifacts; W&B retains only the compact eight-metric
paper profile and does not create per-layer probe panels.

The default diagnostic batch is one 128-token sequence. Dense and block covariance inverse
applications use the mathematically exact Woodbury form, so the probe does not need a
4096-by-4096 Cholesky factor. The update direction still uses the same five-step Newton--Schulz
matrix-sign approximation as training. Exact SVD is available as an opt-in follow-up and is not
enabled in P0. The diagnostic objective is evaluated in deterministic `model.eval()` mode; this
removes dropout noise from HVP and line-search comparisons and is recorded in `probe_config.json`.

One-command dry-run:

```powershell
python scripts/06_kstate_spectrum/run_quadratic_score_probe.py --dry-run --python-exe python
```

Formal-shape GPU memory/HVP preflight at initialization, not a paper result:

```powershell
python scripts/06_kstate_spectrum/run_quadratic_score_probe.py `
  --python-exe python --wandb-mode disabled `
  --max-iters 1 --probe-steps 0 --eval-iters 1 `
  --run-prefix mainconf_quadprobe_gpu_preflight
```

This uses the full 24L/D1024 model, all three target layers, five directions, exact HVP, and both
line-search splits, but performs no training trajectory. Run it once before spending 10,000 steps.
If it exceeds device memory, first retry with `--probe-block-size 64`; do not weaken the formal
training configuration itself.

Formal seed2026 P0 run:

```powershell
python scripts/06_kstate_spectrum/run_quadratic_score_probe.py --python-exe python
```

The default run stops immediately after completing the step-10,000 probe (`max_iters=10001`).
Expected output is 15 direction rows and 150 line-search rows under:

```text
${SNM_RESULTS_ROOT}/06_kstate_spectrum/
  quadratic_probe_p0/<run_name>/
```

Generated files:

- `quadratic_probe_long.csv`
- `quadratic_probe_summary.csv`
- `line_search_results.csv`
- `probe_config.json`
- `probe_metadata.json`
- `probe_data_quality_checks.csv`

Cheap implementation smoke, not a paper experiment:

```powershell
python scripts/06_kstate_spectrum/run_quadratic_score_probe.py `
  --python-exe python --wandb-mode disabled `
  --n-layer 2 --n-head 2 --n-embd 32 `
  --batch-size 2 --block-size 16 --max-iters 2 `
  --probe-steps 0 --probe-layers 0 1 `
  --probe-batch-size 1 --probe-block-size 8
```

Seed2026 P0 completed on 2026-07-16. The unconstrained quadratic-score ranking does not agree
with the fixed-step held-out ranking: block4/full often have higher `A^2/C`, while diag has the
best three-layer held-out mean at eta 0.01 and 0.02. The unchanged P0 should not be copied to
seeds 2024/2025 yet because it has only one 128-token held-out batch, a measurable scalar/none
line-search control gap, fresh K rather than optimizer EMA K, and no exact-SVD control.

Reproduce the archived analysis:

```powershell
python scripts/06_kstate_spectrum/analyze_quadratic_probe_p0.py `
  --run-dir <archived-run-directory> `
  --historical-summary ${SNM_RESULTS_ROOT}/06_kstate_spectrum/summaries/combined_val_curves_all_seeds.csv
```

Build the canonical technical-report input:

```powershell
python scripts/06_kstate_spectrum/build_quadratic_probe_p0_report.py `
  --run-dir <archived-run-directory>
```

The detailed result, limitations, processed tables, hashes, and seed gate are stored with the
archived run. The broader preregistered expansion logic remains in
`docs/reports/20260716_mathematical_mechanism_plan.md`.

## M1-P1 repeated fresh-K cross-fit probe

P1 keeps the same single seed2026 none/release84 trajectory and step-10000 checkpoint, but fixes
the main P0 measurement limitations before any seed expansion:

- four independent 128-token direction-building batches;
- eight fixed 128-token held-out batches shared by every build repeat, layer, and mode;
- strict float32 probe forward/HVP/line-search with CUDA TF32 disabled and training precision
  restored afterward;
- an exact cloned `none_repeat` direction to measure the line-search numerical floor;
- `diag_normmatch`, `block4_normmatch`, and `full_normmatch`, each rescaled to the none direction
  Frobenius norm;
- exact-SVD cosine diagnostics for the first build repeat only, avoiding 60 full-size SVDs while
  covering all three layers and all effective variants.

The nine effective directions are `none`, `scalar`, `diag`, `block4`, `full`, `none_repeat`,
`diag_normmatch`, `block4_normmatch`, and `full_normmatch`. The default output is 108 direction
rows and 4,860 line-search rows. All detailed data stays in local CSV/JSON; W&B remains on the
eight-metric paper profile.

In addition to the P0 files, P1 writes:

- `line_search_paired_results.csv`, with every row paired against the identical none and
  none-repeat observation;
- `line_search_crossfit_summary.csv`, aggregating mean, population SD, median, and paired
  win/loss counts by mode, evaluation kind, and step size.

P1 is explicitly a fresh-K cross-fit experiment. A none trajectory does not contain the
counterfactual EMA K and momentum states that diag/block4/full would have accumulated. Those
stateful controls belong to a separate shadow-state experiment and are not approximated here.

Formal-shape GPU preflight:

```powershell
python scripts/06_kstate_spectrum/run_quadratic_score_probe_p1.py `
  --python-exe python --wandb-mode disabled `
  --max-iters 1 --probe-steps 0 --eval-iters 1 `
  --run-prefix mainconf_quadprobe_p1_gpu_preflight
```

This is the exact P1 diagnostic shape at initialization, not a scientific result. It must produce
108 direction rows, 4,860 line-search rows, and 12 passing data-quality checks. If memory is
insufficient, retry the preflight with `--probe-block-size 64` before changing any other setting.

Formal dry-run:

```powershell
python scripts/06_kstate_spectrum/run_quadratic_score_probe_p1.py `
  --dry-run --python-exe python
```

Formal seed2026 run:

```powershell
python scripts/06_kstate_spectrum/run_quadratic_score_probe_p1.py --python-exe python
```

The formal output is written under:

```text
${SNM_RESULTS_ROOT}/06_kstate_spectrum/
  quadratic_probe_p1/mainconf_quadprobe_p1_none_L24_D1024_step10000_seed2026/
```

Do not start seeds 2024/2025 until the P1 seed2026 result passes all quality gates and shows a
held-out diag effect larger than the `none_repeat`/scalar numerical floor.

P1 completed on 2026-07-16. All numerical controls passed, but the held-out diag effect shrank
to the \(10^{-6}\) scale, changed sign across build/layer aggregates, and failed the seed gate.
The unchanged P1 must not be copied to seeds 2024/2025.

## M1-P2 optimizer-state and exact-polar probe

P2 keeps the model on the same c_proj `none` trajectory while maintaining non-intervening
`diag` and `full` shadow EMA K plus shadow momentum for the three probed c_proj layers only.
At step 10000 it compares:

- fresh-K/current-gradient NS5;
- EMA-K/current-gradient NS5;
- EMA-K/shadow-momentum NS5;
- fresh-K/current-gradient exact-SVD polar directions.

The formal seed2026 run evaluates `none/diag/full`, four build batches, three layers, eight
held-out batches, and exact SVD in all four builds. It writes 144 direction rows and 6480
line-search rows. Detailed outputs remain local; W&B stays on the eight-metric paper profile.

Full-model memory preflight:

```powershell
python scripts/06_kstate_spectrum/run_quadratic_score_probe_p2.py `
  --python-exe python --wandb-mode disabled `
  --max-iters 2 --probe-steps 1 `
  --build-repeats 1 --heldout-batches 1 --exact-svd-repeats 1 `
  --eval-iters 1 --run-prefix mainconf_quadprobe_p2_gpu_preflight
```

Formal dry-run:

```powershell
python scripts/06_kstate_spectrum/run_quadratic_score_probe_p2.py `
  --dry-run --python-exe python
```

Formal seed2026 run:

```powershell
python scripts/06_kstate_spectrum/run_quadratic_score_probe_p2.py --python-exe python
```

The output is written under:

```text
${SNM_RESULTS_ROOT}/06_kstate_spectrum/
  quadratic_probe_p2/
  mainconf_quadprobe_p2_none_shadowdiagfull_L24_D1024_step10000_seed2026/
```

P2 disables checkpoint saving because the diagnostic shadow optimizer state is not needed after
the CSV/JSON export. Restricting shadow state to layers 0/11/23 cuts its expected size from about
5.25 GiB to about 672 MiB. The shadow states increase measured peak memory but do not change the
model update or the algorithmic K-state metric.

## M1-P3 all-layer temporal map and FP64 polar control

P2 recovered the long-run qualitative ordering only after historical momentum was included, but
the small diag benefit was concentrated in layer 0 and its FP32 SVD arm failed the predeclared
row-orthogonality gate. P3 therefore:

- maintains non-intervening diag/full shadow K and momentum for all 24 c_proj layers;
- reports every layer plus preregistered early/middle/late depth bands;
- keeps four build repeats and eight shared held-out batches;
- disables exact HVP because it was not decision-relevant in P2;
- computes exact SVD internally in float64, casts the applied direction back to float32, and
  checks the applied direction against the same `1e-4` orthogonality threshold.

The formal output contains 1,152 direction rows and 51,840 line-search rows. Estimated persistent
diagnostic shadow state is about 5.25 GiB, so run the full-layer memory preflight first:

```powershell
python scripts/06_kstate_spectrum/run_quadratic_score_probe_p3.py `
  --python-exe python --wandb-mode disabled `
  --max-iters 2 --lr-decay-iters 2 --probe-steps 1 `
  --build-repeats 1 --heldout-batches 1 --exact-svd-repeats 1 `
  --no-line-search --eval-iters 1 --eval-interval 1 --log-interval 1 `
  --run-prefix mainconf_quadprobe_p3_gpu_preflight
```

Formal dry-run:

```powershell
python scripts/06_kstate_spectrum/run_quadratic_score_probe_p3.py `
  --dry-run --python-exe python
```

Formal seed2026:

```powershell
python scripts/06_kstate_spectrum/run_quadratic_score_probe_p3.py --python-exe python
```

The formal output is written under:

```text
${SNM_RESULTS_ROOT}/06_kstate_spectrum/
  quadratic_probe_p3_layer_map/
  mainconf_quadprobe_p3_layermap_fp64svd_none_shadowdiagfull_
  L24_D1024_step10000_seed2026/
```

Do not add seed2024/2025 until the seed2026 all-layer/depth-band gate passes and every FP64-SVD
direction passes the orthogonality check. The full protocol is
`docs/reports/20260716_quadratic_probe_p3_protocol.md`.
