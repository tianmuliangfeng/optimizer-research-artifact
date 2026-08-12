# Reference-Aligned LR and Ordering Sanity Runs

This family is experiment B in the reference-alignment correction plan. It
uses the same explicit optimizer settings as the 60-run formal reset and adds
a reference-pipeline dense `mlp.c_proj` K control. The 12L suite runs for 5000
steps so it observes the late Muon instability that began near step 2500;
the 24L suite remains a 3000-step early-ordering check.

The five methods are:

- public/blog-aligned Muon;
- paper-structured block4 Newton-Muon;
- no `mlp.c_proj` K (`none`);
- diagonal `mlp.c_proj` K (`diag`);
- dense `(4d) x (4d)` `mlp.c_proj` K (`full`, mechanism control only).

The default B matrix contains 30 runs: two model sizes, three learning rates,
five methods, and seed 2026. It is a sensitivity and early-ordering test, not
a replacement for the three-seed long-run experiment.

Both model sizes scan matrix LR `0.005`, `0.01`, and `0.02`, with `0.01` as
the shared formal reference LR. The prior 12L `0.04` point was removed because
`0.02` already produced repeated instability. The 12L jobs stop at step 5000
but retain the formal 12L/100M run's `lr_decay_iters=12208`, so they are an
actual prefix of the formal schedule.
The 24L formal schedule already uses `lr_decay_iters=3000`, which is retained.
The matrix learning rate is constant in the shared project training pipeline.

Inspect the full 30-run matrix without starting training:

```bash
python scripts/13_reference_lr_sanity/run_reference_lr_sanity.py \
  --dry-run \
  --python-exe python
```

Run the fastest useful conclusion check: five 24L runs at the formal learning
rate 0.01:

```bash
python scripts/13_reference_lr_sanity/run_reference_lr_sanity.py \
  --quick-ordering-gate \
  --suites owt24l_3k \
  --python-exe python \
  --continue-on-error
```

Run the ten-run two-scale ordering gate:

```bash
python scripts/13_reference_lr_sanity/run_reference_lr_sanity.py \
  --quick-ordering-gate \
  --python-exe python \
  --continue-on-error
```

After that ten-run gate, complete the LR grid without repeating the reference
LR runs (20 remaining runs):

```bash
python scripts/13_reference_lr_sanity/run_reference_lr_sanity.py \
  --lr-grid-remainder \
  --python-exe python \
  --continue-on-error
```

If only the five-run 24L gate was run first, finish B without repetition using
15 full-grid 12L runs plus 10 remaining 24L runs:

```bash
python scripts/13_reference_lr_sanity/run_reference_lr_sanity.py \
  --suites owt12l_3k \
  --python-exe python \
  --continue-on-error

python scripts/13_reference_lr_sanity/run_reference_lr_sanity.py \
  --lr-grid-remainder \
  --suites owt24l_3k \
  --python-exe python \
  --continue-on-error
```

Run the complete 30-run LR sanity matrix:

```bash
python scripts/13_reference_lr_sanity/run_reference_lr_sanity.py \
  --python-exe python \
  --continue-on-error
```

All runs force the compact `paper` W&B profile and force
`wandb_log_tables=False`; the runner intentionally provides no switch that can
enable W&B tables. The logged scalar curves and counters
are sufficient to recover best/final/late validation loss, steps/tokens/time to
a common loss, current and peak CUDA allocation, and exact K-state bytes.
