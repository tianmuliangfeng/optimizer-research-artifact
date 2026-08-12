# Paper-Block4 Correction Runs

This family replaces legacy cross-dataset and depth comparisons with four
fully rerun methods under one shared Muon pipeline:

- blog-aligned Muon;
- paper-structured block4 Newton-Muon;
- no `mlp.c_proj` K (`none`);
- diagonal `mlp.c_proj` K (`diag`).

The corrected baseline follows the Newton-Muon public implementation's matrix
partition:

- packed QKV: one full `d x d` K state;
- attention output: one full `d x d` K state;
- MLP expansion: one full `d x d` K state;
- MLP contraction (`mlp.c_proj`): four independent `d x d` K states.

All commands explicitly set both the Muon pipeline (Nesterov, EMA-form
momentum, separate Q/K/V orthogonalization, aspect-ratio scaling, BF16
Newton-Schulz) and the Newton-Muon K structure/hyperparameters
(`beta=0.95`, ridge `0.2`, first refresh on optimizer step 31 and then every
32 steps, covariance initialization `0.001 I`, identity initial inverse, and
all activations from the refresh batch). They do not rely on `train.py`
defaults.

All five formal suites now use one shared constant matrix learning rate
`0.01`. Earlier 12L/18L `0.02` runs showed repeated Muon instability and are
retained only as learning-rate stress controls. The new commands use new W&B
projects and a new run prefix so those exploratory runs cannot be mixed into
the formal reset.

Default dry run:

```bash
python scripts/12_paper_block4_correction/run_paper_block4_correction.py \
  --dry-run \
  --python-exe python
```

Two-run GPU preflight (Muon + block4, 33 steps so the first K refresh is
exercised):

```bash
python scripts/12_paper_block4_correction/run_paper_block4_correction.py \
  --preflight \
  --python-exe python
```

Formal overnight queue:

```bash
python scripts/12_paper_block4_correction/run_paper_block4_correction.py \
  --python-exe python \
  --continue-on-error
```

The default formal plan contains 60 runs split across three new W&B projects:

- `Selective-Newton-Muon-MainConf-ReferenceReset-LR001-12L-20260717`: 24 runs;
- `Selective-Newton-Muon-MainConf-ReferenceReset-LR001-18L-20260717`: 12 runs;
- `Selective-Newton-Muon-MainConf-ReferenceReset-LR001-24L-20260717`: 24 runs.

- five primary configurations;
- four methods per configuration;
- seeds 2024, 2025, and 2026.

The short `owt12l_5k` configuration remains available for a focused pilot but
is not in the default queue. The unstable optional WikiText-24L LR=0.02 suite
has been removed. Existing dense-`full` runs remain valid only as mechanism
controls and must no longer be labelled as the paper Newton-Muon baseline.

Every formal command forces `wandb_log_profile=paper` and
`wandb_log_tables=False`; the runner exposes no option to enable diagnostic
tables.
