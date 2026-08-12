# OWT 12L depth × c_proj K-mode experiment

This experiment reruns the old layer-position question as a paired
`none`/`diag` design.  Selected `mlp.c_proj` depths use the requested K
representation; all unselected c_proj matrices and all other matrix
families stay on full Newton-Muon.

The formal batch contains 36 runs:

- 5 depth rules × 2 K modes × 3 seeds = 30 treatment cells;
- full Newton-Muon and Muon anchors × 3 seeds = 6 anchors.

The primary statistics and interpretation boundaries are frozen in
`ANALYSIS_CONTRACT_20260724.md`.

## Required source synchronization

The source checkout must contain the new `cproj_k_layers` option in
`train.py`, `optimizer_factory.py`, and `optimizers.py`.  The runner checks
for those markers before emitting or starting jobs.

## Remote setup

```bash
PROJECT_ROOT="${SNM_REPO:-$(pwd)}"
SOURCE_REPO="${SELECTIVE_NEWTON_MUON_SOURCE_REPO:-$PROJECT_ROOT/backends/nanogpt}"
CTRL_PY=${SNM_CONTROLLER_PYTHON}
TRAIN_PY=${SNM_TRAINING_PYTHON}
RUNNER="$PROJECT_ROOT/scripts/25_owt_depth_kmode/run_owt_depth_kmode.py"

export SNM_REPO="$PROJECT_ROOT"
export SELECTIVE_NEWTON_MUON_SOURCE_REPO="$SOURCE_REPO"
export CUDA_VISIBLE_DEVICES=0
cd "$PROJECT_ROOT"
```

Inspect all 36 formal commands without launching:

```bash
"$CTRL_PY" "$RUNNER" --python-exe "$TRAIN_PY" --dry-run
```

Run the six-cell numerical smoke.  It covers center/all × none/diag plus
full/Muon and reaches the first K refresh:

```bash
"$CTRL_PY" "$RUNNER" \
  --python-exe "$TRAIN_PY" \
  --numerical-smoke \
  --smoke-steps 34 \
  --wandb-mode disabled
```

After smoke passes, run the complete three-seed formal batch:

```bash
"$CTRL_PY" "$RUNNER" \
  --python-exe "$TRAIN_PY" \
  --formal \
  --wandb-mode online \
  --continue-on-error
```

One old OWT 12L/50M run took roughly 10.5 minutes on the matched H100.
Thirty-six sequential cells therefore imply about 6.3 hours of training;
reserve a 9-hour host window for validation, startup variance, and any
failed-cell continuation.  If another GPU on the node is training, keep
quality and per-process memory but mark timing ineligible.

## Completed three-seed analysis

The 2026-07-24 delivery contains all 36 preregistered runs.  Analyze the
17 metric exports with:

```bash
python scripts/25_owt_depth_kmode/analyze_owt_depth_kmode_exports.py \
  --inputs /path/to/wandb_export_*.csv \
  --output-dir /path/to/25_owt_depth_kmode/analysis_20260724_multiseed \
  --skip-notebook-execution
```

The frozen step-4500 paired contrast is negative for all 15
rule/seed pairs.  The all-depth mean `diag-none` is `-0.013930`
(`SD=0.001057`, `3/3` negative).  Direction is uniform, while magnitude
is depth-modulated: center/late are weaker than all in `3/3` seeds and
edge is stronger than all in `3/3` seeds.  No additional OWT seeds are
needed.  Cross-dataset or R1 work should begin with a seed2026 screen and
test this frozen magnitude pattern before adding confirmation seeds.
