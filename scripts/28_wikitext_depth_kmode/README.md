# WikiText-103 12L depth × c_proj K-mode

This family is the three-seed cross-dataset replication of family 25.
It preserves the OWT model, optimizer, depth rules, schedule, evaluation
grid, and paired `none/diag` design, changing only the dataset to the pinned
GPT-2-tokenized WikiText-103 subset.

The runner is backward-compatible with both the original and refactored
family-25 controller interfaces; updating family 28 does not require changing
the already-completed OWT runner on the remote host.

Formal matrix:

- five depth rules: early, center, late, edge, all;
- two selected-layer K modes: none, diag;
- three seeds: 2024, 2025, 2026;
- full Newton-Muon and Muon anchors for every seed;
- total: 36 runs.

The analysis decisions are frozen in
`ANALYSIS_CONTRACT_20260724.md`. The formal run is confirmatory for the OWT
direction and depth-amplitude pattern; it must not be used to select a new
post-hoc best mask.

## Pinned data

The runner requires:

```text
backends/nanogpt/data/wikitext103_gpt2_50m/
  train.bin
  val.bin
  meta.pkl
  prepare_summary.json
```

Before emitting commands it verifies semantic metadata, exact byte counts,
and the SHA-256 hashes of `train.bin` and `val.bin`. The resulting audit JSON
is written under:

```text
runs/
  28_wikitext_depth_kmode/data_audit/
```

If data are missing, prepare them with
`scripts/04_dataset_generalization/prepare_wikitext103_gpt2.py`. Do not use
`--force` on an existing dataset until its current fingerprint has been
checked.

## Remote setup

```bash
PROJECT_ROOT="${SNM_REPO:-$(pwd)}"
SOURCE_REPO="${SELECTIVE_NEWTON_MUON_SOURCE_REPO:-$PROJECT_ROOT/backends/nanogpt}"
CTRL_PY=${SNM_CONTROLLER_PYTHON}
TRAIN_PY=${SNM_TRAINING_PYTHON}
RUNNER="$PROJECT_ROOT/scripts/28_wikitext_depth_kmode/run_wikitext_depth_kmode.py"

export SNM_REPO="$PROJECT_ROOT"
export SELECTIVE_NEWTON_MUON_SOURCE_REPO="$SOURCE_REPO"
export CUDA_VISIBLE_DEVICES=0
cd "$PROJECT_ROOT"
```

Audit the data and inspect all 36 formal commands without launching:

```bash
"$CTRL_PY" "$RUNNER" \
  --python-exe "$TRAIN_PY" \
  --dry-run \
  --wandb-mode disabled
```

Run the six-path numerical smoke. It covers center/all × none/diag plus
full/Muon and crosses the first step-32 K refresh:

```bash
"$CTRL_PY" "$RUNNER" \
  --python-exe "$TRAIN_PY" \
  --numerical-smoke \
  --smoke-steps 34 \
  --wandb-mode disabled
```

After smoke passes, launch the complete three-seed batch:

```bash
"$CTRL_PY" "$RUNNER" \
  --python-exe "$TRAIN_PY" \
  --formal \
  --seeds 2024 2025 2026 \
  --wandb-mode online \
  --continue-on-error
```

The expected sequential training time is approximately 6.3 hours on the
matched H100 based on family 25. Reserve a 9-hour compute window, or 10 hours
when the remote host duration must be selected in advance. Do not run another
training process on the same physical GPU; otherwise timing becomes
ineligible, although loss and memory remain usable if no OOM occurs.

The formal configuration disables checkpoints. Results are uploaded to W&B
and command/data-audit records remain in the experiment artifact directory.

## Completed three-seed analysis

The 2026-07-25 delivery contains all 36 preregistered runs and passes the
frozen cross-dataset confirmation contract. Analyze the 17 W&B metric exports
with:

```bash
python scripts/28_wikitext_depth_kmode/analyze_wikitext_depth_kmode_exports.py \
  --inputs /path/to/wandb_export_*.csv \
  --output-dir /path/to/28_wikitext_depth_kmode/analysis_20260725_multiseed \
  --project-root /path/to/NanoGPT
```

All five rule means favor diag. The all-depth `diag-none` effect is
`-0.015348 +/- 0.006156 SD` with 3/3 negative seeds. The frozen magnitude
pattern also transfers: edge is stronger than all by `-0.005012`, while center
and late are weaker by `+0.003876` and `+0.006709`. No additional WikiText
seeds or masks are required; the next depth-specific gate is the R1
architecture/implementation screen.
