# Official R1 depth × c_proj K-mode

This family migrates the completed OWT/WikiText depth experiment to the
official R1 architecture and implementation.

Selected `mlp.c_proj` layers use `none` or efficient `diag`; unselected layers
retain official `block4`. This makes the experiment a localized intervention
on the official Newton-Muon baseline rather than a new dense-full hybrid.

The formal matrix contains 36 runs:

- five rules (`early`, `center`, `late`, `edge`, `all`);
- two selected-layer modes (`none`, `diag`);
- three seeds (2024, 2025, 2026);
- official block4 and Muon anchors for each seed.

The frozen statistical and interpretation rules are in
`ANALYSIS_CONTRACT_20260725.md`.

## Remote setup

```bash
PROJECT_ROOT="${SNM_REPO}"
OFFICIAL_REPO="${SNM_OFFICIAL_REPO}"
CTRL_PY="${SNM_CONTROLLER_PYTHON}"
TRAIN_PY="${SNM_TRAINING_PYTHON}"
RUNNER="$PROJECT_ROOT/scripts/29_r1_depth_kmode/run_r1_depth_kmode.py"
BATCH="$PROJECT_ROOT/scripts/29_r1_depth_kmode/run_three_seed_batch.py"
RESULTS="${SNM_RESULTS_ROOT}/29_r1_depth_kmode/results"

cd "$PROJECT_ROOT"
```

Inspect generated sources and the full method matrix without using data/GPU:

```bash
CUDA_VISIBLE_DEVICES=0 "$CTRL_PY" "$RUNNER" \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY" \
  --seed 2026 \
  --dry-run \
  --wandb-mode disabled
```

Run a one-GPU data/runtime/initialization preflight:

```bash
CUDA_VISIBLE_DEVICES=0 "$CTRL_PY" "$RUNNER" \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY" \
  --seed 2026 \
  --preflight \
  --wandb-mode online
```

`CTRL_PY` and `TRAIN_PY` are intentionally different. The controller owns
W&B upload and must be able to import `wandb`; the training interpreter owns
the pinned PyTorch/CUDA/Triton runtime and does not need W&B.

After preflight passes, one command runs all three seeds on both GPUs. Every
six-method shard first obtains its own 34-step smoke certificate and then
starts its matching formal run:

```bash
"$CTRL_PY" "$BATCH" \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY" \
  --seeds 2024 2025 2026 \
  --devices 0 1 \
  --smoke-steps 34 \
  --results-dir "$RESULTS" \
  --wandb-mode online
```

The batch performs an additional fail-fast controller check before launching
either smoke shard. It rejects a controller/training interpreter collision and
verifies that the controller can import `wandb`; the July 25 failure therefore
cannot recur after consuming GPU time merely because the training interpreter
was used as the controller.

The maintained command wrapper exposes the same sequence as `check`,
`dry-run`, `preflight`, and `formal`:

```bash
bash commands/29_r1_depth_kmode/20260806_ex29_r1_depth_kmode.sh check
bash commands/29_r1_depth_kmode/20260806_ex29_r1_depth_kmode.sh dry-run
bash commands/29_r1_depth_kmode/20260806_ex29_r1_depth_kmode.sh preflight
bash commands/29_r1_depth_kmode/20260806_ex29_r1_depth_kmode.sh formal
```

The batch writes full stdout/stderr to per-shard logs and only prints compact
START/DONE/FAILED messages, avoiding terminal or tmux flooding. Concurrent
same-seed shards use separate result namespaces, so their smoke manifests
cannot collide. Every `none/diag` pair stays on one physical GPU;
all-none/all-diag and the block4/Muon anchors also share a shard. The two
shards cross over GPUs between seeds to avoid complete method–GPU confounding.

Formal checkpoints are disabled. Metrics, summaries, manifests, source
diffs/hashes, logs, memory accounting, and W&B histories are retained.

Based on completed R1 runtimes, each six-run shard is approximately 12–13
hours. Two GPUs require three waves, approximately 36–40 hours total. Reserve
a 60-hour host window to cover smoke, initialization, W&B, and runtime
variance.

## Accepted formal analysis

The accepted three-seed batch is `20260806T100702+0000`. Analyze a downloaded
bundle and the nine metric-specific W&B CSV exports with
`analyze_r1_depth_kmode_formal.py`. The analyzer hard-checks the 36-run grid,
completion/source/init lineage, validation steps, memory accounting, W&B run
coverage, and every exported local/W&B value. It also reads the frozen OWT and
WikiText summaries beneath the explicitly supplied
`--reference-results-root` to produce the cross-environment depth table.

Primary outputs are:

- `analysis_verdict.json`;
- `R1_DEPTH_KMODE_FORMAL_ANALYSIS_20260809.md`;
- `paired_depth_contrasts.csv` and `rule_multiseed_summary.csv`;
- `cross_environment_depth_transfer.csv`;
- `data_quality_checks.json` and `wandb_local_crosscheck.csv`;
- two dependency-free SVG figures and exact input inventories/hashes.

The accepted classification is
`uniform_diag_direction_transfer_with_environment_specific_depth_amplitude`:
all 15 R1 endpoint pairs favor diag, while the OWT/WikiText edge-strong
magnitude ordering does not transfer. Timing remains ineligible. Do not use
the output to claim a universal edge mask or individual-layer causality.
