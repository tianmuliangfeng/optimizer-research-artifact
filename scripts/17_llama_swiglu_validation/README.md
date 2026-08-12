# LLaMA / SwiGLU architecture validation

This experiment asks whether the `down_proj` K-structure effect transfers from
the R1 GPT/GELU model to a LLaMA-style model. It is a separate experiment
family: old-host and new-host timing or CUDA peak-memory values must not be
pooled, but quality comparisons among methods in one complete batch on the new
host are valid.

## Architecture audit

The model fixes:

- 12 layers, width 768, 12 attention heads, context 1024;
- learned RMSNorm with epsilon `1e-6`;
- RoPE with base 10000;
- separate Q, K, V, and O projections;
- SwiGLU with `d_ff=2048` and gate/up/down projections;
- no bias and no dropout;
- tied token embedding and language-model head;
- FineWeb10B, global batch 512, 6200 updates, and the R1 validation budget.

It has 123,551,232 parameters. R1 has 123,532,032, so the difference is only
19,200 parameters (about 0.0155%); those extra parameters are the learned
RMSNorm gains.

## Five methods and causal boundary

| Method | Hidden matrices | `down_proj` K |
|---|---|---|
| `adamw` | AdamW recipe | none |
| `muon` | public/reference Muon | none |
| `newton_full` | reference Muon plus right K | dense `2048 x 2048` |
| `down_none` | same Newton pipeline | none only for down |
| `down_diag` | same Newton pipeline | diagonal length 2048 |

`newton_full`, `down_none`, and `down_diag` use identical learning rates and
identical optimizer mathematics except for the down-input K representation.
They are the controlled structural comparison. AdamW and Muon are method-level
baselines, not K-only causal contrasts.

Q/K/V share one dense K per layer because their right-hand activation is
identical. Gate/up likewise share one K. O and down each have their own group.
Sharing is mathematically identical to storing redundant copies, and the
manifest records this mapping explicitly.

With FP32 covariance and applied inverse, exact persistent K-state is:

| Method | K-state |
|---|---:|
| `down_none` | 162.0000 MiB |
| `down_diag` | 162.1875 MiB |
| `newton_full` | 546.0000 MiB |
| `muon`, `adamw` | 0 MiB |

The public/reference Muon path uses EMA momentum, Nesterov lookahead, five BF16
Newton--Schulz steps, separate Q/K/V orthogonalization, and the public
width/height shape adjustment. The shared matrix LR is `0.01`. Backup AdamW
uses LR `0.0036`; the pure AdamW hidden-matrix LR is `0.000576`. All methods use
the same linear 1800-step warmdown and zero weight decay, matching the R1
FineWeb protocol as closely as the modern Muon convention permits.

`block4` is deliberately absent. SwiGLU's gated `d_ff=2048` down input has no
four-`d` concatenation corresponding to the official GPT `c_proj` construction.

## Safety and evidence

- preflight launches every method in a fresh process and requires one identical
  SHA-256 over all initialized named parameters;
- the architecture mapping, parameter allocation, K groups, runtime, data
  shards, and source hash are certified;
- the exact-shape smoke is 34 steps, not 10, so it reaches the first K refresh
  at update 32;
- formal training requires that smoke certificate;
- training writes only local logs while timing; W&B uploads after a completed,
  validated run;
- checkpoints are atomic and contain model, optimizers, K state, activation
  statistics, loader cursor, already-fetched next batch, and all RNG states;
- a resumed method continues from its last 128-step checkpoint. Its loss data
  remain valid, but its speed result is marked `timing_comparable=false` because
  compilation is warmed again after a restart;
- completed local evidence is retained even if W&B upload is interrupted.

## Commands on the new H100 host

Run from `selective-newton-muon`. The controller owns W&B; the
training interpreter owns CUDA. The current new-host data inventory points to
`Newton-Muon-official`, not `Newton-Muon-official-r0`.

```bash
cd ${SNM_REPO}

export CUDA_VISIBLE_DEVICES=0
CTRL_PY=${SNM_CONTROLLER_PYTHON}
TRAIN_PY=${SNM_TRAINING_PYTHON}
OFFICIAL_REPO=${SNM_OFFICIAL_REPO}
RUNNER=scripts/17_llama_swiglu_validation/run_llama_swiglu_validation.py
```

Inspect the immutable plan without GPU or data access:

```bash
"$CTRL_PY" "$RUNNER" \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY" \
  --wandb-mode disabled \
  --dry-run
```

Run the complete runtime, data, architecture, initialization, and W&B
preflight:

```bash
"$CTRL_PY" "$RUNNER" \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY" \
  --wandb-mode online \
  --preflight
```

Run the required five-method, exact-formal-shape 34-step smoke:

```bash
"$CTRL_PY" "$RUNNER" \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY" \
  --wandb-mode disabled \
  --numerical-smoke
```

Then substitute the printed smoke manifest and start formal seed 2026:

```bash
SMOKE_MANIFEST=/absolute/path/to/smoke_batch/llama_manifest.json

"$CTRL_PY" "$RUNNER" \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY" \
  --seed 2026 \
  --smoke-manifest "$SMOKE_MANIFEST" \
  --wandb-mode online
```

If power is lost, substitute the printed formal artifact directory. Use the
same method order and settings:

```bash
FORMAL_ARTIFACT=/absolute/path/to/2026..._formal_seed2026

"$CTRL_PY" "$RUNNER" \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY" \
  --seed 2026 \
  --resume-batch "$FORMAL_ARTIFACT" \
  --wandb-mode online
```

Default W&B project:

```text
Selective-Newton-Muon-MainConf-LLaMA-SwiGLU-20260720
```

Local artifacts contain `terminal.log`, `metrics.csv`, `summary.json`, the
exact training source, atomic checkpoint, W&B upload receipt, batch CSV,
`llama_plan.json`, and `llama_manifest.json`.

## Three-seed result record (2026-07-22)

Formal seeds 2024, 2025, and 2026 are complete for all five methods at 6200
steps / 3,250,585,600 tokens. The three-seed mean endpoint losses are
`newton_full=3.266853`, `down_diag=3.266926`, `down_none=3.266973`,
`muon=3.267578`, and `adamw=3.363416`.

The primary paired contrast `down_diag - down_none` is `-0.000310`,
`-0.001187`, and `+0.001357` for seeds 2024, 2025, and 2026 respectively;
its mean is `-0.000047`. This does not reproduce the stable GPT-R1 mean
contrast of `-0.005567`. The matched-seed architecture interaction
`LLaMA(diag-none) - GPT(diag-none)` is `+0.005520` on average and has the
same positive direction in all three seeds. Together with the completed
host bridge, this supports architecture-associated optimizer behavior.

The full reproducible analysis, evidence checks, derived CSVs, and portable
HTML report are stored under:

```text
${SNM_RESULTS_ROOT}/17_llama_swiglu_validation/analysis/wandb_20260722_multiseed/
```

No practical non-inferiority margin was frozen before the remaining-seed
results were inspected, so this record must not be described as a formal
equivalence or non-inferiority test. Timing remains diagnostic-only.
