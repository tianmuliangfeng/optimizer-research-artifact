# MECH-01 unified K-diagnostics validation

MECH-01 is an implementation gate, not a mechanism-result experiment.  It
must pass before MECH-02/03 numbers are interpreted scientifically.

The runner verifies:

- read-only loading and schema of official R1, GPT-bridge, LLaMA-124M, and
  LLaMA-1B checkpoints;
- exact target routing:
  `F.gelu(c_fc(x)) -> c_proj` for R1 and
  `silu(gate_proj(x))*up_proj(x) -> down_proj` for LLaMA;
- two disjoint K-build batches and two disjoint held-out gradient batches;
- fresh covariance, exact diagonal, off-diagonal energy, damped spectrum,
  Cholesky inverse health, and shadow-update geometry;
- family-specific production momentum:
  R1 sum-form momentum/Nesterov versus LLaMA EMA-form momentum/Nesterov;
- the exact Newton–Schulz function imported from the source saved with the
  run, plus the paired `triton_kernels.py`;
- repeatability and content-sensitive invariance of model,
  optimizer, loader, RNG, next-batch, and checkpoint-file state;
- sequential one-layer-at-a-time dense-K processing, so LLaMA-1B never needs
  every layer's dense K resident simultaneously;
- a fixed tensor bundle that can be replayed under both pinned runtimes.

No stage calls `optimizer.step()`, changes the checkpoint, or uses W&B.

## Frozen numerical conventions

- `K = X^T X / N` is constructed only from the two build batches.
- Held-out gradients never contribute rows to K.
- `ridge_mult=0.2`, `ridge_eps=1e-8`.
- R1 `diag` and `block4` use a separate diagonal-mean ridge for each of the
  four architecture-defined GELU quarters.  R1 `dense_full` uses the global
  diagonal mean.
- LLaMA `diag` and `dense_full` use the global diagonal mean.  `block4` is
  excluded because SwiGLU `down_proj` has no architecture-defined four-way
  partition.
- Newton–Schulz uses the production BF16 five-step path.
- The smoke layers default to first, middle, and last.
- Runtime replay tolerance defaults to `atol=5e-4`, `rtol=5e-3`; changing
  these values creates a different analysis contract and must be recorded.

## Stages

`run_mech01.py` is a standard-library controller.  Always pass the pinned
training Python through `--python-exe`.

1. `--preflight` loads the checkpoint on CPU and audits schema/source/route.
2. `--numerical-smoke` performs the three-layer GPU smoke twice and exports
   `tensor_bundle.pt`.
3. `--replay-bundle` replays exactly one bundle in one runtime.
4. `--compare-replays` compares two `replay.json` files without importing
   PyTorch.

For the host/runtime equivalence control, both replay jobs must use the same
scientific bundle, the same exact training-source file, the same
`triton_kernels.py`, and the same MECH-01 worker version.  Only the pinned
Python/runtime/host may differ.  The comparison stage enforces all four
fingerprints before accepting numerical tolerances.

R1 preflight skeleton:

```bash
"$CTRL_PY" scripts/27_mech01_unified_k_diagnostics/run_mech01.py \
  --preflight \
  --python-exe "$TRAIN_PY" \
  --family r1 \
  --checkpoint "$R1_CHECKPOINT" \
  --source-script "$R1_WORKSPACE/train_r1_none.py" \
  --triton-kernels "$R1_WORKSPACE/triton_kernels.py" \
  --host-id r1-native-h100 \
  --execution-domain r1-native
```

Numerical smoke adds:

```bash
  --numerical-smoke \
  --data-pattern "$R1_REPO/data/fineweb10B/fineweb_val_*.bin" \
  --layers 0 6 11 \
  --device-batch-size 1 \
  --sequence-length 128 \
  --probe-offsets 0 4096 8192 12288 \
  --candidates none diag block4 dense_full
```

The first remote execution should be `--preflight`, followed by the numerical
smoke only after its manifest says `"passed": true`.

For LLaMA-1B, `--source-script` must point to the
`train_llama_swiglu_base.py` copied into the run artifact, because that file
contains the exact model/optimizer implementation.  Also pass the saved
1B shape wrapper through `--profile-script`; both hashes are then retained
alongside the checkpoint-inferred 18L/D2048/FF5504 architecture.

## Formal smoke artifacts

- `checkpoint_schema.json`
- `route_audit.json`
- `batch_contract.json`
- `momentum_audit.json`
- `repeatability.json`
- `production_path_audit.json`
- `diagnostics.json` and `diagnostics.csv`
- `state_invariance.json`
- `streaming_contract.json`
- `tensor_bundle.pt` and `tensor_bundle_manifest.json`
- `runtime.json`, `checks.csv`, `status.json`, and `mech01_manifest.json`

The tensor bundle stores activation rows, held-out gradient, historical
momentum, fresh covariance, numerical contract, and provenance.  It is a
diagnostic runtime control and is not a new training checkpoint.
