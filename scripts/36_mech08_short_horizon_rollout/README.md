# MECH-08: matched-start short-horizon rollout

MECH-08 tests whether the one-step family ranking measured by MECH-07 predicts
what happens after real optimizer updates. It restarts four frozen LLaMA-1B
checkpoints and applies Muon, original Newton-Muon, Selective-diag, and
Selective-none for 128 optimizer steps on exactly matched data streams.
The bridge predictor is the MECH-07 shadow loss at the common production step
multiplier 1.0, matching the actual rollout learning rate. Per-algorithm best
line-search values are retained only as secondary provenance.

The primary comparisons are:

1. Selective-diag versus Muon.
2. Selective-none versus Muon.
3. Selective-diag versus original Newton-Muon.
4. Selective-none versus original Newton-Muon.

Original Newton-Muon versus Muon is a baseline family contrast.
Selective-diag versus Selective-none is explicitly not a primary contrast.

## Frozen intervention

Each origin/replica shares model parameters, backup AdamW state, historical
matrix momentum, learning rates, training tokens, and held-out evaluation
tokens. Origin-specific K state is discarded. Newton candidates receive a
fresh K built from a frozen validation build split before optimizer step 1.
The build and evaluation validation windows are disjoint.

The formal matrix is:

- 2 stages (step 1000 and step 6200);
- 2 origins per stage (Muon and original Newton-Muon);
- 4 applied algorithms;
- 3 disjoint data-order replicas;
- 128 optimizer steps per cell.

That is 48 cells and 6144 optimizer steps in total. Evaluations occur at
steps 0, 16, ..., 128.

## Scope boundary

This is a causal mechanism bridge, not a final-training or systems benchmark.
Its pass/fail flag certifies execution and data integrity only. Worker timing
and CUDA allocation fields are diagnostic and must not be cited as the final
paper throughput or peak-memory benchmark. Fair tokens/s, steps/s, peak memory,
and equal-budget tuning sensitivity remain a separate submission benchmark.

No MECH-08 outcome may be used to tune the frozen algorithms.

## Files

- `rollout_contract.json`: immutable scientific and execution contract.
- `mech07_prediction_reference.csv`: frozen MECH-07 predictions.
- `mech08_worker.py`: one origin/algorithm/data-replica rollout.
- `run_mech08.py`: checkpoint certification, smoke gate, scheduling, resume,
  and final analysis.
- `analyze_mech08.py`: paired endpoint and prediction-realization analysis.
- `test_mech08_contract.py`: CPU-only contract and analysis unit tests.
- `test_mech08_worker.py`: PyTorch tensor-transfer regression tests.

The sole remote entry point is:

`commands/36_mech08_short_horizon_rollout/20260727_mech08_short_horizon_rollout.sh`

Outputs are written directly under:

`${SNM_RESULTS_ROOT}/36_mech08_short_horizon_rollout/<timestamp>`

No archive workflow is part of this experiment.
