# 42 - LLaMA-1B isolated efficiency audit contract

Version: 2026-07-29.3

## Purpose

This experiment measures the implementation throughput and CUDA-memory
trade-off of four frozen LLaMA-1B optimizer roles on one certified idle H100
node:

1. `muon`;
2. `newton_full`, the original Newton-Muon control for LLaMA;
3. `down_none`, Selective-none;
4. `down_diag`, Selective-diag.

Quality conclusions remain frozen to experiment 20's three-seed, 6200-update
formal training. The 544-update timing runs are not quality evidence.

The runtime code and FineWeb cache come from the clean, commit-pinned
`Newton-Muon-official-r0` checkout. The LF-byte SHA-256 of
`triton_kernels.py` is pinned in the JSON contract; this is the same canonical
source identity used by the official R0 provenance validator.

## Frozen execution

- Seed: 2026, with the experiment-20 initialization fingerprint pinned.
- Global batch: 512 sequences.
- Device batch: 8 sequences.
- Sequence length: 1024.
- Tokens per optimizer update: 524,288.
- Warm-up: 32 complete optimizer updates, excluded from the official timer.
- Timed window: updates 33 through 544, exactly 512 updates and 268,435,456
  tokens.
- Validation: one microbatch at step 0 and after update 544, used only as a
  finite-value gate.
- No warmdown, checkpoint, resume, or W&B process.
- Physical GPU0 performs every timed run; physical GPUs 0 and 1 must both be
  idle before and after every timed cell.
- A ten-second continuous process monitor must observe GPU1 idle throughout
  each timed cell and at most the one trainer process on GPU0.

There are four cyclic method-order rotations. Every method therefore occupies
positions one through four exactly once. A formal bundle contains 16 timed
runs.

## Measurement

The audited trainer synchronizes CUDA around every optimizer update.
`steady_train_s` excludes the first 32 updates and excludes validation,
checkpoint I/O, W&B, logging, and controller overhead. It is therefore
reported explicitly as synchronized update-only throughput, not end-to-end
job throughput.

The minimally derived source synchronizes and resets CUDA peak statistics
immediately after update 32. It records the allocated and reserved peaks over
updates 33 through 544 before the final validation can affect them. The
legacy full-run `peak_allocated_bytes` field is retained for compatibility but
is not used by the experiment-42 analysis.

Every raw run, source, terminal log, summary, runtime, full data-cache
fingerprint, initialization, continuous monitor, and exclusive-node
certificate is retained and hashed.

## Comparisons

Selective-none and Selective-diag are each compared separately with Muon and
original Newton-Muon. Original Newton-Muon versus Muon is the baseline
contrast. Diag versus none is not a primary comparison.

The analysis reports raw repeats, aggregate throughput, paired repeat deltas,
position effects, state bytes, and measured allocated/reserved peaks. The
predeclared +/-1% throughput band is descriptive. With four rotations, an
integrity pass is not a claim of statistical equivalence or superiority.

## Recovery and stopping

An interrupted cell is never continued from optimizer state. Its partial
attempt stays in place, and recovery launches a new attempt for that exact
repeat, position, and method. Completed cells are reused only after their
summary, hashes, source, runtime, initialization, state bytes, continuous
monitor, and both exclusive-node certificates validate.

Only NaN/Inf, OOM, source/runtime/data drift, certificate failure, or another
integrity failure stops the experiment. Observed throughput or loss ordering
never triggers early stopping.
