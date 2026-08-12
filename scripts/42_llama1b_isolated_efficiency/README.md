# 42 - LLaMA-1B isolated efficiency audit

This directory is the frozen, recoverable implementation of the paper-facing
LLaMA-1B efficiency benchmark.

The remote runtime and 50-shard FineWeb cache are taken from the clean
`Newton-Muon-official-r0` checkout pinned by the experiment contract.

The only remote entry point is:

`commands/42_llama1b_isolated_efficiency/20260729_llama1b_isolated_efficiency.sh`

The benchmark runs Muon, the paper's full-K Newton-Muon-family LLaMA control
(`newton_full`), Selective-none, and Selective-diag sequentially on physical
GPU0. `newton_full` is a LLaMA adaptation/control, not an upstream official
Newton-Muon LLaMA benchmark. Physical GPU1 must remain idle. The benchmark
uses 32 warm-up updates, 512 timed updates, and four balanced cyclic
method-order rotations.

The main outputs are synchronized update-only tokens/s and steps/s, plus CUDA
allocated/reserved peaks measured over training updates 33 through 544. The
derived trainer resets CUDA peak statistics after update 32 and seals the peak
before final validation. A continuous GPU-process monitor supplements the
before/after exclusive-node certificates for every formal cell.

Experiment 42 contributes implementation performance and memory evidence only.
Experiment 20 remains the source of three-seed optimizer-quality evidence.
Concurrent long-training timing, including a possible 10B-token screen, must
not be combined with this benchmark.

Recovery never deletes an interrupted attempt. Re-run the command with the
printed `RUN_DIR`; completed, fully certified cells are reused and an
incomplete cell starts a new attempt.

Accepted formal run: `20260729T105505+0000`. The independent acceptance report
is `docs/reports/20260730_llama1b_isolated_efficiency_review.md`.
