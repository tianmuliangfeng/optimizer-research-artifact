# Experiment 44 — Modded-NanoGPT Medium Track Record #17 (455M)

This directory implements a paper-grade paired quality comparison on a
single-H100 adaptation of the archived Modded-NanoGPT Medium Track Record #17:

- `muon`;
- `original_newton_muon`;
- `selective_none`;
- `selective_diag`.

The primary comparisons are each Selective method against both Muon and
original Newton–Muon. `diag` versus `none` is secondary. The experiment must
be described as a **single-H100 adaptation of the upstream Record #17
recipe**, not as an unchanged official reproduction.

## Frozen provenance and budget

The vendored upstream trainer is:

```text
repository: https://github.com/KellerJordan/modded-nanogpt
commit:     9e7218468ea864a33053142c196d90bbf3ed48e1
path:       records/track_2_medium/2025-11-12_BlockMaskRedundantOp/train_gpt_medium.py
blob:       8504813a5ba0b1bf981fd6ad9d6348bfa1754b0f
LF SHA256:  03d91174eed5e8cbf57063a1e997eb98570dde7a09ba9b2c94aa36e9d5eb94cb
```

The formal design has 12 cells: four methods × seeds 2024, 2025, and 2026.
Every cell performs 5,960 optimizer updates with 524,288 tokens per update,
for exactly 3,124,756,480 training tokens. On one GPU, each logical update is
eight 65,536-token microbatches. The loader first obtains the original global
batch and then splits it, preserving the upstream shard-boundary semantics.
For Newton-family cells, K statistics intentionally aggregate all eight
sequential microbatches (524,288 tokens per counted update). This is a frozen
single-H100 adaptation rule; it is not claimed to reproduce a hypothetical
eight-rank Newton implementation whose parameter owner used only rank-local
activations.

The smoke gate runs four methods at seed 2026 for 27 counted updates, crossing
the first Newton refresh at update 24. A separate 26-update instrumentation
warmup compiles the refresh path and is fully rolled back before counted
training.

## Remote entry point

Run from the repository root on the otherwise-idle LLaMA H100 host. This
pre-formal host amendment changes provenance only; the frozen model, methods,
seeds, data order, hyperparameters, endpoints, and analysis are unchanged:

```bash
bash commands/44_newton_muon_record17_455m/20260730_newton_muon_record17_455m.sh
```

The command defaults to physical GPU1, leaving GPU0 available for experiment
43. Override it only when intentional:

```bash
EXP44_GPUS=1 bash commands/44_newton_muon_record17_455m/20260730_newton_muon_record17_455m.sh
```

The controller uses `${SNM_CONTROLLER_PYTHON}`; training uses the
existing pinned runtime
`${SNM_TRAINING_PYTHON}`
(Python 3.10.12, PyTorch `2.8.0+cu126`, CUDA 12.6, Triton 3.4.0). No additional
virtual environment or Torch download is required.

The archived upstream execution log records PyTorch `2.7.0+cu126`, but
experiment 44 is a controlled single-H100 adaptation rather than an unchanged
upstream reproduction. Its Newton activation-statistics overlay uses the
PyTorch 2.8 FP32 `out_dtype` GEMM path. The generated unified trainer keeps
FlexAttention and `torch.compile(model)` while count-checked disabling the
optional `compiled_autograd` wrapper for all four methods. This avoids the
PyTorch 2.8 nested FX/FlexAttention failure without changing attention
mathematics or any method-specific training rule.
The official
Newton–Muon/data checkout is
`${SNM_OFFICIAL_REPO}`.

To recover the same run, reuse the printed `RUN_DIR`:

```bash
RUN_DIR="${SNM_RESULTS_ROOT}/44_newton_muon_record17_455m/<timestamp>" \
EXP44_GPUS=1 \
bash commands/44_newton_muon_record17_455m/20260730_newton_muon_record17_455m.sh
```

The first launch seals a self-contained source snapshot. Recovery executes
that snapshot, reuses only integrity-verified completed cells, preserves
failed attempts, and restarts an incomplete cell from initialization. W&B is
uploaded by the controller only after local scientific evidence and analysis
are sealed; upload failure never retrains a valid cell.

Experiment 43 on GPU0 and experiment 44 on GPU1 may run concurrently for
quality. All timing from this experiment is deliberately ineligible for
throughput or wall-clock claims; use the isolated efficiency audits instead.
The recorded dynamic CUDA peak is labeled
`counted_run_after_warmup_reset_including_validation`; it includes scheduled
and final validation and must not be described as a training-step-only peak.

The machine-readable authority is `record17_contract.json`; the Chinese
review document is `RECORD17_455M_CONTRACT.md`.
