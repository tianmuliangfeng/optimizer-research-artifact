# Experiment 44: Medium Track Record #17 paired analysis

All 12 formal cells passed local integrity checks. The seed is the inferential unit; validation checkpoints are not treated as replicates.

## Mean final validation loss

- `muon`: 2.919811
- `original_newton_muon`: 2.918297
- `selective_none`: 2.917476
- `selective_diag`: 2.918910

## Paired final-loss contrasts

Deltas are candidate minus comparator, so negative is better.

- `selective_none_vs_muon`: -0.002335 (95% t CI -0.002764, -0.001905); direction_candidate_better_but_practically_unresolved.
- `selective_none_vs_original`: -0.000821 (95% t CI -0.001904, +0.000262); quality_equivalent_within_margin.
- `selective_diag_vs_muon`: -0.000900 (95% t CI -0.001582, -0.000219); quality_equivalent_within_margin.
- `selective_diag_vs_original`: +0.000614 (95% t CI -0.001264, +0.002491); direction_candidate_worse_but_practically_unresolved.
- `original_vs_muon`: -0.001514 (95% t CI -0.003026, -0.000003); direction_candidate_better_but_practically_unresolved.
- `diag_vs_none`: +0.001435 (95% t CI +0.000484, +0.002385); direction_candidate_worse_but_practically_unresolved.

## Statistical boundary

- Frozen practical margin: +/-0.002.
- `diag_vs_none` is secondary and cannot drive seed expansion.
- Original Newton-Muon versus Muon is a mandatory benchmark anchor.
- Concurrent quality runs are timing-ineligible; no throughput or wall-clock claim is derived here.
- CUDA peaks cover the counted run after warmup reset, including validation; they are not training-step-only peaks.
- Newton-family K statistics aggregate all eight sequential single-H100 microbatches; strict equivalence to a hypothetical eight-rank owner-local K path is not claimed.
- Statistical append trigger: True.
