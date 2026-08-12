# Experiment 43: Record #28-scale paired analysis

All 16 formal cells passed local integrity checks. The seed is the inferential unit; validation checkpoints are not treated as replicates.

## Mean final validation loss

- `muon`: 3.277772
- `original_newton_muon`: 3.274799
- `selective_none`: 3.274724
- `selective_diag`: 3.274963

## Paired final-loss contrasts

Deltas are candidate minus comparator, so negative is better.

- `selective_none_vs_muon`: -0.003048 (95% t CI -0.007387, +0.001292); direction_candidate_better_but_practically_unresolved.
- `selective_none_vs_original`: -0.000074 (95% t CI -0.001937, +0.001788); quality_equivalent_within_margin.
- `selective_diag_vs_muon`: -0.002809 (95% t CI -0.008234, +0.002617); ci_spans_both_practical_boundaries.
- `selective_diag_vs_original`: +0.000165 (95% t CI -0.002526, +0.002856); ci_spans_both_practical_boundaries.
- `original_vs_muon`: -0.002973 (95% t CI -0.007379, +0.001433); direction_candidate_better_but_practically_unresolved.
- `diag_vs_none`: +0.000239 (95% t CI -0.001060, +0.001538); quality_equivalent_within_margin.

## Statistical boundary

- Frozen practical margin: ±0.002.
- `diag_vs_none` is secondary and cannot drive seed expansion.
- Original Newton–Muon versus Muon is a mandatory benchmark anchor.
- Concurrent quality runs are timing-ineligible; no throughput or wall-clock claim is derived here.
- Statistical append trigger: True.
