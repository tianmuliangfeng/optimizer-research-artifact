# Selective Newton–Muon primary comparison

This CPU-only analysis makes the two proposed Selective Newton–Muon methods
the subject of the primary comparisons:

1. each proposal versus Muon;
2. each proposal versus the original Newton–Muon baseline.

Original Newton–Muon versus Muon is retained as a baseline contrast.
`diag` versus `none` is deliberately excluded from the primary contract.

The analysis reads the authoritative three-seed run summaries for GPT-R1,
LLaMA-124M, and LLaMA-1B. It performs no training and does not alter any
MECH-05/06 frozen artifact.
