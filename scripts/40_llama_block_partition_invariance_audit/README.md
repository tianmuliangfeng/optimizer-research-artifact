# 40 — LLaMA block-partition invariance audit

This is a read-only structural audit, not a new optimizer baseline and not a
training experiment.

It asks whether splitting the 5504-dimensional LLaMA-1B SwiGLU
`down_proj` input into four contiguous blocks defines an intrinsic algorithm.
The audit applies function-preserving hidden-neuron coordinate permutations,
maps every computed update back to the original coordinates, and measures:

- off-block covariance energy;
- mapped-back update drift and cosine;
- held-out shadow-loss variation;
- exact `none`, `diag`, and `dense_full` permutation-equivariance controls;
- a within-block permutation control that block4 must preserve.

The global block4 effect is deliberately excluded from worker pass/fail.
Integrity gates cover only provenance, numerical controls, fixed batches,
finite outputs, and checkpoint/model invariance. The frozen analyzer can return
non-invariance, approximate invariance, inconclusive, or invalid.

Contract revision `2026-07-29.2` transparently separates exact
pre-Newton–Schulz equivariance from the BF16/Triton production-update numerical
envelope. It supersedes the overly strict first smoke gate; classification
thresholds were not changed.

`newton_full` remains the original Newton–Muon-family control for LLaMA.
`muon` remains the optimizer baseline. This audit never relabels block4 as
either one.

Run the two-GPU host command in:

`commands/40_llama_block_partition_invariance_audit/20260729_llama_block_partition_invariance_audit.sh`
