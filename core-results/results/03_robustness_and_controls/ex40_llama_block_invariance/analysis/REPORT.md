# LLaMA block-partition invariance audit

- Integrity passed: `true`
- Classification: `strong_non_invariance`
- Pooled median mapped-back block4 update drift: `0.344723`
- Maximum equivariant-control drift: `0.0159316`
- Effect/control multiple: `21.6377`

## Interpretation boundary

This audit tests whether a contiguous four-way partition of the 5504-dimensional SwiGLU hidden coordinate is invariant to function-preserving neuron permutations. It is not a training comparison and does not promote block4 to a primary baseline.

For LLaMA, `newton_full` remains the original Newton–Muon-family control; `muon` remains the optimizer baseline. Selective `none` and `diag` are each compared with those controls.
