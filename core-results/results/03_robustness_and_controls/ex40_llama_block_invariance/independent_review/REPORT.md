# Independent review — LLaMA block-partition invariance audit

- Passed: `true`
- Classification: `strong_non_invariance`
- Archive SHA-256: `9382d9b8602bf45fee155cfe067ac95e8ae64e3a5b22ded9dd4deb0d4f652f73`
- Source artifacts: `106`
- Global block4 observations: `48`
- Pooled median update drift: `0.344722876`
- Maximum equivariant-control drift: `0.015931610`
- Effect/control multiple: `21.637667`

## Stage medians

- early step 1000: `0.310525521`
- late step 6200: `0.392174200`

## Interpretation

Function-preserving cross-block hidden-neuron permutations change the mapped-back block4 update substantially at both checkpoints. Exact preconditioner controls remain at numerical zero, while production BF16/Triton controls remain far below the block4 effect.

This supports omitting block4 as a primary LLaMA baseline. `newton_full` remains the original Newton–Muon-family control. The shadow-loss probe is secondary and does not authorize a full-training performance claim.
