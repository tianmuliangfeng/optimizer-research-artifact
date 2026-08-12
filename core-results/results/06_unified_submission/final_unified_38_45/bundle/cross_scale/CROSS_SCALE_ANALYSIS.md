# Cross-scale paired analysis

Each scale is analyzed independently. No seeds are pooled across scales, and no cross-scale p-value is reported.

Practical loss margin: ±0.0020.

| Scale | Contrast (A − B) | n | Mean Δ final loss | 95% CI | Decision |
|---|---:|---:|---:|---:|---|
| gpt124m | diag − muon | 3 | -0.016033 | [-0.018522, -0.013545] | robust_material_improvement |
| gpt124m | diag − original | 3 | -0.001100 | [-0.002483, +0.000283] | inconclusive |
| gpt124m | none − muon | 3 | -0.010467 | [-0.011717, -0.009216] | robust_material_improvement |
| gpt124m | none − original | 3 | +0.004467 | [+0.003216, +0.005717] | robust_material_degradation |
| gpt124m | diag − none | 3 | -0.005567 | [-0.006841, -0.004292] | robust_material_improvement |
| gpt275m | diag − muon | 4 | -0.002809 | [-0.008234, +0.002617] | inconclusive |
| gpt275m | diag − original | 4 | +0.000165 | [-0.002526, +0.002856] | inconclusive |
| gpt275m | none − muon | 4 | -0.003048 | [-0.007387, +0.001292] | inconclusive |
| gpt275m | none − original | 4 | -0.000074 | [-0.001937, +0.001788] | practical_equivalence_supported |
| gpt275m | diag − none | 4 | +0.000239 | [-0.001060, +0.001538] | practical_equivalence_supported |
| gpt455m | diag − muon | 3 | -0.000900 | [-0.001582, -0.000219] | practical_equivalence_supported |
| gpt455m | diag − original | 3 | +0.000614 | [-0.001264, +0.002491] | inconclusive |
| gpt455m | none − muon | 3 | -0.002335 | [-0.002764, -0.001905] | inconclusive |
| gpt455m | none − original | 3 | -0.000821 | [-0.001904, +0.000262] | practical_equivalence_supported |
| gpt455m | diag − none | 3 | +0.001435 | [+0.000484, +0.002385] | inconclusive |

Interpretation is scale-stratified: consistent signs across scales are replication evidence, not extra IID samples.
