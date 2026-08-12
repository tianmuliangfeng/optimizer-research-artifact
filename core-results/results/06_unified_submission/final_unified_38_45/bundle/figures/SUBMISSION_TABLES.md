# Submission-ready analysis tables

## Scale-stratified final loss

| Scale | Method | Seeds | Final loss (mean ± SD) | Tail-5 mean | Normalized AUC | K state (MiB) |
|---|---|---:|---:|---:|---:|---:|
| gpt124m | muon | 3 | 3.277133 ± 0.000252 | 3.285740 | 3.633386 | 0.000 |
| gpt124m | original | 3 | 3.262200 ± 0.001136 | 3.270480 | 3.615968 | 378.000 |
| gpt124m | diag | 3 | 3.261100 ± 0.001114 | 3.269300 | 3.615159 | 162.281 |
| gpt124m | none | 3 | 3.266667 ± 0.000666 | 3.274993 | 3.624126 | 162.000 |
| gpt275m | muon | 4 | 3.277772 ± 0.002901 | 3.297060 | 3.776393 | 0.000 |
| gpt275m | original | 4 | 3.274799 ± 0.000587 | 3.293523 | 3.761563 | 468.000 |
| gpt275m | diag | 4 | 3.274963 ± 0.002232 | 3.293763 | 3.763216 | 252.281 |
| gpt275m | none | 4 | 3.274724 ± 0.001674 | 3.293986 | 3.768082 | 252.000 |
| gpt455m | muon | 3 | 2.919811 ± 0.000813 | 2.926824 | 3.260498 | 0.000 |
| gpt455m | original | 3 | 2.918297 ± 0.000325 | 2.924894 | 3.250735 | 1120.000 |
| gpt455m | diag | 3 | 2.918910 ± 0.000864 | 2.925520 | 3.250847 | 608.500 |
| gpt455m | none | 3 | 2.917476 ± 0.000659 | 2.924402 | 3.254059 | 608.000 |

## Paired contrasts

| Scale | A − B | n | Mean Δ final loss | 95% CI | Interval decision |
|---|---|---:|---:|---:|---|
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

Scales are reported as separate replication environments; seed counts are never pooled across model sizes.
