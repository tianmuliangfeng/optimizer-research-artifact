# Experiment 49 controlled 124M R1 MALT-family analysis

The accepted Experiment-45 eight-method panel is reused read-only and MALT-R1 adaptation and MALTER-Eq17-R1 adaptation are added as the ninth and tenth methods.

All quality-run timing remains ineligible. Input and output files are sealed by SHA-256 in `analysis_manifest.json`.

## Ten-method endpoint

| rank | method | final val mean | seed SD |
|---:|---|---:|---:|
| 1 | Newton-Muon diag | 3.261100 | 0.001114 |
| 2 | Newton-Muon block4 | 3.262200 | 0.001136 |
| 3 | Newton-Muon none | 3.266667 | 0.000666 |
| 4 | Mousse-R1 | 3.268033 | 0.000252 |
| 5 | MALT-R1 adaptation | 3.272733 | 0.000902 |
| 6 | Moonlight Muon | 3.274967 | 0.000451 |
| 7 | Muon | 3.277133 | 0.000252 |
| 8 | NorMuon | 3.334467 | 0.000924 |
| 9 | AdamW | 3.400433 | 0.004692 |
| 10 | MALTER-Eq17-R1 adaptation | 3.645933 | 0.002888 |

## Frozen paired contrasts

Negative means the named left method has lower final validation loss.

| contrast | role | mean | descriptive 95% paired-t CI | direction | within 0.002 |
|---|---|---:|---:|---:|:---:|
| malt_minus_muon | anchor | -0.004400 | [-0.006921, -0.001879] | 3/3 malt-better | no |
| malt_minus_original_newton_muon | anchor | +0.010533 | [+0.005792, +0.015275] | 0/3 malt-better | no |
| malt_minus_selective_none | primary | +0.006067 | [+0.002297, +0.009837] | 0/3 malt-better | no |
| malt_minus_selective_diag | primary | +0.011633 | [+0.006657, +0.016610] | 0/3 malt-better | no |
| malt_minus_mousse | external_curvature_baseline | +0.004700 | [+0.002179, +0.007221] | 0/3 malt-better | no |
| malter_eq17_minus_muon | anchor | +0.368800 | [+0.361055, +0.376545] | 0/3 malter_eq17-better | no |
| malter_eq17_minus_original_newton_muon | anchor | +0.383733 | [+0.374622, +0.392845] | 0/3 malter_eq17-better | no |
| malter_eq17_minus_selective_none | primary | +0.379267 | [+0.371281, +0.387252] | 0/3 malter_eq17-better | no |
| malter_eq17_minus_selective_diag | primary | +0.384833 | [+0.376836, +0.392830] | 0/3 malter_eq17-better | no |
| malter_eq17_minus_mousse | external_curvature_baseline | +0.377900 | [+0.370155, +0.385645] | 0/3 malter_eq17-better | no |
| malt_minus_malter_eq17 | family_internal | -0.373200 | [-0.380925, -0.365475] | 3/3 malt-better | no |

With only three paired seeds, the t intervals are descriptive. A mean inside the 0.002 practical margin is not an equivalence test and must not be reported as established statistical equivalence.

## Evidence-transfer caveats

- Experiment 45 is reused through its accepted eight-method analysis summary and manifest; Experiment 49 does not rewrite its historical rows.
- MALTER-Eq17-R1 adaptation denotes the frozen Equation-17 single-eta paper-derived interpretation and is not an official MALTER reproduction.
