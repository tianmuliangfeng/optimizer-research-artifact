# Experiment 45 controlled 124M R1 Mousse analysis

Historical identity/reuse certificate: **passed with caveats**. Timing remains diagnostic only.
Historical experiment-15/19 rows are frozen W&B exports plus accepted analysis manifests; their original per-run local source/runtime/checkpoint manifests are not present in this evidence bundle. Quality pairing is accepted at the frozen protocol level, but this limitation must be disclosed and timing comparisons remain prohibited.

## Eight-method endpoint

| rank | method | final val mean | seed SD |
|---:|---|---:|---:|
| 1 | Newton-Muon diag | 3.261100 | 0.001114 |
| 2 | Newton-Muon block4 | 3.262200 | 0.001136 |
| 3 | Newton-Muon none | 3.266667 | 0.000666 |
| 4 | Mousse-R1 | 3.268033 | 0.000252 |
| 5 | Moonlight Muon | 3.274967 | 0.000451 |
| 6 | Muon | 3.277133 | 0.000252 |
| 7 | NorMuon | 3.334467 | 0.000924 |
| 8 | AdamW | 3.400433 | 0.004692 |

## Preregistered paired contrasts

Negative means the left method has lower loss.

| contrast | role | mean | 95% paired-t CI | direction |
|---|---|---:|---:|---:|
| selective_diag_minus_mousse | primary | -0.006933 | [-0.009422, -0.004445] | 3/3 left-better |
| selective_none_minus_mousse | primary | -0.001367 | [-0.002617, -0.000116] | 3/3 left-better |
| mousse_minus_muon | anchor | -0.009100 | [-0.009100, -0.009100] | 3/3 left-better |
| mousse_minus_original_newton_muon | anchor | +0.005833 | [+0.003539, +0.008128] | 0/3 left-better |
| mousse_minus_moonlight | external_background | -0.006933 | [-0.007874, -0.005993] | 3/3 left-better |
| mousse_minus_normuon | external_background | -0.066433 | [-0.069313, -0.063554] | 3/3 left-better |
| mousse_minus_adamw | external_background | -0.132400 | [-0.143803, -0.120997] | 3/3 left-better |

With n=3, confidence intervals are descriptive; no large-sample significance language is warranted.
