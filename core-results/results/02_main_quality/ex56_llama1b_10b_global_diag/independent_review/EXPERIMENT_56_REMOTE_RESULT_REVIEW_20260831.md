# Experiment 56 remote-result acceptance review

## Verdict

Experiment 56 is **accepted for scientific use**. The suite and handoff manifests pass, all 648 recorded integrity checks pass, the source snapshot verifies, the native full-checkpoint receipt passes, and all reported paired statistics were independently recomputed.

W&B data were not provided and do not invalidate the run under the frozen contract. Timing is ineligible.

## Endpoint results

| Tokens | Global-diagonal final validation loss, mean $\pm$ sample SD | Retained diagonal K state |
|---:|---:|---:|
| 3.2506B | 2.971265 $\pm$ 0.000388 | 1.600 MiB |
| 6.9694B | 2.860101 $\pm$ 0.000198 | 1.600 MiB |
| 9.9997B | 2.815139 $\pm$ 0.000374 | 1.600 MiB |

## Paired final-loss contrasts

Differences are global diagonal minus comparator; negative values favor global diagonal.

| Budget | Comparator | Mean delta | 95% CI | Global diagonal wins |
|---|---|---:|---:|---:|
| 3.2506B | identity | -0.002970 | [-0.003989, -0.001952] | 3/3 |
| 3.2506B | local diagonal | -0.004759 | [-0.006639, -0.002878] | 3/3 |
| 3.2506B | full-K | -0.004890 | [-0.005042, -0.004737] | 3/3 |
| 3.2506B | Muon | +0.001376 | [-0.000085, +0.002836] | 0/3 |
| 6.9694B | identity | -0.002702 | [-0.004126, -0.001279] | 3/3 |
| 6.9694B | local diagonal | -0.003600 | [-0.005016, -0.002185] | 3/3 |
| 6.9694B | full-K | -0.003913 | [-0.004418, -0.003408] | 3/3 |
| 6.9694B | Muon | +0.001302 | [+0.000050, +0.002555] | 0/3 |
| 9.9997B | identity | -0.002650 | [-0.004443, -0.000857] | 3/3 |
| 9.9997B | local diagonal | -0.002977 | [-0.005995, +0.000041] | 3/3 |
| 9.9997B | full-K | -0.003569 | [-0.004253, -0.002884] | 3/3 |
| 9.9997B | Muon | +0.001071 | [+0.000572, +0.001570] | 0/3 |

## Scientific interpretation and boundary

Global diagonal beats local diagonal, identity, and full-K in all 27 paired seed-budget comparisons. Muon nevertheless beats global diagonal in all nine paired comparisons. The three mean global-diagonal-minus-Muon gaps are inside the descriptive $\pm0.002$ practical band, but the experiment did not prespecify an equivalence or non-inferiority test; the result must therefore not be reported as formal equivalence.
