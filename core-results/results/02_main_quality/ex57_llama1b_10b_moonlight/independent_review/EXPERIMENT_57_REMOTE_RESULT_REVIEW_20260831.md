# Experiment 57 remote-result acceptance review

## Verdict

Experiment 57 is **accepted for scientific use**. The completion manifest passes, all three formal units are present, sealed-analysis and source-snapshot hashes verify, full-checkpoint hashes verify, and all paired statistics were independently recomputed.

W&B data were not provided and are not required by the frozen contract. Timing is permanently excluded.

## Endpoint results

| Tokens | Moonlight final validation loss, mean $\pm$ sample SD | Optimizer state |
|---:|---:|---:|
| 3.2506B | 2.972485 $\pm$ 0.000471 | 4259.845 MiB |
| 6.9694B | 2.865562 $\pm$ 0.000186 | 4259.845 MiB |
| 9.9997B | 2.824618 $\pm$ 0.000281 | 4259.845 MiB |

The frozen tuning choice is `lr0010`; formal seeds do not overlap tuning seeds. The recorded Moonlight matrix/momentum state is 3474.000 MiB.

## Paired final-loss contrasts

Differences are Moonlight minus comparator; negative values favor Moonlight.

| Budget | Comparator | Mean delta | 95% CI | Moonlight wins |
|---|---|---:|---:|---:|
| 3.2506B | local diagonal | -0.003539 | [-0.007080, +0.000002] | 3/3 |
| 3.2506B | identity | -0.001751 | [-0.003316, -0.000186] | 3/3 |
| 3.2506B | Muon | +0.002595 | [-0.000054, +0.005245] | 0/3 |
| 3.2506B | full-K | -0.003670 | [-0.005341, -0.001999] | 3/3 |
| 6.9694B | local diagonal | +0.001861 | [+0.000003, +0.003719] | 0/3 |
| 6.9694B | identity | +0.002759 | [+0.001343, +0.004175] | 0/3 |
| 6.9694B | Muon | +0.006764 | [+0.005078, +0.008449] | 0/3 |
| 6.9694B | full-K | +0.001549 | [+0.000936, +0.002161] | 0/3 |
| 9.9997B | local diagonal | +0.006502 | [+0.003723, +0.009282] | 0/3 |
| 9.9997B | identity | +0.006829 | [+0.005337, +0.008321] | 0/3 |
| 9.9997B | Muon | +0.010550 | [+0.010200, +0.010900] | 0/3 |
| 9.9997B | full-K | +0.005910 | [+0.005255, +0.006565] | 0/3 |

## Scientific interpretation and boundary

Moonlight is not a uniformly stronger long-budget LLaMA-1B baseline. It trails Muon at every tested budget, and by 6.97B tokens it trails every tested core route for every seed. The gap widens further at approximately 10B tokens.

Experiment 57 is independent of Experiment 54. The near-identical directions at 3.25B and 6.97B tokens are valuable replication evidence, but the two experiments must remain separate analyses.
