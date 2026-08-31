# Experiment 54 remote-result acceptance review

## Verdict

Experiment 54 is **accepted for scientific use**. The completion manifest passes, all six formal units are present, the sealed analysis and source-snapshot hashes verify, the full-checkpoint hashes verify, and every reported paired mean, sample standard deviation, direction count, and two-sided 95% Student-$t$ interval was independently recomputed.

W&B data were not provided and are not required by the frozen contract. Timing is permanently excluded from scientific claims.

## Endpoint results

| Scale | Tokens | Moonlight final validation loss, mean $\pm$ sample SD |
|---|---:|---:|
| LLaMA-124M | 3.2506B | 3.251607 $\pm$ 0.001148 |
| LLaMA-1B | 3.2506B | 2.972615 $\pm$ 0.000707 |
| LLaMA-1B | 6.9694B | 2.865405 $\pm$ 0.000374 |

The frozen tuning choice is `lr0030` for 124M and `lr0010` for 1B. Tuning and formal seeds do not overlap.

## Paired final-loss contrasts

Differences are Moonlight minus comparator; negative values favor Moonlight. Each interval uses three paired formal seeds.

| Scale / budget | Comparator | Mean delta | 95% CI | Moonlight wins |
|---|---|---:|---:|---:|
| 124M / 3.2506B | local diagonal | -0.015319 | [-0.018672, -0.011965] | 3/3 |
| 124M / 3.2506B | identity (`none` in artifact) | -0.015365 | [-0.020333, -0.010398] | 3/3 |
| 124M / 3.2506B | Muon | -0.015971 | [-0.018591, -0.013350] | 3/3 |
| 124M / 3.2506B | full-K | -0.015246 | [-0.018501, -0.011991] | 3/3 |
| 1B / 3.2506B | local diagonal | -0.003409 | [-0.007594, +0.000776] | 3/3 |
| 1B / 3.2506B | identity | -0.001621 | [-0.003712, +0.000471] | 3/3 |
| 1B / 3.2506B | Muon | +0.002726 | [-0.000446, +0.005898] | 0/3 |
| 1B / 3.2506B | full-K | -0.003540 | [-0.005958, -0.001121] | 3/3 |
| 1B / 6.9694B | local diagonal | +0.001704 | [-0.000527, +0.003934] | 0/3 |
| 1B / 6.9694B | identity | +0.002602 | [+0.000927, +0.004276] | 0/3 |
| 1B / 6.9694B | Muon | +0.006606 | [+0.004542, +0.008670] | 0/3 |
| 1B / 6.9694B | full-K | +0.001391 | [+0.000330, +0.002452] | 0/3 |

## Scientific interpretation and boundary

Moonlight is strongly competitive at LLaMA-124M under its independently frozen recipe. That ranking does not transfer unchanged to LLaMA-1B: it already trails Muon at 3.25B tokens, and by 6.97B tokens it trails every core comparator for all three seeds. This is evidence for a scale/training-stage boundary, not a universal baseline ranking.

Experiment 54 and Experiment 57 are independent runs. Their overlapping 1B budgets may be used as replication evidence but must not be pooled into a single $n=6$ analysis.
