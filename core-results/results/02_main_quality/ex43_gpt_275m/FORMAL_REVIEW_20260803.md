# Experiment 43 formal review (2026-08-03)

## Evidence status

- Authoritative run: `20260731T014352+0000`.
- Suite status: `completed`; scientific integrity passed; W&B complete.
- Accepted formal cells: 16/16 (four methods × seeds 2024–2027).
- Handoff audit: 340/340 listed files passed byte-size and SHA-256 checks.
- The inferential unit is the training seed. Timing is ineligible because quality
  cells ran concurrently.

## Final-loss and trajectory summary

| Method | Final loss mean | Seed SD | Tail-5 mean | Normalized AUC |
|---|---:|---:|---:|---:|
| Muon | 3.277772 | 0.002901 | 3.297060 | 3.776393 |
| Original Newton–Muon | 3.274799 | 0.000587 | 3.293523 | 3.761563 |
| Selective-none | 3.274724 | 0.001674 | 3.293986 | 3.768082 |
| Selective-diag | 3.274963 | 0.002232 | 3.293763 | 3.763216 |

Candidate-minus-comparator paired final-loss contrasts:

| Contrast | Mean delta | 95% paired-t CI | Frozen classification |
|---|---:|---:|---|
| none − Muon | -0.003048 | [-0.007387, +0.001292] | direction better, practically unresolved |
| none − original | -0.000074 | [-0.001937, +0.001788] | practical equivalence supported |
| diag − Muon | -0.002809 | [-0.008234, +0.002617] | inconclusive across both margins |
| diag − original | +0.000165 | [-0.002526, +0.002856] | inconclusive across both margins |
| original − Muon | -0.002973 | [-0.007379, +0.001433] | direction better, practically unresolved |
| diag − none | +0.000239 | [-0.001060, +0.001538] | practical equivalence supported; secondary |

Original beat Muon at the endpoint in 4/4 seeds; none and diag each beat Muon in
3/4 seeds. Seed 2027 is the main source of selective-method uncertainty.

## State and memory

| Method | K state (MiB) | Optimizer state (MiB) | Peak allocated (GiB) |
|---|---:|---:|---:|
| Original | 468.000 | 2216.254 | 34.059 |
| None | 252.000 | 1784.254 | 33.418 |
| Diag | 252.281 | 1892.535 | 33.523 |

Relative to original, none removes 216 MiB of persistent K state and 432 MiB
of optimizer state; diag removes 215.719 MiB of K state and 323.719 MiB of
optimizer state. None also lowers measured peak allocation by about 0.642 GiB.

## Interpretation

The stable claim is **quality retention under contraction-state removal**, not
an additional quality gain from none over original. The full four-seed result
does not preserve the earlier partial-seed hypothesis that original−Muon and
none−original improvements are comparable: their means are -0.002973 and
-0.000074, respectively. This is positive for the quality–state Pareto story and
for the claim that the contraction block4 state is largely unnecessary here,
but it is not evidence that removing it systematically improves loss.

The frozen statistical append gate is triggered but does not authorize seeds
automatically. Any extension must be decided from the cross-scale paper claim
and must add all four methods together on seeds 2028 then 2029.

The 2026-08-03 W&B export contains all 560 expected validation checkpoints for
the 16 runs and agrees exactly with accepted local loss, token, and diagnostic
train-time histories. Local artifacts remain authoritative.
