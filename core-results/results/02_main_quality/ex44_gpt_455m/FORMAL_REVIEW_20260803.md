# Experiment 44 formal review (2026-08-03)

## Evidence status

- Authoritative run: `20260730T104121+0000`.
- Suite status: `completed`; scientific integrity passed; W&B complete.
- Accepted formal cells: 12/12 (four methods × seeds 2024–2026).
- Handoff audit: 281/281 listed files passed byte-size and SHA-256 checks.
- The inferential unit is the training seed. Timing is ineligible. CUDA peaks
  include validation and are not training-step-only measurements.

## Final-loss and trajectory summary

| Method | Final loss mean | Seed SD | Tail-5 mean | Normalized AUC |
|---|---:|---:|---:|---:|
| Muon | 2.919811 | 0.000813 | 2.926824 | 3.260498 |
| Original Newton–Muon | 2.918297 | 0.000325 | 2.924894 | 3.250735 |
| Selective-none | 2.917476 | 0.000659 | 2.924402 | 3.254059 |
| Selective-diag | 2.918910 | 0.000864 | 2.925520 | 3.250847 |

Candidate-minus-comparator paired final-loss contrasts:

| Contrast | Mean delta | 95% paired-t CI | Frozen classification |
|---|---:|---:|---|
| none − Muon | -0.002335 | [-0.002764, -0.001905] | direction better, practically unresolved |
| none − original | -0.000821 | [-0.001904, +0.000262] | practical equivalence supported |
| diag − Muon | -0.000900 | [-0.001582, -0.000219] | practical equivalence supported |
| diag − original | +0.000614 | [-0.001264, +0.002491] | direction worse, practically unresolved |
| original − Muon | -0.001514 | [-0.003026, -0.000003] | direction better, practically unresolved |
| diag − none | +0.001435 | [+0.000484, +0.002385] | direction worse, practically unresolved; secondary |

None beats Muon and original at the endpoint in all 3/3 seeds. Diag beats Muon
in all 3/3 seeds, but none beats diag in all 3/3 seeds. With only three seeds,
the practical-margin classifications remain the correct claim boundary.

## State and memory

| Method | K state (MiB) | Optimizer state (MiB) | Peak allocated (GiB) |
|---|---:|---:|---:|
| Original | 1120.000 | 4738.266 | 54.714 |
| None | 608.000 | 3714.266 | 53.214 |
| Diag | 608.500 | 3970.766 | 53.465 |

Relative to original, none removes 512 MiB of persistent K state and 1024 MiB
of optimizer state, while lowering peak allocation by about 1.500 GiB. Diag
removes 511.5 MiB of K state and 767.5 MiB of optimizer state.

## Interpretation

Experiment 44 is strong positive evidence for Selective-none as a
quality–state Pareto point. However, its endpoint advantage does not imply full
trajectory dominance: original and diag have lower normalized AUC than none.
The defensible statement is that none preserves original-level quality and has
a consistent favorable endpoint direction while using materially less state.

The earlier “two similar gains” pattern is only partly stable. Original−Muon is
-0.001514 and none−original is -0.000821 on average; the latter is about 54% of
the former and varies substantially at seed 2026. It is suggestive decomposition
evidence, not a fixed additive law.

The frozen statistical append gate is triggered but not automatic. If the final
cross-scale claim truly depends on resolving the remaining interval ambiguity,
all four methods must be appended together on seeds 2027 then 2028.

The 2026-08-03 W&B export contains all 588 expected validation checkpoints for
the 12 runs and agrees exactly with accepted local loss, token, and diagnostic
train-time histories. Local artifacts remain authoritative.
