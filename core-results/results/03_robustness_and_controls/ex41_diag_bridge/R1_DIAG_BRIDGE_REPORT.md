# Experiment 41D: R1 diagonal bridge

## Technical summary

With `c_fc K` fixed to full, diagonal `c_proj K` materially improves final
validation loss over no `c_proj K` while using only 0.28125 MiB additional
persistent K state. Diag also matches block4 within the preregistered 0.002
practical-loss margin while saving
215.71875 MiB
(57.07%) of K state.

The accepted classification is
**diag_recovers_block4_quality_at_near_none_state_cost**. No new training is recommended.

## Diag recovers the omitted scale information at near-none state cost

| c_proj K | Mean final val loss | K-state MiB | Peak memory MiB |
|---|---:|---:|---:|
| diag | 3.261100 | 162.28125 | 38304 |
| block4 | 3.262200 | 378.00000 | 39168 |
| none | 3.266667 | 162.00000 | 38304 |

Diag-minus-none final loss is -0.005567, with a 95% t interval of
[-0.006841, -0.004292] and 3/3 seeds in
the beneficial direction. This is both statistically directional in this
small sample and larger than the 0.002 practical margin.

Diag-minus-block4 final loss is -0.001100, with a 95% t interval of
[-0.002483, 0.000283]. All three seeds
are numerically favorable to diag, but the mean magnitude is below the
practical margin and the interval crosses zero. The defensible claim is
quality matching or slight numerical improvement, not superiority.

## Scope and metric definitions

- Architecture and recipe: R1 Modded-NanoGPT, 6200 updates.
- Statistical unit: paired seed; seeds 2024, 2025, and 2026.
- Fixed factor: `c_fc K=full`.
- Varied factor: `c_proj K` in `none`, `diag`, and `block4`.
- Primary metric: final validation loss; lower is better.
- Practical margin: 0.002 final-loss units.
- Confidence intervals: paired-effect t intervals with two degrees of freedom.

## Methodology and source linkage

The analysis reads the frozen experiment-15 three-seed summary. Its block4 and
none rows are checked field-by-field against experiment 41's frozen reused-cell
reference. Their method/seed coverage, initial losses, final losses, tail-five
losses, normalized AUC, K state, and peak memory must match before any output is
written. The accepted experiment-41 result is also hash-pinned and its block4
and none means are rechecked.

Source SHA-256 values:

- experiment-15 run summary:
  `ab91afd37db5559c031ed1ffac5441b57856386e62175d94f3f55b3f2568dcc3`
- experiment-41 frozen reference:
  `21737784b9fccc69053bf7e507bd91a0a8a8ea26d02579d5c3d87f512e0ecb13`
- experiment-41 accepted result:
  `b5506b6e0902b4c1b338dac2336d24c7e3d2d7e3147e3a39e3f6d42b1357698f`
- 41D contract:
  `28e2f9bb897ff5b99c3efc0d9af1adacc03232dd2bd3607a2c3af70324def21d`

## Limitations and robustness boundaries

- `n=3`; intervals are small-sample t intervals.
- The result establishes the diag effect only when `c_fc K` is full.
- It does not estimate the missing `c_fc=none,c_proj=diag` cell or a
  diag-by-`c_fc` interaction.
- It is not a Muon comparison.
- Concurrent timing is not used; experiment 39 remains the isolated-efficiency
  source.
- Peak-memory equality for diag and none is reported at the measurement
  resolution of the original runs.

## Recommended paper use

Place this three-level slice next to the experiment-41 2×2 factorial:

1. Experiment 41 shows that full `c_fc K` and block4 `c_proj K` make
   approximately additive quality contributions.
2. Experiment 41D shows that the diagonal approximation retains the useful
   `c_proj` scale signal at essentially none-level state cost.
3. State that diag materially improves over none and quality-matches block4;
   do not claim statistically established superiority over block4.

## Further question

Only add the missing `c_fc=none,c_proj=diag` cell if the paper later requires a
claim that the diag benefit is independent of `c_fc K`. It is not required for
the current deployed-configuration claim.
