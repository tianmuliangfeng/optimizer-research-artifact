# R1 block-alpha multi-seed confirmatory contract

This contract was frozen after inspecting the exploratory seed-2026 pilot and
before inspecting seed-2024/2025 block-alpha results.  The original pilot gate
did not authorize an automatic expansion.  This is therefore a separate
contradiction-resolution study, not a retroactive continuation of that pilot.

## Runs and controls

For seeds 2024 and 2025, run the complete dense block-local path at
`alpha = 0, 0.25, 0.50, 0.75`.  Reuse the already completed, same-seed official
R1 `block4` run as the exact mathematical `alpha = 1` endpoint.  Keep the
official R1 data order, initialization, learning rates, EMA, ridge, refresh
schedule, evaluation grid, and 6,200-step horizon fixed.

The dense `alpha = 0` cell is retained in both new seeds.  It is an
implementation-path control against efficient `diag`; it must not be omitted
even though the two methods are expected to be numerically close.  Every newly
run alpha cell stores dense block4 state, so this study makes no memory or
throughput claim.

## Primary endpoint and contrast

The primary endpoint is validation loss at step 6,200.  For each seed define

`C_seed = L(alpha=0.5) - 0.5 * (L(alpha=0) + L(alpha=1))`.

Negative values indicate that the midpoint is better than the mean of the two
endpoints and support a non-monotone/U-shaped response.  Report `C_seed` for
2024, 2025, and the already observed exploratory 2026 seed separately, followed
by the three-seed mean, sample standard deviation, and sign count.  Do not use
the 2026 observation as an independent confirmation because it motivated this
study.

Also report whether `alpha=0.5` beats both endpoints in each seed.  A strong
confirmatory statement requires this strict relation in both new seeds.

## Secondary analyses

1. mean validation loss over the final five checkpoints;
2. normalized validation AUC on the common evaluation grid;
3. the full five-point curve for each seed;
4. dense-alpha0 minus efficient-diag implementation delta;
5. exact local evidence checks and failure accounting.

The best alpha selected from the same five points is descriptive.  Do not claim
that 0.5 is universally optimal.

## Integrity requirements

- matching seed-specific exact-shape numerical smoke certificates;
- identical initialization hashes across methods within each seed;
- the pinned official R1 checkout and original R1 PyTorch runtime;
- all four dense-alpha cells reach step 6,200;
- no duplicated validation steps or W&B run names;
- same-seed official `diag` and `block4` endpoints only;
- timing remains ineligible, especially under concurrent GPU training.

