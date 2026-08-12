# R1 dense-full alpha multi-seed confirmatory contract

This contract was frozen after the seed-2026 dense-full pilot and before
inspecting any seed-2024/2025 dense-full-alpha result. The pilot's automatic
seed-expansion gate did not pass. Seeds 2024 and 2025 are therefore a new,
explicitly authorized confirmation and contradiction-resolution study, not a
retroactive claim that the old gate passed.

## Runs and controls

For seeds 2024 and 2025, run the complete dense-full path at
`alpha = 0, 0.25, 0.50, 0.75, 1.00`. Every cell maintains one complete
3072 x 3072 `c_proj` covariance and inverse per layer and applies

`K_alpha = diag(K_full) + alpha * (K_full - diag(K_full))`.

Keep the official R1 data order, initialization, learning rates, EMA, ridge,
refresh schedule, evaluation grid, and 6,200-step horizon fixed. The five
cells within a seed must have an identical initialization fingerprint.

## Primary endpoint and contrast

The primary endpoint is validation loss at step 6,200. For each seed define

`C_seed = L(alpha=0.5) - 0.5 * (L(alpha=0) + L(alpha=1))`.

Negative values support the shallow non-monotone/U-shaped response observed in
the exploratory seed-2026 pilot. Report `C_seed` separately for 2024, 2025,
and exploratory 2026, followed by the three-seed mean, sample standard
deviation, and sign count. Seed 2026 is not an independent confirmation
because it motivated this study.

Also report whether `alpha=0.5` beats both endpoints in each seed. A strong
confirmatory statement requires this strict relation in both new seeds. Do not
claim that 0.5 is universally optimal.

## Secondary analyses

1. mean validation loss over the final five checkpoints;
2. normalized validation AUC on the common evaluation grid;
3. the full five-point curve for each seed;
4. `alpha=1 minus alpha=0` endpoint effect;
5. seed-wise matched dense-full minus block-alpha topology contrasts at
   `alpha = 0, 0.25, 0.50, 0.75`;
6. diagnostic refresh coverage, update cosine, and update norm ratio.

The block-alpha `alpha=1` endpoint is the same official block4 mathematical
endpoint, while dense-full `alpha=1` is a distinct full-covariance topology.
Do not substitute one for the other in topology contrasts.

## Integrity requirements

- matching seed-specific exact-shape numerical smoke certificates;
- all ten new formal cells reach exactly step 6,200;
- the exact 63-point validation grid is present once per cell;
- source/runtime/data/initialization fingerprints match the frozen controls;
- dense-full diagnostic refreshes exist at steps
  31, 1023, 2047, 3071, 4095, 5119, and 6143;
- Cholesky failure count is zero;
- complete local evidence is written before any network operation;
- every formal cell is uploaded to the dedicated W&B confirmatory project;
- duplicate W&B run names are forbidden, while idempotent upload retry by the
  deterministic run id is allowed.

## Parallel-host boundary

The block-alpha confirmation may run concurrently on the other physical H100.
Each process must expose exactly one distinct GPU. Quality-vs-step, diagnostics,
and per-process state evidence remain usable. All timing and throughput fields
from both concurrent families are ineligible for paper comparisons.
