# R1 dense-full alpha seed2026 analysis contract

This contract is frozen before the seed2026 formal curves are observed.  The experiment is a
mechanism pilot, not a new optimizer baseline and not a performance experiment.

## Intervention and estimands

For every `mlp.c_proj`, maintain the complete 3072 x 3072 EMA covariance and apply

`K_alpha = diag(K_full) + alpha * (K_full - diag(K_full))`

at alpha `0, 0.25, 0.50, 0.75, 1`.  All five cells use identical dense-full storage, ridge, inverse
refresh, data order, initialization and learning rates.

The primary endpoint is validation loss at step 6200.  Secondary endpoints are the final-five
validation mean and normalized trapezoidal validation AUC over steps 0:100:6200.

The first estimand is the full-path dose response.  The second is the matched topology contrast

`Delta_topology(alpha) = loss_full_alpha(alpha) - loss_block_alpha(alpha)`

using the already completed family 22 curve.  The block curve restores within-block covariance;
the full curve restores within- and cross-block covariance together.

## Quality and implementation gates

1. Dense-full alpha=0 must be within 0.001 of efficient diag at final and tail-five loss.
2. Every run must complete exactly 6200 updates and the exact 63-point validation grid.
3. Initialization, data, runtime, official base hash, learning rates and refresh rule must match.
4. Every requested diagnostic refresh must be present and Cholesky failures must be zero.

An OWT-like monotone signal is descriptive only when at least three of four adjacent final-loss
deltas are nonnegative, Spearman rho(alpha, final loss) is at least 0.8, the alpha=1 minus alpha=0
final delta is at least 0.001, and tail-five/AUC have the same direction.

A topology effect is considered materially visible when at least one matched-alpha absolute final
contrast is at least 0.002 and the corresponding tail-five and AUC contrasts have the same sign.
This threshold was chosen before observing the full-alpha data and is not an equivalence margin.

The pilot becomes a candidate for confirmatory seeds only if the implementation gates pass and
either the monotone signal has total final effect at least 0.002 or the topology-effect rule passes.
Even then, seeds 2024/2025 require an explicit follow-up command with `--allow-seed-expansion`;
the controller never launches them automatically.

## Diagnostics and boundaries

At refresh steps 31, 1023, 2047, 3071, 4095, 5119 and 6143, save:

- raw cross-block/within-block covariance Frobenius ratio;
- scaled offdiagonal/diagonal covariance ratio;
- Cholesky diagonal spread and inverse offdiagonal/diagonal ratio;
- inverse diagonal RMS;
- preconditioned-update norm ratio and cosine against the dense diagonal reference.

These quantities diagnose inverse-induced scale/direction crossing.  They are descriptive and do
not replace the loss endpoints.  Their GPU reductions synchronize execution, so all timing from
this family is ineligible.  A seed2026 optimum must not be described as a universal alpha optimum.

