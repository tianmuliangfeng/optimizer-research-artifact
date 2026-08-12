# Selective Newton–Muon Unified Mechanism Synthesis

## Technical summary

The evidence supports a **Newton–Muon-family mechanism with stage-dependent
curvature refresh effects**, not a primary “diag versus none” story. In formal
three-seed training, the 12 frozen Selective contrasts contain
3 materially better, 3 materially worse, and
6 within-margin outcomes; therefore each Selective method must be
reported separately against both Muon and original Newton–Muon.

The strongest causal result is MECH-09R: all three frozen directional predictions
passed under exact shared prefixes, identifying the scheduled down-projection
refresh as a mediator of post-refresh short-horizon degradation. The R1 alpha
experiments independently show a nonmonotonic response on both block and dense-full
topologies, with alpha=0.5 beating both endpoints for every tested seed. This does
not establish a universal best alpha.

The negative result is equally important: MECH-03 failed its prediction gate,
MECH-06 remained uncertain, and MECH-08 prediction/trajectory sign concordance is
only descriptive. One-step shadow loss must not be presented as a validated
long-horizon selector.

The previously omitted 24-layer OWT and WikiText-103 studies now form the
foundational module-allocation layer. Diagonal c_proj retention ranks first and
none ranks second on both datasets. The complementary bridge is worse than
none in all six paired seeds, showing that c_proj-only K state is not the
useful part of the historical Newton contribution.

## The evidence chain narrows from numerical validity to a local causal mediator

| Stage | Evidence | Status | Result | Boundary |
| --- | --- | --- | --- | --- |
| foundational_module_allocation | supportive | replicated_across_two_datasets | top-two modes: OWT=diag,none; WikiText-103=diag,none; cproj-only worse in 6/6 paired seeds | historical 24L GPT evidence; R1 module allocation remains unisolated |
| architecture_transfer_boundary | limiting | strong_non_invariance | LLaMA contiguous block4 median update drift=0.3447; equivariant-control max=0.0159; effect/control=21.64x | coordinate-partition dependence, not a full-training performance ranking; official LLaMA original control remains newton_full |
| numerical_implementation | supportive | passed | fixed-tensor and cross-runtime numerical checks passed | implementation validity, not optimizer superiority |
| k_geometry | descriptive | candidate_signal | 5/9 geometry metric gates passed | geometry did not authorize the next mechanism stage |
| one_step_crossfit_prediction | limiting | failed_prediction_gate | 0/4 material layers; 19/32 positive paired cells | one-step shadow loss is not a validated long-horizon proxy |
| llama1b_one_step_confirmation | limiting | uncertain | early=uncertain; late=uncertain | no retrospective ranking was used to rescue the proxy |
| frozen_checkpoint_family_shadow | supportive | stage_dependent | early and late checkpoint counterfactual rankings differ | counterfactual shadow steps are not real training trajectories |
| real_128_step_rollout | supportive | mixed_and_stage_dependent | all-stage primary AUC/step128 rows: 4 left-better, 1 left-worse, 3 uncertain | 128 steps do not replace full-budget training |
| down_projection_refresh_mediation | confirmatory | full_support | 3/3 frozen directional predictions passed | causal within the frozen MECH-09R intervention tree |
| alpha_dose_response | confirmatory | strong_confirmatory_support | 2/2 topologies have negative curvature in all seeds | tested-grid nonmonotonicity, not universal alpha optimality |
| formal_training_outcomes | confirmatory | authoritative_for_method_performance | three architectures × three seeds × five frozen contrasts | mechanism evidence explains but does not replace this result |

The chain deliberately separates implementation validation, descriptive geometry,
counterfactual shadow evidence, real rollout evidence, causal intervention, and
full-budget training. A later stage does not retroactively turn an earlier
descriptive diagnostic into a confirmatory predictor.

## Foundational OWT/WikiText module allocation is now part of the evidence chain

These historical three-seed studies vary the c_proj K structure while retaining
the same 24-layer GPT training family. The WikiText Muon row is a matched-recipe
historical reference rather than a row from the dual-alpha launch; consequently
its block4 paired delta is intentionally left blank.

| Dataset | Mode | Final loss | Rank | Delta vs block4 | K state MiB | c_proj K MiB | Peak MiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OWT | diag | 5.4581 | 1 | -0.0588179 | 864.75 | 0.75 | 6204.54 |
| OWT | none | 5.47082 | 2 | -0.0460909 | 864 | 0 | 6203.73 |
| OWT | block4 | 5.51692 | 3 | 0 | 2016 | 1152 | 7355.79 |
| OWT | dense_full | 5.5334 | 4 | 0.0164851 | 5472 | 4608 | 10811.8 |
| OWT | muon | 5.75915 | 5 | 0.242233 | 0 | 0 | 5147.57 |
| WikiText-103 | diag | 4.90307 | 1 | -0.0604981 | 864.75 | 0.75 | 6204.59 |
| WikiText-103 | none | 4.92224 | 2 | -0.0413267 | 864 | 0 | 6203.68 |
| WikiText-103 | block4 | 4.96357 | 3 | 0 | 2016 | 1152 | 7355.84 |
| WikiText-103 | dense_full | 4.96441 | 4 | 0.000843048 | 5472 | 4608 | 10811.8 |
| WikiText-103 | muon | 5.23657 | 5 | — | 0 | 0 | 5147.57 |

`none` removes c_proj K while retaining non-c_proj K. `dense_full` is a dense
mechanism control, not the official block4 contraction. Timing from these sources
is not reused for paper claims; memory and K-state accounting remain explicit.

The complementary bridge reverses the allocation: it retains c_proj K but removes
non-c_proj K. Positive deltas below mean that the c_proj-only bridge is worse.

| Dataset | c_proj-only loss | none loss | Paired delta | 95% t CI | Worse seeds |
| --- | --- | --- | --- | --- | --- |
| OWT | 5.89779 | 5.47082 | 0.426967 | [0.341521, 0.512412] | 3/3 |
| WikiText-103 | 5.39064 | 4.92453 | 0.466106 | [0.352878, 0.579334] | 3/3 |

This isolates the historical result more sharply than a loss ranking alone:
removing the expensive c_proj state preserved the useful contribution, whereas
retaining only that state did not. The evidence is supportive and does not assert
that every architecture must share the same module allocation.

## LLaMA exposes a strict block4 transfer boundary

Experiment 40 is limiting evidence, not a training-performance comparison. Under
global hidden-coordinate permutations, the contiguous block4 LLaMA update changes
far more than the equivariant controls:

| Architecture | Candidate | Classification | Pooled median drift | Control max | Effect/control | Original control |
| --- | --- | --- | --- | --- | --- | --- |
| LLaMA-1B | contiguous_block4 | strong_non_invariance | 0.344723 | 0.0159316 | 21.64x | newton_full |

Therefore block4 is not treated as original Newton–Muon or as a primary LLaMA
baseline. The official LLaMA original Newton–Muon control remains `newton_full`.
This result does not authorize a loss ranking among full-training optimizers.

## Formal training requires both baselines for every Selective proposal

All deltas below are left minus right validation loss; negative values favor the
left algorithm. Each row aggregates seeds 2024, 2025, and 2026 using the already
accepted primary analysis.

| Family | Priority | Contrast | Final Δ | 95% CI | Classification |
| --- | --- | --- | --- | --- | --- |
| GPT-R1 | primary | selective_diag_vs_muon | -0.0160333 | [-0.0185216, -0.0135451] | selective_or_left_materially_better |
| GPT-R1 | primary | selective_none_vs_muon | -0.0104667 | [-0.011717, -0.00921634] | selective_or_left_materially_better |
| GPT-R1 | primary | selective_diag_vs_original_newton_muon | -0.0011 | [-0.00248311, 0.000283109] | within_practical_margin |
| GPT-R1 | primary | selective_none_vs_original_newton_muon | 0.00446667 | [0.00321634, 0.00571699] | selective_or_left_materially_worse |
| GPT-R1 | baseline | original_newton_muon_vs_muon | -0.0149333 | [-0.0172281, -0.0126386] | selective_or_left_materially_better |
| LLaMA-124M | primary | selective_diag_vs_muon | -0.000652075 | [-0.00267006, 0.00136591] | within_practical_margin |
| LLaMA-124M | primary | selective_none_vs_muon | -0.000605424 | [-0.002959, 0.00174815] | within_practical_margin |
| LLaMA-124M | primary | selective_diag_vs_original_newton_muon | 7.25587e-05 | [-0.00146811, 0.00161323] | within_practical_margin |
| LLaMA-124M | primary | selective_none_vs_original_newton_muon | 0.000119209 | [-0.00181572, 0.00205414] | within_practical_margin |
| LLaMA-124M | baseline | original_newton_muon_vs_muon | -0.000724634 | [-0.0015397, 9.04349e-05] | within_practical_margin |
| LLaMA-1B | primary | selective_diag_vs_muon | 0.00562263 | [0.00327867, 0.00796658] | selective_or_left_materially_worse |
| LLaMA-1B | primary | selective_none_vs_muon | 0.0044709 | [0.00247082, 0.00647099] | selective_or_left_materially_worse |
| LLaMA-1B | primary | selective_diag_vs_original_newton_muon | -0.000964164 | [-0.00178464, -0.000143693] | within_practical_margin |
| LLaMA-1B | primary | selective_none_vs_original_newton_muon | -0.00211589 | [-0.00371767, -0.000514097] | selective_or_left_materially_better |
| LLaMA-1B | baseline | original_newton_muon_vs_muon | 0.00658679 | [0.00366509, 0.00950849] | selective_or_left_materially_worse |

This is the authoritative performance layer. The fact that a Selective method can
beat Muon while matching or losing to original Newton–Muon is precisely why the
two baselines cannot be collapsed.

## Real 128-step trajectories are mixed rather than a universal ranking

The table uses the all-stage MECH-08 comparisons at normalized AUC and step 128.
Intervals come from 5000 hierarchical bootstrap draws with
checkpoint origin as the outer cluster and replicas resampled within origin.

| Stage | Priority | Contrast | Metric | Mean Δ | Cluster 95% CI | Result |
| --- | --- | --- | --- | --- | --- | --- |
| all | primary | selective_diag_vs_muon | step 128 | 0.00189954 | [-0.000142477, 0.00385944] | uncertain |
| all | primary | selective_diag_vs_muon | AUC | -0.00017542 | [-0.00309914, 0.00271577] | uncertain |
| all | primary | selective_none_vs_muon | step 128 | 0.00158297 | [1.72286e-05, 0.00311421] | left_worse |
| all | primary | selective_none_vs_muon | AUC | -0.000881887 | [-0.00360552, 0.00171497] | uncertain |
| all | primary | selective_diag_vs_original_newton_muon | step 128 | -0.00123983 | [-0.00154534, -0.000988412] | left_better |
| all | primary | selective_diag_vs_original_newton_muon | AUC | -0.00176966 | [-0.00225412, -0.00130629] | left_better |
| all | primary | selective_none_vs_original_newton_muon | step 128 | -0.0015564 | [-0.00188155, -0.0011924] | left_better |
| all | primary | selective_none_vs_original_newton_muon | AUC | -0.00247613 | [-0.00291332, -0.00205116] | left_better |
| all | baseline | original_newton_muon_vs_muon | step 128 | 0.00313937 | [0.00117276, 0.00494289] | left_worse |
| all | baseline | original_newton_muon_vs_muon | AUC | 0.00159424 | [-0.00124348, 0.00429952] | uncertain |

Short-horizon results diagnose when relative behavior changes; they do not replace
the 6200-step formal result. MECH-08 timing is excluded from all efficiency claims.

## One-step predictions do not reliably bridge to the rollout

| Metric | Step | Pearson | Spearman | Sign concordance | Units |
| --- | --- | --- | --- | --- | --- |
| normalized_loss_auc | AUC_0_128 | 0.0741 | -0.03824 | 0.5 | 16 |
| normalized_heldout_loss | 128 | 0.3192 | 0.5794 | 0.5625 | 16 |

The AUC and step-128 correlations/sign agreement are descriptive and based on 16
primary origin-contrast units. Together with the failed MECH-03 gate and uncertain
MECH-06 result, they rule out using one-step shadow loss as a stand-alone selection
criterion.

## Alpha confirms a nonmonotonic dose response, not a universal optimum

Curvature is defined as
`L(alpha=0.5) - 0.5 * [L(alpha=0) + L(alpha=1)]` at validation step 6200.
Negative values mean the midpoint beats the linear interpolation of endpoints.

| Topology | Seeds | Mean curvature C | All C<0 | α=.5 beats endpoints | Class |
| --- | --- | --- | --- | --- | --- |
| block | 3 | -0.00126667 | True | True | strong_confirmatory_support |
| dense_full | 3 | -0.00196667 | True | True | strong_confirmatory_support |

The block and dense-full confirmations support the same response-curve claim.
Topology itself showed no material threshold effect in the accepted dense audit,
and timing from these concurrent runs remains ineligible for paper efficiency
claims.

## Scope, definitions, and statistical design

- Primary comparison set: four Selective-versus-baseline contrasts; original
  Newton–Muon versus Muon is a baseline contrast.
- Formal-training unit: seed within architecture.
- MECH-08 paired unit: checkpoint origin × data replica; checkpoint origin is the
  bootstrap cluster.
- MECH-09R causal unit: a branch from an exact shared prefix under the frozen
  intervention tree.
- Evidence levels: confirmatory, supportive, descriptive, and limiting as frozen
  in `UNIFIED_MECHANISM_CONTRACT.md`.

No checkpoint, raw W&B API, or unregistered training log was read by this
synthesis.

## Claim audit keeps negative and limiting evidence visible

| ID | Claim type | Level | Status | Evidence | Caveat |
| --- | --- | --- | --- | --- | --- |
| C01 | validation | supportive | supported | 5/5 registered manifests passed | does not compare optimizer quality |
| C02 | descriptive | descriptive | supported_with_limit | 5/9 geometry metric gates passed | MECH-03 prediction gate failed |
| C03 | predictive_validation | limiting | supported | primary sign concordance: AUC=0.500, step128=0.562 | alignment is descriptive and uses only 16 origin-contrast units |
| C04 | trajectory | supportive | supported | early/late shadow and rollout contrasts change sign or certainty | 128-step rollouts are not full training runs |
| C05 | causal | confirmatory | supported | 3/3 frozen directional predictions passed with exact shared prefixes | scope is the frozen 1B intervention tree and 128-step horizon |
| C06 | confirmatory_response_curve | confirmatory | supported | 2/2 topologies: alpha=0.5 beats both endpoints in all seeds | best alpha was descriptive; no universal alpha=0.5 claim |
| C07 | method_comparison | confirmatory | supported | 12 primary contrasts: 3 materially better, 3 materially worse, 6 within margin | diag-versus-none is intentionally not a primary contrast |
| C08 | module_allocation | supportive | replicated_with_architecture_boundary | diagonal ranked first in 2/2 datasets; none ranked second in 2/2; cproj-only was worse in 6/6 paired seeds; none removed 84.21% of dense K state | supportive OWT/WikiText-103 24L evidence; R1 does not reproduce the exact none-versus-block4 ordering and needs a module factorial |
| C09 | architecture_transfer_boundary | limiting | supported | pooled block4 median update drift=0.3447; equivariant-control max=0.0159; effect/control=21.64x | does not authorize a full-training performance ordering; the official LLaMA original Newton–Muon control remains newton_full |

## Limitations and robustness boundaries

- MECH-09R establishes a local refresh mediator in the frozen LLaMA-1B design; it
  does not prove that every scale or architecture has the same mediator.
- MECH-08 contains only four checkpoint origins and a 128-step horizon, so its
  clustered intervals are intentionally conservative and should not be treated as
  a replacement for multi-seed formal training.
- The alpha result covers the tested five-point grid. Selecting the best point
  from that grid is descriptive.
- The OWT/WikiText module-allocation studies use a 24-layer GPT family. Their
  replicated direction motivates, but cannot substitute for, an R1 module
  factorial. The WikiText Muon row is a matched-recipe reference from a separate
  accepted launch.
- Experiment 40 establishes coordinate-partition dependence of contiguous block4
  on LLaMA. It is not evidence that block4 underperforms or outperforms another
  optimizer in full training.
- Existing mechanism and concurrent alpha timing is not paper-ready efficiency
  evidence.

## Recommended next step

Keep the 39 submission-efficiency and sensitivity audit as the currently running
R1 job. After it finishes, run the frozen 41 R1 module 2x2 factorial. The new
factorial adds only the missing cproj-only and all-K-off cells and reuses the
accepted block4 and none cells. Any further mechanism experiment is
conditional on the interaction estimate and seed consistency from that result.

## Further questions

- Does the down-projection refresh mediator reproduce outside the frozen LLaMA-1B
  branch design?
- Which existing throughput and peak-memory records meet exclusive-GPU,
  same-domain, same-shape, warmup, synchronization, and repetition requirements?
- Does a shared learning-rate multiplier grid preserve the formal ranking under an
  equal tuning budget?
- Does the R1 c_fc-by-c_proj factorial reproduce the historical allocation result,
  or reveal an architecture-specific interaction?
