# MDP-05 / Experiment 46 final acceptance and scientific report

## Frozen identity

- Remote run: `20260804T021302+0000`
- Source ZIP: `20260804T021302+0000.zip`
- ZIP bytes: `13,049,790`
- ZIP SHA-256: `52189521ca8399ede2e33d1f96a6d9a68dc9d975e27f0e9fa1b3176718f2a1b4`
- MDP-05 contract SHA-256: `127883e8a277e77a900d06f3319b66b6418ad9d6d798854a5d083a89097cabce`
- Derived execution contract SHA-256: `734738b284cc63d2e1ffa8ab51c8e8b8c12630302fe627d0e455ebfc6ec35e1f`
- Final status: integrity `PASS`; scientific result `partial_or_null`; claim success `false`.

## Acceptance audit

The local read-only inspector passed identity, completed status, analysis
integrity, handoff, artifact hashes, and all 12 selected formal units. All 12
selected units are `attempt_002` and passed. The source-repair activation also
passed: before the repair, zero formal units had been accepted and no analysis
had been opened; the contract and base workers were unchanged; the two failed
`attempt_001` units were not reused. The repaired snapshot and both predecessor
and active formal plans remain preserved.

The accepted analysis contains 12 units, 24 unit-event rows, 432 nested layer
rows, and eight finite float64 calibration slices. Training/evaluation hashes
use the new held-out offsets. All exact shadow-to-actual gradient and NS5
fingerprint gates passed.

## Direct causal loss effect

Positive `oriented_endpoint_loss_harm` means that refreshing the down-projection
preconditioner is worse than keeping down refresh frozen at the matched start.

| Event | Frozen contrast and endpoint | Positive units | Mean normalized harm | Median | Range |
|---|---|---:|---:|---:|---:|
| production refresh at step 32 | production refresh vs frozen at step 48 | 12/12 | 0.004484 | 0.004143 | [0.002929, 0.006641] |
| delayed refresh at step 64 | delayed refresh vs frozen at step 80 | 12/12 | 0.002325 | 0.002292 | [0.001711, 0.003088] |

Thus the independent held-out replay robustly confirms a short-horizon adverse
loss impulse from down-projection refresh. The average production-window harm
is about 0.448% of the matched normalized loss level; the delayed-window harm
is about 0.232%. These sign counts and means are descriptive causal-effect
summaries, not additional preregistered significance tests.

## Preregistered update-shock mediation tests

| Event | Primary mediator | Spearman rho | Within-origin r | LOO rho range | Exact p | Holm p | Formal pass |
|---|---|---:|---:|---:|---:|---:|---|
| production@32 | matched-G preconditioned change | 0.7832 | 0.3749 | [0.4833, 0.9833] | 0.1088 | 0.4352 | no |
| production@32 | actual NS5 update change | 0.7063 | -0.2037 | [0.4667, 0.7667] | 0.6296 | 1.0000 | no |
| delayed@64 | matched-G preconditioned change | 0.7133 | -0.1312 | [0.4667, 0.7833] | 0.5448 | 1.0000 | no |
| delayed@64 | actual NS5 update change | 0.5664 | -0.0095 | [0.2333, 0.8333] | 0.3503 | 1.0000 | no |

All four pooled directions and all leave-one-origin-out directions are
positive, but only production@32 matched-G remains positive after origin
centering. None of the four within-origin exact permutation tests is
significant even before Holm correction, and zero of four passes the frozen
multiplicity gate. The supportive AUC analysis has the same pooled ordering;
only production@32 matched-G has a positive within-origin correlation
(`0.3957`). AUC was frozen as supportive and cannot rescue the primary result.

## Why pooled and within-origin conclusions differ

Checkpoint origin explains most of the observed variation. Descriptively, the
between-origin share of total sum of squares is 96.2% for production loss harm
and 96.0% for delayed loss harm. It is 84.8--91.0% for the two primary shock
metrics. Therefore the pooled Spearman correlations mainly say that checkpoint
origins with different optimizer history/stage jointly differ in shock size
and loss harm. The three new held-out replicas within a fixed origin generally
do not show the same quantitative association.

This is not evidence that matched-G or NS5 shock is irrelevant. It is evidence
against the stronger preregistered claim that either scalar metric provides a
stable origin-independent quantitative mediator of short-horizon harm.

## Numerical diagnostics

The resolvent diagnostic was correctly excluded from the MDP-05 primary hard
gate. Five of 432 rows exceed the old MDP-04 threshold of `0.01`, with maximum
`0.0128927`; all five are `late_muon`, layer 3. The eight preregistered float64
slices are finite. These rows must remain reported, but they do not invalidate
the exact matched-G/actual-NS5 experiment.

## Frozen paper and project decision

1. Claim-eligible: refreshing the down-projection preconditioner produces a
   reproducible short-horizon held-out loss impulse under the matched replay.
2. Not claim-eligible: the magnitude of that impulse is generally mediated or
   predicted by the 18-layer median matched-G or actual NS5 update shock after
   controlling for checkpoint origin.
3. MDP-05 belongs in the appendix/limitations as a valid partial/null
   confirmatory result. It strengthens the loss-level causal result while
   narrowing the proposed quantitative mechanism.
4. MDP-04 remains `numeric_gate_failed / descriptive partial`; MDP-05 does not
   retroactively promote it.
5. The frozen stop rule remains active: no extra replicas, aggregation change,
   MDP-06, Pythia HVP, LLaMA 10B-token run, or Mousse extension is triggered.
6. The next work is local: rebuild the method-deepening evidence package,
   claim-evidence matrix, negative-evidence table, mechanism figure, and final
   submission bundle; then write the paper under these bounded claims.

