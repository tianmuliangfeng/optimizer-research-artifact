# R1 depth × c_proj K-mode three-seed formal analysis

Analysis date: 2026-08-09  
Formal batch: `20260806T100702+0000`  
Primary endpoint: step-6200 validation loss  
Primary contrast: seed-matched `diag - none`; negative favors diagonal K  
Status: 36/36 formal runs accepted; all local/W&B integrity checks passed; timing is ineligible.

## Executive conclusion

The direction transfer succeeds more strongly than expected: all 15 R1 depth pairs
favor `diag` at step 6200, and all five rule means are negative in 3/3 seeds. The
all-depth effect is `-0.005367 ± 0.000289`.
Tail-5 and normalized validation AUC are also negative in
`15/15` and `15/15` pairs.

The frozen local magnitude pattern does not fully transfer. Center and late remain
weaker than all, but edge reverses: its interaction versus all is
`+0.002067` and is positive in
`3/3` seeds. Therefore the defensible
claim is that retaining coordinate-wise diagonal scale is broadly useful across
depth, while the best depth allocation is environment-dependent. The data do not
support a universal edge mask or individual-layer causality.

## R1 paired results

| Rule | Mean diag-none ± SD | Negative seeds | Tail-5 mean | AUC mean | Interaction vs all |
|---|---:|---:|---:|---:|---:|
| early | -0.004867 ± 0.000416 | 3/3 | -0.004913 | -0.006938 | +0.000500 |
| center | -0.003833 ± 0.000115 | 3/3 | -0.003813 | -0.005303 | +0.001533 |
| late | -0.002500 ± 0.000265 | 3/3 | -0.002620 | -0.004630 | +0.002867 |
| edge | -0.003300 ± 0.000529 | 3/3 | -0.003407 | -0.005467 | +0.002067 |
| all | -0.005367 ± 0.000289 | 3/3 | -0.005493 | -0.008968 | +0.000000 |

## Seed-level contrasts

| Seed | Rule | Step-6200 | Tail-5 | Normalized AUC |
|---:|---|---:|---:|---:|
| 2024 | early | -0.005000 | -0.004980 | -0.005582 |
| 2024 | center | -0.003700 | -0.003660 | -0.004860 |
| 2024 | late | -0.002200 | -0.002380 | -0.004521 |
| 2024 | edge | -0.003100 | -0.003260 | -0.004825 |
| 2024 | all | -0.005200 | -0.005300 | -0.008719 |
| 2025 | early | -0.005200 | -0.005340 | -0.007984 |
| 2025 | center | -0.003900 | -0.003880 | -0.005041 |
| 2025 | late | -0.002600 | -0.002740 | -0.004789 |
| 2025 | edge | -0.003900 | -0.004000 | -0.006006 |
| 2025 | all | -0.005700 | -0.005880 | -0.009420 |
| 2026 | early | -0.004400 | -0.004420 | -0.007247 |
| 2026 | center | -0.003900 | -0.003900 | -0.006009 |
| 2026 | late | -0.002700 | -0.002740 | -0.004581 |
| 2026 | edge | -0.002900 | -0.002960 | -0.005570 |
| 2026 | all | -0.005200 | -0.005300 | -0.008765 |

## Official anchors and state trade-off

Across seeds, all-depth diag minus block4 is
`-0.000667`; all-depth none minus block4 is
`+0.004700`. Thus diag remains within and
slightly improves the ±0.002 practical neighborhood of block4 in all three seeds,
whereas removing c_proj K completely costs about
`+0.004700` validation loss.

Mean K-state is `162.281` MiB for all-diag,
`162.000` MiB for all-none, and
`378.000` MiB for block4. All-diag saves
`215.719` MiB
of K-state and `864.0` MiB
of peak allocation relative to block4. Muon remains the lightest anchor. Concurrent
quality-run timing is permanently ineligible.

Block4 minus Muon is `-0.015500`, but this is a
recipe-level comparison because Muon retains its official lower LR.

## Cross-environment synthesis

OWT and WikiText-103 established negative all-depth effects of `-0.013930` and
`-0.015348`; official R1 confirms the direction with a smaller all-depth effect of
`-0.005367`. OWT/WikiText both placed edge
as the strongest rule, while official R1 places all first and edge below all. This
combination is stronger for the paper than either extreme claim: the representation
benefit is robust, but the location of the largest benefit is not architecture/
implementation invariant.

No further depth seeds, denser masks, or post-hoc layer search are warranted. The
unified depth result should receive a short main-text paragraph or compact figure;
the complete 45 paired contrasts and implementation audits belong in the appendix.

## Integrity

The accepted batch contains six smoke and six formal shard manifests, 36 formal run
manifests, 36 local metric histories, and nine W&B metric exports covering the exact
same 36 run names. Local and W&B values match at every exported point. Source/init
lineage, validation grids, completion, W&B upload, checkpoint-disable policy, and
timing-ineligibility gates all pass. Exact hashes are recorded in
`input_manifest.csv` and `remote_bundle_inventory.csv`.
