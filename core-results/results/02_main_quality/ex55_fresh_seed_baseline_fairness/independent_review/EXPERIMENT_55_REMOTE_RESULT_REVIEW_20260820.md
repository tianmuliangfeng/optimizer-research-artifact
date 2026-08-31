# Experiment 55 remote-result review (2026-08-20)

## Verdict

The received remote run is complete and scientifically usable. All ten frozen methods were evaluated on the fresh, selection-independent seed 2027; the repaired panel contains the prespecified seeds 2024/2025/2027; and timing remains ineligible. The subsequently received W&B exports also pass exact secondary-mirror reconciliation; the frozen local result files remain authoritative.

## Preserved source

- Remote run: `20260817T021952+0000`
- Received ZIP: `archive/55_r1_fresh_seed_baseline_fairness.zip`
- ZIP SHA-256: `e03fb2327decaa93baabdf31286feac2d569e4b8119dcafef2a8394f03d085e3`
- Analysis manifest SHA-256: `00f1adf557010d75459fad1cc29b1913c74d348118ef28de75ea97921b7b8f2e`
- Handoff manifest SHA-256: `bd12087fc5f84ef78f2ced4fd374fa92f5f131824f21eacce029237535d7a763`
- Checkpoint verification manifest SHA-256: `738d8c078fa694e5842159a18ba3f579253eaff701825b6724394c42ff94d9ae`
- W&B export ZIP: `../wandb/archive/55.zip`
- W&B export ZIP SHA-256: `2dad85e745d2d3397886b1ac886644adba84e8513dffd150e35a6f69814746bf`

All ten analysis artifacts match the hashes sealed by the analysis manifest; the handoff binds both the analysis and checkpoint-verification manifests; all 60 source-snapshot records match; ten retained endpoint checkpoints passed the remote full-hash gate; and the independently recomputed 100 aggregate/statistical fields match the received tables.

## W&B secondary-mirror reconciliation

The eleven W&B CSV exports contain exactly the ten expected seed-2027 formal run identities and no extras; the `_upload01` suffix on the two re-uploaded W&B run names is normalized only for identity matching. Against the frozen per-run metric and summary files, all 15,247 comparable exported values match within `1e-9`: validation loss 630/630, training loss 3,100/3,100, matrix/AdamW/auxiliary learning rates 6,220/6,220, memory/state values 24/24, step-average values 2,163/2,163, cumulative training time 2,177/2,177, and token counts 933/933. The CSV `MIN`/`MAX` companion columns also equal their single-run base values throughout.

The exact raw exports, normalized point-level reconciliation, metric summary, and audit manifest are preserved under `../wandb/`. This cross-check does not change the repaired-panel ranking or make the excluded timing measurements scientific efficiency evidence.

## Fresh seed 2027 endpoints

| Method | Final loss | Tail-5 mean | Normalized AUC |
|---|---:|---:|---:|
| block-4 | 3.2624 | 3.27066 | 3.616936 |
| diag | 3.2627 | 3.27090 | 3.617497 |
| Mousse | 3.2666 | 3.27730 | 3.630451 |
| none | 3.2676 | 3.27602 | 3.626535 |
| MALT | 3.2730 | 3.29848 | 3.696098 |
| Moonlight | 3.2745 | 3.28628 | 3.651006 |
| Muon | 3.2771 | 3.28570 | 3.634265 |
| NorMuon | 3.3353 | 3.34350 | 3.728639 |
| AdamW | 3.4035 | 3.41324 | 3.865094 |
| MALTER-Eq17 | 3.6467 | 3.68394 | 4.079281 |

All ten fresh runs share initial validation loss 10.9937 and initialization SHA-256 `b328c4699491699620a5979160450eac4f41f9107cd73fb42974db9d8e4ed7b4`.

## Repaired three-seed panel

The repaired panel replaces the selection seed 2026 with fresh seed 2027. Values are mean +/- sample SD over seeds 2024/2025/2027.

| Rank | Method | Mean final loss | Sample SD |
|---:|---|---:|---:|
| 1 | diag | 3.261300 | 0.001400 |
| 2 | block-4 | 3.262100 | 0.001082 |
| 3 | none | 3.266833 | 0.000862 |
| 4 | Mousse | 3.267567 | 0.000874 |
| 5 | MALT | 3.273133 | 0.000416 |
| 6 | Moonlight | 3.274967 | 0.000451 |
| 7 | Muon | 3.277133 | 0.000252 |
| 8 | NorMuon | 3.334567 | 0.001021 |
| 9 | AdamW | 3.403167 | 0.001528 |
| 10 | MALTER-Eq17 | 3.645600 | 0.002629 |

The rank order is identical to both the historical 2024/2025/2026 panel and the prespecified two-seed leave-out diagnostic. Relative to Muon, block-4, diag, none, Mousse, MALT, and Moonlight have lower final loss in all three repaired-panel seeds. AdamW, NorMuon, and MALTER-Eq17 are worse in all three.

## Scientific interpretation

This is a successful fairness repair. The broad ten-method ranking is not an artifact of using seed 2026 for both selection and reporting: it is unchanged when seed 2026 is removed and fresh seed 2027 is inserted.

The top-two ordering must remain descriptive. On seed 2027, block-4 is 0.0003 lower than diag, whereas the repaired three-seed mean favors diag by 0.0008 and only two of three seeds favor diag. Thus the evidence supports “diag has the lowest observed repaired-panel mean,” not equivalence, non-inferiority, or universal dominance over block-4. The much larger separations from Muon and the broad ranking stability are the stronger fairness conclusions.
