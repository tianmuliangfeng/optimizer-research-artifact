# Experiment 53 remote-result review (2026-08-20)

## Verdict

The received remote run is complete and scientifically usable. The engineering and analysis gates passed, the formal grid contains all 15 planned cells (five arms times seeds 2024/2025/2026), and timing remains ineligible. The subsequently received W&B exports also pass exact secondary-mirror reconciliation; the frozen local result files remain authoritative.

## Preserved source

- Remote run: `20260817T013015+0000`
- Received ZIP: `archive/53_r1_matched_diag_module_placement.zip`
- ZIP SHA-256: `af51dbb9582f439d0f81c1d7f1c03856827d3dd7589008a259717d07985c266d`
- Analysis manifest SHA-256: `87e13c29cac8932cb9a924a3c098ce1c84282a93c265b84cb6185c58e6d52628`
- Handoff manifest SHA-256: `ef598568d326a3110bce60deb6b56449bd87fb49c39d1b78378b18d7dc8e5cd4`
- W&B export ZIP: `../wandb/archive/53.zip`
- W&B export ZIP SHA-256: `073ee2287dcaf201b277215c887b8b29eb4ffd07f195b2ad3738e06576287b04`

The seven analysis artifacts match the hashes sealed by the analysis manifest; the handoff binds that analysis manifest; all 29 source-snapshot records match; and the independently recomputed 50 aggregate/statistical fields match the received tables to numerical tolerance.

## W&B secondary-mirror reconciliation

The nine W&B CSV exports contain exactly the 15 expected formal run identities and no extras. Against the frozen per-run `r1_metrics.csv` and `r1_summary.json` files, all 24,270 comparable exported values match within `1e-9`: validation loss 945/945, training loss 4,650/4,650, matrix and AdamW learning rates 9,330/9,330, memory/state values 45/45, step-average values 4,635/4,635, and cumulative training time 4,665/4,665. The CSV `MIN`/`MAX` companion columns also equal their single-run base values throughout.

The exact raw exports, normalized point-level reconciliation, metric summary, and audit manifest are preserved under `../wandb/`. This cross-check does not change the result or make concurrently measured timing scientifically usable.

## Endpoint means

Lower validation loss is better. Values are mean +/- sample SD across three formal seeds.

| Arm | Mean final loss | Sample SD | Retained K state | Rank |
|---|---:|---:|---:|---:|
| `c_fc_c_proj_diag` | 3.271967 | 0.000814 | 0.351562 MiB | 1 |
| `c_proj_diag` | 3.273000 | 0.000436 | 0.281250 MiB | 2 |
| `o_proj_diag` | 3.276767 | 0.000651 | 0.070312 MiB | 3 |
| `c_fc_diag` | 3.276800 | 0.001136 | 0.070312 MiB | 4 |
| `all_none` | 3.277600 | 0.000693 | 0 MiB | 5 |

## Matched-diagonal placement result

All route-on arms use the same diagonal representation. Paired effects are method A minus method B, so negative values favor method A.

| Contrast/effect | Mean paired delta | 95% paired t interval | Seed direction |
|---|---:|---:|---:|
| `c_proj_diag - all_none` | -0.004600 | [-0.005257, -0.003943] | 3/3 lower |
| `c_fc_diag - all_none` | -0.000800 | [-0.001938, +0.000338] | 3/3 lower |
| `o_proj_diag - all_none` | -0.000833 | [-0.001774, +0.000107] | 3/3 lower |
| `c_fc_diag - c_proj_diag` | +0.003800 | [+0.002061, +0.005539] | `c_proj` lower in 3/3 |
| `c_proj_diag - o_proj_diag` | -0.003767 | [-0.004801, -0.002732] | `c_proj` lower in 3/3 |
| `c_fc` factorial main effect | -0.000917 | [-0.001951, +0.000118] | 3/3 negative |
| `c_proj` factorial main effect | -0.004717 | [-0.005434, -0.004000] | 3/3 negative |
| factorial interaction | -0.000233 | [-0.000520, +0.000054] | 3/3 negative |

## Scientific interpretation

This is a successful confirmatory placement experiment. Once the representation is matched, retaining diagonal activation state at `c_proj` produces a much larger and more stable endpoint benefit than retaining it only at `c_fc` or `o_proj` in the tested Modded GPT 124M environment. The combined `c_fc+c_proj` arm has the lowest observed mean, but the additional `c_fc` main effect is small and its interval crosses zero; the interaction interval also crosses zero.

The safe claim is therefore placement-specific, not universal: this experiment removes the prior full-versus-block-4 representation confound, but it does not equalize input dimension or retained-state bytes and covers one architecture/scale/training environment. It supports `c_proj` as the strongest tested single-module diagonal placement in this environment; it does not establish a universal module ranking.
