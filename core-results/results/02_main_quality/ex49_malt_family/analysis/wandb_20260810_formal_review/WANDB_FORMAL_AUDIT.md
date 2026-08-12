# Experiment 49 W&B formal audit

Audit date: 2026-08-10  
Accepted run: `20260809T054721+0000`  
W&B project: `Selective-Newton-Muon-MainConf-R1-MALTFamilyFormalV4-20260809`

## Acceptance result

- Six formal runs are complete: two methods × seeds 2024/2025/2026.
- Every W&B endpoint exactly matches the accepted ZIP run summary.
- Each run contains 63 validation rows (`0:100:6200`), 310 training-loss rows
  (`20:20:6200`), 311 learning-rate/token rows (`0:20:6200`), and one final
  memory row. No missing, duplicated, non-finite, or non-monotone steps were found.
- The accepted local analysis is `completed_valid`; all six formal manifests are
  `completed_valid`; all six W&B uploads are `uploaded_online`.

## Scientific summary

| Method | selected LR | final loss mean ± sample SD | 124M rank | optimizer state | peak memory |
|---|---:|---:|---:|---:|---:|
| MALT-R1 adaptation | 0.0125 | 3.272733 ± 0.000902 | 5/10 | 619.107426 MiB | 37708.667 MiB |
| MALTER-Eq17-R1 adaptation | 0.015 | 3.645933 ± 0.002888 | 10/10 | 619.107700 MiB | 37708.000 MiB |

MALT is a competitive lightweight baseline: it beats Muon by `0.004400` final
loss in all three paired seeds and also finishes ahead of Moonlight. It remains
behind Mousse (`+0.004700`), none (`+0.006067`), block4 (`+0.010533`), and diag
(`+0.011633`). MALTER-Eq17 remains far behind MALT at steps 1000, 1800, 4400,
and 6200; there is no late-training reversal.

The MALTER statement is deliberately narrow. This is the frozen
`MALTER-Eq17-R1 adaptation` using the paper's single-outer-eta Equation (17)
interpretation, not an official-code reproduction and not evidence that every
possible MALTER implementation fails.

## Evidence boundary

- Quality-run timing is permanently `timing_eligible=false`. The approximately
  2.1 h diagnostic training time per run is not a fair isolated wall-clock claim.
- The source ZIP contains manifests, analysis, curves, paths, byte counts, and
  checkpoint SHA-256 certificates, but not the six checkpoint binaries.
- The ZIP's unified analysis references two accepted Experiment 45 inputs that
  are hash-certified but not embedded in this ZIP. Those inputs remain preserved
  under Experiment 45's authoritative local archive.

## Raw W&B export SHA-256

| Metric | File suffix | SHA-256 |
|---|---|---|
| `lr/matrix` | `09_26_00.658+08_00.csv` | `75eee1a2da2d763611d96dd9c207ebd2fa80a8bf8df0a81100c786dd169c7a47` |
| `lr/auxiliary` | `09_26_03.709+08_00.csv` | `bb2a2213e36d32ad33f2b7b64632887be77c06a340989c1ed9fea01f749ecd33` |
| `memory/peak_allocated_mib` | `09_26_08.414+08_00.csv` | `c566fb4bc9a970ea1efc21c05c62bd8816d35a9761cc1703cef5e2505e764fde` |
| `memory/optimizer_state_mib` | `09_26_11.402+08_00.csv` | `4f148a2c5e27289f8261ea892e96ebb711e799ec1e2ac3b87fd4267796cb1615` |
| `tokens/seen` | `09_26_16.408+08_00.csv` | `00ab2f793d491af45dda08cb3237c982e01f050faa929e99e52b6cb71f1cb33e` |
| `train/loss_step` | `09_26_20.353+08_00.csv` | `9138663883968d01cfc5a6b5e91d4ba375140cf4b1e725bac814531a30cd1780` |
| `val/loss` | `09_26_25.028+08_00.csv` | `3f3e48e92520af9b8cbacbca56c539c8c3955b3fb4f8896193336d2e45c377dd` |
