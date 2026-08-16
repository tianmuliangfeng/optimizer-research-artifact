# 50 R1 Global Activation Diag

- Status: `implemented`
- Code: `../../scripts/50_r1_global_activation_diag`
- Source-freeze tier: `sealed_source_snapshot`
- Generic artifact verification: `true`
- Fresh rerun: `false`
- One-click rerun: `false`
- Native resume: `true`
- Native verification: `true`

## Accepted result

Three formal seeds establish that global activation diagonals are worse
than selective contraction diagonals at ModdedGPT 124M. Quality claims
exclude concurrent-run timing; W&B is a pointwise-checked secondary mirror.

Inspect the frozen public entrypoints:

```bash
python reproducibility/reproduce.py inspect 50_r1_global_activation_diag
```

A fresh rerun is not declared for this archived experiment.
Use the native archival validator below when available.

Verify an existing result without training. The run path must
match a declared legacy result root or carry a matching sealed
source-snapshot lineage:

```bash
python reproducibility/reproduce.py verify \
  50_r1_global_activation_diag --results-root /path/to/results \
  --run-dir /path/to/results/50_r1_global_activation_diag/RUN_ID
```

Resume an interrupted native run:

```bash
python reproducibility/reproduce.py resume \
  50_r1_global_activation_diag --results-root /path/to/results \
  --run-dir /path/to/results/50_r1_global_activation_diag/RUN_ID
```

Run the experiment-specific native validator:

```bash
python reproducibility/reproduce.py native-verify \
  50_r1_global_activation_diag --results-root /path/to/results \
  --run-dir /path/to/results/50_r1_global_activation_diag/RUN_ID
```

## Entrypoints

- `command:20260814_ex50_r1_global_activation_diag` → `commands/50_r1_global_activation_diag/20260814_ex50_r1_global_activation_diag.sh` (native-verify, resume)
- `script:run_global_diag` → `scripts/50_r1_global_activation_diag/run_global_diag.py` (explicit selection only)
- `script:run_global_diag_suite` → `scripts/50_r1_global_activation_diag/run_global_diag_suite.py` (explicit selection only)

Historical results are not stored in this source repository. See
`../../docs/REPRODUCIBILITY.md` for evidence tiers and limitations.
