# 51 Moddedgpt Global Diag Scale

- Status: `implemented`
- Code: `../../scripts/51_moddedgpt_global_diag_scale`
- Source-freeze tier: `sealed_source_snapshot`
- Generic artifact verification: `true`
- Fresh rerun: `false`
- One-click rerun: `false`
- Native resume: `true`
- Native verification: `true`

## Accepted result

The 275M and 455M scale confirmation does not reverse the ModdedGPT result:
global activation diagonals do not beat the accepted selective controls.
Quality claims exclude concurrent-run timing.

Inspect the frozen public entrypoints:

```bash
python reproducibility/reproduce.py inspect 51_moddedgpt_global_diag_scale
```

A fresh rerun is not declared for this archived experiment.
Use the native archival validator below when available.

Verify an existing result without training. The run path must
match a declared legacy result root or carry a matching sealed
source-snapshot lineage:

```bash
python reproducibility/reproduce.py verify \
  51_moddedgpt_global_diag_scale --results-root /path/to/results \
  --run-dir /path/to/results/51_moddedgpt_global_diag_scale/RUN_ID
```

Resume an interrupted native run:

```bash
python reproducibility/reproduce.py resume \
  51_moddedgpt_global_diag_scale --results-root /path/to/results \
  --run-dir /path/to/results/51_moddedgpt_global_diag_scale/RUN_ID
```

Run the experiment-specific native validator:

```bash
python reproducibility/reproduce.py native-verify \
  51_moddedgpt_global_diag_scale --results-root /path/to/results \
  --run-dir /path/to/results/51_moddedgpt_global_diag_scale/RUN_ID
```

## Entrypoints

- `command:20260814_ex51_moddedgpt_global_diag_scale` → `commands/51_moddedgpt_global_diag_scale/20260814_ex51_moddedgpt_global_diag_scale.sh` (native-verify, resume)
- `script:run_global_diag_scale_cell` → `scripts/51_moddedgpt_global_diag_scale/run_global_diag_scale_cell.py` (explicit selection only)
- `script:run_global_diag_scale_suite` → `scripts/51_moddedgpt_global_diag_scale/run_global_diag_scale_suite.py` (explicit selection only)

Historical results are not stored in this source repository. See
`../../docs/REPRODUCIBILITY.md` for evidence tiers and limitations.
