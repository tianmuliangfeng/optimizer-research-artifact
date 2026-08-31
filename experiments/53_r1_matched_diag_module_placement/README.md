# 53 R1 Matched Diag Module Placement

- Status: `implemented`
- Code: `../../scripts/53_r1_matched_diag_module_placement`
- Source-freeze tier: `sealed_source_snapshot`
- Generic artifact verification: `true`
- Fresh rerun: `false`
- One-click rerun: `false`
- Native resume: `true`
- Native verification: `true`

## Accepted result

A representation-matched diagonal factorial isolates feed-forward module
placement at Modded GPT 124M. The accepted three-seed analysis supports
placement-specific attribution without conflating full and block-4 state.

Inspect the frozen public entrypoints:

```bash
python reproducibility/reproduce.py inspect 53_r1_matched_diag_module_placement
```

A fresh rerun is not declared for this archived experiment.
Use the native archival validator below when available.

Verify an existing result without training. The run path must
match a declared legacy result root or carry a matching sealed
source-snapshot lineage:

```bash
python reproducibility/reproduce.py verify \
  53_r1_matched_diag_module_placement --results-root /path/to/results \
  --run-dir /path/to/results/53_r1_matched_diag_module_placement/RUN_ID
```

Resume an interrupted native run:

```bash
python reproducibility/reproduce.py resume \
  53_r1_matched_diag_module_placement --results-root /path/to/results \
  --run-dir /path/to/results/53_r1_matched_diag_module_placement/RUN_ID
```

Run the experiment-specific native validator:

```bash
python reproducibility/reproduce.py native-verify \
  53_r1_matched_diag_module_placement --results-root /path/to/results \
  --run-dir /path/to/results/53_r1_matched_diag_module_placement/RUN_ID
```

## Entrypoints

- `command:20260817_ex53_r1_matched_diag_module_placement` → `commands/53_r1_matched_diag_module_placement/20260817_ex53_r1_matched_diag_module_placement.sh` (native-verify, resume)
- `script:run_matched_diag` → `scripts/53_r1_matched_diag_module_placement/run_matched_diag.py` (explicit selection only)
- `script:run_matched_diag_suite` → `scripts/53_r1_matched_diag_module_placement/run_matched_diag_suite.py` (explicit selection only)

Historical results are not stored in this source repository. See
`../../docs/REPRODUCIBILITY.md` for evidence tiers and limitations.
