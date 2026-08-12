# Commands for 44 Newton Muon Record17 455M

- Status: `implemented`
- Code: `../../scripts/44_newton_muon_record17_455m`
- Source-freeze tier: `sealed_source_snapshot`
- Generic artifact verification: `true`
- Fresh rerun: `true`
- One-click rerun: `true`
- Native resume: `false`
- Native verification: `false`

Inspect the frozen public entrypoints:

```bash
python reproducibility/reproduce.py inspect 44_newton_muon_record17_455m
```

Build a fresh reproduction plan (read-only by default).
Machine-specific settings must use repeated `--env KEY=VALUE`
arguments, while runner-specific options use repeated `--arg`
values, so both are covered by the plan receipt. A plan may be
inspected with missing required arguments, but execution is refused
until every declared argument group is satisfied:

```bash
python reproducibility/reproduce.py reproduce 44_newton_muon_record17_455m
```

Verify an existing result without training. The run path must
match a declared legacy result root or carry a matching sealed
source-snapshot lineage:

```bash
python reproducibility/reproduce.py verify \
  44_newton_muon_record17_455m --results-root /path/to/results \
  --run-dir /path/to/results/44_newton_muon_record17_455m/RUN_ID
```

## Entrypoints

- `command:20260730_newton_muon_record17_455m` -> `commands/44_newton_muon_record17_455m/20260730_newton_muon_record17_455m.sh` (reproduce)
- `script:run_record17_cell` -> `scripts/44_newton_muon_record17_455m/run_record17_cell.py` (explicit selection only)
- `script:run_record17_suite` -> `scripts/44_newton_muon_record17_455m/run_record17_suite.py` (explicit selection only)

Historical results are not stored in this source repository. See
`../../docs/REPRODUCIBILITY.md` for evidence tiers and limitations.
