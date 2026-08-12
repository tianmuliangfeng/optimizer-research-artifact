# Commands for 43 Newton Muon Record28 275M

- Status: `implemented`
- Code: `../../scripts/43_newton_muon_record28_275m`
- Source-freeze tier: `sealed_source_snapshot`
- Generic artifact verification: `true`
- Fresh rerun: `true`
- One-click rerun: `true`
- Native resume: `false`
- Native verification: `false`

Inspect the frozen public entrypoints:

```bash
python reproducibility/reproduce.py inspect 43_newton_muon_record28_275m
```

Build a fresh reproduction plan (read-only by default).
Machine-specific settings must use repeated `--env KEY=VALUE`
arguments, while runner-specific options use repeated `--arg`
values, so both are covered by the plan receipt. A plan may be
inspected with missing required arguments, but execution is refused
until every declared argument group is satisfied:

```bash
python reproducibility/reproduce.py reproduce 43_newton_muon_record28_275m
```

Verify an existing result without training. The run path must
match a declared legacy result root or carry a matching sealed
source-snapshot lineage:

```bash
python reproducibility/reproduce.py verify \
  43_newton_muon_record28_275m --results-root /path/to/results \
  --run-dir /path/to/results/43_newton_muon_record28_275m/RUN_ID
```

## Entrypoints

- `command:20260730_newton_muon_record28_275m` -> `commands/43_newton_muon_record28_275m/20260730_newton_muon_record28_275m.sh` (reproduce)
- `script:run_record28_cell` -> `scripts/43_newton_muon_record28_275m/run_record28_cell.py` (explicit selection only)
- `script:run_record28_suite` -> `scripts/43_newton_muon_record28_275m/run_record28_suite.py` (explicit selection only)

Historical results are not stored in this source repository. See
`../../docs/REPRODUCIBILITY.md` for evidence tiers and limitations.
