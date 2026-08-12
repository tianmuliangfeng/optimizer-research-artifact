# Commands for 34 Selective Primary Comparison

- Status: `analysis_only`
- Code: `../../scripts/34_selective_primary_comparison`
- Source-freeze tier: `legacy_command_or_live_source`
- Generic artifact verification: `true`
- Fresh rerun: `true`
- One-click rerun: `true`
- Native resume: `false`
- Native verification: `false`

Inspect the frozen public entrypoints:

```bash
python reproducibility/reproduce.py inspect 34_selective_primary_comparison
```

Build a fresh reproduction plan (read-only by default).
Machine-specific settings must use repeated `--env KEY=VALUE`
arguments, while runner-specific options use repeated `--arg`
values, so both are covered by the plan receipt. A plan may be
inspected with missing required arguments, but execution is refused
until every declared argument group is satisfied:

```bash
python reproducibility/reproduce.py reproduce 34_selective_primary_comparison
```

Verify an existing result without training. The run path must
match a declared legacy result root or carry a matching sealed
source-snapshot lineage:

```bash
python reproducibility/reproduce.py verify \
  34_selective_primary_comparison --results-root /path/to/results \
  --run-dir /path/to/results/34_selective_primary_comparison/RUN_ID
```

## Entrypoints

- `command:20260727_selective_primary_comparison` -> `commands/34_selective_primary_comparison/20260727_selective_primary_comparison.sh` (reproduce)
- `script:analyze_primary` -> `scripts/34_selective_primary_comparison/analyze_primary.py` (explicit selection only)

Historical results are not stored in this source repository. See
`../../docs/REPRODUCIBILITY.md` for evidence tiers and limitations.
