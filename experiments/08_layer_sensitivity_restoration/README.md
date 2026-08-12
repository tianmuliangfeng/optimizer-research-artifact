# 08 Layer Sensitivity Restoration

- Status: `implemented`
- Code: `../../scripts/08_layer_sensitivity_restoration`
- Source-freeze tier: `legacy_command_or_live_source`
- Generic artifact verification: `true`
- Fresh rerun: `true`
- One-click rerun: `true`
- Native resume: `false`
- Native verification: `false`

Inspect the frozen public entrypoints:

```bash
python reproducibility/reproduce.py inspect 08_layer_sensitivity_restoration
```

Build a fresh reproduction plan (read-only by default).
Machine-specific settings must use repeated `--env KEY=VALUE`
arguments, while runner-specific options use repeated `--arg`
values, so both are covered by the plan receipt. A plan may be
inspected with missing required arguments, but execution is refused
until every declared argument group is satisfied:

```bash
python reproducibility/reproduce.py reproduce 08_layer_sensitivity_restoration
```

Verify an existing result without training. The run path must
match a declared legacy result root or carry a matching sealed
source-snapshot lineage:

```bash
python reproducibility/reproduce.py verify \
  08_layer_sensitivity_restoration --results-root /path/to/results \
  --run-dir /path/to/results/08_layer_sensitivity_restoration/RUN_ID
```

## Entrypoints

- `script:run_12l_50m_seed2026_completion` → `scripts/08_layer_sensitivity_restoration/run_12l_50m_seed2026_completion.py` (reproduce)

Historical results are not stored in this source repository. See
`../../docs/REPRODUCIBILITY.md` for evidence tiers and limitations.
