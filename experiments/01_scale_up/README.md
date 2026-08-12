# 01 Scale Up

- Status: `implemented`
- Code: `../../scripts/01_scale_up`
- Source-freeze tier: `legacy_command_or_live_source`
- Generic artifact verification: `true`
- Fresh rerun: `true`
- One-click rerun: `false`
- Native resume: `false`
- Native verification: `false`

Inspect the frozen public entrypoints:

```bash
python reproducibility/reproduce.py inspect 01_scale_up
```

Build a fresh reproduction plan (read-only by default).
Machine-specific settings must use repeated `--env KEY=VALUE`
arguments, while runner-specific options use repeated `--arg`
values, so both are covered by the plan receipt. A plan may be
inspected with missing required arguments, but execution is refused
until every declared argument group is satisfied:

```bash
python reproducibility/reproduce.py reproduce 01_scale_up --entrypoint script:run_24l_module_bridge
python reproducibility/reproduce.py reproduce 01_scale_up --entrypoint script:run_scale_up_300m
```

Verify an existing result without training. The run path must
match a declared legacy result root or carry a matching sealed
source-snapshot lineage:

```bash
python reproducibility/reproduce.py verify \
  01_scale_up --results-root /path/to/results \
  --run-dir /path/to/results/01_scale_up/RUN_ID
```

## Entrypoints

- `script:run_24l_module_bridge` → `scripts/01_scale_up/run_24l_module_bridge.py` (reproduce)
- `script:run_scale_up_300m` → `scripts/01_scale_up/run_scale_up_300m.py` (reproduce)

Historical results are not stored in this source repository. See
`../../docs/REPRODUCIBILITY.md` for evidence tiers and limitations.
