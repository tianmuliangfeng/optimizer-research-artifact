# Commands for 25 Owt Depth Kmode

- Status: `implemented`
- Code: `../../scripts/25_owt_depth_kmode`
- Source-freeze tier: `legacy_command_or_live_source`
- Generic artifact verification: `true`
- Fresh rerun: `true`
- One-click rerun: `false`
- Native resume: `false`
- Native verification: `false`

Inspect the frozen public entrypoints:

```bash
python reproducibility/reproduce.py inspect 25_owt_depth_kmode
```

Build a fresh reproduction plan (read-only by default).
Machine-specific settings must use repeated `--env KEY=VALUE`
arguments, while runner-specific options use repeated `--arg`
values, so both are covered by the plan receipt. A plan may be
inspected with missing required arguments, but execution is refused
until every declared argument group is satisfied:

```bash
python reproducibility/reproduce.py reproduce 25_owt_depth_kmode --arg=--formal
```

Verify an existing result without training. The run path must
match a declared legacy result root or carry a matching sealed
source-snapshot lineage:

```bash
python reproducibility/reproduce.py verify \
  25_owt_depth_kmode --results-root /path/to/results \
  --run-dir /path/to/results/25_owt_depth_kmode/RUN_ID
```

## Entrypoints

- `script:run_owt_depth_kmode` -> `scripts/25_owt_depth_kmode/run_owt_depth_kmode.py` (reproduce; requires receipt-bound --arg values)

Historical results are not stored in this source repository. See
`../../docs/REPRODUCIBILITY.md` for evidence tiers and limitations.
