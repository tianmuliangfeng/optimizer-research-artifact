# Commands for 13 Reference Lr Sanity

- Status: `implemented`
- Code: `../../scripts/13_reference_lr_sanity`
- Source-freeze tier: `legacy_command_or_live_source`
- Generic artifact verification: `true`
- Fresh rerun: `true`
- One-click rerun: `true`
- Native resume: `false`
- Native verification: `false`

Inspect the frozen public entrypoints:

```bash
python reproducibility/reproduce.py inspect 13_reference_lr_sanity
```

Build a fresh reproduction plan (read-only by default).
Machine-specific settings must use repeated `--env KEY=VALUE`
arguments, while runner-specific options use repeated `--arg`
values, so both are covered by the plan receipt. A plan may be
inspected with missing required arguments, but execution is refused
until every declared argument group is satisfied:

```bash
python reproducibility/reproduce.py reproduce 13_reference_lr_sanity
```

Verify an existing result without training. The run path must
match a declared legacy result root or carry a matching sealed
source-snapshot lineage:

```bash
python reproducibility/reproduce.py verify \
  13_reference_lr_sanity --results-root /path/to/results \
  --run-dir /path/to/results/13_reference_lr_sanity/RUN_ID
```

## Entrypoints

- `script:run_reference_lr_sanity` -> `scripts/13_reference_lr_sanity/run_reference_lr_sanity.py` (reproduce)

Historical results are not stored in this source repository. See
`../../docs/REPRODUCIBILITY.md` for evidence tiers and limitations.
