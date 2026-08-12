# 32 Mech05 Frozen Selection Rule

- Status: `analysis_only`
- Code: `../../scripts/32_mech05_frozen_selection_rule`
- Source-freeze tier: `legacy_command_or_live_source`
- Generic artifact verification: `true`
- Fresh rerun: `true`
- One-click rerun: `true`
- Native resume: `false`
- Native verification: `false`

Inspect the frozen public entrypoints:

```bash
python reproducibility/reproduce.py inspect 32_mech05_frozen_selection_rule
```

Build a fresh reproduction plan (read-only by default).
Machine-specific settings must use repeated `--env KEY=VALUE`
arguments, while runner-specific options use repeated `--arg`
values, so both are covered by the plan receipt. A plan may be
inspected with missing required arguments, but execution is refused
until every declared argument group is satisfied:

```bash
python reproducibility/reproduce.py reproduce 32_mech05_frozen_selection_rule
```

Verify an existing result without training. The run path must
match a declared legacy result root or carry a matching sealed
source-snapshot lineage:

```bash
python reproducibility/reproduce.py verify \
  32_mech05_frozen_selection_rule --results-root /path/to/results \
  --run-dir /path/to/results/32_mech05_frozen_selection_rule/RUN_ID
```

## Entrypoints

- `command:20260727_mech05_freeze_selection_rule` → `commands/32_mech05_frozen_selection_rule/20260727_mech05_freeze_selection_rule.sh` (reproduce)
- `script:analyze_mech05` → `scripts/32_mech05_frozen_selection_rule/analyze_mech05.py` (explicit selection only)

Historical results are not stored in this source repository. See
`../../docs/REPRODUCIBILITY.md` for evidence tiers and limitations.
