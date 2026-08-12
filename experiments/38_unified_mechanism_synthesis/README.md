# 38 Unified Mechanism Synthesis

- Status: `analysis_only`
- Code: `../../scripts/38_unified_mechanism_synthesis`
- Source-freeze tier: `legacy_command_or_live_source`
- Generic artifact verification: `true`
- Fresh rerun: `true`
- One-click rerun: `true`
- Native resume: `false`
- Native verification: `false`

Inspect the frozen public entrypoints:

```bash
python reproducibility/reproduce.py inspect 38_unified_mechanism_synthesis
```

Build a fresh reproduction plan (read-only by default).
Machine-specific settings must use repeated `--env KEY=VALUE`
arguments, while runner-specific options use repeated `--arg`
values, so both are covered by the plan receipt. A plan may be
inspected with missing required arguments, but execution is refused
until every declared argument group is satisfied:

```bash
python reproducibility/reproduce.py reproduce 38_unified_mechanism_synthesis
```

Verify an existing result without training. The run path must
match a declared legacy result root or carry a matching sealed
source-snapshot lineage:

```bash
python reproducibility/reproduce.py verify \
  38_unified_mechanism_synthesis --results-root /path/to/results \
  --run-dir /path/to/results/38_unified_mechanism_synthesis/RUN_ID
```

## Entrypoints

- `command:20260729_unified_mechanism_synthesis` → `commands/38_unified_mechanism_synthesis/20260729_unified_mechanism_synthesis.sh` (reproduce)
- `script:analyze_unified_mechanism` → `scripts/38_unified_mechanism_synthesis/analyze_unified_mechanism.py` (explicit selection only)
- `script:validate_unified_mechanism` → `scripts/38_unified_mechanism_synthesis/validate_unified_mechanism.py` (explicit selection only)

Historical results are not stored in this source repository. See
`../../docs/REPRODUCIBILITY.md` for evidence tiers and limitations.
