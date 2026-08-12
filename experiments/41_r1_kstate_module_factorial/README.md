# 41 R1 Kstate Module Factorial

- Status: `implemented`
- Code: `../../scripts/41_r1_kstate_module_factorial`
- Source-freeze tier: `partial_or_run_specific_snapshot`
- Generic artifact verification: `true`
- Fresh rerun: `true`
- One-click rerun: `true`
- Native resume: `false`
- Native verification: `false`

Inspect the frozen public entrypoints:

```bash
python reproducibility/reproduce.py inspect 41_r1_kstate_module_factorial
```

Build a fresh reproduction plan (read-only by default).
Machine-specific settings must use repeated `--env KEY=VALUE`
arguments, while runner-specific options use repeated `--arg`
values, so both are covered by the plan receipt. A plan may be
inspected with missing required arguments, but execution is refused
until every declared argument group is satisfied:

```bash
python reproducibility/reproduce.py reproduce 41_r1_kstate_module_factorial
```

Verify an existing result without training. The run path must
match a declared legacy result root or carry a matching sealed
source-snapshot lineage:

```bash
python reproducibility/reproduce.py verify \
  41_r1_kstate_module_factorial --results-root /path/to/results \
  --run-dir /path/to/results/41_r1_kstate_module_factorial/RUN_ID
```

## Entrypoints

- `command:20260729_r1_kstate_module_factorial` → `commands/41_r1_kstate_module_factorial/20260729_r1_kstate_module_factorial.sh` (reproduce)
- `script:run_r1_module_factorial` → `scripts/41_r1_kstate_module_factorial/run_r1_module_factorial.py` (explicit selection only)
- `script:run_r1_module_factorial_suite` → `scripts/41_r1_kstate_module_factorial/run_r1_module_factorial_suite.py` (explicit selection only)

Historical results are not stored in this source repository. See
`../../docs/REPRODUCIBILITY.md` for evidence tiers and limitations.
