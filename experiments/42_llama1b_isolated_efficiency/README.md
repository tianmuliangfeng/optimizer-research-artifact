# 42 Llama1B Isolated Efficiency

- Status: `implemented`
- Code: `../../scripts/42_llama1b_isolated_efficiency`
- Source-freeze tier: `partial_or_run_specific_snapshot`
- Generic artifact verification: `true`
- Fresh rerun: `true`
- One-click rerun: `true`
- Native resume: `false`
- Native verification: `false`

Inspect the frozen public entrypoints:

```bash
python reproducibility/reproduce.py inspect 42_llama1b_isolated_efficiency
```

Build a fresh reproduction plan (read-only by default).
Machine-specific settings must use repeated `--env KEY=VALUE`
arguments, while runner-specific options use repeated `--arg`
values, so both are covered by the plan receipt. A plan may be
inspected with missing required arguments, but execution is refused
until every declared argument group is satisfied:

```bash
python reproducibility/reproduce.py reproduce 42_llama1b_isolated_efficiency
```

Verify an existing result without training. The run path must
match a declared legacy result root or carry a matching sealed
source-snapshot lineage:

```bash
python reproducibility/reproduce.py verify \
  42_llama1b_isolated_efficiency --results-root /path/to/results \
  --run-dir /path/to/results/42_llama1b_isolated_efficiency/RUN_ID
```

## Entrypoints

- `command:20260729_llama1b_isolated_efficiency` → `commands/42_llama1b_isolated_efficiency/20260729_llama1b_isolated_efficiency.sh` (reproduce)
- `script:run_llama1b_efficiency` → `scripts/42_llama1b_isolated_efficiency/run_llama1b_efficiency.py` (explicit selection only)

Historical results are not stored in this source repository. See
`../../docs/REPRODUCIBILITY.md` for evidence tiers and limitations.
