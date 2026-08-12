# Commands for 45 R1 Mousse Strong Baseline

- Status: `implemented`
- Code: `../../scripts/45_r1_mousse_strong_baseline`
- Source-freeze tier: `partial_or_run_specific_snapshot`
- Generic artifact verification: `true`
- Fresh rerun: `true`
- One-click rerun: `true`
- Native resume: `false`
- Native verification: `false`

Inspect the frozen public entrypoints:

```bash
python reproducibility/reproduce.py inspect 45_r1_mousse_strong_baseline
```

Build a fresh reproduction plan (read-only by default).
Machine-specific settings must use repeated `--env KEY=VALUE`
arguments, while runner-specific options use repeated `--arg`
values, so both are covered by the plan receipt. A plan may be
inspected with missing required arguments, but execution is refused
until every declared argument group is satisfied:

```bash
python reproducibility/reproduce.py reproduce 45_r1_mousse_strong_baseline
```

Verify an existing result without training. The run path must
match a declared legacy result root or carry a matching sealed
source-snapshot lineage:

```bash
python reproducibility/reproduce.py verify \
  45_r1_mousse_strong_baseline --results-root /path/to/results \
  --run-dir /path/to/results/45_r1_mousse_strong_baseline/RUN_ID
```

## Entrypoints

- `command:20260730_r1_mousse_strong_baseline` -> `commands/45_r1_mousse_strong_baseline/20260730_r1_mousse_strong_baseline.sh` (reproduce)
- `script:run_r1_mousse` -> `scripts/45_r1_mousse_strong_baseline/run_r1_mousse.py` (explicit selection only)

Historical results are not stored in this source repository. See
`../../docs/REPRODUCIBILITY.md` for evidence tiers and limitations.
