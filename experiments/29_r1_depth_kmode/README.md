# 29 R1 Depth Kmode

- Status: `implemented`
- Code: `../../scripts/29_r1_depth_kmode`
- Source-freeze tier: `partial_or_run_specific_snapshot`
- Generic artifact verification: `true`
- Fresh rerun: `true`
- One-click rerun: `true`
- Native resume: `false`
- Native verification: `false`

Inspect the frozen public entrypoints:

```bash
python reproducibility/reproduce.py inspect 29_r1_depth_kmode
```

Build a fresh reproduction plan (read-only by default).
Machine-specific settings must use repeated `--env KEY=VALUE`
arguments, while runner-specific options use repeated `--arg`
values, so both are covered by the plan receipt. A plan may be
inspected with missing required arguments, but execution is refused
until every declared argument group is satisfied:

```bash
python reproducibility/reproduce.py reproduce 29_r1_depth_kmode
```

Verify an existing result without training. The run path must
match a declared legacy result root or carry a matching sealed
source-snapshot lineage:

```bash
python reproducibility/reproduce.py verify \
  29_r1_depth_kmode --results-root /path/to/results \
  --run-dir /path/to/results/29_r1_depth_kmode/RUN_ID
```

## Entrypoints

- `command:20260806_ex29_r1_depth_kmode` → `commands/29_r1_depth_kmode/20260806_ex29_r1_depth_kmode.sh` (explicit selection only)
- `command:reproduce_full` → `commands/29_r1_depth_kmode/reproduce_full.sh` (reproduce)
- `script:analyze_r1_depth_kmode_formal` → `scripts/29_r1_depth_kmode/analyze_r1_depth_kmode_formal.py` (explicit selection only; requires receipt-bound --arg values)
- `script:run_r1_depth_kmode` → `scripts/29_r1_depth_kmode/run_r1_depth_kmode.py` (explicit selection only; requires receipt-bound --arg values)
- `script:run_three_seed_batch` → `scripts/29_r1_depth_kmode/run_three_seed_batch.py` (explicit selection only; requires receipt-bound --arg values)

Historical results are not stored in this source repository. See
`../../docs/REPRODUCIBILITY.md` for evidence tiers and limitations.
