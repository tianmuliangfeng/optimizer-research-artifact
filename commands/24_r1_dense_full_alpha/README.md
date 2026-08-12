# Commands for 24 R1 Dense Full Alpha

- Status: `implemented`
- Code: `../../scripts/24_r1_dense_full_alpha`
- Source-freeze tier: `partial_or_run_specific_snapshot`
- Generic artifact verification: `true`
- Fresh rerun: `true`
- One-click rerun: `true`
- Native resume: `false`
- Native verification: `false`

Inspect the frozen public entrypoints:

```bash
python reproducibility/reproduce.py inspect 24_r1_dense_full_alpha
```

Build a fresh reproduction plan (read-only by default).
Machine-specific settings must use repeated `--env KEY=VALUE`
arguments, while runner-specific options use repeated `--arg`
values, so both are covered by the plan receipt. A plan may be
inspected with missing required arguments, but execution is refused
until every declared argument group is satisfied:

```bash
python reproducibility/reproduce.py reproduce 24_r1_dense_full_alpha
```

Verify an existing result without training. The run path must
match a declared legacy result root or carry a matching sealed
source-snapshot lineage:

```bash
python reproducibility/reproduce.py verify \
  24_r1_dense_full_alpha --results-root /path/to/results \
  --run-dir /path/to/results/24_r1_dense_full_alpha/RUN_ID
```

## Entrypoints

- `command:20260727_r1_dense_full_alpha_confirmatory_multiseed` -> `commands/24_r1_dense_full_alpha/20260727_r1_dense_full_alpha_confirmatory_multiseed.sh` (reproduce)
- `script:run_confirmatory_batch` -> `scripts/24_r1_dense_full_alpha/run_confirmatory_batch.py` (explicit selection only)
- `script:run_r1_dense_full_alpha` -> `scripts/24_r1_dense_full_alpha/run_r1_dense_full_alpha.py` (explicit selection only)

Historical results are not stored in this source repository. See
`../../docs/REPRODUCIBILITY.md` for evidence tiers and limitations.
