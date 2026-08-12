# Commands for 47 Update Geometry Curvature

- Status: `implemented`
- Code: `../../scripts/47_update_geometry_curvature`
- Source-freeze tier: `sealed_source_snapshot`
- Generic artifact verification: `true`
- Fresh rerun: `true`
- One-click rerun: `true`
- Native resume: `true`
- Native verification: `true`

Inspect the frozen public entrypoints:

```bash
python reproducibility/reproduce.py inspect 47_update_geometry_curvature
```

Build a fresh reproduction plan (read-only by default).
Machine-specific settings must use repeated `--env KEY=VALUE`
arguments, while runner-specific options use repeated `--arg`
values, so both are covered by the plan receipt. A plan may be
inspected with missing required arguments, but execution is refused
until every declared argument group is satisfied:

```bash
python reproducibility/reproduce.py reproduce 47_update_geometry_curvature
```

Verify an existing result without training. The run path must
match a declared legacy result root or carry a matching sealed
source-snapshot lineage:

```bash
python reproducibility/reproduce.py verify \
  47_update_geometry_curvature --results-root /path/to/results \
  --run-dir /path/to/results/_shared/analysis/method_deepening_geo01_update_curvature/RUN_ID
```

Resume an interrupted native run:

```bash
python reproducibility/reproduce.py resume \
  47_update_geometry_curvature --results-root /path/to/results \
  --run-dir /path/to/results/_shared/analysis/method_deepening_geo01_update_curvature/RUN_ID
```

Run the experiment-specific native validator:

```bash
python reproducibility/reproduce.py native-verify \
  47_update_geometry_curvature --results-root /path/to/results \
  --run-dir /path/to/results/_shared/analysis/method_deepening_geo01_update_curvature/RUN_ID
```

## Entrypoints

- `command:20260804_ex47_update_geometry_curvature` -> `commands/47_update_geometry_curvature/20260804_ex47_update_geometry_curvature.sh` (native-verify, resume)
- `command:reproduce_full` -> `commands/47_update_geometry_curvature/reproduce_full.sh` (reproduce)
- `script:run_geo01` -> `scripts/47_update_geometry_curvature/run_geo01.py` (explicit selection only)

Historical results are not stored in this source repository. See
`../../docs/REPRODUCIBILITY.md` for evidence tiers and limitations.
