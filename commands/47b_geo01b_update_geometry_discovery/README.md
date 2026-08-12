# Commands for 47B Geo01B Update Geometry Discovery

- Status: `implemented`
- Code: `../../scripts/47b_geo01b_update_geometry_discovery`
- Source-freeze tier: `sealed_source_snapshot`
- Generic artifact verification: `true`
- Fresh rerun: `true`
- One-click rerun: `true`
- Native resume: `true`
- Native verification: `true`

Inspect the frozen public entrypoints:

```bash
python reproducibility/reproduce.py inspect 47b_geo01b_update_geometry_discovery
```

Build a fresh reproduction plan (read-only by default).
Machine-specific settings must use repeated `--env KEY=VALUE`
arguments, while runner-specific options use repeated `--arg`
values, so both are covered by the plan receipt. A plan may be
inspected with missing required arguments, but execution is refused
until every declared argument group is satisfied:

```bash
python reproducibility/reproduce.py reproduce 47b_geo01b_update_geometry_discovery
```

Verify an existing result without training. The run path must
match a declared legacy result root or carry a matching sealed
source-snapshot lineage:

```bash
python reproducibility/reproduce.py verify \
  47b_geo01b_update_geometry_discovery --results-root /path/to/results \
  --run-dir /path/to/results/47b_geo01b_update_geometry_discovery/RUN_ID
```

Resume an interrupted native run:

```bash
python reproducibility/reproduce.py resume \
  47b_geo01b_update_geometry_discovery --results-root /path/to/results \
  --run-dir /path/to/results/47b_geo01b_update_geometry_discovery/RUN_ID
```

Run the experiment-specific native validator:

```bash
python reproducibility/reproduce.py native-verify \
  47b_geo01b_update_geometry_discovery --results-root /path/to/results \
  --run-dir /path/to/results/47b_geo01b_update_geometry_discovery/RUN_ID
```

## Entrypoints

- `command:20260804_ex47b_geo01b_discovery` -> `commands/47b_geo01b_update_geometry_discovery/20260804_ex47b_geo01b_discovery.sh` (native-verify, resume)
- `command:reproduce_full` -> `commands/47b_geo01b_update_geometry_discovery/reproduce_full.sh` (reproduce)
- `script:run_geo01b` -> `scripts/47b_geo01b_update_geometry_discovery/run_geo01b.py` (explicit selection only)

Historical results are not stored in this source repository. See
`../../docs/REPRODUCIBILITY.md` for evidence tiers and limitations.
