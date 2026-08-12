# Commands for 30 Mech02 K Geometry

- Status: `implemented`
- Code: `../../scripts/30_mech02_k_geometry`
- Source-freeze tier: `legacy_command_or_live_source`
- Generic artifact verification: `true`
- Fresh rerun: `true`
- One-click rerun: `false`
- Native resume: `false`
- Native verification: `false`

Inspect the frozen public entrypoints:

```bash
python reproducibility/reproduce.py inspect 30_mech02_k_geometry
```

Build a fresh reproduction plan (read-only by default).
Machine-specific settings must use repeated `--env KEY=VALUE`
arguments, while runner-specific options use repeated `--arg`
values, so both are covered by the plan receipt. A plan may be
inspected with missing required arguments, but execution is refused
until every declared argument group is satisfied:

```bash
python reproducibility/reproduce.py reproduce 30_mech02_k_geometry --entrypoint command:20260727_mech02_llama_host_endpoint
python reproducibility/reproduce.py reproduce 30_mech02_k_geometry --entrypoint command:20260727_mech02_r1_endpoint
```

Verify an existing result without training. The run path must
match a declared legacy result root or carry a matching sealed
source-snapshot lineage:

```bash
python reproducibility/reproduce.py verify \
  30_mech02_k_geometry --results-root /path/to/results \
  --run-dir /path/to/results/30_mech02_k_geometry/RUN_ID
```

## Entrypoints

- `command:20260727_mech02_llama_host_endpoint` -> `commands/30_mech02_k_geometry/20260727_mech02_llama_host_endpoint.sh` (reproduce)
- `command:20260727_mech02_r1_endpoint` -> `commands/30_mech02_k_geometry/20260727_mech02_r1_endpoint.sh` (reproduce)
- `script:run_mech02` -> `scripts/30_mech02_k_geometry/run_mech02.py` (explicit selection only)

Historical results are not stored in this source repository. See
`../../docs/REPRODUCIBILITY.md` for evidence tiers and limitations.
