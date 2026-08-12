# Commands for 37 Mech09 Downproj Refresh Mediation

- Status: `implemented`
- Code: `../../scripts/37_mech09_downproj_refresh_mediation`
- Source-freeze tier: `partial_or_run_specific_snapshot`
- Generic artifact verification: `true`
- Fresh rerun: `true`
- One-click rerun: `false`
- Native resume: `false`
- Native verification: `false`

Inspect the frozen public entrypoints:

```bash
python reproducibility/reproduce.py inspect 37_mech09_downproj_refresh_mediation
```

Build a fresh reproduction plan (read-only by default).
Machine-specific settings must use repeated `--env KEY=VALUE`
arguments, while runner-specific options use repeated `--arg`
values, so both are covered by the plan receipt. A plan may be
inspected with missing required arguments, but execution is refused
until every declared argument group is satisfied:

```bash
python reproducibility/reproduce.py reproduce 37_mech09_downproj_refresh_mediation --entrypoint command:20260728_mech09_downproj_refresh_mediation
python reproducibility/reproduce.py reproduce 37_mech09_downproj_refresh_mediation --entrypoint command:20260728_mech09r_causal_tree_repair
```

Verify an existing result without training. The run path must
match a declared legacy result root or carry a matching sealed
source-snapshot lineage:

```bash
python reproducibility/reproduce.py verify \
  37_mech09_downproj_refresh_mediation --results-root /path/to/results \
  --run-dir /path/to/results/37_mech09_downproj_refresh_mediation/RUN_ID
```

## Entrypoints

- `command:20260728_mech09_downproj_refresh_mediation` -> `commands/37_mech09_downproj_refresh_mediation/20260728_mech09_downproj_refresh_mediation.sh` (reproduce)
- `command:20260728_mech09r_causal_tree_repair` -> `commands/37_mech09_downproj_refresh_mediation/20260728_mech09r_causal_tree_repair.sh` (reproduce)
- `script:run_mech09` -> `scripts/37_mech09_downproj_refresh_mediation/run_mech09.py` (explicit selection only)
- `script:run_mech09r` -> `scripts/37_mech09_downproj_refresh_mediation/run_mech09r.py` (explicit selection only)

Historical results are not stored in this source repository. See
`../../docs/REPRODUCIBILITY.md` for evidence tiers and limitations.
