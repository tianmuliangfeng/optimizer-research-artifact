# Commands for 27 Mech01 Unified K Diagnostics

- Status: `implemented`
- Code: `../../scripts/27_mech01_unified_k_diagnostics`
- Source-freeze tier: `legacy_command_or_live_source`
- Generic artifact verification: `true`
- Fresh rerun: `true`
- One-click rerun: `false`
- Native resume: `false`
- Native verification: `false`

Inspect the frozen public entrypoints:

```bash
python reproducibility/reproduce.py inspect 27_mech01_unified_k_diagnostics
```

Build a fresh reproduction plan (read-only by default).
Machine-specific settings must use repeated `--env KEY=VALUE`
arguments, while runner-specific options use repeated `--arg`
values, so both are covered by the plan receipt. A plan may be
inspected with missing required arguments, but execution is refused
until every declared argument group is satisfied:

```bash
python reproducibility/reproduce.py reproduce 27_mech01_unified_k_diagnostics --entrypoint command:20260727_mech01_llama_host_remaining
python reproducibility/reproduce.py reproduce 27_mech01_unified_k_diagnostics --entrypoint command:20260727_mech01_r1_none6200
```

Verify an existing result without training. The run path must
match a declared legacy result root or carry a matching sealed
source-snapshot lineage:

```bash
python reproducibility/reproduce.py verify \
  27_mech01_unified_k_diagnostics --results-root /path/to/results \
  --run-dir /path/to/results/27_mech01_unified_k_diagnostics/RUN_ID
```

## Entrypoints

- `command:20260727_mech01_llama_host_remaining` -> `commands/27_mech01_unified_k_diagnostics/20260727_mech01_llama_host_remaining.sh` (reproduce)
- `command:20260727_mech01_r1_none6200` -> `commands/27_mech01_unified_k_diagnostics/20260727_mech01_r1_none6200.sh` (reproduce)
- `script:run_mech01` -> `scripts/27_mech01_unified_k_diagnostics/run_mech01.py` (explicit selection only)

Historical results are not stored in this source repository. See
`../../docs/REPRODUCIBILITY.md` for evidence tiers and limitations.
