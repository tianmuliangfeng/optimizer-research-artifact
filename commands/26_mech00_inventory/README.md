# Commands for 26 Mech00 Inventory

- Status: `implemented`
- Code: `../../scripts/26_mech00_inventory`
- Source-freeze tier: `legacy_command_or_live_source`
- Generic artifact verification: `true`
- Fresh rerun: `true`
- One-click rerun: `false`
- Native resume: `false`
- Native verification: `false`

Inspect the frozen public entrypoints:

```bash
python reproducibility/reproduce.py inspect 26_mech00_inventory
```

Build a fresh reproduction plan (read-only by default).
Machine-specific settings must use repeated `--env KEY=VALUE`
arguments, while runner-specific options use repeated `--arg`
values, so both are covered by the plan receipt. A plan may be
inspected with missing required arguments, but execution is refused
until every declared argument group is satisfied:

```bash
python reproducibility/reproduce.py reproduce 26_mech00_inventory --entrypoint command:20260727_mech00_llama_followup_full_hash
python reproducibility/reproduce.py reproduce 26_mech00_inventory --entrypoint command:20260727_mech00_llama_full_hash
python reproducibility/reproduce.py reproduce 26_mech00_inventory --entrypoint command:20260727_mech00_r1_full_hash
```

Verify an existing result without training. The run path must
match a declared legacy result root or carry a matching sealed
source-snapshot lineage:

```bash
python reproducibility/reproduce.py verify \
  26_mech00_inventory --results-root /path/to/results \
  --run-dir /path/to/results/26_mech00_inventory/RUN_ID
```

## Entrypoints

- `command:20260727_mech00_llama_followup_full_hash` -> `commands/26_mech00_inventory/20260727_mech00_llama_followup_full_hash.sh` (reproduce)
- `command:20260727_mech00_llama_full_hash` -> `commands/26_mech00_inventory/20260727_mech00_llama_full_hash.sh` (reproduce)
- `command:20260727_mech00_r1_full_hash` -> `commands/26_mech00_inventory/20260727_mech00_r1_full_hash.sh` (reproduce)
- `script:run_mech00_inventory` -> `scripts/26_mech00_inventory/run_mech00_inventory.py` (explicit selection only)

Historical results are not stored in this source repository. See
`../../docs/REPRODUCIBILITY.md` for evidence tiers and limitations.
