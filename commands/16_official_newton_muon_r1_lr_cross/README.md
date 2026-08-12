# Commands for 16 Official Newton Muon R1 Lr Cross

- Status: `implemented`
- Code: `../../scripts/16_official_newton_muon_r1_lr_cross`
- Source-freeze tier: `legacy_command_or_live_source`
- Generic artifact verification: `true`
- Fresh rerun: `true`
- One-click rerun: `false`
- Native resume: `false`
- Native verification: `false`

Inspect the frozen public entrypoints:

```bash
python reproducibility/reproduce.py inspect 16_official_newton_muon_r1_lr_cross
```

Build a fresh reproduction plan (read-only by default).
Machine-specific settings must use repeated `--env KEY=VALUE`
arguments, while runner-specific options use repeated `--arg`
values, so both are covered by the plan receipt. A plan may be
inspected with missing required arguments, but execution is refused
until every declared argument group is satisfied:

```bash
python reproducibility/reproduce.py reproduce 16_official_newton_muon_r1_lr_cross --entrypoint script:run_all_seeds_gpu1
python reproducibility/reproduce.py reproduce 16_official_newton_muon_r1_lr_cross --entrypoint script:run_official_newton_muon_r1_lr_cross
```

Verify an existing result without training. The run path must
match a declared legacy result root or carry a matching sealed
source-snapshot lineage:

```bash
python reproducibility/reproduce.py verify \
  16_official_newton_muon_r1_lr_cross --results-root /path/to/results \
  --run-dir /path/to/results/16_official_newton_muon_r1_lr_cross/RUN_ID
```

## Entrypoints

- `script:run_all_seeds_gpu1` -> `scripts/16_official_newton_muon_r1_lr_cross/run_all_seeds_gpu1.sh` (reproduce)
- `script:run_official_newton_muon_r1_lr_cross` -> `scripts/16_official_newton_muon_r1_lr_cross/run_official_newton_muon_r1_lr_cross.py` (reproduce)

Historical results are not stored in this source repository. See
`../../docs/REPRODUCIBILITY.md` for evidence tiers and limitations.
