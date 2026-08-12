# Commands for 46 Mdp05 Confirmatory Update Shock

- Status: `implemented`
- Code: `../../scripts/46_mdp05_confirmatory_update_shock`
- Source-freeze tier: `sealed_source_snapshot`
- Generic artifact verification: `true`
- Fresh rerun: `true`
- One-click rerun: `true`
- Native resume: `true`
- Native verification: `true`

Inspect the frozen public entrypoints:

```bash
python reproducibility/reproduce.py inspect 46_mdp05_confirmatory_update_shock
```

Build a fresh reproduction plan (read-only by default).
Machine-specific settings must use repeated `--env KEY=VALUE`
arguments, while runner-specific options use repeated `--arg`
values, so both are covered by the plan receipt. A plan may be
inspected with missing required arguments, but execution is refused
until every declared argument group is satisfied:

```bash
python reproducibility/reproduce.py reproduce 46_mdp05_confirmatory_update_shock
```

Verify an existing result without training. The run path must
match a declared legacy result root or carry a matching sealed
source-snapshot lineage:

```bash
python reproducibility/reproduce.py verify \
  46_mdp05_confirmatory_update_shock --results-root /path/to/results \
  --run-dir /path/to/results/_shared/analysis/method_deepening_mdp05_confirmatory_update_shock/RUN_ID
```

Resume an interrupted native run:

```bash
python reproducibility/reproduce.py resume \
  46_mdp05_confirmatory_update_shock --results-root /path/to/results \
  --run-dir /path/to/results/_shared/analysis/method_deepening_mdp05_confirmatory_update_shock/RUN_ID \
  --env MDP05_PILOT_CERTIFICATE=/path/to/value
```

Run the experiment-specific native validator:

```bash
python reproducibility/reproduce.py native-verify \
  46_mdp05_confirmatory_update_shock --results-root /path/to/results \
  --run-dir /path/to/results/_shared/analysis/method_deepening_mdp05_confirmatory_update_shock/RUN_ID
```

## Entrypoints

- `command:20260804_ex46_mdp05_confirmatory_update_shock` -> `commands/46_mdp05_confirmatory_update_shock/20260804_ex46_mdp05_confirmatory_update_shock.sh` (native-verify, resume)
- `command:reproduce_full` -> `commands/46_mdp05_confirmatory_update_shock/reproduce_full.sh` (reproduce)
- `script:run_mdp05` -> `scripts/46_mdp05_confirmatory_update_shock/run_mdp05.py` (explicit selection only)

Historical results are not stored in this source repository. See
`../../docs/REPRODUCIBILITY.md` for evidence tiers and limitations.
