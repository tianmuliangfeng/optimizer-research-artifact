# Commands for 04 Dataset Generalization

- Status: `implemented`
- Code: `../../scripts/04_dataset_generalization`
- Source-freeze tier: `legacy_command_or_live_source`
- Generic artifact verification: `true`
- Fresh rerun: `true`
- One-click rerun: `false`
- Native resume: `false`
- Native verification: `false`

Inspect the frozen public entrypoints:

```bash
python reproducibility/reproduce.py inspect 04_dataset_generalization
```

Build a fresh reproduction plan (read-only by default).
Machine-specific settings must use repeated `--env KEY=VALUE`
arguments, while runner-specific options use repeated `--arg`
values, so both are covered by the plan receipt. A plan may be
inspected with missing required arguments, but execution is refused
until every declared argument group is satisfied:

```bash
python reproducibility/reproduce.py reproduce 04_dataset_generalization --entrypoint script:run_wikitext103_100m
python reproducibility/reproduce.py reproduce 04_dataset_generalization --entrypoint script:run_wikitext103_24l_12k_lr_sweep
```

Verify an existing result without training. The run path must
match a declared legacy result root or carry a matching sealed
source-snapshot lineage:

```bash
python reproducibility/reproduce.py verify \
  04_dataset_generalization --results-root /path/to/results \
  --run-dir /path/to/results/04_dataset_generalization/RUN_ID
```

## Entrypoints

- `script:run_wikitext103_100m` -> `scripts/04_dataset_generalization/run_wikitext103_100m.py` (reproduce)
- `script:run_wikitext103_24l_12k_lr_sweep` -> `scripts/04_dataset_generalization/run_wikitext103_24l_12k_lr_sweep.py` (reproduce)

Historical results are not stored in this source repository. See
`../../docs/REPRODUCIBILITY.md` for evidence tiers and limitations.
