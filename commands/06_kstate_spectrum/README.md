# Commands for 06 Kstate Spectrum

- Status: `implemented`
- Code: `../../scripts/06_kstate_spectrum`
- Source-freeze tier: `legacy_command_or_live_source`
- Generic artifact verification: `true`
- Fresh rerun: `true`
- One-click rerun: `false`
- Native resume: `false`
- Native verification: `false`

Inspect the frozen public entrypoints:

```bash
python reproducibility/reproduce.py inspect 06_kstate_spectrum
```

Build a fresh reproduction plan (read-only by default).
Machine-specific settings must use repeated `--env KEY=VALUE`
arguments, while runner-specific options use repeated `--arg`
values, so both are covered by the plan receipt. A plan may be
inspected with missing required arguments, but execution is refused
until every declared argument group is satisfied:

```bash
python reproducibility/reproduce.py reproduce 06_kstate_spectrum --entrypoint script:run_cproj_k_structure
python reproducibility/reproduce.py reproduce 06_kstate_spectrum --entrypoint script:run_quadratic_score_probe
python reproducibility/reproduce.py reproduce 06_kstate_spectrum --entrypoint script:run_quadratic_score_probe_p1
python reproducibility/reproduce.py reproduce 06_kstate_spectrum --entrypoint script:run_quadratic_score_probe_p2
python reproducibility/reproduce.py reproduce 06_kstate_spectrum --entrypoint script:run_quadratic_score_probe_p3
python reproducibility/reproduce.py reproduce 06_kstate_spectrum --entrypoint script:run_targeted_diag_generalization
```

Verify an existing result without training. The run path must
match a declared legacy result root or carry a matching sealed
source-snapshot lineage:

```bash
python reproducibility/reproduce.py verify \
  06_kstate_spectrum --results-root /path/to/results \
  --run-dir /path/to/results/06_kstate_spectrum/RUN_ID
```

## Entrypoints

- `script:run_cproj_k_structure` -> `scripts/06_kstate_spectrum/run_cproj_k_structure.py` (reproduce)
- `script:run_quadratic_score_probe` -> `scripts/06_kstate_spectrum/run_quadratic_score_probe.py` (reproduce)
- `script:run_quadratic_score_probe_p1` -> `scripts/06_kstate_spectrum/run_quadratic_score_probe_p1.py` (reproduce)
- `script:run_quadratic_score_probe_p2` -> `scripts/06_kstate_spectrum/run_quadratic_score_probe_p2.py` (reproduce)
- `script:run_quadratic_score_probe_p3` -> `scripts/06_kstate_spectrum/run_quadratic_score_probe_p3.py` (reproduce)
- `script:run_targeted_diag_generalization` -> `scripts/06_kstate_spectrum/run_targeted_diag_generalization.py` (reproduce)

Historical results are not stored in this source repository. See
`../../docs/REPRODUCIBILITY.md` for evidence tiers and limitations.
