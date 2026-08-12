# 39 Submission Efficiency And Sensitivity

- Status: `implemented`
- Code: `../../scripts/39_submission_efficiency_and_sensitivity`
- Source-freeze tier: `partial_or_run_specific_snapshot`
- Generic artifact verification: `true`
- Fresh rerun: `true`
- One-click rerun: `false`
- Native resume: `false`
- Native verification: `false`

Inspect the frozen public entrypoints:

```bash
python reproducibility/reproduce.py inspect 39_submission_efficiency_and_sensitivity
```

Build a fresh reproduction plan (read-only by default).
Machine-specific settings must use repeated `--env KEY=VALUE`
arguments, while runner-specific options use repeated `--arg`
values, so both are covered by the plan receipt. A plan may be
inspected with missing required arguments, but execution is refused
until every declared argument group is satisfied:

```bash
python reproducibility/reproduce.py reproduce 39_submission_efficiency_and_sensitivity --entrypoint command:20260729_r1_isolated_efficiency_followup
python reproducibility/reproduce.py reproduce 39_submission_efficiency_and_sensitivity --entrypoint command:20260729_r1_shared_lr_sensitivity_followup
python reproducibility/reproduce.py reproduce 39_submission_efficiency_and_sensitivity --entrypoint command:20260729_submission_efficiency_and_sensitivity_audit
python reproducibility/reproduce.py reproduce 39_submission_efficiency_and_sensitivity --entrypoint command:20260729_submission_efficiency_and_sensitivity_full
```

Verify an existing result without training. The run path must
match a declared legacy result root or carry a matching sealed
source-snapshot lineage:

```bash
python reproducibility/reproduce.py verify \
  39_submission_efficiency_and_sensitivity --results-root /path/to/results \
  --run-dir /path/to/results/39_submission_efficiency_and_sensitivity/RUN_ID
```

## Entrypoints

- `command:20260729_r1_isolated_efficiency_followup` → `commands/39_submission_efficiency_and_sensitivity/20260729_r1_isolated_efficiency_followup.sh` (reproduce)
- `command:20260729_r1_shared_lr_sensitivity_followup` → `commands/39_submission_efficiency_and_sensitivity/20260729_r1_shared_lr_sensitivity_followup.sh` (reproduce)
- `command:20260729_submission_efficiency_and_sensitivity_audit` → `commands/39_submission_efficiency_and_sensitivity/20260729_submission_efficiency_and_sensitivity_audit.sh` (reproduce)
- `command:20260729_submission_efficiency_and_sensitivity_full` → `commands/39_submission_efficiency_and_sensitivity/20260729_submission_efficiency_and_sensitivity_full.sh` (reproduce)
- `script:run_r1_lr_sensitivity` → `scripts/39_submission_efficiency_and_sensitivity/run_r1_lr_sensitivity.py` (explicit selection only)

Historical results are not stored in this source repository. See
`../../docs/REPRODUCIBILITY.md` for evidence tiers and limitations.
