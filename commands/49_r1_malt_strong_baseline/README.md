# Experiment 49 commands

These launchers reproduce the paper-derived MALT and MALTER-Eq17 R1
adaptations. They do not claim equivalence to unpublished author code. Set
`SNM_OFFICIAL_REPO`, `SNM_CONTROLLER_PYTHON`, `SNM_TRAINING_PYTHON`,
`SNM_RESULTS_ROOT`, and `EX49_GPUS` for the target host.

Use the guarded dispatcher from the repository root:

```bash
python reproducibility/reproduce.py reproduce 49_r1_malt_strong_baseline
```

`reproduce_full.sh` executes `preflight → pilot → formal → verify`. Native
resume is accepted-unit/upload level only; interrupted training units restart
from their frozen initialization rather than an arbitrary optimizer step.

