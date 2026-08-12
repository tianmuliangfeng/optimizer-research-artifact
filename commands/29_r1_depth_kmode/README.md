# Experiment 29 commands

These launchers are portable copies of the accepted R1 depth × K-mode
workflow. Set `SNM_OFFICIAL_REPO`, `SNM_CONTROLLER_PYTHON`,
`SNM_TRAINING_PYTHON`, `SNM_RESULTS_ROOT`, and `EX29_GPUS` for the target host.

Use the guarded dispatcher from the repository root:

```bash
python reproducibility/reproduce.py reproduce 29_r1_depth_kmode
```

Planning is read-only. Execution requires the printed SHA-256 receipt. The
formal quality workflow deliberately has no mid-run checkpoint/resume claim.

