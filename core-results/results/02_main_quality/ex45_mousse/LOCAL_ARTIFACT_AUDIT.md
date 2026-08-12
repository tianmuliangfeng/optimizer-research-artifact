# Experiment 45 local artifact acceptance audit

## Result

- Quality evidence: accepted with documented evidence caveats.
- Pilot: 3/3 valid; local selection certificate independently verifies mousse_lr100 (0.015).
- Formal smoke: 3/3 valid, each 34 steps and four refreshes per logical matrix.
- Formal: 3/3 valid at seeds 2024/2025/2026, 6200 steps and 3,250,585,600 tokens each.
- W&B uploads: 6/6 quality runs uploaded; local metrics and W&B histories match exactly.
- Local/W&B comparisons: 4572 values across 42 run/metric series; maximum absolute difference 0.
- Source/runtime: one shared derived source and one shared accepted H100 runtime fingerprint.
- Official eight-method analyzer: completed_valid; historical identity/reuse passed_with_caveats and paired quality is eligible.
- Timing: ineligible by the frozen two-GPU-concurrency contract.

## Checkpoint boundary

The three formal manifests record non-empty remote checkpoints totaling 10,565,077,695 bytes,
and the training controller verified their existence and size before accepting each run. The uploaded
ZIP omits the checkpoint binaries, so this workstation cannot independently rehash or reload them.
This does not change the validated loss histories, but strict checkpoint-payload completeness is false.

## Archive

- SHA-256: `732a7411561e15de31f9bafeb37834c0b014cca5407305deb690994b0829fc1f`
- Entries: 422; corrupt/unsafe/duplicate: 0/0/0.
- The original ZIP is retained unchanged under experiment 45 `source_archives/`.
