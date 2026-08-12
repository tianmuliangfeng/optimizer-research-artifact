# MECH-07 LLaMA-1B family contrast

This is the only new experiment authorized by the corrected comparison
hierarchy. It is read-only and uses eight existing seed2026 checkpoints:

- `down_diag`, `down_none`, original `newton_full`, and `muon`;
- steps 1000 and 6200.

At every checkpoint, all four algorithm candidates are reconstructed on the
same build batches and the same checkpoint historical momentum:

- Muon: no input preconditioner anywhere;
- original Newton-Muon: dense input preconditioner on family-core and down;
- Selective-diag: dense family-core plus diagonal down;
- Selective-none: dense family-core plus no down preconditioner.

The primary contrasts are each Selective proposal versus Muon and versus the
original Newton-Muon baseline. `diag` versus `none` is excluded from the
primary contract.

Fresh build-split covariance is used for all candidates. Therefore this is a
matched local intervention, not a claim that every candidate reproduces its
full historical production trajectory. Long-run family ranking remains the
existing three-seed training result.

Files live in this numbered script directory and the executable instruction is
`commands/35_mech07_llama1b_family_contrast/20260727_mech07_llama1b_family_contrast.sh`.

The contract stores one frozen base offset per repeat. Controller version
`2026-07-27.2` and later expands each base into disjoint interleaved A/B batch windows
using the exact token-window length. To resume an interrupted timestamp without
re-hashing unchanged certified checkpoints, set `MECH07_RESUME_STAMP` when
running the command file.

Worker version `2026-07-27.3` resolves the intentionally ambiguous
`muon_or_adamw` model-state label using the matrix-optimizer state: every state
entry must carry Muon's `momentum` tensor and no AdamW `exp_avg` or
`exp_avg_sq`. Version-2 non-Muon cells are computation-compatible and may be
reused by the resume controller because this change only strengthens Muon
identity auditing.
