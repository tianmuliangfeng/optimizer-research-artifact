# OWT 12L depth × c_proj K-mode analysis contract

This contract is frozen before the new formal results are inspected.

## Question and intervention

Use the historical OpenWebText 12L/D768/H12, batch-16, block-512,
5,000-update protocol.  For each depth rule, apply either `none` or `diag`
only to the selected `mlp.c_proj` matrices.  Every unselected `mlp.c_proj`
and every non-`mlp.c_proj` matrix retains full Newton-Muon.

The paired depth rules are:

- early: h0-h7
- center: h2-h9
- late: h4-h11
- edge: h0-h3 and h8-h11
- all: h0-h11

Run seeds 2024, 2025, and 2026.  Rerun full Newton-Muon and Muon anchors
under the same launcher rather than relying on historical curves.

## Primary estimand

The primary endpoint is validation loss at the last common validation
checkpoint, step 4,500.  For every depth rule and seed, compute the paired
contrast

`Delta_diag-none(rule, seed) = L_diag(rule, seed) - L_none(rule, seed)`.

Negative values favor diagonal K.  Report the three-seed mean, sample
standard deviation, and sign count.  The all-depth contrast is the primary
reviewer-facing result.  The four partial rules test depth interaction.

## Secondary estimands

1. normalized validation AUC over the common step 0:500:4,500 grid;
2. mean of the final three validation checkpoints;
3. best validation loss, retained only for continuity with the historical
   layer-mask table;
4. exact K-state bytes, peak allocated memory, and elapsed time.

Timing is descriptive if another process trains on the same node.

## Interaction and decision language

Compute

`I(rule) = Delta_diag-none(rule) - Delta_diag-none(all)`.

Do not select a "best layer rule" from the same data and call it
confirmatory.  The acceptable conclusions are:

- uniform diag benefit: all five paired means are negative and at least
  four rules have 3/3 negative seed deltas;
- depth-dependent diag benefit: signs or magnitudes materially differ
  across preregistered rules;
- no reliable diag benefit: the all-depth mean is near zero or seed signs
  conflict, with no consistent partial-rule pattern.

The experiment does not test global diagonalization of attention and
`mlp.c_fc`; it varies only the `mlp.c_proj` K representation.

## Integrity checks

Every run must have:

- identical seed-matched step-0 validation loss within numerical tolerance;
- exactly ten validation checkpoints through step 4,500;
- training logs through step 4,980;
- mode counts matching the target layers;
- `retained + released = full` K-state bytes;
- no duplicate W&B run names or duplicate metric steps.

The `none` and `diag` cells must use the same
`CProjKModeNewtonMuon` implementation path.  Historical selective-mask
results are context only and are not pooled into the primary paired table.
