# Submission efficiency and sensitivity audit contract

Version: 2026-07-29.2

## Decision

This is a read-only evidence audit. A successful audit does not mean that every
paper-facing metric is already available. The manifest must distinguish:

- `paper_ready`: directly usable under the frozen comparison contract;
- `supporting_only`: useful robustness evidence with an explicit limitation;
- `historical_only`: informative, but not comparable enough for a paper claim;
- `missing`: no accepted artifact meets the contract.

## Frozen method roles

The primary four-method set is:

1. `muon`;
2. `original_newton_muon` (`block4` or `newton_full`);
3. `selective_none` (`none` or `down_none`);
4. `selective_diag` (`diag` or `down_diag`).

`selective_diag` versus `selective_none` is never promoted to the primary
comparison. Each Selective method is interpreted against Muon and original
Newton-Muon.

## Efficiency eligibility

Paper-ready throughput requires one GPU/runtime, identical model/data/batch/
sequence/compile settings, an exclusive node, at least 32 warm-up steps, at
least 512 timed optimizer steps, four fully balanced rotated-order repeats, raw
per-repeat records, and a complete provenance manifest. Training logs collected
for quality experiments are historical timing only.

Paper-ready memory and capacity require the same model/runtime, explicit CUDA
peak-reset measurements, recorded allocated and reserved peaks, exact optimizer
and K-state bytes, and a resolved success/OOM boundary. Timing from capacity
searches is excluded.

## Sensitivity eligibility

Learning-rate sensitivity must use the four frozen roles, the final R1 recipe,
the same token budget, the same validation schedule, and the same multiplier
grid applied to each role's frozen recipe learning rates. A short, single-seed
grid is `supporting_only`; tuned-best claims require equal trial budgets,
predeclared selection, and held-out confirmation.

Alpha response curves test the Newton-Muon interpolation mechanism. They are
confirmatory mechanism evidence but do not substitute for learning-rate
sensitivity.

## Required outputs

- `source_inventory.csv`
- `metric_eligibility.csv`
- `capacity_boundary.csv`
- `fixed_batch_memory.csv`
- `sensitivity_coverage.csv`
- `gap_matrix.csv`
- `minimal_followup_contract.json`
- `SUBMISSION_EVIDENCE_AUDIT_REPORT.md`
- `report_source_notes.json`
- `artifact.json`
- `submission_evidence_manifest.json`

The audit passes only when all required inputs and computations validate. The
separate `submission_ready` field is true only when no blocking metric is
classified as `missing`.
