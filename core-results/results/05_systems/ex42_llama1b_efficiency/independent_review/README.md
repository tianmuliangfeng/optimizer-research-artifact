# Experiment 42 independent review

This directory contains the local, read-only acceptance audit for formal run
`20260729T105505+0000`. It is separate from the remote controller's
`analysis/` output.

Key files:

- `archive_receipt.json`: received ZIP identity and safe-extraction receipt.
- `source_artifact_hashes.csv`: SHA-256 inventory of every received formal-run
  file, excluding this local review directory.
- `independent_audit.json`: integrity, rotation, formula, isolation, and scope
  checks independently recomputed from the received artifacts.
- `important_results.json`: compact paper-facing values and claim boundaries.
- `independent_audit_manifest.json`: hashes of the local review artifacts.

The narrative review is maintained at
`docs/reports/20260730_llama1b_isolated_efficiency_review.md`.

No checkpoint is required or expected for this experiment.
