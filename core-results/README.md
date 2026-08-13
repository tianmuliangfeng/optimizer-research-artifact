# core-results

This directory is the compact, portable evidence release for the paper.
It contains 502 explicitly selected artifacts plus integrity metadata.
Every package reference is relative to this directory; source-machine roots are not retained.

## Validation

Run from this directory after copying or moving it:

```text
python tools/validate_core_results_package.py .
```

The validator checks the complete file inventory, SHA-256 hashes, JSON/CSV parsing,
privacy and path portability, evidence statuses, and the release-mode gates.

## Intentional compact-package omissions

10 byte-verified full-archive inputs are intentionally not bundled.
Their source hashes, sizes, reasons, and compact anchor rows are registered in
`provenance/omission_ledger.json`; omission never means missing evidence.

## Release state

This is the final validated package.
EX48 formal, endpoint-checkpoint, analysis, verification, and resume-lineage gates passed.
