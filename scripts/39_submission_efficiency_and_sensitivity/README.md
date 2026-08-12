# 39 - submission efficiency and sensitivity audit

This directory audits accepted local evidence before requesting new remote
experiments. It does not modify training artifacts.

Run the command in:

`commands/39_submission_efficiency_and_sensitivity/20260729_submission_efficiency_and_sensitivity_audit.sh`

The audit deliberately separates `audit_passed` from `submission_ready`.
Missing paper-grade throughput or final-recipe LR sensitivity produces a valid
audit with explicit follow-up requirements.

Source resolution is portable but strict.  The analyzer first checks the
canonical experiment path, then any registry-scoped relocation patterns, and
finally `source_snapshot/`.  The snapshot contains only the small immutable
manifests and summary tables consumed by this audit.  Relocated sources must
match uniquely; the analyzer refuses ambiguous matches instead of guessing.
Every portable input is pinned by SHA-256 in `evidence_registry.json`, and the
preflight verifies the complete snapshot even when canonical inputs exist.

The audit command runs a no-write `--preflight-only` pass before creating the
timestamped output directory.  That pass resolves all required sources and
validates the foundation, capacity, fixed-batch memory, and historical
sensitivity inputs.  Expensive follow-ups should only be started after this
preflight succeeds on the target host.

The shared learning-rate sensitivity follow-up uses two concurrent lanes:
GPU0 runs `diag/none`, and GPU1 runs `block4/muon`.  Rerunning the wrapper
reuses `LATEST_RUN_DIR.txt`, so completed multiplier cells are skipped and
incomplete cells resume.  Set `RESUME=0` or provide a new `RUN_DIR` only when a
genuinely fresh grid is intended.

The isolated performance bundle uses a new timestamped subdirectory on each
launch.  It binds four fully balanced method-order rotations to pre/post
exclusive-node certificates and a pinned, clean official-repository
certificate.  Interrupted performance bundles are not reused.

For an unattended end-to-end run, use:

`commands/39_submission_efficiency_and_sensitivity/20260729_submission_efficiency_and_sensitivity_full.sh`

The controller performs a path-scoped Git trust/provenance preflight, verifies
that both H100 GPUs are idle, runs the two LR lanes, waits for them, runs the
isolated GPU0 efficiency benchmark, and finishes only if the independently
validated evidence audit is submission-ready.
