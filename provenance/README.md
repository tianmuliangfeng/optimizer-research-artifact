# Provenance records

This directory separates sanitized lineage records from the public API.

- `legacy_command_inventory.json` records hashes of the 37 original launchers
  and their sanitized public counterparts. Original contents remain in the
  private research archive because they contained machine-specific paths.
- `command_path_migration.json` maps every historical launcher path to its
  organized public path.
- `source_copy_inventory.json` records source-relative hashes before and after
  publication packaging without publishing private absolute paths.
- `public_contract_lineage.json` identifies hash-bound contracts whose public
  bytes differ only because of path/label portability or explicit public-hash
  lineage; it preserves both hashes and marks scientific settings unchanged.
- `accepted_result_anchors.json` records the accepted run IDs and artifact
  hashes that can be checked against a separately obtained results archive. It
  also records archive-restoration lineage without implying that result
  payloads are bundled here. The MDP-05 and GEO-01A gaps were resolved on
  2026-08-05; MDP-05 retains both the recovered exact historical archive and a
  content-identical re-export whose outer hash differs because its ZIP entry
  timestamps changed.
- Experiment 29 anchors its accepted 36-run depth analysis and byte-complete
  result ZIP. Experiment 49 anchors its accepted six-unit analysis, W&B
  reconciliation, and lightweight result ZIP. Both are included in the compact
  `core-results` package and have passed its package-level inventory, hash,
  parsing, privacy, and path-portability validation. This establishes release
  integrity, not an independent scientific recomputation: neither accepted
  anchor record currently identifies a separate independent-review certificate,
  so any independent scientific audit remains a distinct outstanding step.
- Experiment 48 anchors its accepted 12-unit, 36-endpoint LLaMA-1B long-token
  analysis, a generic archive certificate, the persisted remote full-checkpoint
  re-hash receipt, pilot/resume lineage, an independent local recomputation, and
  W&B full-trajectory reconciliation. The 36 endpoint tensors remain external;
  their 439,092,884,892-byte certificate set is bound by the lineage certificate.
  The compact receipt is a semantic verifier certificate, not a forensic process
  execution log, and four-GPU concurrent timing is excluded from efficiency claims.

Formal result directories and their source snapshots are not rewritten by
this packaging process.
