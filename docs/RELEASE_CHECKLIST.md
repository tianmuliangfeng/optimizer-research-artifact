# Public release checklist

- [ ] Confirm the paper title, author list, affiliations, and preferred
      citation; then add `CITATION.cff`.
- [x] Confirm that the project-level MIT license covers newly authored optimizer
      and experiment code. Third-party MIT notices are retained separately.
- [ ] Replace any anonymous-review placeholders after the review period.
- [ ] Publish or document access to the exact accepted result archives.
- [x] Build and independently validate the relocatable `core-results` package
      from an explicit evidence-selection contract.
- [x] Rebuild the compact package from empty staging with the Experiment 29
      whitelist and ten-entry omission ledger, then verify every
      internal JSON/CSV/TSV/sidecar certificate and public-source Catalog and
      accepted-anchor parity gate.
- [x] Accept Experiment 48, rebuild `core-results` in final mode, bind all four
      EX48 gate certificates, and rerun the complete package audit from the
      moved directory.
- [x] Accept Experiments 50–52, add their 15 result anchors and 111/111 W&B
      reconciliation, and rebuild the final compact package.
- [x] Accept Experiments 53–57, publish their sealed controllers and portable
      launchers, add compact independent-review and verification evidence, and
      rebuild the final compact package without checkpoints or raw W&B exports.
- [x] Run all CPU tests and `bash -n` over every launcher.
- [ ] Run native archive verification for every released formal result.
- [ ] Validate the external Newton-Muon revision and FineWeb inventory on a
      clean machine.
- [x] Confirm that no private hostnames, credentials, W&B API keys, usernames,
      or absolute personal paths remain.
- [ ] Tag the exact code revision used for camera-ready results.

Packaging verification on 2026-08-05: 270 unit tests passed, 3 were explicitly
skipped because accepted external evidence is not bundled, and 41 shell
launchers passed `bash -n`. Release-integrity scans found no private machine
identifiers or generated Python caches.

Archive update on 2026-08-05: the previously missing MDP-05 complete raw run
and GEO-01A engineering-pilot run were restored to the private results tree,
validated against their handoff and source-snapshot hashes, and recorded in
`provenance/accepted_result_anchors.json`. The exact historical MDP-05 ZIP was
subsequently recovered. Its 627 file entries are content-identical to the
timestamp-only re-export; an earlier apparent two-file omission was a Windows
long-path visibility issue, not an archive-content defect. Their result
payloads remain outside the public source package, so publication or an access
procedure is still a release prerequisite.

EX48 finalization on 2026-08-12: 12/12 formal units and 36 endpoints passed;
the generic archive certificate, persisted remote full-checkpoint re-hash
receipt, pilot/resume lineage, and independent local analysis were accepted.
The compact package was rebuilt from empty staging in final mode and validated.
The endpoint tensors remain external; the receipt is a semantic verifier
certificate rather than a forensic execution log, and concurrent timing is
excluded from efficiency claims.

Supplemental-control update on 2026-08-16: Experiments 50–52 passed their
native analyses; the joint W&B audit reconciled 111/111 metric rows. The public
Catalog now contains 54 identifiers, the full CPU release suite passes 62/62,
and all 47 public shell launchers pass syntax validation. Experiment 52's
stitched W&B history is supporting-only; local formal artifacts remain primary.

Post-review supplement on 2026-08-31: Experiments 53–57 passed independent
acceptance review. The public Catalog contains 59 identifiers. Experiments 53
and 55 retain compact W&B reconciliation while raw exports remain restricted;
Experiments 54, 56, and 57 have no W&B dependency. All supplemental timing is
excluded, and EX54/EX57 are not pooled.
