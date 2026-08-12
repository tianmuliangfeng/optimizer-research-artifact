# LLaMA-1B isolated efficiency audit

Integrity status: **PASS**.

This status certifies evidence integrity only. It is not an optimizer-quality or algorithm-superiority decision; experiment 20 remains the frozen quality source.

## Frozen design

Four methods were run for 4 rotated repeats on physical GPU 0 while both node GPUs were certified idle before and after every timed cell; a sealed process monitor also audited GPU 1 throughout each cell. The first 32 updates were excluded and 512 updates were timed.

## Method-level measurements

| Method | median ms/update | mean tokens/s | throughput CV | peak allocated MiB | peak reserved MiB | K-state MiB |
|---|---:|---:|---:|---:|---:|---:|
| muon | 7,320.651 | 71,620 | 0.11% | 26,906.2 | 27,930.0 | 0.0 |
| newton_full | 7,731.668 | 67,792 | 0.06% | 35,982.3 | 36,956.0 | 5,888.2 |
| down_none | 7,432.821 | 70,535 | 0.05% | 29,559.8 | 30,544.0 | 1,728.0 |
| down_diag | 7,435.942 | 70,507 | 0.05% | 29,560.9 | 30,546.0 | 1,728.8 |

## Frozen primary contrasts

The table reports paired candidate-minus-reference changes across the four rotations. The ±1% band is descriptive only.

| Candidate | Reference | mean paired tokens/s change | 95% t CI | descriptive classification |
|---|---|---:|---:|---|
| down_none | muon | -1.52% | [-1.66%, -1.38%] | candidate_lower_tokens_per_s_outside_descriptive_band |
| down_none | newton_full | 4.05% | [4.00%, 4.09%] | candidate_higher_tokens_per_s_outside_descriptive_band |
| down_diag | muon | -1.55% | [-1.78%, -1.32%] | candidate_lower_tokens_per_s_outside_descriptive_band |
| down_diag | newton_full | 4.01% | [3.92%, 4.09%] | candidate_higher_tokens_per_s_outside_descriptive_band |
| newton_full | muon | -5.35% | [-5.51%, -5.18%] | candidate_lower_tokens_per_s_outside_descriptive_band |

Selective-diag versus Selective-none is deliberately not a primary contrast and is not added to this table.

## Rotation-position diagnostic

| Position | mean tokens/s | position versus overall |
|---:|---:|---:|
| 0 | 70,127 | 0.02% |
| 1 | 70,091 | -0.03% |
| 2 | 70,102 | -0.02% |
| 3 | 70,135 | 0.03% |

## Evidence fingerprints

- Contract SHA256: `6658273d80051ef713b8ea0fa8fd40950823c59effb5cce8bae87e9caa65e646`
- Preflight SHA256: `d596faf10d655f35e79a3b402d689017ab25d52a15c70ce03becb7a8f4587b2a`
- Execution manifest SHA256: `fe0d85dbdb87782d9062f2cd88e1baba61637dbbb48459a917c452f8e29f483c`
- Initialization SHA256: `521d734608ff08ab6a191c4c1a41412c40bf53985ee3cbcefc3ab15457d39fec`
- Derived trainer SHA256: `a21f364b73ea859e1ca62bc75600de348b00a0ce5fcbd315ed7e848eb2b4666a`
- Profile wrapper SHA256: `043c758f3d5eb5d1abc9e1f9029a8d085a238cf169ef69ba86580014699dc401`
- Runtime fingerprint: `2951a374071707c9de19b1dd8e8406579538e75cc4d12960da0db928fae6cefc`
- Data fingerprint: `097478adc4e938305a50d8fa888a9b11508523dc41c403db339f0ba5e9424d0d`

## Integrity checks

- contract_identity: `PASS`
- contract_structure: `PASS`
- frozen_method_and_primary_contrast_priority: `PASS`
- rotation_exact: `PASS`
- formal_cell_inventory: `PASS`
- root_evidence_present: `PASS`
- run_contract_matches_analysis_contract: `PASS`
- preflight_readable: `PASS`
- preflight_contract_runtime: `PASS`
- preflight_official_source: `PASS`
- preflight_data_fingerprint: `PASS`
- preflight_initialization_fingerprint: `PASS`
- preflight_gpu_inventory: `PASS`
- preflight_source_snapshot_hashes: `PASS`
- preflight_exclusive_certificate_readable: `PASS`
- preflight_exclusive_certificate: `PASS`
- execution_manifest_present: `PASS`
- execution_manifest_readable: `PASS`
- execution_smoke_manifest_hash: `PASS`
- execution_postflight_data_hash: `PASS`
- execution_postflight_data_fingerprint: `PASS`
- execution_manifest_and_cell_hash_inventory: `PASS`
- cell_manifests_readable: `PASS`
- cell_coordinates_and_contract: `PASS`
- cell_evidence_locality: `PASS`
- cell_evidence_hashes: `PASS`
- worker_manifests_readable: `PASS`
- trainer_summaries_readable: `PASS`
- exclusive_certificates_readable: `PASS`
- gpu_isolation_monitors_readable: `PASS`
- worker_sealed_artifact_hashes: `PASS`
- worker_manifest_integrity: `PASS`
- frozen_measurement_counts_and_no_resume: `PASS`
- frozen_trainer_configuration: `PASS`
- frozen_architecture: `PASS`
- summary_runtime_fingerprint: `PASS`
- worker_observed_matches_summary: `PASS`
- worker_metric_sequence_audit: `PASS`
- finite_timing_and_memory: `PASS`
- exact_k_state_bytes: `PASS`
- exact_model_and_optimizer_state_bytes: `PASS`
- initialization_hash_format: `PASS`
- source_hash_pins: `PASS`
- exclusive_certificates: `PASS`
- continuous_gpu_isolation_monitors: `PASS`
- gpu_identity_stable_before_during_after: `PASS`
- worker_derived_throughput_matches: `PASS`
- formal_rows_complete: `PASS`
- rotation_observed_exact: `PASS`
- common_preflight_fingerprint: `PASS`
- common_runtime_fingerprint: `PASS`
- common_data_fingerprint: `PASS`
- common_initialization_fingerprint: `PASS`
- common_derived_source_fingerprint: `PASS`
- common_wrapper_source_fingerprint: `PASS`
- preflight_hash_format: `PASS`
- initialization_hash_common_and_valid: `PASS`
- all_integrity_checks_passed: `PASS`
