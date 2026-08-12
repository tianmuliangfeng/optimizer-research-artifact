# Local method-deepening and submission-analysis pipeline

This directory is the read-only post-training layer for the accepted project
evidence. It does not train models, alter accepted experiment artifacts, or
update `HANDOFF.md`. Outputs are new immutable analysis bundles with input
hashes.

## Main stages

1. `build_evidence_ledger.py` adapts the accepted 124M R1, Record #28/275M,
   Record #17/455M, and experiment-45 tables to one run-level schema.
2. `analyze_cross_scale_pareto.py` computes paired contrasts separately within
   each GPT scale. Seeds are never pooled across model sizes.
3. `build_submission_tables_figures.py` emits Markdown tables and deterministic
   SVG forest/Pareto figures.
4. `audit_method_formulation.py` performs independent numerical reference
   checks for dimensions, ridge, resolvent, matched-G, and alpha derivatives.
5. `audit_routing_complexity.py` checks full/block/diag/none formulas, required
   measurement coverage, derivations, source hashes, and config provenance.
6. `audit_route_equivariance.py` tests full, none, diag, and block transformation
   groups. It labels its result as a numerical reference, not a formal proof.
7. `inventory_method_deepening_artifacts.py` inventories local tensors,
   experiment-37 audit payloads, ZIP members, and checkpoint availability.
8. `analyze_refresh_stability.py` v2 separates exact resolvent checks from
   production/runtime inverse fidelity and enforces a formal snapshot contract.
9. `build_method_deepening_package.py` produces the method status, claim and
   negative-evidence matrices, source hashes, and audited figures.
10. `audit_submission_bundle.py` re-hashes committed outputs and enforces the
    synthetic-data claim gate.

`run_local_pipeline.py` executes the cross-scale submission stages. The
method-deepening v2 components are also independently runnable so an absent
refresh export cannot be mistaken for a completed MDP-04 result.

## Frozen statistical rules

- Pairing unit for training comparisons: seed, within one environment.
- Primary sign: candidate minus reference; negative favors the candidate.
- Practical final-loss margin: ±0.002.
- Core GPT panel: Muon, original/block4, diag, none.
- Different scales are replication environments, not IID seed extensions.
- Refresh origins and replicas are nested replay units; four origins times
  three replicas is not reported as 12 training seeds.
- Synthetic fixtures are always invalid for scientific claims.
- Mousse is an external 124M comparison family and is not required at 275M or
  455M by the accepted evidence.

## Refresh formal contract v2

The diagnostic minimum is an NPZ with:

- `k_before`, `k_after` as `[N,d,d]`;
- `gradient_before`, `gradient_after` as `[N,m,d]`.

Paper-eligible formal analysis additionally requires:

- `matched_gradient`;
- production `inverse_before`, `inverse_after`;
- explicit `ridge_before`, `ridge_after`;
- `loss_impulse_step48`, `loss_impulse_step80`, `loss_impulse_auc`;
- metadata with `unit_id,origin,replica,stage,module_id,layer_index`,
  `refresh_event_step,source_method,checkpoint_step,gradient_semantics`;
- a verified `mdp_refresh_snapshot_manifest_v2` containing input hashes,
  source hashes, runtime-contract hash, and fingerprint validation.

The MECH-09R reconstruction default is `ridge_scale=0.2` plus `1e-8`, but a
reconstructed ridge or inverse remains diagnostic-only. Exact float64 inverses
verify the algebra; exported production inverses pass a separate fidelity gate.
The SVD polar metric is explicitly a polar factor, not automatically the full
Muon optimizer update.

Local inventory currently finds no real experiment-37 paired tensor export.
MDP-04 therefore remains `blocked_data`; synthetic fixtures cannot fill it.

## Runtime and commands

Use the repository environment, which already contains NumPy and Matplotlib:

```powershell
$PY = '${SNM_WORKSPACE_ROOT}\venv\muonTest\Scripts\python.exe'
& $PY run_local_pipeline.py --config <filled_pipeline.json> --output-root ${SNM_WORKSPACE_ROOT}\tmp\mdp_analysis_dryrun --dry-run
& $PY run_local_pipeline.py --config <filled_pipeline.json> --output-root <new_output_directory>
& $PY audit_method_formulation.py --output-dir <new_formulation_directory>
& $PY audit_routing_complexity.py --config <complexity.json> --output-dir <new_complexity_directory>
& $PY audit_route_equivariance.py --output-dir <new_equivariance_directory>
& $PY inventory_method_deepening_artifacts.py --workspace-root ${SNM_WORKSPACE_ROOT} --output-dir <new_inventory_directory>
& $PY -m unittest discover -s . -p 'test_*.py' -v
```

Dry-run validates inputs and numerical stages without writing to its requested
output root. A full run refuses to overwrite a committed stage manifest.

## Output contract

Each stage writes CSV/Markdown/SVG artifacts followed by a
`*_manifest.json`. Manifests contain SHA-256 hashes, synthetic flags, and
claim-eligibility boundaries. The hardened method package is committed at:

```text
${SNM_RESULTS_ROOT}/_shared/analysis/method_deepening_20260803_v2/
```

Its package status is intentionally `partial`: MDP-01--03 are ready, while
MDP-04 needs the frozen deterministic original-host replay/export.
