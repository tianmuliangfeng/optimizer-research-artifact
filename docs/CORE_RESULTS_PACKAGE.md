# Core results package

`core-results` is the compact, movable evidence package for the paper. It is not the source-code repository and it is not the full raw experiment archive. Its job is to make the accepted statistical tables, claim boundaries, figures, and audit anchors easy to inspect without copying checkpoints, optimizer states, raw tracking exports, or failed engineering runs.

The directory name is always exactly `core-results`. A draft-mode build may be incomplete while Experiment 48 is running, but draft mode never changes that directory name and must not create an empty Experiment 48 directory.

## Three publication layers

1. The public source repository contains implementations, contracts, reproduction commands, the experiment catalog, and accepted-result hash anchors.
2. `core-results` contains compact accepted results, supporting controls, claim-boundary documents, and a provenance index.
3. The full archive contains raw formal and pilot directories, logs, source snapshots, and other audit material that is too large or too operational for the compact package.

The compact package does not replace the full archive. A result can be scientifically valid while only its compact analysis is included here.

Several evidence boundaries are intentionally stricter than a simple accepted/not-accepted label:

- Experiment 17 did not pre-freeze an equivalence margin, so it is descriptive architecture evidence rather than a formal equivalence claim.
- The Experiment 20 compact payload contains no checkpoints and therefore supports the accepted quality and capacity summaries, not checkpoint replay.
- Experiment 22 excludes checkpoint transfer; that exclusion does not block the accepted alpha analysis.
- For Experiment 39, the independently recomputed numerical tables are authoritative. The older narrative report is not included.
- Experiments 41, 43, 44, and 45 are used only for their eligible quality, state, or module-structure results where applicable; their timing fields are excluded.
- Experiment 45 contains no checkpoint tensors. It does not support checkpoint replay, and its pilot and formal scopes remain separate.
- Experiment 29 supports broad transfer of the `diag`-versus-`none` direction, not a universal edge mask, single-layer causal claim, or transferred effect-size ordering; its concurrent timing is excluded.
- Experiment 49 contains paper-derived independent R1 adaptations rather than official MALT/MALTER code reproductions. MALT is eligible for the main quality/state panel, MALTER-Eq17 only for appendix limitations, and all concurrent timing is excluded.
- MDP-04 completed its computation but failed the numerical gate; MDP-05 is a partial-or-null confirmatory result; and Experiment 47B is a discovery-stage negative result rather than confirmatory evidence.

## Selection model

The machine-readable specification is [`../reproducibility/core_results_selection.json`](../reproducibility/core_results_selection.json). Every selected record keeps four independent labels:

The workstream and release-gate relationships are summarized in the editable [draw.io evidence map](core_results_evidence_map.drawio) and its [SVG rendering](core_results_evidence_map.svg).

- `integrity_status`: whether the retained evidence passed its engineering and audit gates;
- `scientific_status`: what the experiment actually found, including null or partial findings;
- `claim_eligibility`: the exact level at which it can support a paper claim;
- `paper_role`: where the evidence belongs in the paper narrative.

These labels must not be collapsed into a single `passed` field. For example, MDP-05 passed its formal integrity checks but its scientific result is partial or null; MDP-04 completed computation but failed its numerical hard gate and therefore remains descriptive support rather than formal paper evidence.

The file policy is explicit-whitelist only. Directory recursion and glob expansion are forbidden except for the one sealed CSV dependency graph described below. This keeps raw trajectories and operational artifacts from leaking into the compact release. Deliberate exclusions that still require provenance are recorded separately in the generated `provenance/omission_ledger.json`; an omitted file is never inferred merely because it is absent.

## Included workstreams

### Core evidence

- Discovery and regime map: the merged 60-run analysis built from Experiments 01, 02, 04, and 12.
- Main quality: Experiment 15 at 124M, Experiment 17 on LLaMA 124M, Experiment 20 on LLaMA 1B, Experiments 43 and 44 at 275M and 455M, Experiment 45 with Mousse and the unified eight-method panel, and Experiment 49 with the independent MALT-family adaptations and unified ten-method panel.
- Capacity and systems: the exact LLaMA 1B capacity boundary from Experiment 20, isolated R1 efficiency and sensitivity from Experiment 39, and isolated LLaMA 1B efficiency from Experiment 42.
- Robustness and structure: Experiments 22 and 24 for the two alpha families, Experiment 29 for three-seed cross-environment depth-direction transfer, Experiment 40 for LLaMA block-partition invariance, and Experiment 41 plus its diagonal bridge for module structure. The Experiment 29 compact whitelist contains exactly 16 files, including `analysis_20260809_formal/input_manifest.csv` as its input-provenance anchor.
- Mechanism: Experiment 34, the accepted refresh-mediation result from Experiment 37, the unified synthesis from Experiment 38, method-deepening v2, and the final mechanism-closure package.
- Submission synthesis: the final unified Experiment 38-45 analysis, including cross-scale tables, Pareto summaries, evidence and source ledgers, figures, claim matrix, and limitations matrix.

### Supporting evidence

- Experiment 14 anchors the official R0 reproduction.
- Experiment 16 supplies the crossed learning-rate control.
- Experiment 19 preserves provenance for the extended baseline panel later superseded by Experiment 45.
- Experiment 21 is the host-context bridge.
- The Experiment 38 source-audit graph freezes the compact inputs from Experiments 01, 04, 06, 27, 30, 31, 33, 35, and 36 that are not otherwise copied directly.
- MDP-04 is retained as diagnostic negative evidence with its failed numerical hard gate visible.
- MDP-05 is retained as the accepted confirmatory boundary result; only its supported loss-shock claim is eligible.
- Experiment 47B is retained as a discovery-stage negative mechanism result and an input to mechanism closure.

The MDP-05 ZIP receipt remains in the full archive because its historical run-directory field uses archive-relative traversal that is not valid inside a freely movable compact package. The released accepted-result-anchor snapshot still preserves the archive hash binding; omitting the receipt from `core-results` does not change the accepted MDP-05 numerical evidence.

The dependency graph is non-recursive. The builder reads only the `relative_path` and `sha256` columns of the hash-anchored Experiment 38 `source_audit.csv`, verifies each listed input, and copies those files to `supporting/mechanism_source_inputs`. To remain portable on filesystems with conservative path limits, graph destinations use collision-resistant shortened names derived from the experiment prefix, source basename, and relative-path hash. A file referenced by one of those inputs does not trigger another traversal.

## Deliberately not in the compact package

Experiments 03, 08, 13, 18, 23, 25, 26, 28, 32, and 47 are archive-only historical, superseded, intermediate, or engineering records. Experiments 05, 07, 09, 10, and 11 are planned or placeholder records and are excluded.

Raw tracking exports, checkpoints, optimizer states, training logs, caches, failed runs, smoke-only runs, dry runs, and unlisted source snapshots are excluded from `core-results`. The compact package also omits large history or layer-level trajectory tables when an accepted aggregate, paired table, audit manifest, or frozen source graph is sufficient. This is a compactness decision, not a deletion policy for the full archive.

Experiment 29 has ten explicit, machine-checked omissions: one complete result ZIP and nine raw W&B CSV exports. For every omitted source, the selection freezes its SHA-256 and byte count, and the builder requires exactly one matching row in Experiment 29's packaged `input_manifest.csv`. The generated omission ledger records that each source is hash-anchored and intentionally not bundled. These ten entries are therefore deliberate compact-package boundaries, not missing files. The complete ZIP and raw exports remain part of the full archive layer.

## Portable paths

The package must continue to work after being moved or renamed by its parent directory. Therefore:

- source paths in the selection are relative to a caller-supplied results root or source root;
- released paths are relative to the `core-results` directory;
- manifests and documentation in the released package may not contain drive-qualified, UNC, user-home, cluster-absolute, or `file:` paths;
- links between files use package-relative paths;
- the builder records both the original source SHA-256 and the released SHA-256 when private path relocation changes a text file.

Internal references in JSON, CSV, TSV, and standard SHA-256 sidecars are finalized from dependency leaves toward their owners. When sanitization or reference normalization changes a packaged child, the builder re-signs the package-relative path, SHA-256, and byte count in the owner while retaining the distinct source certificate. This leaf-first process continues until the dependency graph is stable; cycles, unresolved targets, stale sidecars, and source/package certificate conflation are rejected.

Path relocation may alter path-valued strings only. It must not alter measurements, statistical values, seeds, method labels, acceptance decisions, or claim-boundary fields. Binary research artifacts are copied byte-for-byte or rejected; they are never rewritten in place.

## Build and validation interface

Run the builder from the public source repository. All paths below are parameters or relative paths; no machine-specific location is embedded in the package:

```text
python reproducibility/build_core_results_package.py \
  --selection reproducibility/core_results_selection.json \
  --source-root results=<RESULTS_ROOT> \
  --source-root source=. \
  --output ../core-results \
  --mode final
```

Validation is independent of the package's parent directory:

```text
cd ../core-results
python tools/validate_core_results_package.py .
```

The builder starts from an empty staging directory, verifies each source anchor, resolves only explicitly selected files, relocates private path-valued text, finalizes internal references leaf-first, writes source and released hashes, and atomically installs the completed directory. At the end of the build it can compare the packaged experiment Catalog and accepted-result anchors byte-for-byte with the supplied public source tree. This public-source parity gate prevents a package from being accepted against an older embedded Catalog after reproduction metadata has changed. The validator checks at least:

- exact file inventory and absence of unlisted files;
- source-anchor and released-file SHA-256 values;
- JSON parsing and CSV/TSV header and row integrity;
- internal JSON, CSV, TSV, and sidecar path/hash/byte certificates;
- the omission ledger, its frozen source hashes and sizes, and its unique packaged input-manifest anchors;
- optional build-end parity with the public experiment Catalog and accepted-result-anchor snapshot;
- duplicate or escaping paths, symlinks, and unexpectedly large files;
- drive-qualified, UNC, home-directory, cluster-absolute, and file-URI leakage;
- separation of integrity, scientific result, claim eligibility, and paper role;
- the release-mode rules below.

## Experiment 48 final gate

Experiment 48 is accepted and the selection is final. Its four distinct hard-gate
artifacts are the formal analysis manifest, the portable generic archive-verification
certificate, the persisted remote full-checkpoint native-verification receipt, and
the pilot/source/data/resume-lineage manifest. The accepted-result snapshot binds
all four paths and hashes, while the independent audit and lineage certificate bind
the native receipt to the exact run and all 36 retained endpoint certificates.

The complete package contains 494 selected artifacts plus generated provenance and
validation files. It was rebuilt from an empty staging directory; appending an EX48
folder to the former draft was not used. Ten full-archive-only inputs remain recorded
in the omission ledger. The 36 endpoint checkpoint tensors are intentionally absent
from this compact package; their certified total is 439,092,884,892 bytes.

The native receipt is a semantic certificate tied to hash-frozen verifier code. It
does not itself record a forensic command line, process exit code, host identity, or
Python environment. Four-GPU concurrent timing is permanently excluded from
efficiency claims.

## Interpreting the package

The package supports a bounded paper story: K-state allocation is environment-dependent; the preferred quality/state trade-off changes across model size and architecture; the `diag` direction transfers across the three depth environments while its amplitude does not; Mousse and MALT are useful external neighbors at 124M but do not dominate the selective variants; and refresh interventions have a reproducible short-horizon loss effect while the stronger proposed geometry mediation remains unsupported. Null and partial mechanism findings are retained because they define the claim boundary rather than weakening the integrity of the accepted positive results.

The claim matrix and limitations matrix in `results/06_unified_submission/final_unified_38_45` and the closure documents in `results/04_mechanism/mechanism_closure` are authoritative for wording. No individual CSV should be used to widen a claim beyond those boundaries.
