# Selective Newton-Muon

Selective Newton-Muon is a model-agnostic optimizer research artifact for
studying where Newton-style input preconditioning is useful inside
Muon-family matrix updates. The method implementation is intentionally
separated from the training backend: NanoGPT is included as one reference
backend, while the optimizer code under `src/selective_newton_muon/` has no
dependency on a particular transformer architecture.

This repository packages the retained experiment code, sanitized launchers,
and an explicit catalog of implemented, analysis-only, and never-implemented
plans in a publication-oriented layout. Training data, model checkpoints, W&B
exports, and private machine paths are not part of the source release.

Newly authored code is released under the MIT license in [`LICENSE`](LICENSE).
Third-party attributions and preserved license texts are listed in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Anonymous-review release

This snapshot is prepared for double-blind review. The accompanying compact,
self-validating evidence is in `core-results/`; the internal `full-archive`,
raw W&B exports, checkpoints, and private Git history are intentionally absent.

During review, cite and share only the audited anonymous mirror:
`https://anonymous.4open.science/r/optimizer-research-artifact-B032/`.
Do not cite the identifiable origin repository in the manuscript.

## Repository layout

| Path | Purpose |
|---|---|
| `src/selective_newton_muon/` | Model-agnostic optimizer implementation. |
| `backends/nanogpt/` | NanoGPT-compatible reference training backend. |
| `scripts/` | Experiment controllers, workers, analyses, and tests grouped by experiment. |
| `commands/<experiment>/` | Organized, parameterized launchers. |
| `experiments/<experiment>/` | Reproduction metadata and experiment-specific instructions. |
| `reproducibility/` | Guarded rerun, resume, and artifact-verification tools. |
| `provenance/` | Sanitized migration records and hashes of private-era sources. |
| `core-results/` | Compact, anonymized, self-validating reviewer evidence. |
| `runs/` | Local result root; generated artifacts are ignored by version control. |

## Install the optimizer package

```bash
python -m pip install -e .
```

The training experiments use additional dependencies:

```bash
python -m pip install -e '.[experiments,test]'
```

Formal H100 experiments used a stricter frozen environment documented in
[`docs/ENVIRONMENT_AND_DATA.md`](docs/ENVIRONMENT_AND_DATA.md).

## Reproduction interface

List all 59 registered experiment identifiers:

```bash
python reproducibility/reproduce.py list
```

Inspect one experiment and its source hashes:

```bash
python reproducibility/reproduce.py inspect 48_llama1b_10b_multibudget
```

Build a fresh reproduction plan:

```bash
python reproducibility/reproduce.py reproduce 48_llama1b_10b_multibudget
```

Planning is read-only. Execution requires repeating the same request with
`--execute --receipt <plan_sha256>`. This prevents an inspection command from
silently starting a multi-GPU experiment.

Verify an existing result without training:

```bash
python reproducibility/reproduce.py verify \
  43_newton_muon_record28_275m \
  --results-root /path/to/experiment-results \
  --run-dir /path/to/experiment-results/43_newton_muon_record28_275m/RUN_ID
```

The common verifier checks JSON integrity and sealed source-snapshot hashes.
It does not replace an experiment's native scientific validator. The precise
distinction between `reproduce`, `resume`, `verify`, and `native-verify` is
documented in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Portable command profile

Public launchers no longer embed the original private host paths. When calling
a launcher directly, the following variables select external resources:

```bash
export SNM_RESULTS_ROOT=/path/to/experiment-results
export SNM_OFFICIAL_REPO=/path/to/Newton-Muon-official-r0
export SNM_CONTROLLER_PYTHON=/path/to/controller-python
export SNM_TRAINING_PYTHON=/path/to/frozen-training-python
```

Defaults are suitable for local inspection. GPU formal runs should always set
the exact training interpreter and data locations explicitly.
The guarded `reproduce.py` dispatcher intentionally ignores ambient `SNM_*`
variables; pass each machine-specific value as `--env KEY=VALUE` so it is
included in the signed plan receipt.

## Reproducibility scope

- 51 experiment directories contain runnable training or diagnostic code.
- 3 directories are analysis-only.
- 5 historical identifiers were planning placeholders and are explicitly
  marked `planned_not_implemented`.
- Later experiments provide sealed source snapshots and native resume/verify
  paths. Earlier experiments retain command hashes and sanitized public
  launchers, but cannot be
  retroactively upgraded to bitwise replay if their original runtime or raw
  artifacts were not preserved.
- Experiment 29 includes its accepted three-seed formal analyzer; Experiment
  49 includes the hash-bound MALT-family implementation and accepted-unit
  resume/verification controller. Both retain public-path packaging lineage,
  while their result payloads remain in the separate result packages.
- Experiments 50–52 close the global-vs-selective diagonal control across
  ModdedGPT 124M/275M/455M and LLaMA 124M/1B. Each has a portable launcher,
  native verification, accepted-result anchors, and a W&B reconciliation;
  concurrent timing remains excluded.
- Experiments 53–57 add the representation-matched module-placement control,
  a fresh-seed baseline-fairness panel, Moonlight scale/budget controls, and
  the long-budget LLaMA-1B global-diagonal test. All five expose portable
  launchers and native resume/verification modes; compact claims exclude timing.

See the generated [experiment index](docs/EXPERIMENT_INDEX.md) for the status
of every identifier.

## Core results package

The compact, reviewer-facing result payload is published separately as a
directory named `core-results`. It is a deterministic view of accepted CSV,
JSON, reports, and figure inputs; raw W&B exports, checkpoints, worker logs,
and machine-specific paths are deliberately excluded. Experiment 29 has a
16-file compact whitelist, including its input manifest. Its complete result
ZIP and nine raw W&B exports are recorded in the machine-readable omission
ledger at `provenance/omission_ledger.json` with frozen hashes and byte counts,
so their deliberate exclusion
cannot be confused with missing evidence.

Build the current package from an accepted private results tree:

```bash
python reproducibility/build_core_results_package.py \
  --selection reproducibility/core_results_selection.json \
  --source-root results=<RESULTS_ROOT> \
  --source-root source=. \
  --output ../core-results \
  --mode final
```

The output is relocatable. From inside the copied or unpacked directory, its
full integrity and portability audit is self-contained:

```bash
python tools/validate_core_results_package.py .
```

During an empty-staging build, internal JSON, CSV, TSV, and SHA-256 sidecar
references are re-signed leaf-first with separate source and packaged
certificates. The build-end audit can also require the embedded experiment
Catalog and accepted-result anchors to match the supplied public source tree.

Experiment 48 is accepted. The final package binds its 12 formal units, 36
endpoint certificates, native full-rehash receipt, generic verification,
pilot/resume lineage, and independent analysis through four distinct hard-gate
artifacts and accepted-result anchors. The final build remains fail-closed and
must be rebuilt from an empty staging directory. See
[`docs/CORE_RESULTS_PACKAGE.md`](docs/CORE_RESULTS_PACKAGE.md) for the evidence
tiers and release gate.

The final compact package also includes Experiments 50–52 and their joint
111/111 metric reconciliation. Experiment 52's reused 1B W&B history stitches
the screen and formal continuation, so accepted local formal CSVs and manifests
remain the primary scientific record. Raw W&B exports are retained only in the
restricted audit archive.

The 2026-08-31 supplement adds Experiments 53–57. Raw W&B exports for
Experiments 53 and 55 remain outside the compact package, while their compact
reconciliation manifests are included. Experiments 54, 56, and 57 were
accepted from checkpoint-free transport bundles with native verification
receipts; the package preserves their accepted analyses, unit manifests,
source-snapshot manifests, and independent review certificates. Timing is not
used for any of these claims.
