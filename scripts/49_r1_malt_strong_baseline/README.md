# Experiment 49: controlled 124M R1 MALT-family baselines

Experiment 49 evaluates MALT and MALTER-Eq17 in the frozen controlled-124M R1
environment. They are **paper-derived independent implementations**, not
unchanged official reproductions. At the 2026-08-07 freeze, the MALT v1 paper
did not publish usable author code and did not disclose its Newton--Schulz
implementation. MALTER also
contains the documented `v`/`nu` and duplicated-`eta` ambiguity; this repository
uses Equation (17), exactly one outer learning-rate factor, and names that arm
`malter_eq17`.

The immutable scientific and provenance rules are in `malt_contract.json`; the
derivation choices are summarized in `PAPER_DERIVATION.md`.

## Evidence roles

- `malt`: focused six-point v4 LR grid `0.0160, 0.0125, 0.0100, 0.0090,
  0.0080, 0.0064`, run in that descending order. The grid retains the v3
  boundary winner `0.0080` and its immediate lower neighbor `0.0064`, then
  searches upward. It is eligible for formal training only after the
  independent pilot analyzer issues a positive selection certificate.
- `malter_eq17`: six-point LR grid for the explicitly documented Equation-(17)
  adaptation. It receives its own independent formal-recipe selection, but it
  remains an independent controlled adaptation and cannot support a claim of
  reproducing unpublished official MALTER code.
- Existing R1 Muon/Newton--Muon rows remain the frozen controls. Absolute loss
  values from the OpenWebText MALT paper are not cross-paper baselines.

## Required sequence

Run every phase from the repository root and preserve the printed run directory.
The command wrapper resolves this repository structurally. External code, data,
runtime, and result locations are supplied through the documented `SNM_*`
environment variables; do not move accepted files between phases.

On a newly allocated host, Git may reject the shared official checkout because
its owner differs from the container user. If preflight reports `dubious
ownership`, authorize only the exact frozen checkout before retrying:

```bash
git config --global --add safe.directory \
  "${SNM_OFFICIAL_REPO}"
```

1. **Preflight.** Audit the pinned R1 repository, exact training runtime
   (`Python 3.10.12`, `torch 2.8.0+cu126`, CUDA `12.6`, Triton `3.4.0`, NumPy
   `2.2.6`), parameter routing, independent source derivation, and small-matrix
   numerical invariants. The suite full-hashes exactly train shards
   `000001--000050` plus validation shard `000000`; later shards added for
   Experiment 48 are ignored by the generated loader. No scientific run is
   allowed after a failed preflight or data-content audit.
2. **Pilot.** Run all twelve seed-2026, 1000-step cells: six MALT LRs and six
   MALTER-Eq17 LRs. The MALT cells are queued in descending order
   `0.0160, 0.0125, 0.0100, 0.0090, 0.0080, 0.0064`, followed by the unchanged
   MALTER-Eq17 grid `0.007, 0.009, 0.012, 0.015, 0.018, 0.025`. On two GPUs the
   first two MALT cells run concurrently. Partial grids cannot select a formal
   recipe. A W&B upload
   failure may leave `completed_valid_local_wandb_incomplete`; locally accepted
   evidence remains analyzable and upload can be retried without retraining.
3. **Independent pilot analysis.** The suite runs and seals this automatically.
   For a standalone audit, point the analyzer at either the batch directory or
   its `pilot_manifest.json`:

   ```bash
   python scripts/49_r1_malt_strong_baseline/analyze_malt_pilot.py \
     /absolute/path/to/pilot_batch \
     --output-dir /absolute/path/to/pilot_batch/pilot_analysis
   ```

   The analyzer requires the sibling `pilot_summary.csv`, verifies the exact
   method/LR grid, seed, budget, endpoint, and `evidence_valid` flags, then
   cross-checks any runner-generated `pilot_selection.json`.

   MALT and MALTER-Eq17 are ranked separately by step-1000 validation loss.
   The old MALT paper center `0.0013` is not in the focused v4 grid and receives
   no synthetic center preference: MALT selects the raw endpoint-loss minimum
   (an exact internal tie is broken by lower LR, then cell ID), with boundaries
   `0.0064/0.0160`. MALTER-Eq17 retains paper center `0.012`,
   boundaries `0.007/0.025`, and the rule that the center is preferred when it
   lies within `best + 0.002`; otherwise its interior raw best is selected. If
   either method's reported minimum includes one of its two boundaries,
   including an exact tie at the four-decimal loss precision emitted by the
   frozen trainer, that method is `boundary_inconclusive` and the complete
   formal stage is blocked fail-closed. Any later grid amendment must be
   frozen separately.
4. **Formal.** Continue only when
   `pilot_analysis/pilot_selection_verified.json` has `status=selected` and
   `formal_allowed=true`, has certificate role
   `independent_pilot_analysis_selection`, and contains accepted selections for
   both required methods. Use the selected MALT and MALTER-Eq17 LRs, complete
   six method-by-seed exact-shape smokes, then run six 6200-step formal units
   (both methods at seeds 2024/2025/2026). Do not retune other optimizer
   fields.
5. **Verify.** Validate local manifests, checkpoints, optimizer-state bytes,
   source/runtime lineage, all six method-by-seed formal units, and W&B upload
   state. Upload repair must reuse accepted local artifacts rather than rerun
   training.
6. **Analysis.** Run the formal analyzer only on verified formal artifacts and
   the accepted historical R1 controls. The sealed output is a ten-method,
   30-run panel with method-family and control contrasts. Concurrent
   quality-run wall clock is diagnostic; any efficiency claim requires the
   separate isolated R1 timing contract.

## Pilot analyzer outputs

`analyze_malt_pilot.py` creates a new output directory containing:

- `pilot_ranking.csv`: a six-cell MALT ranking and a separate six-cell
  MALTER-Eq17 ranking;
- `pilot_selection_verified.json`: the only analyzer-issued dual-method
  selection certificate (or the formal-blocking `boundary_inconclusive`
  result); runner preselection is cross-check evidence and cannot self-authorize
  formal training;
- `pilot_analysis_manifest.json`: input hashes, integrity gates, provenance,
  selection outcome, and output hashes;
- `pilot_analysis_manifest.sha256`: SHA-256 sidecar for the analysis manifest.

These files describe a controlled R1 adaptation. They do not certify equivalence
to unpublished author code.

## V4 focused post-boundary amendment

The completed v3 pilot `20260807T104741+0000` placed the MALT raw optimum at
its `0.0080` upper boundary, while MALTER-Eq17 selected an interior point. That
outcome legitimately blocked v3 formal training. V4 therefore keeps MALT
`0.0064/0.0080` as the local overlap pair, removes already uninformative lower
MALT rates, and freezes four upward points `0.0090/0.0100/0.0125/0.0160`. The
MALTER-Eq17 six-point grid is rerun unchanged so both methods still share one
fresh pilot protocol, initialization audit, runtime, seed, budget, and data
lineage. V4 does not import or merge numerical artifacts from v1, v2, or v3.

Because the scripts, contract, protocols, and source hashes changed, do not
resume any old v1/v2/v3 run directory. Create one new `EX49_RUN_DIR` and use it
for all v4 stages. A non-finite cell remains a hard pilot failure; it is not
silently converted to an infinite loss for selection.

## Resume boundary

Resume is exact at the **accepted-unit and upload** level, not at an arbitrary
training step. A locally accepted pilot/formal unit is never retrained, and a
failed W&B upload is retried from the saved local evidence under a fresh upload
ID. An interrupted, non-accepted 1000- or 6200-step training unit starts a new
attempt from its frozen initialization; Experiment 49 does not claim mid-run
checkpoint continuation. Controller SIGTERM/SIGHUP/KeyboardInterrupt cleanup
terminates the full runner/trainer process group before releasing GPU locks.
