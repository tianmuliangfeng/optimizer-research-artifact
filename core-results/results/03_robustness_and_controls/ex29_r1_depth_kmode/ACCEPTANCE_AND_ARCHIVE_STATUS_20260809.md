# Experiment 29 acceptance and archive status

Acceptance date: 2026-08-09  
Accepted formal batch: `20260806T100702+0000`  
Scientific status: accepted three-seed formal evidence  
Timing status: permanently ineligible for efficiency claims

## Authoritative retained inputs

- The byte-complete remote bundle is retained at
  `source_bundle/29_r1_depth_kmode_20260809.zip`.
- ZIP SHA-256:
  `f44ec4c7ad50c9d8f4fd94be59767146f3e6fa9871e9ec109167bbf5d3056b20`.
- The archive contains 1,424 files; Python `ZipFile.testzip()` returned no bad member.
- The nine W&B CSV exports are retained under
  `analysis_20260809_formal/raw_wandb_exports/`, with per-file SHA-256 values in
  `analysis_20260809_formal/input_manifest.csv`.

The local `batches/` and `results/` trees are a convenience extraction. Some remote
workspace log paths exceed the ordinary Windows path limit, so that extraction must
not be treated as a byte-complete mirror of the remote directory. Use the retained ZIP
for full audit or re-extraction. All files required for the accepted formal analysis
are present in the convenience extraction.

## Acceptance evidence

The frozen analysis is in `analysis_20260809_formal/`. It establishes:

- six accepted smoke manifests and six accepted formal manifests;
- 36/36 `completed_valid` formal runs and 15 seed-matched depth contrasts;
- exact step-0:100:6200 validation grids;
- exact agreement between all nine W&B metric exports and the local metrics at every
  exported point (324 run-by-metric checks, maximum absolute error
  `1.1368683772161603e-13`);
- identical rerun hashes for the 11 core numerical/report artifacts when the analyzer
  was run a second time against this canonical local directory.

The primary report is
`analysis_20260809_formal/R1_DEPTH_KMODE_FORMAL_ANALYSIS_20260809.md`; the machine-
readable verdict is `analysis_20260809_formal/analysis_verdict.json`.

## Claim boundary

In official R1, all 15 step-6200 `diag - none` pairs favor `diag`, as do all 15
tail-5 and all 15 normalized-AUC contrasts. The local OWT/WikiText ordering in which
`edge` showed the largest effect does not transfer: R1 places `all` first and `edge`
below `all`. The accepted claim is broad transfer of the benefit direction with
environment-dependent depth amplitude, not a universal edge mask or layer-level
causal law.
