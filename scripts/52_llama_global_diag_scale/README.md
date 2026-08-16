# Experiment 52: LLaMA global-diagonal scale extension

This experiment derives a `global_diag` trainer from the accepted Experiment-17/20 LLaMA/SwiGLU source. Attention input, attention output, MLP expansion input, and MLP contraction input factors are all coordinate-wise diagonal. The 124M and 1.014B recipes retain the accepted 6200-update, approximately 3.25B-token budget. Experiment 48's 10B-token recipe is explicitly out of scope.

The frozen stages are:

1. `preflight`: runtime, architecture, initialization, exact parent-source lineage, and full-content SHA-256 of the accepted 50+1 data view;
2. `pilot`: 34 updates for all three seeds at both scales (engineering only);
3. `screen`: mandatory 1000-update 1B runs for all three seeds (screening only);
4. `formal`: three 124M plus three 1B quality runs;
5. `verify`: six endpoints and paired accepted controls.

Every child process sees exactly one H100. Two physical GPUs only parallelize independent seeds. The accepted controllers retain 128-step checkpoints and batch-level resume for 124M formal and 1B medium/formal. Always resume with the same `EX52_RUN_DIR`. Timing is ineligible.

The code/runtime repository is always `Newton-Muon-official-r0`. Because that r0 tree may also contain the additional shards downloaded for Experiment 48, Experiment 52 accepts an explicit `EX52_DATA_DIR`. The command wrapper automatically runs `prepare_frozen50_view.py` before `preflight`/`all` to build or verify a non-copying, read-only selection view of shards 1--50 plus the validation shard inside r0. Suite preflight then hashes all 10.2 GB and requires the accepted full-content fingerprint `1202c308...a8c68`; matching names and sizes alone are not sufficient.

Contract version `2026-08-14.4` also pins the accepted Experiment-17/20 parent source hashes, the three per-scale initialization fingerprints, the frozen-control CSV hash, and the exact LLaMA recipe. The parent controllers include the absolute data directory in their lightweight inventory fingerprint, so EX52 derives that path-bound fingerprint from the full-content certificate for the active frozen view; the historical `57d23f...ceb8a2` value remains provenance rather than an invalid cross-path gate. Pilot certificates validate the parent controllers' intentional short-run settings (`val_every=34`, reduced validation tokens, and no checkpoint), whereas screen/formal certificates require their production settings. Every pilot, screen, and formal unit receives a content-bound `ex52_unit_manifest.json`; later stages require passed manifests rather than file existence. A resume validates all artifact hashes before skipping a unit.

The runs `20260814T071735+0000` and `20260814T080903+0000` contain only engineering preflights made with retired `.1` and `.2` snapshots. Do not resume them. A `.3` run that reached the pilot-certificate gate may be resumed in place: the wrapper freezes a `.4` certificate-only amendment, certifies already completed raw pilot units without retraining, and leaves its frozen training source and scientific contract unchanged. Fresh runs use `.4` throughout.
