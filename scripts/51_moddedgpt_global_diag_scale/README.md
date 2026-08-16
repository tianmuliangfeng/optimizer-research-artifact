# Experiment 51: ModdedGPT global-diagonal scale extension

This experiment adds one frozen `global_diag` method to the accepted Experiment-43 (approximately 275M) and Experiment-44 (approximately 455M) recipes. It matches each accepted parent grid: four formal seeds (2024--2027) at 275M and three formal seeds (2024--2026) at 455M. No learning rate, token budget, validation schedule, or optimizer hyperparameter is retuned.

The pilot is an engineering gate only: seed 2026 at each scale. Formal quality evidence consists of exactly seven units. Each unit is single-GPU; the suite accepts either one fixed GPU or two fixed GPUs and runs at most one unit per selected GPU. The GPU allocation is sealed by preflight and cannot be expanded or changed inside the same run directory. Timing is ineligible.

Stages are `preflight`, `pilot`, `formal`, optional `upload`, and `verify`. Reuse the printed `EX51_RUN_DIR` for every later stage. Accepted units are skipped; a failed unit is retained as `attempt_NNN` and restarted from initialization because the inherited 43/44 cell format has no mid-step checkpoint resume.

Both the code provenance and the data source are rooted at `Newton-Muon-official-r0`. Because the shared r0 data link can contain the later Experiment-48 shards, the launcher first creates/verifies `Newton-Muon-official-r0/ex51_frozen50_data_repo/data/fineweb10B`: a non-copying view containing only shards 1--50 and the validation shard. Preflight then hashes all 10.2 GB and requires the exact Experiment-43/44 fingerprint `1202c308...8c68`; a merely well-formed but different dataset is rejected.

Pilot contains only seed 2026 at both scales and is engineering-only. Formal contains four 275M seeds and three 455M seeds, matching the corresponding parent experiments. Accepted units are skipped on rerun, while failed attempts remain preserved. A failed unit restarts from the beginning; there is no unsupported mid-step checkpoint claim. Use `EX51_GPUS="0"` for a fixed single-GPU run or `EX51_GPUS="0 1"` for a fixed two-GPU run before preflight; do not change that value between stages. Timing is ineligible because host concurrency is not a scientific endpoint.

The optional `upload` stage operates only on accepted formal attempts. It skips
completed upload receipts and retries incomplete uploads without retraining. A W&B
failure never invalidates or deletes the sealed local scientific result. `verify`
does not require W&B because the local CSV/JSON artifacts are the primary evidence.
