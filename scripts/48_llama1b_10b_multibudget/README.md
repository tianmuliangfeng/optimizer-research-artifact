# Experiment 48: LLaMA-1B three-seed multi-budget long-token confirmation

Experiment 48 is the formal LLaMA/SwiGLU 1.014B long-token experiment.  It
tests all four accepted optimizer routes (`down_none`, `down_diag`,
`newton_full`, `muon`) at seeds 2024, 2025, and 2026.

The scientific endpoints are 3,250,585,600, 6,969,360,384, and
9,999,745,024 training tokens (steps 6200, 13293, and 19073).  Their exact
tokens/parameter values are 3.206685, 6.875236, and 9.864694.

## Why the run is segmented

The accepted step-6200 recipe ends after an 1800-update cooldown.  Continuing
that zero-endpoint schedule is forbidden.  Experiment 48 starts from scratch
and creates plateau checkpoints at steps 4400, 11493, and 17273.  Each one
feeds an independent 1800-update cooldown branch:

```text
plateau 0..4400
  |-- cooldown 4400..6200       -> 3.2506B endpoint
  `-- plateau 4400..11493
        |-- cooldown 11493..13293 -> 6.9694B endpoint
        `-- plateau 11493..17273
              `-- cooldown 17273..19073 -> approximately-10B endpoint
```

Thus the three endpoint schedules have the same peak-LR semantics and equal
cooldown length.  The two intermediate endpoints are counterfactual branches,
not ordinary checkpoints on the final 10B trajectory.

## Integrity and resume behavior

- Formal data must contain at least 101 consecutively numbered train shards
  beginning at index 1 and exactly one validation shard.
- Preflight reads every header and computes a full SHA-256 for every train and
  validation shard.  Workers recheck file size and mtime before every phase.
- The training loader never uses modulo.  A single wrap attempt is a hard
  failure.  Every checkpoint cursor must equal
  `1 + completed_steps * 64` prefetched/consumed microbatches.
- Checkpoints atomically include model, both optimizers, train-loader cursor,
  prefetched `next_x/next_y`, Python/NumPy/CPU-CUDA RNG, absolute step,
  contract/data lineage, phase identity, and resume count.
- Resume uses the same run directory and sealed source snapshot.  Passed
  phases are skipped; an interrupted phase resumes from its own latest
  checkpoint.
- Fork checkpoints are deleted only after all direct children pass and after
  one final full-hash check.  A retirement certificate preserves their size,
  hash, and children.  The 36 primary endpoint checkpoints remain retained.
- Timing from any resumed phase is descriptive and not claim eligible.

The preflight hard-gates the accepted H100/PyTorch/Triton runtime and requires
at least 600 GB free on the artifact filesystem.  This is needed because the
36 retained endpoint checkpoints are estimated at roughly 439 GB before
headroom.

The replacement formal run is pinned to one physical host with exactly four
H100 80GB GPUs and NVIDIA driver `580.95.05`.  The PyTorch runtime remains the
accepted CUDA 12.6 build; the CUDA 13.0 value printed by `nvidia-smi` is the
newer driver's supported API ceiling, not the training runtime.  The earlier incomplete two-GPU run was interrupted and its
artifacts were deleted by the user; it is not resumable and is not accepted as
evidence.  Moving to four GPUs changes only the single-host LPT scheduling and
expected wall-clock.  Methods, seeds, initialization, data order, learning-rate
schedules, endpoints, and frozen analysis rules are unchanged.  Concurrent-run
timing remains ineligible for paper efficiency claims.

The accepted Linux `triton_kernels.py` has SHA-256 `b51ac50c...`.  An initial
preflight-only contract mistakenly pinned `f092ae99...`, which is the
content-identical Windows CRLF representation.  Before any data audit,
initialization, pilot, or training outcome was produced, the runtime field was
amended to the historical Linux LF digest and the outcome-blind amendment was
recorded in the formal contract.  Code and data roots remain independently
overridable; the certificate-safe `Newton-Muon-official-r0` checkout is the
default for both.

## Engineering pilot

The pilot is outcome-free.  It deliberately stops a two-update Muon segment
after one update (return code 75), resumes it in place, branches its checkpoint
into a second directory, verifies hashes/cursors, and retires the two large
pilot checkpoints.  Formal training cannot start without this pilot manifest.

The first remote pilot correctly caught that a CUDA-mapped checkpoint load also
moved the tiny RNG-state ByteTensors to CUDA, while PyTorch's RNG restore APIs
require them on CPU.  Before formal training, the worker was amended to copy
only those RNG tensors back to CPU; model and optimizer states remain directly
CUDA-mapped.  The outcome-free amendment is recorded in the formal contract,
and the failed pilot run is not resumable under the corrected source snapshot.

## Entry point

Only the scripts directory and command file need to be synchronized to the
remote repository:

```bash
bash commands/48_llama1b_10b_multibudget/20260805_ex48_llama1b_10b_multibudget.sh check
```

The ordered preflight, pilot, formal, resume, upload, and verify commands are
documented by the command wrapper and use one explicit `EX48_RUN_DIR`.

Local CSV/JSON/checkpoints are primary.  W&B upload occurs after each endpoint
has completed and can be retried independently.  The frozen project is:

`Selective-Newton-Muon-MainConf-LLaMA-1B-10B-Formal-20260805`

## Local tests

These tests are CPU-only and import no training dependencies:

```bash
python -m unittest scripts/48_llama1b_10b_multibudget/test_protocol.py -v
python scripts/48_llama1b_10b_multibudget/run_formal.py check
```

`HANDOFF.md` is intentionally not modified by this experiment implementation.
