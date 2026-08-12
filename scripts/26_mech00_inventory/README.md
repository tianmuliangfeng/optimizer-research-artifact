# MECH-00 result/checkpoint inventory

`run_mech00_inventory.py` is a standard-library-only, read-only audit. It:

- discovers completed R1, GPT-bridge, LLaMA-124M, and LLaMA-1B summaries;
- verifies that referenced checkpoints exist and records file size/mtime;
- keeps `host_id`, execution domain, source repository, runtime, method, seed,
  and training step attached to every row;
- maps target steps `0/500/1000/3000/6200` to the nearest checkpoint that
  actually exists, without interpolation;
- distinguishes a checkpoint suitable for fresh-batch geometry diagnostics
  from a checkpoint whose exact loader/RNG replay has been verified;
- optionally computes stable full-file SHA-256 hashes after active training has
  stopped.

It does not import PyTorch, load checkpoints, reserve GPU memory, modify source
repositories, or modify experiment artifacts. Its only writes are under
`--output-dir`.

## Important interpretation

- R1/GPT-bridge checkpoints are registered as model checkpoints suitable for
  fresh-batch diagnostics. MECH-00 does not claim exact training resumption.
- Completed LLaMA formal checkpoints are marked `resumable_expected` according
  to their runner contract, but remain `checkpoint_schema_verified=false`
  until MECH-01 performs a read-only load/schema audit.
- `--hash-mode none` is the safe choice while another job is training on the
  same host. It records hashing as deferred. Re-run later with
  `--hash-mode full`; a digest is accepted only when size, mtime, and inode are
  stable throughout the read.
- A repository whose Git commands succeed but whose tracked worktree is
  modified is reported as `ok_dirty` and produces an audit warning rather than
  being mislabeled clean or rejected automatically.
- For the 1B staged runner, `execution_stage` takes precedence over the reused
  base runner's `batch_kind`; this keeps the 1000-step medium screen distinct
  from 6200-step formal evidence.

## Minimal example

```bash
python scripts/26_mech00_inventory/run_mech00_inventory.py \
  --host-id r1-native-h100 \
  --execution-domain r1-native \
  --input r1=/path/to/r1/results \
  --family-hint r1=r1_native \
  --repo r1_source=/path/to/Newton-Muon-official-r0 \
  --methods none muon \
  --hash-mode none \
  --output-dir /path/to/mech00/r1_native
```

The formal output set is:

- `checkpoint_inventory.csv`
- `checkpoint_hashes.csv`
- `available_step_map.csv`
- `input_discovery.csv`
- `source_inventory.csv`
- `runtime_inventory.json`
- `diagnostic_data_contract.json`
- `audit_checks.csv`
- `mech00_manifest.json`
