# Environment and data

## Core package

The optimizer package requires Python 3.10 or newer and PyTorch. It is not
coupled to NanoGPT, a particular tokenizer, or a transformer architecture.

## Accepted H100 runtime

The late formal experiments used the following frozen runtime on single-host
NVIDIA H100 80 GB systems. Most late protocols used two GPUs; experiment 48's
accepted replacement protocol uses four.

| Component | Version |
|---|---|
| Python | 3.10.12 |
| PyTorch | 2.8.0+cu126 |
| CUDA reported by PyTorch | 12.6 |
| Triton | 3.4.0 |
| NumPy | 2.2.6 |

Individual contracts may impose additional environment variables such as
`CUBLAS_WORKSPACE_CONFIG=:4096:8` and deterministic backend settings. The
experiment controller, not this overview, is the final authority.

## Newton-Muon upstream

Experiments aligned to the official Newton-Muon implementation expect a
separate checkout supplied through `SNM_OFFICIAL_REPO`. The accepted historical
revision was commit:

```text
df78af0db523d8bceb25af4919a3e3e7082b80f3
```

The upstream repository is not vendored here. Source builders and formal
contracts verify the required files before training.

## Data

Training binaries and checkpoints are excluded from source control. In
particular, experiment 48 requires:

- at least 101 consecutively numbered FineWeb training shards beginning at
  index 1;
- exactly one validation shard;
- enough unique tokens to reach the 10B-token endpoint without loader wrap;
- a full-content preflight inventory before formal training.

## Experiment 48 execution geometry

Experiment 48 requires one physical host exposing exactly four H100 80GB GPUs
for the accepted replacement protocol. Its frozen contract also records NVIDIA
driver `580.95.05` and treats concurrent timing as ineligible for efficiency
claims.

Set data paths through the experiment-specific variables documented by its
launcher. The common public profile is:

```bash
export SNM_OFFICIAL_REPO=/path/to/Newton-Muon-official-r0
export SNM_RESULTS_ROOT=/path/to/experiment-results
```

These exports are for direct launcher use. With
`reproducibility/reproduce.py`, provide the same values through repeated
`--env KEY=VALUE` options so they are bound to the execution receipt.

## W&B

Local CSV, JSON, manifests, and checkpoints are primary evidence. W&B is a
secondary visualization/upload channel and may be disabled or retried without
changing local scientific outcomes when the controller supports it.
