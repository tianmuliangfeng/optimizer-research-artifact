# Official Newton-Muon R0: two-run H100 reproduction gate

R0 answers one question before any further architecture-bridge experiment:

> Can the current official Newton-Muon repository reproduce the reported
> Newton-Muon-1 direction on the paper's own single-H100 recipe?

It runs exactly two unchanged upstream scripts from a pinned official checkout:

| R0 method | Upstream entry | Official base LR | Muon matrix LR | Explicit seed |
|---|---|---:|---:|---|
| Muon | `train_gpt_muon_1.py` | 0.0036 | 0.00036 | not set upstream |
| block4 Newton-Muon | `train_gpt_newton_muon_1.py` | 0.0040 | 0.00040 | not set upstream |

The different method-specific LRs are intentional and reproduce the official
recipe. R0 is therefore a **paper-recipe reproduction**, not a shared-LR
fairness test. The official scripts do not explicitly seed PyTorch, so these
runs must not be named seed2026 or treated as paired-seed evidence.

The wrapper pins commit:

```text
df78af0db523d8bceb25af4919a3e3e7082b80f3
```

and verifies canonical, line-ending-normalized SHA-256 for both training
scripts, the Triton kernels, and the official data downloader. This keeps the
check identical on Linux LF and Windows CRLF checkouts. It rejects modified tracked files, missing FineWeb
shards, non-H100 hardware, insufficient VRAM, and missing runtime packages.

## 1. Prepare the official checkout once

From the Selective Newton-Muon artifact root (the parent of
`selective-newton-muon`):

```bash
git clone https://github.com/zhehangdu/Newton-Muon.git Newton-Muon-official-r0
git -C Newton-Muon-official-r0 checkout df78af0db523d8bceb25af4919a3e3e7082b80f3
```

Keep the controller and training interpreters separate when needed. The
controller launches the jobs and performs the post-training W&B upload; the
training interpreter only needs the packages imported by the pinned official
scripts. Do not upgrade the environment that produced the 60-run results in
place.

The `torch 2.12.1+cu130` environment at
`${SNM_CONTROLLER_PYTHON}` is explicitly rejected for formal R0: on
2026-07-18 that exact runtime produced non-finite loss from optimizer step 2 in
both official methods. `/usr/bin/python` (`torch 2.3.0a0 nv24.04 / cu124`) is
also rejected: its bundled Triton lacks
`triton.tools.tensor_descriptor.TensorDescriptor`, which the pinned official
`triton_kernels.py` imports.

Create a dedicated environment on the same H100 instead of mutating the 60-run
venv. PyTorch 2.8.0+cu126 with its Triton 3.4.0 dependency is the current
project-tested candidate (the earlier R1 implementation smoke imported and
started the same official kernel path with this pair):

```bash
/usr/bin/python -m venv /path/to/venv-r0-torch280-cu126
${SNM_TRAINING_PYTHON} -m pip install --upgrade pip
${SNM_TRAINING_PYTHON} -m pip install \
  torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
${SNM_TRAINING_PYTHON} -m pip install \
  numpy==2.2.6 typing-extensions==4.15.0
```

Download the official 50-shard FineWeb10B subset (large download; do this only
once):

```bash
python Newton-Muon-official-r0/data/cached_fineweb10B.py 50
```

If the exact shards already exist elsewhere, a symlink named
`Newton-Muon-official-r0/data/fineweb10B` may be used instead. Do not regenerate,
subsample, or rename shards for R0.

## 2. Inspect exactly two commands

Run from `selective-newton-muon`:

```bash
python scripts/14_official_newton_muon_r0/run_official_newton_muon_r0.py \
  --official-repo ../Newton-Muon-official-r0 \
  --python-exe python \
  --dry-run
```

Dry-run checks the pinned official source but does not require the dataset or a
GPU and does not create W&B runs.

## 3. H100/data/runtime preflight

Run the wrapper with the 60-run venv as controller and the isolated R0 runtime
as the official training child:

```bash
${SNM_CONTROLLER_PYTHON} scripts/14_official_newton_muon_r0/run_official_newton_muon_r0.py \
  --official-repo ../Newton-Muon-official-r0 \
  --python-exe ${SNM_TRAINING_PYTHON} \
  --wandb-mode disabled \
  --preflight
```

This checks the pinned code, the exact 50 training shards plus validation shard
and file magic, PyTorch/NumPy/Triton imports from the training child, CUDA, H100
identity, and nominal 80GB VRAM. It resolves directory symlinks, so a plain
`find` count of zero on a symlink is not treated as proof that data is missing.
It does not start training.

## 4. Required exact-shape 10-step numerical smoke

The smoke preserves the official training batch size, device batch size,
sequence length, accumulation count, optimizer mathematics, and Triton kernel
shapes. It only shortens training to 10 updates, reduces validation work, and
disables the final checkpoint. Its generated source, SHA-256, and exact patch
are saved. It is a compatibility gate and is never formal R0 evidence.

```bash
CUDA_VISIBLE_DEVICES=0 ${SNM_CONTROLLER_PYTHON} \
  scripts/14_official_newton_muon_r0/run_official_newton_muon_r0.py \
  --official-repo ../Newton-Muon-official-r0 \
  --python-exe ${SNM_TRAINING_PYTHON} \
  --numerical-smoke \
  --smoke-steps 10 \
  --wandb-mode disabled
```

Both methods must end as `completed_valid_smoke`. Any NaN/Inf is detected from
the live stream and terminates the child immediately. Keep the resulting
`r0_manifest.json`; formal R0 requires it as a runtime-matched certificate.

## 5. Formal two-run R0

```bash
CUDA_VISIBLE_DEVICES=0 ${SNM_CONTROLLER_PYTHON} \
  scripts/14_official_newton_muon_r0/run_official_newton_muon_r0.py \
  --official-repo ../Newton-Muon-official-r0 \
  --python-exe ${SNM_TRAINING_PYTHON} \
  --smoke-manifest /absolute/path/to/successful-smoke/r0_manifest.json \
  --wandb-mode online
```

The jobs run sequentially: official Muon first, official Newton-Muon block4
second. Do not add `--continue-on-error` for the first execution: an environment
failure in Muon should stop the gate before consuming another full run.

If Muon completed and only block4 needs to be retried:

```bash
${SNM_CONTROLLER_PYTHON} scripts/14_official_newton_muon_r0/run_official_newton_muon_r0.py \
  --official-repo ../Newton-Muon-official-r0 \
  --python-exe ${SNM_TRAINING_PYTHON} \
  --methods block4 \
  --smoke-manifest /absolute/path/to/successful-smoke/r0_manifest.json \
  --wandb-mode online
```

## 6. Logging, validity gates, and artifacts

The upstream training process is not instrumented or patched. The wrapper
captures its stdout and official UUID log, parses scalar loss/time/LR/memory
metrics, and uploads scalar W&B histories only **after training finishes**.
This prevents W&B network traffic from altering official benchmark timing.
No W&B tables are created.

The wrapper accepts and exposes `nan`, `inf`, and signed/case variants instead
of silently dropping them. A run is valid only when it has all 6200 train
points, all 63 validation points at steps 0,100,...,6200, finite losses, the
correct terminal steps and total-step fields, and a valid peak-memory record.
Quality validation occurs before W&B upload. Manifests distinguish
`completed_valid`, `invalid_nonfinite`, `invalid_incomplete`,
`training_failed`, and W&B upload failure.

Default W&B project:

```text
Selective-Newton-Muon-MainConf-OfficialR0-H100-20260717
```

Artifacts are stored under:

```text
${SNM_RESULTS_ROOT}/
  14_official_newton_muon_r0/results/<batch-id>/
```

Each run preserves its stdout, copied official source-containing log,
machine-readable scalar history, summary, official checkpoint directory, code
commit/hashes, hardware/runtime information, full command, and W&B status.

The unchanged upstream scripts save final checkpoints. Ensure adequate disk
space before starting both jobs.

## 7. R0 acceptance gate

Use the official README values only as a reference, not an exact numerical
tolerance across machines:

| Method | Official reported final loss |
|---|---:|
| Muon | 3.2793 |
| Newton-Muon | 3.2611 |

R0 passes when both runs finish without provenance/data errors, have plausible
loss curves, and Newton-Muon reproduces the lower-final-loss direction. Wall
time is compared only between the two runs on the same H100. If the direction
does not reproduce, stop before adding none/diag and audit the official
environment/data first.
