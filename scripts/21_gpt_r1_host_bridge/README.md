# GPT R1 on the LLaMA host: diag/none bridge

This bridge isolates the architecture comparison from the H100 host, driver,
CUDA, PyTorch, and Triton change. It runs the original GPT R1 `diag` and `none`
cells on the same host/runtime used by the LLaMA/SwiGLU experiment.

## What remains valid during GPU0/GPU1 concurrency

The LLaMA job may continue on physical GPU0 while this bridge exclusively uses
physical GPU1. The following evidence remains usable if both jobs complete all
step/token/data/checkpoint gates:

- final validation loss, tail-5, normalized AUC, and steps/tokens-to-target;
- the within-seed `diag - none` paired difference;
- exact K/optimizer-state bytes and per-process CUDA peak allocated/reserved.

The following evidence is always marked ineligible for this bridge:

- wall-clock, ms/step, tokens/s, time-to-loss, power, and energy;
- any timing comparison with the old R1 host or with the concurrent LLaMA job.

Shared-node I/O contention can make training slower, but it does not change the
fixed update count or deterministic data order. If contention causes an I/O
error, missing step, non-finite value, incomplete checkpoint, or non-exact
restart, the existing R1 gates reject the affected run.

## Fixed design

- physical GPU: GPU1 via `CUDA_VISIBLE_DEVICES=1`;
- concurrent workload: LLaMA/SwiGLU multi-seed on physical GPU0;
- GPT methods: `diag`, then `none`;
- seed: 2026 first;
- model/data/training: unchanged GPT R1 recipe, 6200 updates and about 3.25B
  tokens per method;
- learning rate: both methods use the same Newton LR (`0.0040`, matrix LR
  `0.00040`);
- W&B project:
  `Selective-Newton-Muon-MainConf-GPT-R1-HostBridge-20260721`;
- local family: `21_gpt_r1_host_bridge`.

The two methods run sequentially on GPU1. Their total elapsed time is therefore
roughly two single-method R1 runs plus preflight/smoke overhead; elapsed time is
for scheduling only and is not paper evidence.

## Commands on the LLaMA H100 host

The bridge wrapper and its audited base runner must be synchronized together.
Before running, verify that the remote base runner contains the native bridge
API:

```bash
grep -n "HOST_BRIDGE_FAMILY" \
  scripts/15_official_newton_muon_r1/run_official_newton_muon_r1.py
```

If this prints nothing, update that base runner before continuing. Copying only
the `scripts/21_gpt_r1_host_bridge` directory is insufficient because the
wrapper deliberately reuses the R1 source/evidence implementation.

The official source worktree must also be clean. Do not reset or overwrite the
directory used by another experiment. If `git status --porcelain
--untracked-files=no` is nonempty, create a detached clean worktree from the
already-local Git objects and link only the immutable FineWeb shards:

```bash
BASE_REPO=${SNM_OFFICIAL_REPO}
BRIDGE_REPO=${SNM_OFFICIAL_REPO}
DATA_DIR="$BASE_REPO/data/fineweb10B"
PINNED_COMMIT=df78af0db523d8bceb25af4919a3e3e7082b80f3

test -d "$DATA_DIR"
test ! -e "$BRIDGE_REPO"
git -C "$BASE_REPO" worktree add --detach "$BRIDGE_REPO" "$PINNED_COMMIT"
ln -s "$DATA_DIR" "$BRIDGE_REPO/data/fineweb10B"

git -C "$BRIDGE_REPO" rev-parse HEAD
git -C "$BRIDGE_REPO" status --porcelain --untracked-files=no
```

If the container user that runs training differs from the worktree owner, Git
may reject these checks with `detected dubious ownership`. Trust only this
explicit experiment worktree for the current user, then repeat the checks:

```bash
git config --global --add safe.directory "$BRIDGE_REPO"
git -C "$BRIDGE_REPO" rev-parse HEAD
git -C "$BRIDGE_REPO" status --porcelain --untracked-files=no
```

Do not configure `safe.directory='*'`; the bridge only needs this one explicit
path.

The first check must print the pinned commit and the second must print nothing.
The symlink is not tracked by the official repository, while both GPU jobs may
safely read the same shards. Set `OFFICIAL_REPO="$BRIDGE_REPO"` for all bridge
commands below.

Run from `selective-newton-muon`:

```bash
cd ${SNM_REPO}

export CUDA_VISIBLE_DEVICES=1
CTRL_PY=${SNM_CONTROLLER_PYTHON}
TRAIN_PY=${SNM_TRAINING_PYTHON}
OFFICIAL_REPO=${SNM_OFFICIAL_REPO}
RUNNER=scripts/21_gpt_r1_host_bridge/run_gpt_r1_host_bridge.py
OTHER_JOB=llama_swiglu_multiseed_gpu0
```

Inspect the two-job plan without data/GPU/W&B:

```bash
"$CTRL_PY" "$RUNNER" \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY" \
  --wandb-mode disabled \
  --dry-run
```

Run source/data/runtime/W&B and identical-initialization preflight:

```bash
"$CTRL_PY" "$RUNNER" \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY" \
  --concurrent-node-training \
  --concurrent-workload "$OTHER_JOB" \
  --wandb-mode online \
  --preflight
```

Run the required exact-shape smoke. It is intentionally 34 steps so both
methods cross the first K refresh at update 32:

```bash
"$CTRL_PY" "$RUNNER" \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY" \
  --concurrent-node-training \
  --concurrent-workload "$OTHER_JOB" \
  --wandb-mode disabled \
  --numerical-smoke \
  --smoke-steps 34
```

Use the printed smoke manifest for formal seed2026:

```bash
SMOKE_MANIFEST=/absolute/path/to/21_gpt_r1_host_bridge/results/<smoke_batch>/r1_manifest.json

"$CTRL_PY" "$RUNNER" \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY" \
  --seed 2026 \
  --smoke-manifest "$SMOKE_MANIFEST" \
  --concurrent-node-training \
  --concurrent-workload "$OTHER_JOB" \
  --wandb-mode online
```

If the batch is interrupted, reuse the exact formal artifact directory. A
completed method is revalidated and skipped; an interrupted method restarts
from step zero because the official R1 program does not persist the full loader
and scheduler state needed for an exact mid-method continuation:

```bash
FORMAL_ARTIFACT=/absolute/path/to/21_gpt_r1_host_bridge/results/<formal_batch>

"$CTRL_PY" "$RUNNER" \
  --official-repo "$OFFICIAL_REPO" \
  --python-exe "$TRAIN_PY" \
  --seed 2026 \
  --resume-batch "$FORMAL_ARTIFACT" \
  --concurrent-node-training \
  --concurrent-workload "$OTHER_JOB" \
  --wandb-mode online
```

Do not add `block4` or Muon to this batch. The bridge estimand is only the GPT
`diag - none` difference under the LLaMA host/runtime. Add seeds2024/2025 only
if seed2026 changes direction, collapses toward zero, or otherwise changes the
architecture interpretation.

## Frozen seed2026 result (2026-07-22)

The formal bridge completed both prescribed methods with 63/63 validation
points through step 6200. W&B endpoint results are:

| Method | Final val | Tail-5 | Normalized AUC | Peak allocated | K state |
|---|---:|---:|---:|---:|---:|
| diag | 3.2620 | 3.270280 | 3.616995 | 38,304 MiB | 162.28125 MiB |
| none | 3.2669 | 3.275220 | 3.625623 | 38,304 MiB | 162.00000 MiB |

The bridge `diag - none` final-loss difference is `-0.0049`, compared with
`-0.0050` on the original GPT R1 host. Moving GPT to the LLaMA host/runtime
therefore changes the paired difference by only `+0.0001`, which is near the
four-decimal logging resolution. On that same host/runtime the LLaMA/SwiGLU
seed2026 difference is `+0.001357`, leaving an architecture-associated
residual shift of `+0.006257`.

The prespecified trigger for GPT bridge seeds2024/2025 did not fire: direction
and magnitude were retained. Do not add bridge seeds. Complete the LLaMA
seeds2024/2025 instead. The result supports an architecture-associated
interaction under the measured systems, but it is not a randomized causal
architecture effect and must not be generalized beyond the tested seed until
the LLaMA multi-seed analysis closes.

The preserved W&B analysis is under
`${SNM_RESULTS_ROOT}/21_gpt_r1_host_bridge/analysis/wandb_20260722_seed2026/`.
Its status is `PASS_WITH_CAVEATS`: all 20 W&B structural/numerical checks pass,
but the remote formal manifest, checkpoint, source/runtime/init fingerprints,
concurrency declaration, and resume count still need to be synchronized and
audited. Timing remains ineligible by design.
