# MECH-03 cross-fit shadow update

MECH-03 tests the prediction frozen after the MECH-02 endpoint geometry gate.
It is read-only with respect to training state: it never calls
`optimizer.step()`, never writes a checkpoint, and never uses W&B.

The formal contract is stored in `prediction_contract.json`. It must be
committed and hashed before remote results are read. The primary hypothesis is
fixed to the same-host GPT-bridge versus LLaMA-124M contrast:

> LLaMA-124M has a weaker held-out `diag`-over-`none` shadow-loss advantage
> than GPT bridge.

For each of four repeats, two disjoint eight-window splits are collected.
Direction A→B builds K and the gradient on A and evaluates shadow loss only on
B; B→A swaps the roles. The prespecified layers are h0/h4/h8/h11. The worker
uses the checkpoint's historical momentum, family-specific Nesterov
convention, production Newton–Schulz function, matrix-optimizer learning rate,
and weight decay. The line-search base is the checkpoint's `initial_lr`, not
the nearly-zero final warmdown LR; production shape/parameter multipliers are
retained.

R1/GPT candidates are `none/diag/block4/dense_full`; LLaMA candidates are
`none/diag/dense_full`. Only `diag` versus `none` enters the primary gate.
Full/block and grouped-layer results are secondary. LLaMA additionally records
gate/up projection outputs, the resulting `down_proj` input, and their
split-to-split second-moment drift.

Every family must first pass a small smoke run. Formal runs must consume the
matching smoke manifest and a passed MECH-02 formal directory. The final
cross-architecture analyzer consumes three formal directories and never
auto-authorizes MECH-04.

Implementation v2026-07-27.2 defines `model_unchanged` using exact persistent
parameter/state-dict content hashes. Reversible in-place shadow perturbations
increment PyTorch's non-persistent autograd `Parameter._version` counter even
after the original bytes are restored; that counter is recorded separately
but is not treated as checkpoint/model content mutation.
