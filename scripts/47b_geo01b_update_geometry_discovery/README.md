# Experiment 47 / GEO-01B: directional-geometry discovery

GEO-01B is the preregistered exploratory follow-up to the accepted GEO-01A
engineering pilot. It asks whether the actual all-down refresh update direction
predicts local and 16-step held-out loss harm, whether it improves over update
norm alone, and whether exact directional curvature improves on first order.

## Frozen scope

- Four checkpoint origins: early/late Muon and early/late Newton-full.
- Three new data replicas: 9, 10, and 11 (12 independent units).
- Two events per unit: production refresh at step 32 and delayed refresh at
  step 64.
- All 18 down-projection layers are the primary scope. The fixed early,
  middle, and late six-layer strata are descriptive support only.
- The predictor family is fixed before launch: relative update norm,
  first-order alignment, and full second-order Taylor prediction.
- Exact directional HVPs are used; no full Hessian is constructed and no full
  direction tensor is persisted.

The discovery can mark a result as a *confirmation candidate*, but cannot make
a paper claim or authorize confirmation. A positive result requires review and
a new frozen GEO-01C contract. GEO-01B cannot trigger LLaMA-1B/10B-token work.

## Workflow

Upload this directory into `scripts/`, synchronize the command file into
`commands/`, and run the launcher modes in order: `check`, `dry-run`, then
`discovery`. Use `resume` only with the same run directory and unchanged source
snapshot; use `verify` after completion.

The controller first runs an outcome-blind smoke test, then runs two fixed GPU
lanes. Each GPU processes six units sequentially, so two jobs share neither a
GPU nor memory. Passed units are hash-checked and skipped on resume. Failed
attempts are never overwritten.

Expected compact outputs include 96 geometry rows (12 × 2 × 4 scopes), 24
unit-event outcome rows, an analysis manifest, an event summary, and a sealed
handoff manifest. Checkpoints, directions, and Hessians are not copied into the
artifact directory.
