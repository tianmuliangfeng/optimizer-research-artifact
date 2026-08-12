# MECH-02 checkpoint K-geometry

MECH-02 is the first scientific mechanism-discovery stage. It reuses frozen
checkpoints and the implementation paths certified by MECH-01; it does not
train, call `optimizer.step()`, write checkpoints, or use W&B.

The first frozen contract covers:

- R1-native `none`, GPT-bridge `none`, and LLaMA-124M `down_none`;
- all 12 contraction/output-projection layers;
- four independent K-build repeats, each made from two disjoint 128-token
  batches (`N=256` activation rows per repeat at device batch size 1);
- FineWeb token offsets
  `0,4096,8192,12288,16384,20480,24576,28672`;
- global damping `0.2 * mean(diag(K)) + 1e-8` for comparable spectra and
  dense inverse-health checks;
- top-32 eigenspace overlap between repeat pairs;
- sequential layer processing so only one layer's repeat covariances and
  eigensystems are resident at a time.

R1 additionally reports within-quarter and cross-quarter off-diagonal energy
for the architecture-defined four GELU quarters. LLaMA does not invent a
`block4` partition.

Required outputs include layer/repeat geometry, cross-repeat stability,
batch/source/runtime/checkpoint provenance, state invariance, and a manifest.
MECH-02 numbers are interpretable only when the supplied MECH-01 numerical
smoke directory passes all certificate checks.

Every family first runs an explicitly labeled MECH-02 smoke. A formal run must
consume that passed smoke manifest and match its family, method, checkpoint
SHA-256, and worker version.

`analyze_mech02.py` freezes the primary same-host contrast before results are
read. A metric becomes a geometry-gate candidate only when at least 8 of 12
layers differ in the same direction by more than the larger family repeat SD.
This script never auto-authorizes MECH-03: a separate held-out `diag-none`
prediction contract must be frozen first.
