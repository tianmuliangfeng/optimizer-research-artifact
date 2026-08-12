# MECH-06 LLaMA-1B scale confirmation

MECH-06 is a read-only CUDA diagnostic on the existing seed2026 `down_none`
LLaMA-1B checkpoints at steps 1000 and 6200. It performs no new training, never
calls `optimizer.step()`, never writes a checkpoint, and does not use W&B.

For each checkpoint it runs smoke before formal:

- exact diagonal and exact low-rank spectrum geometry for all 18 `down_proj`
  inputs, with four repeats in formal;
- held-out cross-fit shadow loss for layers 0/6/12/17 and candidates
  `none/diag/dense_full`;
- exact Woodbury application of the dense inverse, avoiding a 5504×5504 dense
  inverse;
- full checkpoint SHA-256 once before smoke/formal;
- persistent model, optimizer/loader, and checkpoint-file invariance.

MECH-04 was not authorized, so MECH-06 does not run HVP. The existing LLaMA-1B
training rankings were known before MECH-05 and are excluded from diagnostic
prediction generation. Any later comparison with them is retrospective.

Files stay in the numbered script directory. The executable instruction stays
in `commands/33_mech06_llama1b_confirmation/20260727_mech06_llama1b_confirmation.sh`. No archive is created.

Run:

```bash
MECH_GPU=0 bash commands/33_mech06_llama1b_confirmation/20260727_mech06_llama1b_confirmation.sh
```
