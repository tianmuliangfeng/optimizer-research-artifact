# Experiment 53: matched diagonal module placement

- integrity: `passed` (15/15 formal units)
- descriptive lowest endpoint mean: `c_fc_c_proj_diag`
- primary evidence: final-step validation loss with within-seed paired contrasts
- factorial: `c_fc x c_proj` diagonal/none, including per-seed effects
- attention extension: the same diagonal representation at `o_proj`
- QKV route: `none` in every arm
- dense preconditioner workspace: forbidden and observed zero
- timing: ineligible

The rank is descriptive, not a universal module ranking. Interpret paired effects, uncertainty, and state cost together; no arm was selected or removed from pilot loss.
