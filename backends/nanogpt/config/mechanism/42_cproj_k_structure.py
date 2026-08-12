"""Mechanism control that varies only the activation K used by mlp.c_proj."""

optimizer_type = "cproj_k_mode_newton_muon"
cproj_k_mode = "block4"
# Empty applies the mode to mlp.c_proj in every Transformer block.  A
# comma-separated list such as "0,1,2,3" applies it only at those depths and
# keeps every other c_proj on the full Newton path.
cproj_k_layers = ""
cproj_k_blocks = 4
# Shared by dense ``alpha`` and block-local ``block_alpha``.  The latter
# scales within-block off-diagonals only; cross-block entries remain zero.
cproj_k_offdiag_alpha = 0.5

# Expensive eigenspectrum diagnostics are disabled for training runs. The
# optimizer still logs exact K-state byte counts for every mode.
diagnostic_interval = 0
