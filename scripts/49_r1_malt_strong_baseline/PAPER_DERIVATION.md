# Experiment 49 paper-derivation notice

Experiment 49 contains an independent implementation derived from the public
mathematical description in MALT, arXiv:2608.05088v1. No author code was
publicly available when this contract was frozen. The implementation must be
described as **MALT-R1 adaptation** or **MALTER-Eq17-R1 adaptation**, not as an
official reproduction.

Primary source:

- <https://arxiv.org/abs/2608.05088>
- <https://arxiv.org/pdf/2608.05088>

The unambiguous MALT arm follows Algorithm 1. The independently selected
MALTER-Eq17 supporting formal arm follows Equation (17) and the prose
immediately below Algorithm 2: one outer learning-rate factor is used. This
resolves the v1 inconsistency in
which Algorithm 2 mixes `v`/`nu` and prints `eta` both inside `alpha` and again
in the parameter update.

The following choices belong to the R1 adaptation because the paper does not
disclose them completely:

- the pinned R1 five-step Newton--Schulz backend;
- splitting packed QKV into three logical matrices;
- the exact decoupled weight-decay ordering;
- no additional matrix-shape multiplier beyond Algorithm 1.

All such choices are frozen in `malt_contract.json` and echoed in every run
manifest.
