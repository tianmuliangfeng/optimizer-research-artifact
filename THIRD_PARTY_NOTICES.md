# Third-party notices

The repository-level MIT license in `LICENSE` covers the anonymous authors'
contributions. It does not replace the licenses or attribution requirements of
the projects below. Upstream repositories and datasets are not included unless
an affected path is explicitly identified.

## NanoGPT-derived training code

- NanoGPT: <https://github.com/karpathy/nanoGPT>
- Affected path: `backends/nanogpt/`.
- License: MIT. The preserved notice is in `backends/nanogpt/LICENSE`.

The preserved notice states `Copyright (c) 2022 Andrej Karpathy` and must
remain with copies or substantial portions of the derived software.

## Modded-NanoGPT

- Upstream: <https://github.com/KellerJordan/modded-nanogpt>
- Frozen revisions used by the experiment contracts:
  `df78af0db523d8bceb25af4919a3e3e7082b80f3` and
  `9e7218468ea864a33053142c196d90bbf3ed48e1`
- Affected paths include the controlled R1 trainers and the vendored
  Record-17 trainer at
  `scripts/44_newton_muon_record17_455m/upstream/record17_train_gpt_medium.py`.
- License: MIT, `Copyright (c) 2024 Keller Jordan`. The complete upstream
  license is preserved in `third_party/licenses/Modded-NanoGPT-LICENSE`.

## Newton-Muon

- Upstream: <https://github.com/zhehangdu/Newton-Muon>
- Frozen revision: `df78af0db523d8bceb25af4919a3e3e7082b80f3`
- Role: reference implementation, pinned training scaffold, Triton kernels,
  and FineWeb preparation provenance used by multiple experiment families.
  The upstream checkout itself is not included in this package and must be
  obtained separately as described in `docs/ENVIRONMENT_AND_DATA.md`.
- License: MIT. The locally preserved upstream license is reproduced in
  `third_party/licenses/Newton-Muon-LICENSE`.

## Mousse

- Upstream: <https://github.com/Anti-Entrophic/Mousse>
- Frozen revision: `d00c1bf17790fbe56424ee5567cce80d8e75f4b2`
- Affected path: `scripts/45_r1_mousse_strong_baseline/`
- License: MIT. The complete upstream notice, including
  `Copyright (c) 2026 ShikiNatsume`, is preserved in
  `scripts/45_r1_mousse_strong_baseline/THIRD_PARTY_NOTICES.md`.

## NorMuon

- Upstream: <https://github.com/zichongli5/NorMuon>
- Reference file: <https://github.com/zichongli5/NorMuon/blob/main/normuon.py>
- Affected adaptation: the NorMuon compatibility implementation in
  `scripts/19_r1_extended_baselines/extended_optimizers.py`, reused by the
  corresponding LLaMA baseline controller.
- License: MIT, `Copyright (c) 2025 zichongli5`. The complete upstream license
  is preserved in `third_party/licenses/NorMuon-LICENSE`.

## Moonlight Muon

- Upstream: <https://github.com/MoonshotAI/Moonlight>
- Reference file:
  <https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py>
- Affected adaptation: the Moonlight Muon compatibility implementation in
  `scripts/19_r1_extended_baselines/extended_optimizers.py`, reused by the
  corresponding LLaMA baseline controller.
- License: MIT, `Copyright © 2025 Moonshot AI`. The complete upstream license
  is preserved in `third_party/licenses/Moonlight-LICENSE`.

The package does not vendor the complete NorMuon or Moonlight repositories.
Their compact experiment-specific adaptations include packed-QKV handling and
local Triton-kernel integration; the source URLs remain in the affected file.

## Dependencies and data

PyTorch, Triton, NumPy, pandas, W&B, Hugging Face datasets, tiktoken, and other
runtime dependencies are installed separately and remain under their own
licenses. FineWeb/FineWeb10B shards, model checkpoints, and W&B services are
not distributed as source code by this package; users are responsible for the
applicable upstream dataset, model, and service terms.
