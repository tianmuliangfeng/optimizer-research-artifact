from __future__ import annotations

import os
from pathlib import Path


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else None


def find_artifact_root(start: Path | None = None) -> Path:
    """Find the model-agnostic public artifact root."""
    env_root = _env_path("SNM_REPO") or _env_path("SNM_ARTIFACT_ROOT")
    if env_root is not None:
        return env_root

    current = (start or Path(__file__)).resolve()
    for parent in [current, *current.parents]:
        is_public_artifact = (
            (parent / "pyproject.toml").is_file()
            and (parent / "scripts").is_dir()
            and (parent / "backends" / "nanogpt").is_dir()
        )
        if is_public_artifact:
            return parent

    raise RuntimeError(
        "Could not find the Selective Newton-Muon artifact root. Set SNM_REPO "
        "to the repository directory when using a custom layout."
    )


ARTIFACT_ROOT = find_artifact_root()
WORKSPACE_ROOT = _env_path("SNM_WORKSPACE_ROOT") or ARTIFACT_ROOT
_PUBLIC_BACKEND = ARTIFACT_ROOT / "backends" / "nanogpt"
SOURCE_REPO = _env_path("SELECTIVE_NEWTON_MUON_SOURCE_REPO") or _PUBLIC_BACKEND
EXPERIMENT_REPO = (
    _env_path("SELECTIVE_NEWTON_MUON_REPO") or ARTIFACT_ROOT
)
_RESULTS_ROOT = _env_path("SNM_RESULTS_ROOT") or ARTIFACT_ROOT / "runs"
EXPERIMENT_DATA_ROOT = _RESULTS_ROOT
EXPERIMENT_RESULTS_ROOT = _RESULTS_ROOT
TRAIN_PY = SOURCE_REPO / "train.py"
