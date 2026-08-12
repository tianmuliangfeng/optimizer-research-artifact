"""Build audited R1 sources for the block-local off-diagonal alpha sweep.

The official block4 covariance is retained in dense ``(4, d, d)`` storage.
Only the matrix copied into the inverse workspace is changed:

    K_alpha = diag(K) + alpha * (K - diag(K)).

Thus alpha=0 is a dense-storage engineering control, while alpha=1 takes the
unmodified official block4 inverse path.
"""

from __future__ import annotations

import difflib
import hashlib
import math
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
R1_DIR = SCRIPT_DIR.parent / "15_official_newton_muon_r1"
sys.path.insert(0, str(R1_DIR))

import r1_source_builder as base


ALPHA_BY_METHOD: dict[str, float] = {
    "alpha0": 0.0,
    "alpha0p25": 0.25,
    "alpha0p50": 0.50,
    "alpha0p75": 0.75,
}
ALLOWED_METHODS = tuple(ALPHA_BY_METHOD)


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"block-alpha expected one {label} anchor, observed {count}")
    return source.replace(old, new, 1)


def interpolate_dense_block(matrix: list[list[float]], alpha: float) -> list[list[float]]:
    """Small dependency-free reference implementation used by contract tests."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    return [
        [value if row == col else alpha * value for col, value in enumerate(values)]
        for row, values in enumerate(matrix)
    ]


def self_test_alpha_math() -> None:
    matrix = [[2.0, -3.0, 5.0], [7.0, 11.0, -13.0], [17.0, 19.0, 23.0]]
    zero = interpolate_dense_block(matrix, 0.0)
    one = interpolate_dense_block(matrix, 1.0)
    half = interpolate_dense_block(matrix, 0.5)
    if zero != [[2.0, 0.0, 0.0], [0.0, 11.0, 0.0], [0.0, 0.0, 23.0]]:
        raise AssertionError("alpha=0 is not the dense diagonal endpoint")
    if one != matrix:
        raise AssertionError("alpha=1 is not exactly the block4 endpoint")
    if not math.isclose(half[0][1], -1.5) or half[1][1] != 11.0:
        raise AssertionError("interior alpha interpolation is incorrect")


def build_source(repo: Path, method: str) -> base.DerivedSource:
    if method not in ALPHA_BY_METHOD:
        raise ValueError(f"unsupported block-alpha method: {method!r}")
    alpha = ALPHA_BY_METHOD[method]
    block4 = base.build_source(repo, "block4")
    source = block4.source

    source = _replace_once(
        source,
        '''if R1_METHOD not in ("block4", "none", "diag"):
    raise ValueError(f"invalid Newton R1 method={R1_METHOD!r}")
if R1_CPROJ_K_MODE != R1_METHOD:
    raise ValueError("Newton R1 method and cproj_k_mode must match")
''',
        f'''R1_BLOCK_ALPHA = {alpha!r}
if R1_METHOD != "{method}":
    raise ValueError("this derived source requires R1_METHOD={method}")
if R1_CPROJ_K_MODE != "block4":
    raise ValueError("R1 block-alpha requires dense official block4 state")
if not 0.0 <= R1_BLOCK_ALPHA <= 1.0:
    raise ValueError("R1_BLOCK_ALPHA must lie in [0, 1]")
''',
        "method/alpha validation",
    )
    source = _replace_once(
        source,
        '''print(f"R1_METADATA method={R1_METHOD} cproj_k_mode={R1_CPROJ_K_MODE} seed={R1_SEED} init_sha256={r1_init_sha256}")
''',
        '''print(f"R1_METADATA method={R1_METHOD} cproj_k_mode={R1_CPROJ_K_MODE} seed={R1_SEED} init_sha256={r1_init_sha256}")
print(f"R1_BLOCK_ALPHA alpha={R1_BLOCK_ALPHA:.8g} storage=dense_block4")
''',
        "alpha audit line",
    )
    source = _replace_once(
        source,
        '''            else:
                K[i].copy_(st["precond_cov"][sub])

        diag = K.diagonal(dim1=-2, dim2=-1)
''',
        '''            else:
                K[i].copy_(st["precond_cov"][sub])
                # Preserve the raw dense EMA state and official ridge rule.  Only
                # attenuate off-diagonals in the disposable inverse workspace.
                # The branch makes alpha=1 the exact official block4 path.
                if kind == "c_proj" and R1_BLOCK_ALPHA != 1.0:
                    diag_i = K[i].diagonal().clone()
                    K[i].mul_(R1_BLOCK_ALPHA)
                    K[i].diagonal().copy_(diag_i)

        diag = K.diagonal(dim1=-2, dim2=-1)
''',
        "block-local alpha interpolation",
    )
    compile(source, f"<R1-block-alpha-{method}>", "exec")

    official_source = (
        (repo / block4.base_script).read_bytes().replace(b"\r\n", b"\n").decode("utf-8")
    )
    diff = "".join(
        difflib.unified_diff(
            official_source.splitlines(keepends=True),
            source.splitlines(keepends=True),
            fromfile=f"official/{block4.base_script}",
            tofile=f"r1_block_alpha/train_r1_{method}.py",
        )
    )
    return base.DerivedSource(
        method=method,
        base_script=block4.base_script,
        base_canonical_sha256=block4.base_canonical_sha256,
        derived_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        source=source,
        unified_diff=diff,
    )

