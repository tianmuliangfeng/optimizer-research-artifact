"""Build audited R1 sources for the dense-full c_proj alpha experiment.

The experiment follows the OpenWebText intervention exactly at the R1 shape::

    K_alpha = diag(K_full) + alpha * (K_full - diag(K_full)).

Unlike ``22_r1_block_alpha``, ``K_full`` is the complete 3072 x 3072 c_proj
input covariance.  Consequently the intervention restores within-block and
cross-block covariance together.  Every alpha cell keeps the same dense-full
covariance/inverse/workspace allocation.
"""

from __future__ import annotations

import difflib
import hashlib
import math
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
R1_DIR = SCRIPT_DIR.parent / "15_official_newton_muon_r1"
PERF_DIR = SCRIPT_DIR.parent / "18_r1_performance"
for directory in (R1_DIR, PERF_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import r1_source_builder as base  # noqa: E402
import r1_perf_source_builder as perf  # noqa: E402


ALPHA_BY_METHOD: dict[str, float] = {
    "fullalpha0": 0.0,
    "fullalpha0p25": 0.25,
    "fullalpha0p50": 0.50,
    "fullalpha0p75": 0.75,
    "fullalpha1": 1.0,
}
ALLOWED_METHODS = tuple(ALPHA_BY_METHOD)
DIAGNOSTIC_STEPS = (31, 1023, 2047, 3071, 4095, 5119, 6143)


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"dense-full alpha expected one {label} anchor, observed {count}")
    return source.replace(old, new, 1)


def interpolate_dense_full(matrix: list[list[float]], alpha: float) -> list[list[float]]:
    """Dependency-free reference implementation used by contract tests."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    return [
        [value if row == col else alpha * value for col, value in enumerate(values)]
        for row, values in enumerate(matrix)
    ]


def self_test_alpha_math() -> None:
    matrix = [[2.0, -3.0, 5.0], [7.0, 11.0, -13.0], [17.0, 19.0, 23.0]]
    zero = interpolate_dense_full(matrix, 0.0)
    one = interpolate_dense_full(matrix, 1.0)
    half = interpolate_dense_full(matrix, 0.5)
    if zero != [[2.0, 0.0, 0.0], [0.0, 11.0, 0.0], [0.0, 0.0, 23.0]]:
        raise AssertionError("alpha=0 is not the dense diagonal endpoint")
    if one != matrix:
        raise AssertionError("alpha=1 is not exactly dense full")
    if not math.isclose(half[0][1], -1.5) or half[1][1] != 11.0:
        raise AssertionError("dense-full alpha interpolation is incorrect")


def _package(repo: Path, method: str, base_script: str, source: str) -> base.DerivedSource:
    base_raw = (repo / base_script).read_bytes()
    official_source = base.canonical_bytes(base_raw).decode("utf-8")
    compile(source, f"<R1-dense-full-alpha-{method}>", "exec")
    diff = "".join(
        difflib.unified_diff(
            official_source.splitlines(keepends=True),
            source.splitlines(keepends=True),
            fromfile=f"official/{base_script}",
            tofile=f"r1_dense_full_alpha/train_{method}.py",
        )
    )
    return base.DerivedSource(
        method=method,
        base_script=base_script,
        base_canonical_sha256=base.canonical_sha256(base_raw),
        derived_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        source=source,
        unified_diff=diff,
    )


def build_source(repo: Path, method: str) -> base.DerivedSource:
    if method not in ALPHA_BY_METHOD:
        raise ValueError(f"unsupported dense-full alpha method: {method!r}")
    repo = repo.resolve()
    alpha = ALPHA_BY_METHOD[method]
    built = perf.build_perf_source(repo, "dense_full")
    source = built.source

    source = _replace_once(
        source,
        '''if R1_METHOD not in ("block4", "none", "diag", "dense_full"):
    raise ValueError(f"invalid Newton R1 method={R1_METHOD!r}")
if R1_CPROJ_K_MODE != R1_METHOD:
    raise ValueError("Newton R1 method and cproj_k_mode must match")
''',
        f'''R1_DENSE_FULL_ALPHA = {alpha!r}
if R1_METHOD != "{method}":
    raise ValueError("this derived source requires R1_METHOD={method}")
if R1_CPROJ_K_MODE != "dense_full":
    raise ValueError("R1 dense-full alpha requires cproj_k_mode=dense_full")
if not 0.0 <= R1_DENSE_FULL_ALPHA <= 1.0:
    raise ValueError("R1_DENSE_FULL_ALPHA must lie in [0, 1]")
''',
        "method/alpha validation",
    )
    source = _replace_once(
        source,
        '''print(f"R1_METADATA method={R1_METHOD} cproj_k_mode={R1_CPROJ_K_MODE} seed={R1_SEED} init_sha256={r1_init_sha256}")
''',
        '''print(f"R1_METADATA method={R1_METHOD} cproj_k_mode={R1_CPROJ_K_MODE} seed={R1_SEED} init_sha256={r1_init_sha256}")
print(f"R1_DENSE_FULL_ALPHA alpha={R1_DENSE_FULL_ALPHA:.8g} storage=dense_full_3072")
''',
        "alpha audit line",
    )

    source = _replace_once(
        source,
        '''            "inv_proj_full": torch.empty((len(proj_full_params), 4 * d, 4 * d), device=dev, dtype=torch.float32) if proj_full_params else None,
            "g_proj_full": torch.empty((len(proj_full_params), d, 4 * d), device=dev, dtype=torch.float32) if proj_full_params else None,
            "tmp_proj_full": torch.empty((len(proj_full_params), d, 4 * d), device=dev, dtype=torch.float32) if proj_full_params else None,
            "full_refresh_work": torch.empty((len(proj_full_params), 4 * d, 4 * d), device=dev, dtype=torch.float32) if proj_full_params else None,
''',
        '''            "inv_proj_full": torch.empty((len(proj_full_params), 4 * d, 4 * d), device=dev, dtype=torch.float32) if proj_full_params else None,
            "diag_ref_proj_full": torch.empty((len(proj_full_params), 4 * d), device=dev, dtype=torch.float32) if proj_full_params else None,
            "g_proj_full": torch.empty((len(proj_full_params), d, 4 * d), device=dev, dtype=torch.float32) if proj_full_params else None,
            "tmp_proj_full": torch.empty((len(proj_full_params), d, 4 * d), device=dev, dtype=torch.float32) if proj_full_params else None,
            "diag_probe_update": torch.empty((len(proj_full_params), d, 4 * d), device=dev, dtype=torch.float32) if proj_full_params else None,
            "full_refresh_work": torch.empty((len(proj_full_params), 4 * d, 4 * d), device=dev, dtype=torch.float32) if proj_full_params else None,
''',
        "diagnostic buffers",
    )
    source = _replace_once(
        source,
        '''        if plan["inv_proj_full"] is not None:
            plan["inv_proj_full"].zero_()
            plan["inv_proj_full"].diagonal(dim1=-2, dim2=-1).fill_(1.0)
            for i, p in enumerate(proj_full_params):
                self.state[p]["precond_inv_apply"] = plan["inv_proj_full"][i]
''',
        '''        if plan["inv_proj_full"] is not None:
            plan["inv_proj_full"].zero_()
            plan["inv_proj_full"].diagonal(dim1=-2, dim2=-1).fill_(1.0)
            plan["diag_ref_proj_full"].fill_(1.0)
            for i, p in enumerate(proj_full_params):
                self.state[p]["precond_inv_apply"] = plan["inv_proj_full"][i]
''',
        "diagnostic initialization",
    )

    source = _replace_once(
        source,
        '''        if plan["inv_proj_full"] is not None:
            G = plan["g_proj_full"]
            for i, p in enumerate(plan["proj_full_params"]):
                if p.grad is None:
                    G[i].zero_()
                else:
                    G[i].copy_(p.grad, non_blocking=True)
            torch.bmm(G, plan["inv_proj_full"], out=plan["tmp_proj_full"])
            for i, p in enumerate(plan["proj_full_params"]):
                if p.grad is not None:
                    p.grad.copy_(plan["tmp_proj_full"][i], non_blocking=True)
''',
        f'''        if plan["inv_proj_full"] is not None:
            G = plan["g_proj_full"]
            for i, p in enumerate(plan["proj_full_params"]):
                if p.grad is None:
                    G[i].zero_()
                else:
                    G[i].copy_(p.grad, non_blocking=True)
            torch.bmm(G, plan["inv_proj_full"], out=plan["tmp_proj_full"])
            if self.global_step in {DIAGNOSTIC_STEPS!r}:
                torch.mul(
                    G,
                    plan["diag_ref_proj_full"].unsqueeze(1),
                    out=plan["diag_probe_update"],
                )
                actual = plan["tmp_proj_full"].float()
                reference = plan["diag_probe_update"].float()
                actual_norm = actual.norm().clamp_min(1e-30)
                reference_norm = reference.norm().clamp_min(1e-30)
                cosine = (actual * reference).sum() / (actual_norm * reference_norm)
                print(
                    "R1_FULL_ALPHA_UPDATE "
                    f"step={{self.global_step}} alpha={{R1_DENSE_FULL_ALPHA:.8g}} "
                    f"norm_ratio_vs_diag={{(actual_norm / reference_norm).item():.9g}} "
                    f"cosine_vs_diag={{cosine.item():.9g}}"
                )
            for i, p in enumerate(plan["proj_full_params"]):
                if p.grad is not None:
                    p.grad.copy_(plan["tmp_proj_full"][i], non_blocking=True)
''',
        "update diagnostics",
    )

    source = _replace_once(
        source,
        '''        if self._apply_plan["inv_proj_full"] is not None:
            work = self._apply_plan["full_refresh_work"]
            for i, p in enumerate(self._apply_plan["proj_full_params"]):
                work[i].copy_(self.state[p]["precond_cov"])
            diag_full = work.diagonal(dim1=-2, dim2=-1)
            ridge_full = diag_full.mean(dim=-1) * self.precond_ridge_mult + self.precond_eps
            diag_full.add_(ridge_full.unsqueeze(-1))
            factor_full, info_full = torch.linalg.cholesky_ex(work, upper=False, check_errors=False)
            torch.cholesky_inverse(factor_full, upper=False, out=self._apply_plan["inv_proj_full"])
            bad_full = info_full != 0
            if bad_full.any():
                self._apply_plan["inv_proj_full"][bad_full].zero_()
                self._apply_plan["inv_proj_full"][bad_full].diagonal(dim1=-2, dim2=-1).fill_(1.0)
''',
        f'''        if self._apply_plan["inv_proj_full"] is not None:
            work = self._apply_plan["full_refresh_work"]
            for i, p in enumerate(self._apply_plan["proj_full_params"]):
                work[i].copy_(self.state[p]["precond_cov"])
            raw_diag = work.diagonal(dim1=-2, dim2=-1).clone()
            diagnostic_refresh = self.global_step in {DIAGNOSTIC_STEPS!r}
            if diagnostic_refresh:
                raw_total_sq = work.float().square().sum()
                raw_diag_sq = raw_diag.float().square().sum()
                raw_offdiag_sq = (raw_total_sq - raw_diag_sq).clamp_min(0.0)
                raw_within_sq = raw_offdiag_sq.new_zeros(())
                full_d = int(work.size(-1))
                block_d = full_d // 4
                for block_index in range(4):
                    lo = block_index * block_d
                    hi = lo + block_d
                    block = work[:, lo:hi, lo:hi].float()
                    block_diag = block.diagonal(dim1=-2, dim2=-1)
                    raw_within_sq.add_((block.square().sum() - block_diag.square().sum()).clamp_min(0.0))
                raw_cross_sq = (raw_offdiag_sq - raw_within_sq).clamp_min(0.0)
            if R1_DENSE_FULL_ALPHA != 1.0:
                work.mul_(R1_DENSE_FULL_ALPHA)
                work.diagonal(dim1=-2, dim2=-1).copy_(raw_diag)
            diag_full = work.diagonal(dim1=-2, dim2=-1)
            ridge_full = raw_diag.mean(dim=-1) * self.precond_ridge_mult + self.precond_eps
            self._apply_plan["diag_ref_proj_full"].copy_((raw_diag + ridge_full.unsqueeze(-1)).reciprocal())
            diag_full.add_(ridge_full.unsqueeze(-1))
            factor_full, info_full = torch.linalg.cholesky_ex(work, upper=False, check_errors=False)
            torch.cholesky_inverse(factor_full, upper=False, out=self._apply_plan["inv_proj_full"])
            bad_full = info_full != 0
            if bad_full.any():
                self._apply_plan["inv_proj_full"][bad_full].zero_()
                self._apply_plan["inv_proj_full"][bad_full].diagonal(dim1=-2, dim2=-1).fill_(1.0)
            if diagnostic_refresh:
                inv = self._apply_plan["inv_proj_full"].float()
                inv_diag = inv.diagonal(dim1=-2, dim2=-1)
                inv_diag_sq = inv_diag.square().sum().clamp_min(1e-30)
                inv_offdiag_sq = (inv.square().sum() - inv_diag_sq).clamp_min(0.0)
                chol_diag = factor_full.float().diagonal(dim1=-2, dim2=-1).abs()
                chol_spread = chol_diag.amax() / chol_diag.clamp_min(1e-30).amin()
                print(
                    "R1_FULL_ALPHA_K "
                    f"step={{self.global_step}} alpha={{R1_DENSE_FULL_ALPHA:.8g}} "
                    f"raw_cross_to_within={{(raw_cross_sq.sqrt() / raw_within_sq.clamp_min(1e-30).sqrt()).item():.9g}} "
                    f"scaled_offdiag_to_diag={{(R1_DENSE_FULL_ALPHA * raw_offdiag_sq.sqrt() / raw_diag_sq.clamp_min(1e-30).sqrt()).item():.9g}} "
                    f"chol_diag_spread={{chol_spread.item():.9g}} "
                    f"inv_offdiag_to_diag={{(inv_offdiag_sq.sqrt() / inv_diag_sq.sqrt()).item():.9g}} "
                    f"inv_diag_rms={{inv_diag.square().mean().sqrt().item():.9g}} "
                    f"cholesky_failures={{int(bad_full.sum().item())}}"
                )
''',
        "dense-full alpha inverse and diagnostics",
    )

    return _package(repo, method, built.base_script, source)

