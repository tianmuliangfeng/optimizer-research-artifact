"""Build auditable short-run R1 performance sources.

The quality R1 builder remains unchanged.  This module reuses its pinned-source
derivation for the existing methods and adds two performance-only controls:

* ``adamw``: the same GPT with fused AdamW on hidden matrices;
* ``dense_full``: one full 3072 x 3072 right preconditioner for each c_proj.

Every transformation is anchored exactly once.  A changed upstream source
therefore fails closed instead of silently producing a different benchmark.
"""

from __future__ import annotations

import difflib
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path


R1_DIR = Path(__file__).resolve().parents[1] / "15_official_newton_muon_r1"
if str(R1_DIR) not in sys.path:
    sys.path.insert(0, str(R1_DIR))

import r1_source_builder as r1  # noqa: E402


METHODS = ("block4", "diag", "none", "muon", "adamw", "dense_full")
CPROJ_MODE = {
    "block4": "block4",
    "diag": "diag",
    "none": "none",
    "muon": "muon",
    "adamw": "adamw",
    "dense_full": "dense_full",
}


@dataclass(frozen=True)
class PerfSource:
    method: str
    base_script: str
    base_canonical_sha256: str
    derived_sha256: str
    source: str
    unified_diff: str


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"R1-PERF expected one {label!r} anchor, observed {count}"
        )
    return source.replace(old, new, 1)


def _repackage(official_repo: Path, method: str, base_script: str, source: str) -> PerfSource:
    base_path = official_repo / base_script
    base_raw = base_path.read_bytes()
    base_source = r1.canonical_bytes(base_raw).decode("utf-8")
    compile(source, f"<R1-PERF-{method}>", "exec")
    diff = "".join(
        difflib.unified_diff(
            base_source.splitlines(keepends=True),
            source.splitlines(keepends=True),
            fromfile=f"official/{base_script}",
            tofile=f"r1_perf/train_{method}.py",
        )
    )
    return PerfSource(
        method=method,
        base_script=base_script,
        base_canonical_sha256=r1.canonical_sha256(base_raw),
        derived_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        source=source,
        unified_diff=diff,
    )


def _build_adamw(official_repo: Path) -> PerfSource:
    built = r1.build_source(official_repo, "muon")
    source = built.source
    source = _replace_once(
        source,
        '''if R1_METHOD != "muon" or R1_CPROJ_K_MODE != "muon":
    raise ValueError("the R1 Muon source requires method=muon and cproj_k_mode=muon")
''',
        '''if R1_METHOD != "adamw" or R1_CPROJ_K_MODE != "adamw":
    raise ValueError("the R1-PERF AdamW source requires method=adamw and cproj_k_mode=adamw")
''',
        "AdamW mode validation",
    )
    source = _replace_once(
        source,
        "optimizer2 = Muon(raw_model.transformer.h.parameters(), lr=0.1*args.learning_rate, momentum=0.95)\n",
        '''optimizer2 = torch.optim.AdamW(
    raw_model.transformer.h.parameters(), lr=0.000576, betas=(0.9, 0.95),
    weight_decay=args.weight_decay, fused=True
)
''',
        "hidden-matrix AdamW optimizer",
    )
    return _repackage(official_repo, "adamw", built.base_script, source)


DENSE_MLP_INIT = '''        self.c_fc.weight._stats_ref = {"kind": "c_fc", "d": d, "accum": self.fc_xtx_accum, "count": self.fc_xtx_count}
        if R1_CPROJ_K_MODE == "block4":
            self.proj_xtx_accum = nn.Buffer(torch.zeros(4, d, d, dtype=torch.float32), persistent=False)
            self.proj_xtx_tmp = nn.Buffer(torch.empty(4, d, d, dtype=torch.float32), persistent=False)
            self.proj_xtx_count = nn.Buffer(torch.zeros((), dtype=torch.float32), persistent=False)
            self.c_proj.weight._stats_ref = {"kind": "c_proj", "d": d, "accum": self.proj_xtx_accum, "count": self.proj_xtx_count}
        elif R1_CPROJ_K_MODE == "diag":
            self.proj_xtx_accum = nn.Buffer(torch.zeros(4, d, dtype=torch.float32), persistent=False)
            self.proj_xtx_count = nn.Buffer(torch.zeros((), dtype=torch.float32), persistent=False)
            self.c_proj.weight._stats_ref = {"kind": "c_proj_diag", "d": d, "accum": self.proj_xtx_accum, "count": self.proj_xtx_count}
        elif R1_CPROJ_K_MODE == "dense_full":
            self.proj_xtx_accum = nn.Buffer(torch.zeros(4 * d, 4 * d, dtype=torch.float32), persistent=False)
            self.proj_xtx_tmp = nn.Buffer(torch.empty(4 * d, 4 * d, dtype=torch.float32), persistent=False)
            self.proj_xtx_count = nn.Buffer(torch.zeros((), dtype=torch.float32), persistent=False)
            self.c_proj.weight._stats_ref = {"kind": "c_proj_full", "d": d, "accum": self.proj_xtx_accum, "count": self.proj_xtx_count}
        elif R1_CPROJ_K_MODE == "none":
            self.c_proj.weight._stats_ref = None
        else:
            raise ValueError(f"unsupported R1_CPROJ_K_MODE={R1_CPROJ_K_MODE!r}")
'''


DENSE_MLP_FORWARD = '''        if precond_flag and R1_CPROJ_K_MODE != "none":
            z2d = x.flatten(0, -2)
            if R1_CPROJ_K_MODE == "block4":
                torch.ops.nanogpt.accum_xtx_blocks4(z2d, self.proj_xtx_accum, self.proj_xtx_count, self.proj_xtx_tmp)
            elif R1_CPROJ_K_MODE == "dense_full":
                torch.ops.nanogpt.accum_xtx(z2d, self.proj_xtx_accum, self.proj_xtx_count, self.proj_xtx_tmp)
            else:
                torch.ops.nanogpt.accum_xtx_diag4(z2d, self.proj_xtx_accum, self.proj_xtx_count)
'''


DENSE_MLP_APPLY = '''        self.c_fc.weight._stats_ref = {"kind": "c_fc", "d": d, "accum": self.fc_xtx_accum, "count": self.fc_xtx_count}
        if R1_CPROJ_K_MODE == "block4":
            self.c_proj.weight._stats_ref = {"kind": "c_proj", "d": d, "accum": self.proj_xtx_accum, "count": self.proj_xtx_count}
        elif R1_CPROJ_K_MODE == "diag":
            self.c_proj.weight._stats_ref = {"kind": "c_proj_diag", "d": d, "accum": self.proj_xtx_accum, "count": self.proj_xtx_count}
        elif R1_CPROJ_K_MODE == "dense_full":
            self.c_proj.weight._stats_ref = {"kind": "c_proj_full", "d": d, "accum": self.proj_xtx_accum, "count": self.proj_xtx_count}
        else:
            self.c_proj.weight._stats_ref = None
        return self
'''


def _build_dense_full(official_repo: Path) -> PerfSource:
    built = r1.build_source(official_repo, "block4")
    source = built.source
    source = _replace_once(
        source,
        'if R1_METHOD not in ("block4", "none", "diag"):\n',
        'if R1_METHOD not in ("block4", "none", "diag", "dense_full"):\n',
        "dense-full method validation",
    )
    source = _replace_once(source, r1.MLP_INIT_R1, DENSE_MLP_INIT, "dense MLP state")
    source = _replace_once(source, r1.MLP_FORWARD_R1, DENSE_MLP_FORWARD, "dense MLP accumulation")
    source = _replace_once(source, r1.MLP_APPLY_R1, DENSE_MLP_APPLY, "dense MLP apply metadata")

    source = _replace_once(
        source,
        '''        elif kind == "c_proj_diag":
            st["precond_cov"] = torch.full(
                (4, d), self.precond_init_diag, device=p.device, dtype=torch.float32
            )
''',
        '''        elif kind == "c_proj_diag":
            st["precond_cov"] = torch.full(
                (4, d), self.precond_init_diag, device=p.device, dtype=torch.float32
            )
        elif kind == "c_proj_full":
            cov = torch.empty((4 * d, 4 * d), device=p.device, dtype=torch.float32)
            cov.zero_()
            cov.diagonal().fill_(self.precond_init_diag)
            st["precond_cov"] = cov
''',
        "dense-full covariance state",
    )
    source = _replace_once(
        source,
        "        qkv_params, o_params, fc_params, proj_params, proj_diag_params = [], [], [], [], []\n",
        "        qkv_params, o_params, fc_params, proj_params, proj_diag_params, proj_full_params = [], [], [], [], [], []\n",
        "dense-full parameter list",
    )
    source = _replace_once(
        source,
        '''            elif kind == "c_proj_diag":
                proj_diag_params.append(p)
''',
        '''            elif kind == "c_proj_diag":
                proj_diag_params.append(p)
            elif kind == "c_proj_full":
                proj_full_params.append(p)
''',
        "dense-full parameter routing",
    )
    source = _replace_once(
        source,
        '''            "proj_params": proj_params,
            "proj_diag_params": proj_diag_params,

            "g_qkv": alloc_grad_buf(qkv_params, 3),
''',
        '''            "proj_params": proj_params,
            "proj_diag_params": proj_diag_params,
            "proj_full_params": proj_full_params,

            "g_qkv": alloc_grad_buf(qkv_params, 3),
''',
        "dense-full apply-plan params",
    )
    source = _replace_once(
        source,
        '''            "inv_proj4": torch.empty((len(proj_params), 4, d, d), device=dev, dtype=torch.float32) if proj_params else None,
            "inv_proj_diag": torch.empty((len(proj_diag_params), 4, d), device=dev, dtype=torch.float32) if proj_diag_params else None,
            "tmp_proj_blocks": torch.empty((len(proj_params) * 4, d, d), device=dev, dtype=torch.float32) if proj_params else None,
''',
        '''            "inv_proj4": torch.empty((len(proj_params), 4, d, d), device=dev, dtype=torch.float32) if proj_params else None,
            "inv_proj_diag": torch.empty((len(proj_diag_params), 4, d), device=dev, dtype=torch.float32) if proj_diag_params else None,
            "inv_proj_full": torch.empty((len(proj_full_params), 4 * d, 4 * d), device=dev, dtype=torch.float32) if proj_full_params else None,
            "g_proj_full": torch.empty((len(proj_full_params), d, 4 * d), device=dev, dtype=torch.float32) if proj_full_params else None,
            "tmp_proj_full": torch.empty((len(proj_full_params), d, 4 * d), device=dev, dtype=torch.float32) if proj_full_params else None,
            "full_refresh_work": torch.empty((len(proj_full_params), 4 * d, 4 * d), device=dev, dtype=torch.float32) if proj_full_params else None,
            "tmp_proj_blocks": torch.empty((len(proj_params) * 4, d, d), device=dev, dtype=torch.float32) if proj_params else None,
''',
        "dense-full apply-plan buffers",
    )
    source = _replace_once(
        source,
        '''        if plan["inv_proj_diag"] is not None:
            plan["inv_proj_diag"].fill_(1.0)
            for i, p in enumerate(proj_diag_params):
                self.state[p]["precond_inv_apply"] = plan["inv_proj_diag"][i]

        self._precond_ready = True
''',
        '''        if plan["inv_proj_diag"] is not None:
            plan["inv_proj_diag"].fill_(1.0)
            for i, p in enumerate(proj_diag_params):
                self.state[p]["precond_inv_apply"] = plan["inv_proj_diag"][i]

        if plan["inv_proj_full"] is not None:
            plan["inv_proj_full"].zero_()
            plan["inv_proj_full"].diagonal(dim1=-2, dim2=-1).fill_(1.0)
            for i, p in enumerate(proj_full_params):
                self.state[p]["precond_inv_apply"] = plan["inv_proj_full"][i]

        self._precond_ready = True
''',
        "dense-full inverse initialization",
    )
    source = _replace_once(
        source,
        '''        if plan["inv_proj_diag"] is not None:
            for i, p in enumerate(plan["proj_diag_params"]):
                if p.grad is not None:
                    p.grad.view(d, 4, d).mul_(plan["inv_proj_diag"][i].unsqueeze(0))

    @torch.no_grad()
    def _finalize_precond_buffers_(self):
''',
        '''        if plan["inv_proj_diag"] is not None:
            for i, p in enumerate(plan["proj_diag_params"]):
                if p.grad is not None:
                    p.grad.view(d, 4, d).mul_(plan["inv_proj_diag"][i].unsqueeze(0))

        if plan["inv_proj_full"] is not None:
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

    @torch.no_grad()
    def _finalize_precond_buffers_(self):
''',
        "dense-full gradient application",
    )
    source = _replace_once(
        source,
        '            elif kind in ("c_proj", "c_proj_diag"):\n',
        '            elif kind in ("c_proj", "c_proj_diag", "c_proj_full"):\n',
        "dense-full EWMA refresh",
    )
    source = _replace_once(
        source,
        '''        if self._apply_plan["inv_proj_diag"] is not None:
            for p in self._apply_plan["proj_diag_params"]:
                cov = self.state[p]["precond_cov"]
                ridge = cov.mean(dim=-1) * self.precond_ridge_mult + self.precond_eps
                self.state[p]["precond_inv_apply"].copy_((cov + ridge.unsqueeze(-1)).reciprocal())

    def step(self):
''',
        '''        if self._apply_plan["inv_proj_diag"] is not None:
            for p in self._apply_plan["proj_diag_params"]:
                cov = self.state[p]["precond_cov"]
                ridge = cov.mean(dim=-1) * self.precond_ridge_mult + self.precond_eps
                self.state[p]["precond_inv_apply"].copy_((cov + ridge.unsqueeze(-1)).reciprocal())

        if self._apply_plan["inv_proj_full"] is not None:
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

    def step(self):
''',
        "dense-full inverse refresh",
    )
    return _repackage(official_repo, "dense_full", built.base_script, source)


def build_perf_source(official_repo: Path, method: str) -> PerfSource:
    official_repo = official_repo.resolve()
    if method not in METHODS:
        raise ValueError(f"unsupported R1-PERF method {method!r}")
    if method == "adamw":
        return _build_adamw(official_repo)
    if method == "dense_full":
        return _build_dense_full(official_repo)
    built = r1.build_source(official_repo, method)
    return PerfSource(
        method=method,
        base_script=built.base_script,
        base_canonical_sha256=built.base_canonical_sha256,
        derived_sha256=built.derived_sha256,
        source=built.source,
        unified_diff=built.unified_diff,
    )
