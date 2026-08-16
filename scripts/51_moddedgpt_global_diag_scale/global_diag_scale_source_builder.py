#!/usr/bin/env python3
"""Build Experiment-51 global-diagonal sources from the accepted 43/44 trainers."""

from __future__ import annotations

import difflib
import hashlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path


SCRIPT_VERSION = "2026-08-14.4"
HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent


@dataclass(frozen=True)
class DerivedSource:
    scale: str
    source: str
    derived_sha256: str
    parent_sha256: str
    unified_diff: str


SCALE_CONTRACTS = {
    "275m": {
        "parent_dir": "43_newton_muon_record28_275m",
        "builder": "record28_source_builder.py",
        "module": "ex51_record28_source_builder",
        "d_model": 768,
        "layers": 12,
        "attention_layers": 11,
        "k_cov_bytes": 251_904,
        "k_inv_bytes": 251_904,
        "k_state_bytes": 503_808,
        "activation_stat_bytes": 252_088,
        "preconditioned_parameters": 35,
    },
    "455m": {
        "parent_dir": "44_newton_muon_record17_455m",
        "builder": "record17_source_builder.py",
        "module": "ex51_record17_source_builder",
        "d_model": 1024,
        "layers": 16,
        "attention_layers": 15,
        "k_cov_bytes": 450_560,
        "k_inv_bytes": 450_560,
        "k_state_bytes": 901_120,
        "activation_stat_bytes": 450_808,
        "preconditioned_parameters": 47,
    },
}


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Experiment-51 expected one {label}, observed {count}")
    return source.replace(old, new, 1)


def replace_region(source: str, start: str, end: str, replacement: str, label: str) -> str:
    if source.count(start) < 1 or source.count(end) < 1:
        raise RuntimeError(f"Experiment-51 region anchors drifted for {label}")
    begin = source.index(start)
    finish = source.index(end, begin)
    return source[:begin] + replacement + source[finish:]


def load_parent_builder(scale: str):
    cfg = SCALE_CONTRACTS[scale]
    parent = SCRIPTS / cfg["parent_dir"]
    path = parent / cfg["builder"]
    sys.path.insert(0, str(parent))
    try:
        spec = importlib.util.spec_from_file_location(str(cfg["module"]), path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load parent source builder: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


GENERIC_DIAG_RECORD28 = '''
@torch.compile(dynamic=False, fullgraph=True)
def _accum_xtx_diag_global_v51(x_2d: Tensor, accum: Tensor, count: Tensor) -> None:
    with torch.no_grad():
        accum.add_(x_2d.square().mean(dim=0, dtype=torch.float32))
        count.add_(1)

@torch.library.custom_op("nanogpt::accum_xtx_diag_global_v51", mutates_args=("accum", "count"))
def accum_xtx_diag_global_v51(x_2d: Tensor, accum: Tensor, count: Tensor) -> Tensor:
    _accum_xtx_diag_global_v51(x_2d, accum, count)
    return _dummy_scalar_like(accum)

@accum_xtx_diag_global_v51.register_fake
def _(x_2d: Tensor, accum: Tensor, count: Tensor):
    return accum.new_empty(())
'''


def overlay_record28(source: str) -> str:
    source = replace_once(
        source,
        'if RECORD28_METHOD not in ("original_newton_muon", "selective_none", "selective_diag"):',
        'if RECORD28_METHOD not in ("original_newton_muon", "selective_none", "selective_diag", "global_diag"):',
        "Record-28 Newton-family method allowlist",
    )
    source = replace_once(
        source,
        '    "selective_diag": "diag",\n}',
        '    "selective_diag": "diag",\n    "global_diag": "diag",\n}',
        "Record-28 method map",
    )
    op_anchor = '''@accum_xtx_diag4_op_v3.register_fake
def _(x_2d: Tensor, accum: Tensor, count: Tensor):
    return accum.new_empty(())
'''
    source = replace_once(source, op_anchor, op_anchor + GENERIC_DIAG_RECORD28, "Record-28 diag op")

    attention_start = "            # ---- Attention qkvo ----\n"
    fc_start = "            # ---- MLP c_fc ----\n"
    attention = '''            # ---- Attention qkvo: QKV share one diagonal; O owns one diagonal ----
            if getattr(block, "attn", None) is not None:
                p = block.attn.qkvo_w
                d = block.attn.model_dim
                st = self.state[p]
                st["kind"] = "qkvo_global_diag"
                st["d"] = d
                st["cov"] = torch.full((2, d), PRECOND_INIT_DIAG, device=p.device, dtype=torch.float32)
                st["inv"] = torch.ones((2, d), device=p.device, dtype=torch.float32)
                st["precond_buf"] = torch.empty_like(p)
                self._precond_map[p] = dict(
                    kind="qkvo_global_diag", d=d,
                    xtx_qkv=block.attn.precond_xtx_accum,
                    cnt_qkv=block.attn.precond_xtx_count,
                    xtx_o=block.attn.precond_o_xtx_accum,
                    cnt_o=block.attn.precond_o_xtx_count,
                )

'''
    source = replace_region(source, attention_start, fc_start, attention, "Record-28 attention attach")
    proj_start = "            # ---- MLP c_proj: the only experiment-43 Newton-family intervention ----\n"
    fc = '''            # ---- MLP c_fc: input-side diagonal ----
            p = block.mlp.c_fc
            d = block.mlp.model_dim
            st = self.state[p]
            st["kind"] = "c_fc_global_diag"
            st["d"] = d
            st["cov"] = torch.full((d,), PRECOND_INIT_DIAG, device=p.device, dtype=torch.float32)
            st["inv"] = torch.ones((d,), device=p.device, dtype=torch.float32)
            st["precond_buf"] = torch.empty_like(p)
            self._precond_map[p] = dict(
                kind="c_fc_global_diag", d=d,
                xtx=block.mlp.precond_fc_xtx_accum,
                cnt=block.mlp.precond_fc_xtx_count,
            )

'''
    source = replace_region(source, fc_start, proj_start, fc, "Record-28 fc attach")

    q_refresh_start = '        if kind == "qkvo":\n'
    cproj_refresh_start = '        if kind == "c_proj":\n'
    q_refresh = '''        if kind == "qkvo_global_diag":
            qkv = ref["xtx_qkv"]
            out = ref["xtx_o"]
            q_count = ref["cnt_qkv"].to(torch.float32).clamp_min(1.0)
            o_count = ref["cnt_o"].to(torch.float32).clamp_min(1.0)
            cov, inv = st["cov"], st["inv"]
            cov[0].mul_(PRECOND_EWMA).add_(qkv / q_count, alpha=1.0 - PRECOND_EWMA)
            cov[1].mul_(PRECOND_EWMA).add_(out / o_count, alpha=1.0 - PRECOND_EWMA)
            ridge = cov.mean(-1) * 0.2
            inv.copy_((cov + ridge.unsqueeze(-1)).reciprocal())
            qkv.zero_(); out.zero_(); ref["cnt_qkv"].zero_(); ref["cnt_o"].zero_()
            return

'''
    source = replace_region(source, q_refresh_start, cproj_refresh_start, q_refresh, "Record-28 q refresh")
    fc_refresh_start = '        if kind == "c_fc":\n'
    apply_start = "    @torch.no_grad()\n    def _apply_inv"
    fc_refresh = '''        if kind == "c_fc_global_diag":
            xtx, count = ref["xtx"], ref["cnt"]
            cov, inv = st["cov"], st["inv"]
            cov.mul_(PRECOND_EWMA).add_(xtx / count.to(torch.float32).clamp_min(1.0), alpha=1.0 - PRECOND_EWMA)
            ridge = cov.mean() * 0.2
            inv.copy_((cov + ridge).reciprocal())
            xtx.zero_(); count.zero_()
            return

'''
    source = replace_region(source, fc_refresh_start, apply_start, fc_refresh, "Record-28 fc refresh")
    source = replace_once(
        source,
        '        if kind == "qkvo":\n            torch.bmm(grad, inv, out=buf)\n            return buf\n        if kind == "c_fc":\n            torch.mm(inv, grad, out=buf)\n            return buf\n',
        '        if kind == "qkvo_global_diag":\n            buf.copy_(grad)\n            buf[:3].mul_(inv[0])\n            buf[3].mul_(inv[1])\n            return buf\n        if kind == "c_fc_global_diag":\n            buf.copy_(grad)\n            buf.mul_(inv.unsqueeze(-1))\n            return buf\n',
        "Record-28 diagonal apply",
    )
    source = replace_once(
        source,
        '        self.precond_xtx_accum = nn.Buffer(torch.zeros((dim, dim), dtype=torch.float32), persistent=False)\n        self.precond_xtx_tmp   = nn.Buffer(torch.empty((dim, dim), dtype=torch.float32), persistent=False)\n',
        '        self.precond_xtx_accum = nn.Buffer(torch.zeros((dim,), dtype=torch.float32), persistent=False)\n',
        "Record-28 q stats",
    )
    source = replace_once(
        source,
        '        self.precond_o_xtx_accum = nn.Buffer(torch.zeros((dim, dim), dtype=torch.float32), persistent=False)\n        self.precond_o_xtx_tmp   = nn.Buffer(torch.empty((dim, dim), dtype=torch.float32), persistent=False)\n',
        '        self.precond_o_xtx_accum = nn.Buffer(torch.zeros((dim,), dtype=torch.float32), persistent=False)\n',
        "Record-28 o stats",
    )
    source = replace_once(source, 'torch.ops.nanogpt.accum_xtx_v3(x2d, self.precond_xtx_accum, self.precond_xtx_count, self.precond_xtx_tmp)', 'torch.ops.nanogpt.accum_xtx_diag_global_v51(x2d, self.precond_xtx_accum, self.precond_xtx_count)', "Record-28 q accumulation")
    source = replace_once(source, 'torch.ops.nanogpt.accum_xtx_v3(y2d, self.precond_o_xtx_accum, self.precond_o_xtx_count, self.precond_o_xtx_tmp)', 'torch.ops.nanogpt.accum_xtx_diag_global_v51(y2d, self.precond_o_xtx_accum, self.precond_o_xtx_count)', "Record-28 o accumulation")
    source = replace_once(
        source,
        '        self.precond_fc_xtx_accum = nn.Buffer(torch.zeros((dim, dim), dtype=torch.float32), persistent=False)\n        self.precond_fc_xtx_tmp   = nn.Buffer(torch.empty((dim, dim), dtype=torch.float32), persistent=False)\n',
        '        self.precond_fc_xtx_accum = nn.Buffer(torch.zeros((dim,), dtype=torch.float32), persistent=False)\n',
        "Record-28 fc stats",
    )
    source = replace_once(source, 'torch.ops.nanogpt.accum_xtx_v3(x2d, self.precond_fc_xtx_accum, self.precond_fc_xtx_count, self.precond_fc_xtx_tmp)', 'torch.ops.nanogpt.accum_xtx_diag_global_v51(x2d, self.precond_fc_xtx_accum, self.precond_fc_xtx_count)', "Record-28 fc accumulation")
    attach = "optimizer2.attach_preconditioner(model)\n"
    audit = attach + '''if RECORD28_METHOD == "global_diag":
    if len(optimizer2._precond_map) != 35:
        raise RuntimeError("Experiment-51 275M route expected 35 preconditioned parameters")
    if any("Kbuf" in optimizer2.state[p] for p in optimizer2._precond_map):
        raise RuntimeError("Experiment-51 275M global diag allocated dense K workspace")
    print0("EX51_GLOBAL_DIAG_ROUTE scale=275m parameters=35 dense_activation_workspace=0", console=True)
'''
    return replace_once(source, attach, audit, "Record-28 runtime route audit")


GENERIC_DIAG_RECORD17 = '''
@torch.compile(dynamic=False, fullgraph=True)
def record17_accum_xtx_diag_global_v1(x_2d: Tensor, accum: Tensor, count: Tensor) -> Tensor:
    accum.add_(x_2d.square().mean(dim=0, dtype=torch.float32))
    count.add_(1)
    # Keep the scalar construction inside the compiled graph.  Record-17 has
    # no `_record17_dummy_scalar` helper; referring to one only fails when
    # Dynamo captures the first real 455M batch, which ordinary Python
    # compilation cannot detect.
    return accum.new_empty(())

@torch.library.custom_op("record17::accum_xtx_diag_global_v1", mutates_args=("accum", "count"))
@torch.no_grad()
def record17_accum_xtx_diag_global_op_v1(x_2d: Tensor, accum: Tensor, count: Tensor) -> Tensor:
    return record17_accum_xtx_diag_global_v1(x_2d, accum, count)

@record17_accum_xtx_diag_global_op_v1.register_fake
def _(x_2d: Tensor, accum: Tensor, count: Tensor):
    return accum.new_empty(())
'''


def overlay_record17(source: str) -> str:
    source = replace_once(source, '    "selective_diag": "diag",\n}', '    "selective_diag": "diag",\n    "global_diag": "diag",\n}', "Record-17 method map")
    anchor = '@record17_accum_xtx_diag4_v1.register_fake\ndef _(x_2d: Tensor, accum: Tensor, count: Tensor):\n    return accum.new_empty(())\n'
    source = replace_once(source, anchor, anchor + GENERIC_DIAG_RECORD17, "Record-17 diag op")

    attn_start = "            if block.attn is not None:\n"
    fc_start = "            p = block.mlp.fc_w\n"
    attn = '''            if block.attn is not None:
                p = block.attn.qkvo_w
                d = p.size(-1)
                state = self.state[p]
                state["kind"] = "qkvo_global_diag"
                state["d"] = d
                state["cov"] = torch.full((2, d), RECORD17_PRECOND_INIT_DIAG, device=p.device, dtype=torch.float32)
                state["inv"] = torch.ones((2, d), device=p.device, dtype=torch.float32)
                state["precond_buf"] = torch.empty(p.shape, device=p.device, dtype=torch.float32)
                self._precond_map[p] = {
                    "kind": "qkvo_global_diag", "d": d,
                    "xtx_qkv": block.attn.precond_qkv_xtx_accum,
                    "cnt_qkv": block.attn.precond_qkv_xtx_count,
                    "xtx_o": block.attn.precond_o_xtx_accum,
                    "cnt_o": block.attn.precond_o_xtx_count,
                }

'''
    source = replace_region(source, attn_start, fc_start, attn, "Record-17 attention attach")
    proj_start = "            p = block.mlp.proj_w\n"
    fc = '''            p = block.mlp.fc_w
            d = p.size(-1)
            state = self.state[p]
            state["kind"] = "fc_w_global_diag"
            state["d"] = d
            state["cov"] = torch.full((d,), RECORD17_PRECOND_INIT_DIAG, device=p.device, dtype=torch.float32)
            state["inv"] = torch.ones((d,), device=p.device, dtype=torch.float32)
            state["precond_buf"] = torch.empty(p.shape, device=p.device, dtype=torch.float32)
            self._precond_map[p] = {
                "kind": "fc_w_global_diag", "d": d,
                "xtx": block.mlp.precond_fc_xtx_accum,
                "cnt": block.mlp.precond_fc_xtx_count,
            }

'''
    source = replace_region(source, fc_start, proj_start, fc, "Record-17 fc attach")
    q_start = '        if kind == "qkvo":\n'
    generic_start = '        xtx = ref["xtx"]\n'
    q_refresh = '''        if kind == "qkvo_global_diag":
            qkv, out = ref["xtx_qkv"], ref["xtx_o"]
            if ref["cnt_qkv"].item() == 0 or ref["cnt_o"].item() == 0:
                raise RuntimeError("global-diag qkvo refresh has no samples")
            cov[0].mul_(RECORD17_PRECOND_EWMA).add_(qkv / ref["cnt_qkv"], alpha=1.0 - RECORD17_PRECOND_EWMA)
            cov[1].mul_(RECORD17_PRECOND_EWMA).add_(out / ref["cnt_o"], alpha=1.0 - RECORD17_PRECOND_EWMA)
            ridge = cov.mean(-1) * RECORD17_RIDGE_MULT
            inv.copy_((cov + ridge.unsqueeze(-1)).reciprocal())
            qkv.zero_(); out.zero_(); ref["cnt_qkv"].zero_(); ref["cnt_o"].zero_()
            return

'''
    source = replace_region(source, q_start, generic_start, q_refresh, "Record-17 q refresh")
    source = replace_once(source, '        if kind == "proj_w_diag":\n', '        if kind in ("proj_w_diag", "fc_w_global_diag"):\n', "Record-17 generic diagonal refresh")
    source = replace_once(
        source,
        '        if kind == "qkvo":\n            torch.bmm(grad_fp32, inv, out=buf)\n        elif kind == "fc_w":\n            # fc_w is (4d,d): activation covariance acts on the right.\n            torch.mm(grad_fp32, inv, out=buf)\n',
        '        if kind == "qkvo_global_diag":\n            buf.copy_(grad_fp32)\n            buf[:3].mul_(inv[0])\n            buf[3].mul_(inv[1])\n        elif kind == "fc_w_global_diag":\n            buf.copy_(grad_fp32)\n            buf.mul_(inv)\n',
        "Record-17 diagonal apply",
    )
    source = replace_once(source, 'torch.zeros((dim, dim), dtype=torch.float32), persistent=False\n            )\n            self.precond_qkv_xtx_tmp = nn.Buffer(\n                torch.empty((dim, dim), dtype=torch.float32), persistent=False\n            )', 'torch.zeros((dim,), dtype=torch.float32), persistent=False\n            )', "Record-17 q stats")
    source = replace_once(source, 'torch.zeros((dim, dim), dtype=torch.float32), persistent=False\n            )\n            self.precond_o_xtx_tmp = nn.Buffer(\n                torch.empty((dim, dim), dtype=torch.float32), persistent=False\n            )', 'torch.zeros((dim,), dtype=torch.float32), persistent=False\n            )', "Record-17 o stats")
    source = replace_once(source, 'torch.ops.record17.accum_xtx_dense_v1(\n                x2d,\n                self.precond_qkv_xtx_accum,\n                self.precond_qkv_xtx_count,\n                self.precond_qkv_xtx_tmp,\n            )', 'torch.ops.record17.accum_xtx_diag_global_v1(\n                x2d, self.precond_qkv_xtx_accum, self.precond_qkv_xtx_count\n            )', "Record-17 q accumulation")
    source = replace_once(source, 'torch.ops.record17.accum_xtx_dense_v1(\n                y2d,\n                self.precond_o_xtx_accum,\n                self.precond_o_xtx_count,\n                self.precond_o_xtx_tmp,\n            )', 'torch.ops.record17.accum_xtx_diag_global_v1(\n                y2d, self.precond_o_xtx_accum, self.precond_o_xtx_count\n            )', "Record-17 o accumulation")
    source = replace_once(source, 'torch.zeros((dim, dim), dtype=torch.float32), persistent=False\n            )\n            self.precond_fc_xtx_tmp = nn.Buffer(\n                torch.empty((dim, dim), dtype=torch.float32), persistent=False\n            )', 'torch.zeros((dim,), dtype=torch.float32), persistent=False\n            )', "Record-17 fc stats")
    source = replace_once(source, 'torch.ops.record17.accum_xtx_dense_v1(\n                x2d,\n                self.precond_fc_xtx_accum,\n                self.precond_fc_xtx_count,\n                self.precond_fc_xtx_tmp,\n            )', 'torch.ops.record17.accum_xtx_diag_global_v1(\n                x2d, self.precond_fc_xtx_accum, self.precond_fc_xtx_count\n            )', "Record-17 fc accumulation")
    attach = 'if RECORD17_NEWTON_ACTIVE:\n    optimizer2.attach_preconditioner(model)\n'
    audit = attach + '''if RECORD17_METHOD == "global_diag":
    if len(optimizer2._precond_map) != 47:
        raise RuntimeError("Experiment-51 455M route expected 47 preconditioned parameters")
    if any("Kbuf" in optimizer2.state[p] for p in optimizer2._precond_map):
        raise RuntimeError("Experiment-51 455M global diag allocated dense K workspace")
    print0("EX51_GLOBAL_DIAG_ROUTE scale=455m parameters=47 dense_activation_workspace=0", console=True)
'''
    return replace_once(source, attach, audit, "Record-17 runtime route audit")


def assert_source(scale: str, source: str) -> None:
    required = [
        '"global_diag": "diag"',
        "EX51_GLOBAL_DIAG_ROUTE",
        "dense_activation_workspace=0",
        "global_diag",
    ]
    missing = [value for value in required if value not in source]
    if missing:
        raise RuntimeError(f"{scale} global-diag source missing {missing}")
    if scale == "275m" and (
        '"selective_diag", "global_diag")' not in source
        or '"global_diag": "diag"' not in source
    ):
        raise RuntimeError("275M global-diag source is not admitted by the method contract")
    if "return _record17_dummy_scalar(" in source:
        raise RuntimeError("455M global-diag source retained an unresolved dummy helper")
    compile(source, f"<experiment51-{scale}>", "exec")


def build_source(official_repo: Path, scale: str) -> DerivedSource:
    if scale not in SCALE_CONTRACTS:
        raise ValueError(scale)
    parent = load_parent_builder(scale).build_source(official_repo, "selective_diag")
    source = overlay_record28(parent.source) if scale == "275m" else overlay_record17(parent.source)
    assert_source(scale, source)
    diff = "".join(difflib.unified_diff(parent.source.splitlines(True), source.splitlines(True), fromfile=f"accepted_{scale}/selective_diag", tofile=f"experiment51/{scale}_global_diag"))
    return DerivedSource(scale, source, hashlib.sha256(source.encode()).hexdigest(), parent.derived_sha256, diff)


def expected_memory(scale: str) -> dict[str, int]:
    cfg = SCALE_CONTRACTS[scale]
    return {key: int(cfg[key]) for key in ("k_cov_bytes", "k_inv_bytes", "k_state_bytes", "activation_stat_bytes")}


def self_test() -> None:
    for scale, cfg in SCALE_CONTRACTS.items():
        factors = 2 * cfg["attention_layers"] + cfg["layers"] + 4 * cfg["layers"]
        elements = factors * cfg["d_model"]
        assert cfg["k_cov_bytes"] == elements * 4
        assert cfg["k_state_bytes"] == elements * 8
        counts = 2 * cfg["attention_layers"] + 2 * cfg["layers"]
        assert cfg["activation_stat_bytes"] == elements * 4 + counts * 4
