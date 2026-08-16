#!/usr/bin/env python3
"""Derive the Experiment-50 all-activation-diagonal R1 training source."""

from __future__ import annotations

import difflib
import hashlib
import sys
from pathlib import Path


SCRIPT_VERSION = "2026-08-14.1"
SCRIPT_DIR = Path(__file__).resolve().parent
R1_DIR = SCRIPT_DIR.parent / "15_official_newton_muon_r1"
sys.path.insert(0, str(R1_DIR))

from r1_source_builder import DerivedSource, build_source


EXPECTED_K_COV_BYTES = 258_048
EXPECTED_K_INV_BYTES = 258_048
EXPECTED_K_STATE_BYTES = 516_096
EXPECTED_ACTIVATION_STAT_BYTES = 258_240


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"global-diag source expected one {label!r} anchor, observed {count}"
        )
    return source.replace(old, new, 1)


CUSTOM_OP_ANCHOR = '''@accum_xtx_diag4_op.register_fake
def accum_xtx_diag4_fake(x_2d: Tensor, accum: Tensor, count: Tensor):
    return accum.new_empty(())
'''

CUSTOM_OP_REPLACEMENT = CUSTOM_OP_ANCHOR + '''
@torch.compile
def _accum_xtx_diag_impl(x_2d: Tensor, accum: Tensor, count: Tensor) -> Tensor:
    accum.add_(x_2d.square().mean(dim=0))
    count.add_(1.0)
    return _dummy_scalar_like(accum)

@torch.library.custom_op("nanogpt::accum_xtx_diag", mutates_args=("accum", "count"))
@torch.no_grad()
def accum_xtx_diag_op(x_2d: Tensor, accum: Tensor, count: Tensor) -> Tensor:
    return _accum_xtx_diag_impl(x_2d, accum, count)

@accum_xtx_diag_op.register_fake
def accum_xtx_diag_fake(x_2d: Tensor, accum: Tensor, count: Tensor):
    return accum.new_empty(())
'''

INIT_STATE_OLD = '''        if kind in ("qkv", "o", "c_fc"):
            st["precond_cov"] = _fp32_mat()
        elif kind == "c_proj":
'''

INIT_STATE_NEW = '''        if kind in ("qkv_diag", "o_diag", "c_fc_diag"):
            st["precond_cov"] = torch.full(
                (d,), self.precond_init_diag, device=p.device, dtype=torch.float32
            )
        elif kind in ("qkv", "o", "c_fc"):
            st["precond_cov"] = _fp32_mat()
        elif kind == "c_proj":
'''

APPLY_DIAG_OLD = '''        if plan["inv_proj_diag"] is not None:
            for i, p in enumerate(plan["proj_diag_params"]):
                if p.grad is not None:
                    p.grad.view(d, 4, d).mul_(plan["inv_proj_diag"][i].unsqueeze(0))
'''

APPLY_DIAG_NEW = '''        if plan["inv_input_diag"] is not None:
            for i, p in enumerate(plan["input_diag_params"]):
                if p.grad is not None:
                    p.grad.mul_(plan["inv_input_diag"][i].unsqueeze(0))

        if plan["inv_proj_diag"] is not None:
            for i, p in enumerate(plan["proj_diag_params"]):
                if p.grad is not None:
                    p.grad.view(d, 4, d).mul_(plan["inv_proj_diag"][i].unsqueeze(0))
'''

PARAM_LIST_OLD = '''        qkv_params, o_params, fc_params, proj_params, proj_diag_params = [], [], [], [], []
'''
PARAM_LIST_NEW = '''        qkv_params, o_params, fc_params, proj_params, proj_diag_params = [], [], [], [], []
        input_diag_params = []
'''

CATEGORIZE_OLD = '''            if kind == "qkv":
                qkv_params.append(p)
            elif kind == "o":
                o_params.append(p)
            elif kind == "c_fc":
                fc_params.append(p)
            elif kind == "c_proj":
                proj_params.append(p)
            elif kind == "c_proj_diag":
                proj_diag_params.append(p)
'''

CATEGORIZE_NEW = '''            if kind in ("qkv_diag", "o_diag", "c_fc_diag"):
                input_diag_params.append(p)
            elif kind == "qkv":
                qkv_params.append(p)
            elif kind == "o":
                o_params.append(p)
            elif kind == "c_fc":
                fc_params.append(p)
            elif kind == "c_proj":
                proj_params.append(p)
            elif kind == "c_proj_diag":
                proj_diag_params.append(p)
'''

DEVICE_OLD = '''        dev = refresh_map[0][0].device if refresh_map else torch.device("cuda")
'''
DEVICE_NEW = '''        all_diag_params = [*input_diag_params, *proj_diag_params]
        if not refresh_map and not all_diag_params:
            raise RuntimeError("global-diag route found no preconditioned parameters")
        dev = refresh_map[0][0].device if refresh_map else all_diag_params[0].device
'''

PLAN_OLD = '''            "proj_diag_params": proj_diag_params,

            "g_qkv": alloc_grad_buf(qkv_params, 3),
'''
PLAN_NEW = '''            "proj_diag_params": proj_diag_params,
            "input_diag_params": input_diag_params,

            "g_qkv": alloc_grad_buf(qkv_params, 3),
'''

PLAN_INV_OLD = '''            "inv_proj_diag": torch.empty((len(proj_diag_params), 4, d), device=dev, dtype=torch.float32) if proj_diag_params else None,
'''
PLAN_INV_NEW = '''            "inv_input_diag": torch.empty((len(input_diag_params), d), device=dev, dtype=torch.float32) if input_diag_params else None,
            "inv_proj_diag": torch.empty((len(proj_diag_params), 4, d), device=dev, dtype=torch.float32) if proj_diag_params else None,
'''

INIT_INV_OLD = '''        if plan["inv_proj_diag"] is not None:
            plan["inv_proj_diag"].fill_(1.0)
            for i, p in enumerate(proj_diag_params):
                self.state[p]["precond_inv_apply"] = plan["inv_proj_diag"][i]
'''
INIT_INV_NEW = '''        if plan["inv_input_diag"] is not None:
            plan["inv_input_diag"].fill_(1.0)
            for i, p in enumerate(input_diag_params):
                self.state[p]["precond_inv_apply"] = plan["inv_input_diag"][i]

        if plan["inv_proj_diag"] is not None:
            plan["inv_proj_diag"].fill_(1.0)
            for i, p in enumerate(proj_diag_params):
                self.state[p]["precond_inv_apply"] = plan["inv_proj_diag"][i]
'''

EWMA_OLD = '''            if kind in ("qkv", "o", "c_fc"):
                st["precond_cov"].lerp_(stref["accum"] / cnt.clamp_min(1.0), w)
            elif kind in ("c_proj", "c_proj_diag"):
'''
EWMA_NEW = '''            if kind in ("qkv", "o", "c_fc", "qkv_diag", "o_diag", "c_fc_diag"):
                st["precond_cov"].lerp_(stref["accum"] / cnt.clamp_min(1.0), w)
            elif kind in ("c_proj", "c_proj_diag"):
'''

INVERSE_GATE_OLD = '''        if not do_inverse:
            return
        if self._refresh_K is None or not self._refresh_map:
            return
'''
INVERSE_GATE_NEW = '''        if not do_inverse:
            return

        if self._apply_plan["inv_input_diag"] is not None:
            for p in self._apply_plan["input_diag_params"]:
                cov = self.state[p]["precond_cov"]
                ridge = cov.mean() * self.precond_ridge_mult + self.precond_eps
                self.state[p]["precond_inv_apply"].copy_((cov + ridge).reciprocal())

        if self._apply_plan["inv_proj_diag"] is not None:
            for p in self._apply_plan["proj_diag_params"]:
                cov = self.state[p]["precond_cov"]
                ridge = cov.mean(dim=-1) * self.precond_ridge_mult + self.precond_eps
                self.state[p]["precond_inv_apply"].copy_((cov + ridge.unsqueeze(-1)).reciprocal())

        if self._refresh_K is None or not self._refresh_map:
            return
'''

LATE_DIAG_OLD = '''        if self._apply_plan["inv_proj_diag"] is not None:
            for p in self._apply_plan["proj_diag_params"]:
                cov = self.state[p]["precond_cov"]
                ridge = cov.mean(dim=-1) * self.precond_ridge_mult + self.precond_eps
                self.state[p]["precond_inv_apply"].copy_((cov + ridge.unsqueeze(-1)).reciprocal())
'''

ATTENTION_OLD = '''        d = self.n_embd
        self.qkv_xtx_accum = nn.Buffer(torch.zeros(d, d, dtype=torch.float32), persistent=False)
        self.o_xtx_accum   = nn.Buffer(torch.zeros(d, d, dtype=torch.float32), persistent=False)
        self.xtx_tmp       = nn.Buffer(torch.empty(d, d, dtype=torch.float32), persistent=False)
        self.qkv_xtx_count = nn.Buffer(torch.zeros((), dtype=torch.float32), persistent=False)
        self.o_xtx_count   = nn.Buffer(torch.zeros((), dtype=torch.float32), persistent=False)

        self.c_attn.weight._stats_ref = {"kind": "qkv", "d": d, "accum": self.qkv_xtx_accum, "count": self.qkv_xtx_count}
        self.c_proj.weight._stats_ref = {"kind": "o",   "d": d, "accum": self.o_xtx_accum,   "count": self.o_xtx_count}
'''
ATTENTION_NEW = '''        d = self.n_embd
        self.qkv_xtx_accum = nn.Buffer(torch.zeros(d, dtype=torch.float32), persistent=False)
        self.o_xtx_accum = nn.Buffer(torch.zeros(d, dtype=torch.float32), persistent=False)
        self.qkv_xtx_count = nn.Buffer(torch.zeros((), dtype=torch.float32), persistent=False)
        self.o_xtx_count = nn.Buffer(torch.zeros((), dtype=torch.float32), persistent=False)

        self.c_attn.weight._stats_ref = {"kind": "qkv_diag", "d": d, "accum": self.qkv_xtx_accum, "count": self.qkv_xtx_count}
        self.c_proj.weight._stats_ref = {"kind": "o_diag", "d": d, "accum": self.o_xtx_accum, "count": self.o_xtx_count}
'''

ATTENTION_QKV_OLD = '''            torch.ops.nanogpt.accum_xtx(x2d, self.qkv_xtx_accum, self.qkv_xtx_count, self.xtx_tmp)
'''
ATTENTION_QKV_NEW = '''            torch.ops.nanogpt.accum_xtx_diag(x2d, self.qkv_xtx_accum, self.qkv_xtx_count)
'''
ATTENTION_O_OLD = '''            torch.ops.nanogpt.accum_xtx(y2d, self.o_xtx_accum, self.o_xtx_count, self.xtx_tmp)
'''
ATTENTION_O_NEW = '''            torch.ops.nanogpt.accum_xtx_diag(y2d, self.o_xtx_accum, self.o_xtx_count)
'''
ATTENTION_APPLY_OLD = '''        self.c_attn.weight._stats_ref = {"kind": "qkv", "d": d, "accum": self.qkv_xtx_accum, "count": self.qkv_xtx_count}
        self.c_proj.weight._stats_ref = {"kind": "o",   "d": d, "accum": self.o_xtx_accum,   "count": self.o_xtx_count}
'''
ATTENTION_APPLY_NEW = '''        self.c_attn.weight._stats_ref = {"kind": "qkv_diag", "d": d, "accum": self.qkv_xtx_accum, "count": self.qkv_xtx_count}
        self.c_proj.weight._stats_ref = {"kind": "o_diag", "d": d, "accum": self.o_xtx_accum, "count": self.o_xtx_count}
'''

FC_INIT_OLD = '''        self.fc_xtx_accum  = nn.Buffer(torch.zeros(d, d, dtype=torch.float32), persistent=False)
        self.fc_xtx_tmp    = nn.Buffer(torch.empty(d, d, dtype=torch.float32), persistent=False)
        self.fc_xtx_count  = nn.Buffer(torch.zeros((), dtype=torch.float32), persistent=False)

        self.c_fc.weight._stats_ref = {"kind": "c_fc", "d": d, "accum": self.fc_xtx_accum, "count": self.fc_xtx_count}
'''
FC_INIT_NEW = '''        self.fc_xtx_accum = nn.Buffer(torch.zeros(d, dtype=torch.float32), persistent=False)
        self.fc_xtx_count = nn.Buffer(torch.zeros((), dtype=torch.float32), persistent=False)

        self.c_fc.weight._stats_ref = {"kind": "c_fc_diag", "d": d, "accum": self.fc_xtx_accum, "count": self.fc_xtx_count}
'''
FC_FORWARD_OLD = '''            torch.ops.nanogpt.accum_xtx(x2d, self.fc_xtx_accum, self.fc_xtx_count, self.fc_xtx_tmp)
'''
FC_FORWARD_NEW = '''            torch.ops.nanogpt.accum_xtx_diag(x2d, self.fc_xtx_accum, self.fc_xtx_count)
'''
FC_APPLY_OLD = '''        self.c_fc.weight._stats_ref = {"kind": "c_fc", "d": d, "accum": self.fc_xtx_accum, "count": self.fc_xtx_count}
'''
FC_APPLY_NEW = '''        self.c_fc.weight._stats_ref = {"kind": "c_fc_diag", "d": d, "accum": self.fc_xtx_accum, "count": self.fc_xtx_count}
'''

ENV_OLD = '''R1_METHOD = os.environ["R1_METHOD"]
R1_CPROJ_K_MODE = os.environ["R1_CPROJ_K_MODE"]
R1_SEED = int(os.environ["R1_SEED"])
'''
ENV_NEW = '''R1_METHOD = os.environ["R1_METHOD"]
R1_CPROJ_K_MODE = os.environ["R1_CPROJ_K_MODE"]
R1_GLOBAL_DIAG = os.environ["R1_GLOBAL_DIAG"] == "1"
R1_SEED = int(os.environ["R1_SEED"])
'''
VALIDATION_OLD = '''if R1_METHOD not in ("block4", "none", "diag"):
    raise ValueError(f"invalid Newton R1 method={R1_METHOD!r}")
if R1_CPROJ_K_MODE != R1_METHOD:
    raise ValueError("Newton R1 method and cproj_k_mode must match")
'''
VALIDATION_NEW = '''if R1_METHOD != "global_diag":
    raise ValueError(f"invalid Experiment-50 method={R1_METHOD!r}")
if R1_CPROJ_K_MODE != "diag":
    raise ValueError("global-diag must retain the audited c_proj diag route")
if not R1_GLOBAL_DIAG:
    raise ValueError("R1_GLOBAL_DIAG=1 is required")
'''
METADATA_OLD = '''print(f"R1_METADATA method={R1_METHOD} cproj_k_mode={R1_CPROJ_K_MODE} seed={R1_SEED} init_sha256={r1_init_sha256}")
'''
METADATA_NEW = METADATA_OLD + '''print("R1_GLOBAL_DIAG_METADATA route=all_eligible_activation_diagonal dense_workspace=0")
'''

MEMORY_REPORT_OLD = '''    inv_keys = {_r1_storage_key(tensor) for tensor in inv_tensors}
    workspace_tensors = [
        tensor for tensor in [*plan_tensors, *activation_workspace_tensors]
        if _r1_storage_key(tensor) not in inv_keys
    ]
'''
MEMORY_REPORT_NEW = '''    inv_keys = {_r1_storage_key(tensor) for tensor in inv_tensors}
    model_parameter_keys = {_r1_storage_key(parameter) for parameter in model.parameters()}
    workspace_tensors = [
        tensor for tensor in [*plan_tensors, *activation_workspace_tensors]
        if _r1_storage_key(tensor) not in inv_keys
        and _r1_storage_key(tensor) not in model_parameter_keys
    ]
'''

OPTIMIZER_ATTACH_OLD = '''optimizer2 = Muon(raw_model.transformer.h.parameters(), lr=0.1*args.learning_rate, momentum=0.95)
optimizer2.attach_preconditioner()
optimizers = [optimizer1, optimizer2]
'''
OPTIMIZER_ATTACH_NEW = '''optimizer2 = Muon(raw_model.transformer.h.parameters(), lr=0.1*args.learning_rate, momentum=0.95)
optimizer2.attach_preconditioner()
if optimizer2._refresh_map or optimizer2._refresh_K is not None:
    raise RuntimeError("global-diag unexpectedly constructed a dense refresh route")
if optimizer2._apply_plan is None:
    raise RuntimeError("global-diag apply plan was not constructed")
if len(optimizer2._apply_plan["input_diag_params"]) != 36:
    raise RuntimeError("global-diag expected 36 qkv/o/c_fc diagonal parameters")
if len(optimizer2._apply_plan["proj_diag_params"]) != 12:
    raise RuntimeError("global-diag expected 12 c_proj diagonal parameters")
print("R1_GLOBAL_DIAG_ROUTE input_diag_params=36 proj_diag_params=12 dense_refresh_blocks=0")
optimizers = [optimizer1, optimizer2]
'''


def apply_global_diag_overlay(source: str) -> str:
    replacements = (
        (CUSTOM_OP_ANCHOR, CUSTOM_OP_REPLACEMENT, "generic diagonal custom op"),
        (INIT_STATE_OLD, INIT_STATE_NEW, "diagonal optimizer state"),
        (APPLY_DIAG_OLD, APPLY_DIAG_NEW, "diagonal gradient application"),
        (PARAM_LIST_OLD, PARAM_LIST_NEW, "diagonal parameter list"),
        (CATEGORIZE_OLD, CATEGORIZE_NEW, "diagonal parameter categorization"),
        (DEVICE_OLD, DEVICE_NEW, "device selection without dense map"),
        (PLAN_OLD, PLAN_NEW, "diagonal apply plan"),
        (PLAN_INV_OLD, PLAN_INV_NEW, "diagonal inverse allocation"),
        (INIT_INV_OLD, INIT_INV_NEW, "diagonal inverse initialization"),
        (EWMA_OLD, EWMA_NEW, "diagonal covariance EWMA"),
        (LATE_DIAG_OLD, "", "remove unreachable duplicate c_proj diagonal refresh"),
        (INVERSE_GATE_OLD, INVERSE_GATE_NEW, "diagonal inverse refresh"),
        (ATTENTION_OLD, ATTENTION_NEW, "attention diagonal statistics"),
        (ATTENTION_QKV_OLD, ATTENTION_QKV_NEW, "QKV diagonal accumulation"),
        (ATTENTION_O_OLD, ATTENTION_O_NEW, "attention-output diagonal accumulation"),
        (ATTENTION_APPLY_OLD, ATTENTION_APPLY_NEW, "attention stats repair"),
        (FC_INIT_OLD, FC_INIT_NEW, "c_fc diagonal statistics"),
        (FC_FORWARD_OLD, FC_FORWARD_NEW, "c_fc diagonal accumulation"),
        (FC_APPLY_OLD, FC_APPLY_NEW, "c_fc stats repair"),
        (ENV_OLD, ENV_NEW, "global-diag environment"),
        (VALIDATION_OLD, VALIDATION_NEW, "global-diag validation"),
        (METADATA_OLD, METADATA_NEW, "global-diag metadata"),
        (MEMORY_REPORT_OLD, MEMORY_REPORT_NEW, "workspace memory alias exclusion"),
        (OPTIMIZER_ATTACH_OLD, OPTIMIZER_ATTACH_NEW, "runtime route audit"),
    )
    for old, new, label in replacements:
        source = replace_once(source, old, new, label)
    compile(source, "<R1-global-activation-diag>", "exec")
    assert_global_diag_source_contract(source)
    return source


def assert_global_diag_source_contract(source: str) -> None:
    required = (
        'R1_GLOBAL_DIAG = os.environ["R1_GLOBAL_DIAG"] == "1"',
        '"kind": "qkv_diag"',
        '"kind": "o_diag"',
        '"kind": "c_fc_diag"',
        '"kind": "c_proj_diag"',
        'torch.ops.nanogpt.accum_xtx_diag(',
        'torch.ops.nanogpt.accum_xtx_diag4(',
        '"inv_input_diag"',
        "R1_GLOBAL_DIAG_METADATA route=all_eligible_activation_diagonal",
        "model_parameter_keys = {_r1_storage_key(parameter) for parameter in model.parameters()}",
        "R1_GLOBAL_DIAG_ROUTE input_diag_params=36 proj_diag_params=12 dense_refresh_blocks=0",
    )
    missing = [item for item in required if item not in source]
    if missing:
        raise RuntimeError(f"global-diag source missing contract anchors: {missing}")
    forbidden_runtime_allocations = (
        "self.qkv_xtx_accum = nn.Buffer(torch.zeros(d, d,",
        "self.o_xtx_accum   = nn.Buffer(torch.zeros(d, d,",
        "self.fc_xtx_accum  = nn.Buffer(torch.zeros(d, d,",
        "self.fc_xtx_tmp",
        "self.xtx_tmp",
    )
    present = [item for item in forbidden_runtime_allocations if item in source]
    if present:
        raise RuntimeError(f"global-diag source retains dense activation state: {present}")
    if source.count("R1_GLOBAL_DIAG_METADATA") != 1:
        raise RuntimeError("global-diag metadata line count changed")


def build_global_diag_source(official_repo: Path) -> DerivedSource:
    selective_diag = build_source(official_repo, "diag")
    derived_source = apply_global_diag_overlay(selective_diag.source)
    official_source = (
        (official_repo / selective_diag.base_script)
        .read_bytes()
        .replace(b"\r\n", b"\n")
        .decode("utf-8")
    )
    diff = "".join(
        difflib.unified_diff(
            official_source.splitlines(keepends=True),
            derived_source.splitlines(keepends=True),
            fromfile=f"official/{selective_diag.base_script}",
            tofile="r1_global_diag/train_r1_global_diag.py",
        )
    )
    return DerivedSource(
        method="global_diag",
        base_script=selective_diag.base_script,
        base_canonical_sha256=selective_diag.base_canonical_sha256,
        derived_sha256=hashlib.sha256(derived_source.encode("utf-8")).hexdigest(),
        source=derived_source,
        unified_diff=diff,
    )


def build_global_diag_sources(official_repo: Path) -> dict[str, DerivedSource]:
    return {"global_diag": build_global_diag_source(official_repo)}


def expected_memory_contract() -> dict[str, int]:
    return {
        "k_cov_bytes": EXPECTED_K_COV_BYTES,
        "k_inv_bytes": EXPECTED_K_INV_BYTES,
        "k_state_bytes": EXPECTED_K_STATE_BYTES,
        "activation_stat_bytes": EXPECTED_ACTIVATION_STAT_BYTES,
        "precond_workspace_bytes": 0,
    }


def self_test_global_diag_math() -> None:
    rows = (
        (1.0, -2.0, 3.0),
        (2.0, 0.0, -1.0),
        (-1.0, 4.0, 2.0),
        (0.5, -3.0, 1.0),
    )
    diag = [sum(row[j] ** 2 for row in rows) / len(rows) for j in range(3)]
    dense_diagonal = []
    for j in range(3):
        dense_diagonal.append(
            sum(rows[i][j] * rows[i][j] for i in range(len(rows))) / len(rows)
        )
    if diag != dense_diagonal:
        raise RuntimeError("generic activation diagonal disagrees with dense diagonal")

    rows4 = tuple(tuple(float(i + 1 + 12 * r) for i in range(12)) for r in range(3))
    grouped = [
        [sum(row[block * 3 + j] ** 2 for row in rows4) / len(rows4) for j in range(3)]
        for block in range(4)
    ]
    flat = [sum(row[j] ** 2 for row in rows4) / len(rows4) for j in range(12)]
    if grouped != [flat[i * 3 : (i + 1) * 3] for i in range(4)]:
        raise RuntimeError("c_proj diag4 grouping disagrees with the dense diagonal")


self_test_global_diag_math()
