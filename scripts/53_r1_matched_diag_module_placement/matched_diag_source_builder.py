#!/usr/bin/env python3
"""Derive the five-arm matched-diagonal Experiment-53 training source."""

from __future__ import annotations

import difflib
import hashlib
import sys
from pathlib import Path


SCRIPT_VERSION = "2026-08-17.2"
SCRIPT_DIR = Path(__file__).resolve().parent
GLOBAL_DIAG_DIR = SCRIPT_DIR.parent / "50_r1_global_activation_diag"
R1_DIR = SCRIPT_DIR.parent / "15_official_newton_muon_r1"
sys.path.insert(0, str(GLOBAL_DIAG_DIR))
sys.path.insert(0, str(R1_DIR))

from global_diag_source_builder import build_global_diag_source
from r1_source_builder import DerivedSource


ARM_CONFIGS: dict[str, dict[str, str]] = {
    "all_none": {"c_fc": "none", "c_proj": "none", "o_proj": "none", "qkv": "none"},
    "c_fc_diag": {"c_fc": "diag", "c_proj": "none", "o_proj": "none", "qkv": "none"},
    "c_proj_diag": {"c_fc": "none", "c_proj": "diag", "o_proj": "none", "qkv": "none"},
    "c_fc_c_proj_diag": {"c_fc": "diag", "c_proj": "diag", "o_proj": "none", "qkv": "none"},
    "o_proj_diag": {"c_fc": "none", "c_proj": "none", "o_proj": "diag", "qkv": "none"},
}


EXPECTED_MEMORY: dict[str, dict[str, int]] = {
    "all_none": {
        "k_cov_bytes": 0,
        "k_inv_bytes": 0,
        "k_state_bytes": 0,
        "activation_stat_bytes": 0,
        "precond_workspace_bytes": 0,
    },
    "c_fc_diag": {
        "k_cov_bytes": 36_864,
        "k_inv_bytes": 36_864,
        "k_state_bytes": 73_728,
        "activation_stat_bytes": 36_912,
        "precond_workspace_bytes": 0,
    },
    "c_proj_diag": {
        "k_cov_bytes": 147_456,
        "k_inv_bytes": 147_456,
        "k_state_bytes": 294_912,
        "activation_stat_bytes": 147_504,
        "precond_workspace_bytes": 0,
    },
    "c_fc_c_proj_diag": {
        "k_cov_bytes": 184_320,
        "k_inv_bytes": 184_320,
        "k_state_bytes": 368_640,
        "activation_stat_bytes": 184_416,
        "precond_workspace_bytes": 0,
    },
    "o_proj_diag": {
        "k_cov_bytes": 36_864,
        "k_inv_bytes": 36_864,
        "k_state_bytes": 73_728,
        "activation_stat_bytes": 36_912,
        "precond_workspace_bytes": 0,
    },
}


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"matched-diag source expected one {label!r} anchor, observed {count}"
        )
    return source.replace(old, new, 1)


ENV_OLD = '''R1_METHOD = os.environ["R1_METHOD"]
R1_CPROJ_K_MODE = os.environ["R1_CPROJ_K_MODE"]
R1_GLOBAL_DIAG = os.environ["R1_GLOBAL_DIAG"] == "1"
R1_SEED = int(os.environ["R1_SEED"])
'''

ENV_NEW = '''R1_METHOD = os.environ["R1_METHOD"]
R1_CPROJ_K_MODE = os.environ["R1_CPROJ_K_MODE"]
R1_CFC_K_MODE = os.environ["R1_CFC_K_MODE"]
R1_OPROJ_K_MODE = os.environ["R1_OPROJ_K_MODE"]
R1_QKV_K_MODE = os.environ["R1_QKV_K_MODE"]
R1_SEED = int(os.environ["R1_SEED"])
'''

VALIDATION_OLD = '''if R1_METHOD != "global_diag":
    raise ValueError(f"invalid Experiment-50 method={R1_METHOD!r}")
if R1_CPROJ_K_MODE != "diag":
    raise ValueError("global-diag must retain the audited c_proj diag route")
if not R1_GLOBAL_DIAG:
    raise ValueError("R1_GLOBAL_DIAG=1 is required")
'''

VALIDATION_NEW = '''_EX53_ARMS = {
    "all_none": ("none", "none", "none", "none"),
    "c_fc_diag": ("diag", "none", "none", "none"),
    "c_proj_diag": ("none", "diag", "none", "none"),
    "c_fc_c_proj_diag": ("diag", "diag", "none", "none"),
    "o_proj_diag": ("none", "none", "diag", "none"),
}
if R1_METHOD not in _EX53_ARMS:
    raise ValueError(f"invalid Experiment-53 arm={R1_METHOD!r}")
_observed_ex53_modes = (
    R1_CFC_K_MODE, R1_CPROJ_K_MODE, R1_OPROJ_K_MODE, R1_QKV_K_MODE
)
if _observed_ex53_modes != _EX53_ARMS[R1_METHOD]:
    raise ValueError(
        f"Experiment-53 arm/mode mismatch: arm={R1_METHOD!r} "
        f"observed={_observed_ex53_modes!r} expected={_EX53_ARMS[R1_METHOD]!r}"
    )
'''

DEVICE_OLD = '''        d = int(self._precond_d) if self._precond_d is not None else 0
        self._refresh_K = None if not refresh_map else torch.empty(
            (len(refresh_map), d, d),
            device=refresh_map[0][0].device,
            dtype=torch.float32
        )

        all_diag_params = [*input_diag_params, *proj_diag_params]
        if not refresh_map and not all_diag_params:
            raise RuntimeError("global-diag route found no preconditioned parameters")
        dev = refresh_map[0][0].device if refresh_map else all_diag_params[0].device
'''

DEVICE_NEW = '''        if self._precond_d is not None:
            d = int(self._precond_d)
        else:
            first_parameter = self.param_groups[0]["params"][0]
            d = int(first_parameter.shape[-1])
        self._refresh_K = None if not refresh_map else torch.empty(
            (len(refresh_map), d, d),
            device=refresh_map[0][0].device,
            dtype=torch.float32
        )

        all_diag_params = [*input_diag_params, *proj_diag_params]
        dev = (
            refresh_map[0][0].device
            if refresh_map
            else all_diag_params[0].device
            if all_diag_params
            else self.param_groups[0]["params"][0].device
        )
'''

ATTENTION_OLD = '''        d = self.n_embd
        self.qkv_xtx_accum = nn.Buffer(torch.zeros(d, dtype=torch.float32), persistent=False)
        self.o_xtx_accum = nn.Buffer(torch.zeros(d, dtype=torch.float32), persistent=False)
        self.qkv_xtx_count = nn.Buffer(torch.zeros((), dtype=torch.float32), persistent=False)
        self.o_xtx_count = nn.Buffer(torch.zeros((), dtype=torch.float32), persistent=False)

        self.c_attn.weight._stats_ref = {"kind": "qkv_diag", "d": d, "accum": self.qkv_xtx_accum, "count": self.qkv_xtx_count}
        self.c_proj.weight._stats_ref = {"kind": "o_diag", "d": d, "accum": self.o_xtx_accum, "count": self.o_xtx_count}
'''

ATTENTION_NEW = '''        d = self.n_embd
        self.c_attn.weight._stats_ref = None
        if R1_OPROJ_K_MODE == "diag":
            self.o_xtx_accum = nn.Buffer(torch.zeros(d, dtype=torch.float32), persistent=False)
            self.o_xtx_count = nn.Buffer(torch.zeros((), dtype=torch.float32), persistent=False)
            self.c_proj.weight._stats_ref = {
                "kind": "o_diag", "d": d,
                "accum": self.o_xtx_accum, "count": self.o_xtx_count,
            }
        elif R1_OPROJ_K_MODE == "none":
            self.c_proj.weight._stats_ref = None
        else:
            raise ValueError(f"unsupported R1_OPROJ_K_MODE={R1_OPROJ_K_MODE!r}")
'''

QKV_FORWARD_OLD = '''        if precond_flag:
            x2d = x.flatten(0, -2)
            torch.ops.nanogpt.accum_xtx_diag(x2d, self.qkv_xtx_accum, self.qkv_xtx_count)

'''

O_FORWARD_OLD = '''        if precond_flag:
            y2d = y.flatten(0, -2)
            torch.ops.nanogpt.accum_xtx_diag(y2d, self.o_xtx_accum, self.o_xtx_count)
'''

O_FORWARD_NEW = '''        if precond_flag and R1_OPROJ_K_MODE == "diag":
            y2d = y.flatten(0, -2)
            torch.ops.nanogpt.accum_xtx_diag(y2d, self.o_xtx_accum, self.o_xtx_count)
'''

ATTENTION_APPLY_OLD = '''        d = self.n_embd
        self.c_attn.weight._stats_ref = {"kind": "qkv_diag", "d": d, "accum": self.qkv_xtx_accum, "count": self.qkv_xtx_count}
        self.c_proj.weight._stats_ref = {"kind": "o_diag", "d": d, "accum": self.o_xtx_accum, "count": self.o_xtx_count}
        return self
'''

ATTENTION_APPLY_NEW = '''        d = self.n_embd
        self.c_attn.weight._stats_ref = None
        if R1_OPROJ_K_MODE == "diag":
            self.c_proj.weight._stats_ref = {
                "kind": "o_diag", "d": d,
                "accum": self.o_xtx_accum, "count": self.o_xtx_count,
            }
        else:
            self.c_proj.weight._stats_ref = None
        return self
'''

FC_INIT_OLD = '''        self.fc_xtx_accum = nn.Buffer(torch.zeros(d, dtype=torch.float32), persistent=False)
        self.fc_xtx_count = nn.Buffer(torch.zeros((), dtype=torch.float32), persistent=False)

        self.c_fc.weight._stats_ref = {"kind": "c_fc_diag", "d": d, "accum": self.fc_xtx_accum, "count": self.fc_xtx_count}
'''

FC_INIT_NEW = '''        if R1_CFC_K_MODE == "diag":
            self.fc_xtx_accum = nn.Buffer(torch.zeros(d, dtype=torch.float32), persistent=False)
            self.fc_xtx_count = nn.Buffer(torch.zeros((), dtype=torch.float32), persistent=False)
            self.c_fc.weight._stats_ref = {
                "kind": "c_fc_diag", "d": d,
                "accum": self.fc_xtx_accum, "count": self.fc_xtx_count,
            }
        elif R1_CFC_K_MODE == "none":
            self.c_fc.weight._stats_ref = None
        else:
            raise ValueError(f"unsupported R1_CFC_K_MODE={R1_CFC_K_MODE!r}")
'''

CPROJ_INIT_OLD = '''        if R1_CPROJ_K_MODE == "block4":
            self.proj_xtx_accum = nn.Buffer(torch.zeros(4, d, d, dtype=torch.float32), persistent=False)
            self.proj_xtx_tmp = nn.Buffer(torch.empty(4, d, d, dtype=torch.float32), persistent=False)
            self.proj_xtx_count = nn.Buffer(torch.zeros((), dtype=torch.float32), persistent=False)
            self.c_proj.weight._stats_ref = {"kind": "c_proj", "d": d, "accum": self.proj_xtx_accum, "count": self.proj_xtx_count}
        elif R1_CPROJ_K_MODE == "diag":
            self.proj_xtx_accum = nn.Buffer(torch.zeros(4, d, dtype=torch.float32), persistent=False)
            self.proj_xtx_count = nn.Buffer(torch.zeros((), dtype=torch.float32), persistent=False)
            self.c_proj.weight._stats_ref = {"kind": "c_proj_diag", "d": d, "accum": self.proj_xtx_accum, "count": self.proj_xtx_count}
        elif R1_CPROJ_K_MODE == "none":
            self.c_proj.weight._stats_ref = None
        else:
            raise ValueError(f"unsupported R1_CPROJ_K_MODE={R1_CPROJ_K_MODE!r}")
'''

CPROJ_INIT_NEW = '''        if R1_CPROJ_K_MODE == "diag":
            self.proj_xtx_accum = nn.Buffer(torch.zeros(4, d, dtype=torch.float32), persistent=False)
            self.proj_xtx_count = nn.Buffer(torch.zeros((), dtype=torch.float32), persistent=False)
            self.c_proj.weight._stats_ref = {
                "kind": "c_proj_diag", "d": d,
                "accum": self.proj_xtx_accum, "count": self.proj_xtx_count,
            }
        elif R1_CPROJ_K_MODE == "none":
            self.c_proj.weight._stats_ref = None
        else:
            raise ValueError(f"unsupported R1_CPROJ_K_MODE={R1_CPROJ_K_MODE!r}")
'''

FC_FORWARD_OLD = '''        if precond_flag:
            x2d = x.flatten(0, -2)
            torch.ops.nanogpt.accum_xtx_diag(x2d, self.fc_xtx_accum, self.fc_xtx_count)
'''

FC_FORWARD_NEW = '''        if precond_flag and R1_CFC_K_MODE == "diag":
            x2d = x.flatten(0, -2)
            torch.ops.nanogpt.accum_xtx_diag(x2d, self.fc_xtx_accum, self.fc_xtx_count)
'''

CPROJ_FORWARD_OLD = '''        if precond_flag and R1_CPROJ_K_MODE != "none":
            z2d = x.flatten(0, -2)
            if R1_CPROJ_K_MODE == "block4":
                torch.ops.nanogpt.accum_xtx_blocks4(z2d, self.proj_xtx_accum, self.proj_xtx_count, self.proj_xtx_tmp)
            else:
                torch.ops.nanogpt.accum_xtx_diag4(z2d, self.proj_xtx_accum, self.proj_xtx_count)
'''

CPROJ_FORWARD_NEW = '''        if precond_flag and R1_CPROJ_K_MODE == "diag":
            z2d = x.flatten(0, -2)
            torch.ops.nanogpt.accum_xtx_diag4(z2d, self.proj_xtx_accum, self.proj_xtx_count)
'''

FC_APPLY_OLD = '''        d = self.c_fc.weight.size(1)
        self.c_fc.weight._stats_ref = {"kind": "c_fc_diag", "d": d, "accum": self.fc_xtx_accum, "count": self.fc_xtx_count}
'''

FC_APPLY_NEW = '''        d = self.c_fc.weight.size(1)
        if R1_CFC_K_MODE == "diag":
            self.c_fc.weight._stats_ref = {
                "kind": "c_fc_diag", "d": d,
                "accum": self.fc_xtx_accum, "count": self.fc_xtx_count,
            }
        else:
            self.c_fc.weight._stats_ref = None
'''

CPROJ_APPLY_OLD = '''        if R1_CPROJ_K_MODE == "block4":
            self.c_proj.weight._stats_ref = {"kind": "c_proj", "d": d, "accum": self.proj_xtx_accum, "count": self.proj_xtx_count}
        elif R1_CPROJ_K_MODE == "diag":
            self.c_proj.weight._stats_ref = {"kind": "c_proj_diag", "d": d, "accum": self.proj_xtx_accum, "count": self.proj_xtx_count}
        else:
            self.c_proj.weight._stats_ref = None
'''

CPROJ_APPLY_NEW = '''        if R1_CPROJ_K_MODE == "diag":
            self.c_proj.weight._stats_ref = {
                "kind": "c_proj_diag", "d": d,
                "accum": self.proj_xtx_accum, "count": self.proj_xtx_count,
            }
        else:
            self.c_proj.weight._stats_ref = None
'''

PROJ_DIAG_RIDGE_OLD = '''                ridge = cov.mean(dim=-1) * self.precond_ridge_mult + self.precond_eps
                self.state[p]["precond_inv_apply"].copy_((cov + ridge.unsqueeze(-1)).reciprocal())
'''

PROJ_DIAG_RIDGE_NEW = '''                ridge = cov.mean() * self.precond_ridge_mult + self.precond_eps
                self.state[p]["precond_inv_apply"].copy_((cov + ridge).reciprocal())
'''

METADATA_OLD = '''print("R1_GLOBAL_DIAG_METADATA route=all_eligible_activation_diagonal dense_workspace=0")
'''

METADATA_NEW = '''print(
    "R1_MATCHED_DIAG_METADATA "
    f"arm={R1_METHOD} cfc={R1_CFC_K_MODE} cproj={R1_CPROJ_K_MODE} "
    f"oproj={R1_OPROJ_K_MODE} qkv={R1_QKV_K_MODE} dense_workspace=0"
)
'''

OPTIMIZER_AUDIT_OLD = '''optimizer2.attach_preconditioner()
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

OPTIMIZER_AUDIT_NEW = '''optimizer2.attach_preconditioner()
if optimizer2._refresh_map or optimizer2._refresh_K is not None:
    raise RuntimeError("Experiment-53 unexpectedly constructed a dense refresh route")
if optimizer2._apply_plan is None:
    raise RuntimeError("Experiment-53 apply plan was not constructed")
_expected_input_diag = (12 if R1_CFC_K_MODE == "diag" else 0) + (12 if R1_OPROJ_K_MODE == "diag" else 0)
_expected_proj_diag = 12 if R1_CPROJ_K_MODE == "diag" else 0
_observed_input_diag = len(optimizer2._apply_plan["input_diag_params"])
_observed_proj_diag = len(optimizer2._apply_plan["proj_diag_params"])
if _observed_input_diag != _expected_input_diag:
    raise RuntimeError(
        f"Experiment-53 input-diagonal route mismatch: {_observed_input_diag} != {_expected_input_diag}"
    )
if _observed_proj_diag != _expected_proj_diag:
    raise RuntimeError(
        f"Experiment-53 c_proj-diagonal route mismatch: {_observed_proj_diag} != {_expected_proj_diag}"
    )
print(
    "R1_MATCHED_DIAG_ROUTE "
    f"arm={R1_METHOD} input_diag_params={_observed_input_diag} "
    f"proj_diag_params={_observed_proj_diag} dense_refresh_blocks=0"
)
optimizers = [optimizer1, optimizer2]
'''


def apply_matched_diag_overlay(source: str) -> str:
    replacements = (
        (ENV_OLD, ENV_NEW, "matched placement environment"),
        (VALIDATION_OLD, VALIDATION_NEW, "five-arm validation"),
        (DEVICE_OLD, DEVICE_NEW, "state-free apply-plan support"),
        (ATTENTION_OLD, ATTENTION_NEW, "attention target allocation"),
        (QKV_FORWARD_OLD, "", "QKV state removal"),
        (O_FORWARD_OLD, O_FORWARD_NEW, "o_proj conditional accumulation"),
        (ATTENTION_APPLY_OLD, ATTENTION_APPLY_NEW, "attention stats repair"),
        (FC_INIT_OLD, FC_INIT_NEW, "c_fc conditional allocation"),
        (CPROJ_INIT_OLD, CPROJ_INIT_NEW, "c_proj diagonal-only allocation"),
        (FC_FORWARD_OLD, FC_FORWARD_NEW, "c_fc conditional accumulation"),
        (CPROJ_FORWARD_OLD, CPROJ_FORWARD_NEW, "c_proj diagonal-only accumulation"),
        (FC_APPLY_OLD, FC_APPLY_NEW, "c_fc stats repair"),
        (CPROJ_APPLY_OLD, CPROJ_APPLY_NEW, "c_proj diagonal-only stats repair"),
        (
            PROJ_DIAG_RIDGE_OLD,
            PROJ_DIAG_RIDGE_NEW,
            "shared coordinate-diagonal ridge convention",
        ),
        (METADATA_OLD, METADATA_NEW, "matched placement metadata"),
        (OPTIMIZER_AUDIT_OLD, OPTIMIZER_AUDIT_NEW, "runtime route audit"),
    )
    for old, new, label in replacements:
        source = replace_once(source, old, new, label)
    compile(source, "<R1-matched-diag-module-placement>", "exec")
    assert_matched_diag_source_contract(source)
    return source


def assert_matched_diag_source_contract(source: str) -> None:
    required = (
        'R1_CFC_K_MODE = os.environ["R1_CFC_K_MODE"]',
        'R1_OPROJ_K_MODE = os.environ["R1_OPROJ_K_MODE"]',
        'R1_QKV_K_MODE = os.environ["R1_QKV_K_MODE"]',
        '"all_none": ("none", "none", "none", "none")',
        'if R1_CFC_K_MODE == "diag":',
        'if R1_OPROJ_K_MODE == "diag":',
        'self.c_attn.weight._stats_ref = None',
        'R1_MATCHED_DIAG_METADATA',
        'R1_MATCHED_DIAG_ROUTE',
        'precond_init_diag: float = 0.001',
        'precond_ridge_mult: float = 0.2',
        'precond_eps: float = 1e-8',
        'do_refresh = (t % 32 == 0)',
        'precond_ewma = 0.950',
        'ridge = cov.mean() * self.precond_ridge_mult + self.precond_eps',
        'model_parameter_keys = {_r1_storage_key(parameter) for parameter in model.parameters()}',
    )
    missing = [anchor for anchor in required if anchor not in source]
    if missing:
        raise RuntimeError(f"matched-diag source missing contract anchors: {missing}")
    forbidden = (
        "R1_GLOBAL_DIAG",
        '"kind": "qkv_diag"',
        "self.qkv_xtx_accum",
        "self.qkv_xtx_count",
        "self.fc_xtx_tmp",
        "self.o_xtx_tmp",
        "torch.linalg.cholesky_ex(K",  # unreachable code is still present in the shared optimizer
    )
    # The shared optimizer retains a guarded dense branch, but EX53 proves it
    # unreachable with an empty refresh map. Do not reject that shared code
    # anchor; reject target-family dense allocations instead.
    forbidden = forbidden[:-1]
    present = [anchor for anchor in forbidden if anchor in source]
    if present:
        raise RuntimeError(f"matched-diag source retains forbidden anchors: {present}")
    dense_target_allocations = (
        "self.o_xtx_accum = nn.Buffer(torch.zeros(d, d,",
        "self.fc_xtx_accum = nn.Buffer(torch.zeros(d, d,",
        "self.proj_xtx_accum = nn.Buffer(torch.zeros(4, d, d,",
    )
    present_dense = [anchor for anchor in dense_target_allocations if anchor in source]
    if present_dense:
        raise RuntimeError(
            f"matched-diag source retains dense target allocation: {present_dense}"
        )
    if source.count("R1_MATCHED_DIAG_METADATA") != 1:
        raise RuntimeError("matched-diag metadata line count changed")
    if "ridge = cov.mean(dim=-1) * self.precond_ridge_mult" in source:
        raise RuntimeError("c_proj retains a block-specific diagonal ridge convention")


def build_matched_diag_source(official_repo: Path) -> tuple[str, str, str, str]:
    global_derived = build_global_diag_source(official_repo)
    source = apply_matched_diag_overlay(global_derived.source)
    official_source = (
        (official_repo / global_derived.base_script)
        .read_bytes()
        .replace(b"\r\n", b"\n")
        .decode("utf-8")
    )
    diff = "".join(
        difflib.unified_diff(
            official_source.splitlines(keepends=True),
            source.splitlines(keepends=True),
            fromfile=f"official/{global_derived.base_script}",
            tofile="r1_matched_diag/train_r1_matched_diag.py",
        )
    )
    return (
        global_derived.base_script,
        global_derived.base_canonical_sha256,
        source,
        diff,
    )


def build_matched_diag_sources(official_repo: Path) -> dict[str, DerivedSource]:
    base_script, base_sha, source, diff = build_matched_diag_source(official_repo)
    derived_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return {
        arm: DerivedSource(
            method=arm,
            base_script=base_script,
            base_canonical_sha256=base_sha,
            derived_sha256=derived_sha,
            source=source,
            unified_diff=diff,
        )
        for arm in ARM_CONFIGS
    }


def expected_memory_contract(arm: str) -> dict[str, int]:
    try:
        return dict(EXPECTED_MEMORY[arm])
    except KeyError as exc:
        raise ValueError(f"unknown Experiment-53 arm: {arm}") from exc


def self_test_memory_formula() -> None:
    layers, d, fp32 = 12, 768, 4
    generic_cov = layers * d * fp32
    proj_cov = layers * 4 * d * fp32
    generic_stats = layers * (d + 1) * fp32
    proj_stats = layers * (4 * d + 1) * fp32
    assert EXPECTED_MEMORY["c_fc_diag"] == {
        "k_cov_bytes": generic_cov,
        "k_inv_bytes": generic_cov,
        "k_state_bytes": 2 * generic_cov,
        "activation_stat_bytes": generic_stats,
        "precond_workspace_bytes": 0,
    }
    assert EXPECTED_MEMORY["o_proj_diag"] == EXPECTED_MEMORY["c_fc_diag"]
    assert EXPECTED_MEMORY["c_proj_diag"]["k_cov_bytes"] == proj_cov
    assert EXPECTED_MEMORY["c_proj_diag"]["activation_stat_bytes"] == proj_stats
    assert EXPECTED_MEMORY["c_fc_c_proj_diag"]["k_state_bytes"] == 2 * (
        generic_cov + proj_cov
    )


self_test_memory_formula()
