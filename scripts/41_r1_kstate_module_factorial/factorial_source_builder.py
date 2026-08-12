#!/usr/bin/env python3
"""Derive the two missing R1 module-factorial training sources."""

from __future__ import annotations

import difflib
import hashlib
import sys
from pathlib import Path


SCRIPT_VERSION = "2026-07-29.4"
SCRIPT_DIR = Path(__file__).resolve().parent
R1_DIR = SCRIPT_DIR.parent / "15_official_newton_muon_r1"
sys.path.insert(0, str(R1_DIR))

from r1_source_builder import DerivedSource, build_source


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"factorial source expected one {label!r} anchor, observed {count}"
        )
    return source.replace(old, new, 1)


ENV_OLD = '''R1_METHOD = os.environ["R1_METHOD"]
R1_CPROJ_K_MODE = os.environ["R1_CPROJ_K_MODE"]
R1_SEED = int(os.environ["R1_SEED"])
'''

ENV_NEW = '''R1_METHOD = os.environ["R1_METHOD"]
R1_CPROJ_K_MODE = os.environ["R1_CPROJ_K_MODE"]
R1_CFC_K_MODE = os.environ["R1_CFC_K_MODE"]
R1_SEED = int(os.environ["R1_SEED"])
'''

VALIDATION_OLD = '''if R1_METHOD not in ("block4", "none", "diag"):
    raise ValueError(f"invalid Newton R1 method={R1_METHOD!r}")
if R1_CPROJ_K_MODE != R1_METHOD:
    raise ValueError("Newton R1 method and cproj_k_mode must match")
'''

VALIDATION_NEW = '''if R1_METHOD not in ("block4", "none"):
    raise ValueError(f"invalid R1 module-factorial method={R1_METHOD!r}")
if R1_CPROJ_K_MODE != R1_METHOD:
    raise ValueError("R1 module-factorial method and cproj_k_mode must match")
if R1_CFC_K_MODE != "none":
    raise ValueError("41 only trains the missing c_fc=none cells")
'''

FC_INIT_OLD = '''        self.fc_xtx_accum  = nn.Buffer(torch.zeros(d, d, dtype=torch.float32), persistent=False)
        self.fc_xtx_tmp    = nn.Buffer(torch.empty(d, d, dtype=torch.float32), persistent=False)
        self.fc_xtx_count  = nn.Buffer(torch.zeros((), dtype=torch.float32), persistent=False)

        self.c_fc.weight._stats_ref = {"kind": "c_fc", "d": d, "accum": self.fc_xtx_accum, "count": self.fc_xtx_count}
'''

FC_INIT_NEW = '''        if R1_CFC_K_MODE == "full":
            self.fc_xtx_accum = nn.Buffer(torch.zeros(d, d, dtype=torch.float32), persistent=False)
            self.fc_xtx_tmp = nn.Buffer(torch.empty(d, d, dtype=torch.float32), persistent=False)
            self.fc_xtx_count = nn.Buffer(torch.zeros((), dtype=torch.float32), persistent=False)
            self.c_fc.weight._stats_ref = {
                "kind": "c_fc",
                "d": d,
                "accum": self.fc_xtx_accum,
                "count": self.fc_xtx_count,
            }
        elif R1_CFC_K_MODE == "none":
            self.c_fc.weight._stats_ref = None
        else:
            raise ValueError(f"unsupported R1_CFC_K_MODE={R1_CFC_K_MODE!r}")
'''

FC_FORWARD_OLD = '''        if precond_flag:
            x2d = x.flatten(0, -2)
            torch.ops.nanogpt.accum_xtx(x2d, self.fc_xtx_accum, self.fc_xtx_count, self.fc_xtx_tmp)
'''

FC_FORWARD_NEW = '''        if precond_flag and R1_CFC_K_MODE == "full":
            x2d = x.flatten(0, -2)
            torch.ops.nanogpt.accum_xtx(x2d, self.fc_xtx_accum, self.fc_xtx_count, self.fc_xtx_tmp)
'''

FC_APPLY_OLD = '''        self.c_fc.weight._stats_ref = {"kind": "c_fc", "d": d, "accum": self.fc_xtx_accum, "count": self.fc_xtx_count}
'''

FC_APPLY_NEW = '''        if R1_CFC_K_MODE == "full":
            self.c_fc.weight._stats_ref = {
                "kind": "c_fc",
                "d": d,
                "accum": self.fc_xtx_accum,
                "count": self.fc_xtx_count,
            }
        else:
            self.c_fc.weight._stats_ref = None
'''

METADATA_OLD = '''print(f"R1_METADATA method={R1_METHOD} cproj_k_mode={R1_CPROJ_K_MODE} seed={R1_SEED} init_sha256={r1_init_sha256}")
'''

METADATA_NEW = METADATA_OLD + '''print(f"R1_FACTORIAL_METADATA cfc_k_mode={R1_CFC_K_MODE} cproj_k_mode={R1_CPROJ_K_MODE}")
'''


def apply_factorial_overlay(source: str) -> str:
    source = replace_once(source, ENV_OLD, ENV_NEW, "factorial environment")
    source = replace_once(
        source, VALIDATION_OLD, VALIDATION_NEW, "factorial mode validation"
    )
    source = replace_once(source, FC_INIT_OLD, FC_INIT_NEW, "c_fc state allocation")
    source = replace_once(
        source, FC_FORWARD_OLD, FC_FORWARD_NEW, "c_fc activation accumulation"
    )
    source = replace_once(source, FC_APPLY_OLD, FC_APPLY_NEW, "c_fc apply repair")
    source = replace_once(source, METADATA_OLD, METADATA_NEW, "factorial metadata")
    compile(source, "<R1-module-factorial>", "exec")
    assert_factorial_source_contract(source)
    return source


def assert_factorial_source_contract(source: str) -> None:
    required = (
        'R1_CFC_K_MODE = os.environ["R1_CFC_K_MODE"]',
        'if R1_CFC_K_MODE != "none":',
        'if precond_flag and R1_CFC_K_MODE == "full":',
        'elif R1_CFC_K_MODE == "none":',
        "R1_FACTORIAL_METADATA cfc_k_mode=",
    )
    missing = [anchor for anchor in required if anchor not in source]
    if missing:
        raise RuntimeError(f"factorial source is missing anchors: {missing}")
    if source.count("R1_FACTORIAL_METADATA") != 1:
        raise RuntimeError("factorial metadata line count changed")
    if "R1_CFC_K_MODE != \"none\"" not in source:
        raise RuntimeError("41 must reject c_fc=full training")


def build_factorial_source(official_repo: Path, cproj_mode: str) -> DerivedSource:
    if cproj_mode not in {"block4", "none"}:
        raise ValueError(f"unsupported c_proj mode: {cproj_mode}")
    base = build_source(official_repo, cproj_mode)
    derived_source = apply_factorial_overlay(base.source)
    official_source = (
        (official_repo / base.base_script)
        .read_bytes()
        .replace(b"\r\n", b"\n")
        .decode("utf-8")
    )
    diff = "".join(
        difflib.unified_diff(
            official_source.splitlines(keepends=True),
            derived_source.splitlines(keepends=True),
            fromfile=f"official/{base.base_script}",
            tofile=f"r1_module_factorial/train_r1_{cproj_mode}.py",
        )
    )
    return DerivedSource(
        method=cproj_mode,
        base_script=base.base_script,
        base_canonical_sha256=base.base_canonical_sha256,
        derived_sha256=hashlib.sha256(derived_source.encode("utf-8")).hexdigest(),
        source=derived_source,
        unified_diff=diff,
    )


def build_factorial_sources(official_repo: Path) -> dict[str, DerivedSource]:
    built = {
        mode: build_factorial_source(official_repo, mode)
        for mode in ("block4", "none")
    }
    if len({item.derived_sha256 for item in built.values()}) != 1:
        raise RuntimeError(
            "block4/none must share one parameterized module-factorial source"
        )
    return built
