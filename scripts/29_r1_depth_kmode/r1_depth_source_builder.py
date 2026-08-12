"""Build audited official-R1 sources with depth-selective c_proj K modes.

Selected MLP depths use efficient ``none`` or ``diag`` K. Every unselected
``mlp.c_proj`` retains the official Newton-Muon block4 implementation.
"""

from __future__ import annotations

import difflib
import hashlib
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
R1_DIR = SCRIPT_DIR.parent / "15_official_newton_muon_r1"
sys.path.insert(0, str(R1_DIR))

import r1_source_builder as base


RULE_LAYERS = {
    "early": (0, 1, 2, 3, 4, 5, 6, 7),
    "center": (2, 3, 4, 5, 6, 7, 8, 9),
    "late": (4, 5, 6, 7, 8, 9, 10, 11),
    "edge": (0, 1, 2, 3, 8, 9, 10, 11),
    "all": tuple(range(12)),
}
SELECTED_MODES = ("none", "diag")
DEPTH_METHODS = tuple(
    f"{rule}_{mode}" for rule in RULE_LAYERS for mode in SELECTED_MODES
)
ANCHORS = ("block4", "muon")
ALLOWED_METHODS = (*DEPTH_METHODS, *ANCHORS)


def method_contract(method: str) -> tuple[str, str, tuple[int, ...]]:
    if method not in DEPTH_METHODS:
        raise ValueError(f"not an R1 depth treatment: {method!r}")
    rule, mode = method.rsplit("_", 1)
    return rule, mode, RULE_LAYERS[rule]


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"R1 depth derivation expected one {label!r} anchor, observed {count}"
        )
    return source.replace(old, new, 1)


def _patch_depth_route(source: str, method: str) -> str:
    rule, selected_mode, layers = method_contract(method)
    source = _replace_once(
        source,
        'R1_CPROJ_K_MODE = os.environ["R1_CPROJ_K_MODE"]\n',
        '''R1_CPROJ_K_MODE = os.environ["R1_CPROJ_K_MODE"]
R1_DEPTH_RULE = os.environ["R1_DEPTH_RULE"]
R1_CPROJ_K_LAYERS = tuple(
    int(item) for item in os.environ["R1_CPROJ_K_LAYERS"].split(",") if item
)
if not R1_CPROJ_K_LAYERS:
    raise ValueError("R1_CPROJ_K_LAYERS must not be empty")
if len(R1_CPROJ_K_LAYERS) != len(set(R1_CPROJ_K_LAYERS)):
    raise ValueError("R1_CPROJ_K_LAYERS contains duplicates")
if any(layer < 0 or layer >= 12 for layer in R1_CPROJ_K_LAYERS):
    raise ValueError(f"invalid R1 c_proj layer selection: {R1_CPROJ_K_LAYERS}")
''',
        "depth environment controls",
    )
    source = _replace_once(
        source,
        '''if R1_METHOD not in ("block4", "none", "diag"):
    raise ValueError(f"invalid Newton R1 method={R1_METHOD!r}")
if R1_CPROJ_K_MODE != R1_METHOD:
    raise ValueError("Newton R1 method and cproj_k_mode must match")
''',
        f'''if R1_METHOD != {method!r}:
    raise ValueError(f"R1 depth source requires method={method!r}, got {{R1_METHOD!r}}")
if R1_DEPTH_RULE != {rule!r}:
    raise ValueError(f"R1 depth source requires rule={rule!r}, got {{R1_DEPTH_RULE!r}}")
if R1_CPROJ_K_MODE != {selected_mode!r}:
    raise ValueError(
        f"R1 depth source requires selected mode={selected_mode!r}, "
        f"got {{R1_CPROJ_K_MODE!r}}"
    )
if R1_CPROJ_K_LAYERS != {layers!r}:
    raise ValueError(
        f"R1 depth source requires layers={layers!r}, "
        f"got {{R1_CPROJ_K_LAYERS!r}}"
    )
''',
        "depth method validation",
    )

    mlp_start = source.index("class MLP(nn.Module):")
    block_start = source.index("class Block(nn.Module):", mlp_start)
    prefix = source[:mlp_start]
    mlp = source[mlp_start:block_start]
    suffix = source[block_start:]
    # Rewrite only the mode reads inherited from the global R1 implementation.
    # Do this before inserting the per-layer assignment; otherwise a blanket
    # replacement would turn its RHS into a self-reference.
    mlp = mlp.replace("R1_CPROJ_K_MODE", "self.r1_cproj_k_mode")
    mlp = _replace_once(
        mlp,
        '''class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
''',
        '''class MLP(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.r1_layer_idx = int(layer_idx)
        self.r1_cproj_k_mode = (
            R1_CPROJ_K_MODE
            if self.r1_layer_idx in R1_CPROJ_K_LAYERS
            else "block4"
        )
''',
        "MLP layer index",
    )
    source = prefix + mlp + suffix
    source = _replace_once(
        source,
        '''class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
''',
        '''class Block(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config, layer_idx)
''',
        "Block layer index",
    )
    source = _replace_once(
        source,
        "            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),\n",
        "            h = nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),\n",
        "GPT indexed block construction",
    )
    source = _replace_once(
        source,
        '''print(f"R1_METADATA method={R1_METHOD} cproj_k_mode={R1_CPROJ_K_MODE} seed={R1_SEED} init_sha256={r1_init_sha256}")
''',
        '''print(f"R1_METADATA method={R1_METHOD} cproj_k_mode={R1_CPROJ_K_MODE} seed={R1_SEED} init_sha256={r1_init_sha256}")
r1_depth_observed_modes = tuple(block.mlp.r1_cproj_k_mode for block in model.transformer.h)
r1_depth_expected_modes = tuple(
    R1_CPROJ_K_MODE if layer in R1_CPROJ_K_LAYERS else "block4"
    for layer in range(len(model.transformer.h))
)
if r1_depth_observed_modes != r1_depth_expected_modes:
    raise RuntimeError(
        f"R1 depth routing mismatch: observed={r1_depth_observed_modes}, "
        f"expected={r1_depth_expected_modes}"
    )
print(
    "R1_DEPTH_ROUTING "
    f"rule={R1_DEPTH_RULE} selected_mode={R1_CPROJ_K_MODE} "
    f"selected_layers={','.join(str(layer) for layer in R1_CPROJ_K_LAYERS)} "
    f"selected_count={sum(mode == R1_CPROJ_K_MODE for mode in r1_depth_observed_modes)} "
    f"block4_count={sum(mode == 'block4' for mode in r1_depth_observed_modes)}"
)
''',
        "depth routing audit",
    )
    return source


def build_source(official_repo: Path, method: str) -> base.DerivedSource:
    if method not in ALLOWED_METHODS:
        raise ValueError(f"unsupported R1 depth method: {method!r}")
    if method in ANCHORS:
        return base.build_source(official_repo, method)

    block4 = base.build_source(official_repo, "block4")
    source = _patch_depth_route(block4.source, method)
    compile(source, f"<R1-depth-{method}>", "exec")
    official_raw = (official_repo / block4.base_script).read_bytes()
    official_source = base.canonical_bytes(official_raw).decode("utf-8")
    diff = "".join(
        difflib.unified_diff(
            official_source.splitlines(keepends=True),
            source.splitlines(keepends=True),
            fromfile=f"official/{block4.base_script}",
            tofile=f"r1_depth/train_r1_{method}.py",
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


def self_test_contract() -> None:
    expected_rules = {
        "early": tuple(range(0, 8)),
        "center": tuple(range(2, 10)),
        "late": tuple(range(4, 12)),
        "edge": (0, 1, 2, 3, 8, 9, 10, 11),
        "all": tuple(range(12)),
    }
    if RULE_LAYERS != expected_rules:
        raise AssertionError(f"unexpected R1 depth rules: {RULE_LAYERS}")
    if len(ALLOWED_METHODS) != 12 or len(set(ALLOWED_METHODS)) != 12:
        raise AssertionError("R1 depth contract must contain 12 unique methods")
    for method in DEPTH_METHODS:
        rule, mode, layers = method_contract(method)
        if rule not in RULE_LAYERS or mode not in SELECTED_MODES:
            raise AssertionError(f"bad R1 depth method: {method}")
        if layers != RULE_LAYERS[rule]:
            raise AssertionError(f"bad layer mapping for {method}")
