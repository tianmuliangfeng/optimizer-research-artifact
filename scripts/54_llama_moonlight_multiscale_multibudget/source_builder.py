#!/usr/bin/env python3
"""Derive EX54 Moonlight trainers from the accepted LLaMA/EX48 sources.

The derivation is intentionally narrow: model/data/validation/checkpoint logic
is inherited byte-for-byte from the pinned parents.  We register a Moonlight
matrix-optimizer route and bind run lineage through environment variables.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
import hashlib
from pathlib import Path


PACKAGE_REL = Path("scripts/54_llama_moonlight_multiscale_multibudget")
PARENT_SHA256 = {
    "scripts/19_r1_extended_baselines/extended_optimizers.py":
        "bf39d7e1b435ef737833046c564ce8770d858d1aa474c9d7f11a914057253655",
    "scripts/17_llama_swiglu_validation/train_llama_swiglu.py":
        "b72eb0d2a1dfa91b61cd49b4784b3e0739ecebc2fd3228b8f719cec125706f2a",
    "scripts/20_llama_swiglu_1b/train_llama_swiglu_1b.py":
        "043c758f3d5eb5d1abc9e1f9029a8d085a238cf169ef69ba86580014699dc401",
    "scripts/48_llama1b_10b_multibudget/train_segment.py":
        "ab7fd5acd809cc6233be05469cbff922bd7d7a9f5703af107f62455963f0d02f",
}
MOONLIGHT_TRANSFER_SYMBOLS = (
    "zeropower_via_newtonschulz5",
    "logical_matrix_slices",
    "R1MoonlightMuon",
)


@dataclass(frozen=True)
class DerivedSources:
    trainer: str
    wrapper_1b: str
    long_worker_parent: str
    moonlight_optimizer: str


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_pinned(root: Path, relative: str) -> str:
    path = root / relative
    text = path.read_text(encoding="utf-8")
    observed = sha256_text(text)
    expected = PARENT_SHA256[relative]
    if observed != expected:
        raise RuntimeError(
            f"accepted parent source drift for {relative}: {observed} != {expected}"
        )
    return text


def moonlight_transfer_fingerprints(source: str) -> dict[str, str]:
    """Hash the exact EX19 Moonlight algorithm subtrees, not nearby helpers."""
    tree = ast.parse(source)
    nodes = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and node.name in MOONLIGHT_TRANSFER_SYMBOLS
    }
    if set(nodes) != set(MOONLIGHT_TRANSFER_SYMBOLS):
        raise RuntimeError(
            f"Moonlight transfer symbols are incomplete: {sorted(nodes)}"
        )
    result: dict[str, str] = {}
    for name in MOONLIGHT_TRANSFER_SYMBOLS:
        node = copy.deepcopy(nodes[name])
        payload = ast.dump(node, annotate_fields=True, include_attributes=False)
        result[name] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return result


def audit_moonlight_transfer(root: Path, local_source: str) -> dict[str, object]:
    reference = read_pinned(
        root, "scripts/19_r1_extended_baselines/extended_optimizers.py"
    )
    expected = moonlight_transfer_fingerprints(reference)
    observed = moonlight_transfer_fingerprints(local_source)
    checks = {name: observed[name] == expected[name] for name in expected}
    if not all(checks.values()):
        raise RuntimeError(f"EX54 Moonlight algorithm drifted from EX19: {checks}")
    return {
        "passed": True,
        "reference_sha256": PARENT_SHA256[
            "scripts/19_r1_extended_baselines/extended_optimizers.py"
        ],
        "symbol_sha256": expected,
    }


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, observed {count}")
    return text.replace(old, new, 1)


RNG_CPU_COMPAT_MARKER = "EX54_MOONLIGHT_RNG_CPU_BYTE_COMPAT_V1"


def patch_rng_restore(source: str) -> str:
    if RNG_CPU_COMPAT_MARKER in source:
        return source
    return replace_once(
        source,
        "def restore_rng_state(payload: dict[str, Any]) -> None:\n"
        "    random.setstate(payload[\"python\"])\n"
        "    np.random.set_state(payload[\"numpy\"])\n"
        "    torch.set_rng_state(payload[\"torch_cpu\"])\n"
        "    torch.cuda.set_rng_state_all(payload[\"torch_cuda\"])",
        "def restore_rng_state(payload: dict[str, Any]) -> None:\n"
        f"    # {RNG_CPU_COMPAT_MARKER}: checkpoint map_location may be CUDA.\n"
        "    random.setstate(payload[\"python\"])\n"
        "    np.random.set_state(payload[\"numpy\"])\n"
        "    cpu_state = payload[\"torch_cpu\"].detach().cpu().to(dtype=torch.uint8)\n"
        "    cuda_states = [state.detach().cpu().to(dtype=torch.uint8) for state in payload[\"torch_cuda\"]]\n"
        "    torch.set_rng_state(cpu_state)\n"
        "    torch.cuda.set_rng_state_all(cuda_states)",
        "CUDA-mapped RNG restore compatibility",
    )


def derive_trainer(source: str) -> str:
    source = patch_rng_restore(source)
    source = replace_once(
        source,
        "from torch import Tensor, nn\nimport torch.nn.functional as F",
        "from torch import Tensor, nn\nimport torch.nn.functional as F\n\n"
        "from moonlight_optimizer import (\n"
        "    R1MoonlightMuon,\n"
        "    optimizer_state_breakdown as moonlight_state_breakdown,\n"
        "    state_schema as moonlight_state_schema,\n"
        ")",
        "Moonlight import",
    )
    source = replace_once(
        source,
        'METHODS = ("adamw", "muon", "newton_full", "down_none", "down_diag")',
        'METHODS = ("adamw", "muon", "newton_full", "down_none", "down_diag", "moonlight")',
        "method registration",
    )
    source = replace_once(
        source,
        "def make_optimizers(\n    model: LlamaForCausalLM, args: argparse.Namespace\n) -> tuple[list[torch.optim.Optimizer], torch.optim.Optimizer | None]:",
        "def ex54_moonlight_kwargs() -> dict[str, Any]:\n"
        "    required = ('MOONLIGHT_WEIGHT_DECAY', 'MOONLIGHT_CONTRACT_SHA256', 'MOONLIGHT_SELECTION_SHA256')\n"
        "    missing = [name for name in required if not os.environ.get(name)]\n"
        "    if missing:\n"
        "        raise RuntimeError(f'EX54 Moonlight environment is incomplete: {missing}')\n"
        "    return {\n"
        "        'momentum': 0.95,\n"
        "        'nesterov': True,\n"
        "        'ns_steps': 5,\n"
        "        'weight_decay': float(os.environ['MOONLIGHT_WEIGHT_DECAY']),\n"
        "    }\n\n\n"
        "def make_optimizers(\n    model: LlamaForCausalLM, args: argparse.Namespace\n) -> tuple[list[torch.optim.Optimizer], torch.optim.Optimizer | None]:",
        "frozen Moonlight environment",
    )
    source = replace_once(
        source,
        "    elif args.method == \"muon\":\n"
        "        matrix_optimizer = ReferenceMuon(matrix, lr=args.matrix_lr)\n"
        "    else:\n"
        "        matrix_optimizer = SharedInputNewtonMuon(",
        "    elif args.method == \"muon\":\n"
        "        matrix_optimizer = ReferenceMuon(matrix, lr=args.matrix_lr)\n"
        "    elif args.method == \"moonlight\":\n"
        "        matrix_optimizer = R1MoonlightMuon(matrix, lr=args.matrix_lr, **ex54_moonlight_kwargs())\n"
        "    else:\n"
        "        matrix_optimizer = SharedInputNewtonMuon(",
        "Moonlight optimizer route",
    )
    # Bind checkpoints to this fresh experiment lineage using generic Moonlight fields.
    source = replace_once(
        source,
        '        "init_sha256": init_sha256,\n    }\n    tmp = path.with_suffix(path.suffix + ".tmp")',
        '        "init_sha256": init_sha256,\n'
        '        "moonlight_contract_sha256": os.environ.get("MOONLIGHT_CONTRACT_SHA256", ""),\n'
        '        "moonlight_selection_sha256": os.environ.get("MOONLIGHT_SELECTION_SHA256", ""),\n'
        '        "moonlight_data_inventory_sha256": os.environ.get("MOONLIGHT_DATA_INVENTORY_SHA256", ""),\n'
        '    }\n    tmp = path.with_suffix(path.suffix + ".tmp")',
        "checkpoint lineage payload",
    )
    source = replace_once(
        source,
        '        if checkpoint.get("init_sha256") != init_sha256:\n'
        '            raise RuntimeError("checkpoint initialization fingerprint does not match")',
        '        if checkpoint.get("init_sha256") != init_sha256:\n'
        '            raise RuntimeError("checkpoint initialization fingerprint does not match")\n'
        '        if checkpoint.get("moonlight_contract_sha256") != os.environ.get("MOONLIGHT_CONTRACT_SHA256"):\n'
        '            raise RuntimeError("checkpoint Moonlight contract lineage does not match")\n'
        '        if checkpoint.get("moonlight_selection_sha256") != os.environ.get("MOONLIGHT_SELECTION_SHA256"):\n'
        '            raise RuntimeError("checkpoint Moonlight selection lineage does not match")\n'
        '        if checkpoint.get("moonlight_data_inventory_sha256") != os.environ.get("MOONLIGHT_DATA_INVENTORY_SHA256"):\n'
        '            raise RuntimeError("checkpoint Moonlight data lineage does not match")',
        "checkpoint lineage validation",
    )
    source = replace_once(
        source,
        "    if isinstance(matrix_optimizer, SharedInputNewtonMuon):\n"
        "        memory.update(matrix_optimizer.memory_audit())\n"
        "    else:\n"
        "        memory.update(\n"
        "            {\n"
        "                \"k_cov_bytes\": 0,\n"
        "                \"k_inv_bytes\": 0,\n"
        "                \"k_state_bytes\": 0,\n"
        "                \"preconditioner_workspace_bytes\": 0,\n"
        "            }\n"
        "        )",
        "    if isinstance(matrix_optimizer, SharedInputNewtonMuon):\n"
        "        memory.update(matrix_optimizer.memory_audit())\n"
        "    else:\n"
        "        memory.update(\n"
        "            {\n"
        "                \"k_cov_bytes\": 0,\n"
        "                \"k_inv_bytes\": 0,\n"
        "                \"k_state_bytes\": 0,\n"
        "                \"preconditioner_workspace_bytes\": 0,\n"
        "            }\n"
        "        )\n"
        "    if isinstance(matrix_optimizer, R1MoonlightMuon):\n"
        "        moonlight_breakdown = moonlight_state_breakdown([matrix_optimizer])\n"
        "        moonlight_breakdown['moonlight_matrix_optimizer_state_bytes'] = moonlight_breakdown.pop('optimizer_state_bytes')\n"
        "        memory.update(moonlight_breakdown)\n"
        "        memory['moonlight_state_schema'] = moonlight_state_schema(matrix_optimizer)",
        "Moonlight state audit",
    )
    source = replace_once(
        source,
        '        "timing_comparable": resume_count == 0,\n'
        '        "checkpoint_path": str(checkpoint_path) if checkpoint_path.is_file() else "",',
        '        "timing_comparable": False,\n'
        '        "timing_ineligible_reason": "quality-run concurrency and transferred Moonlight tuning",\n'
        '        "moonlight_contract_sha256": os.environ.get("MOONLIGHT_CONTRACT_SHA256", ""),\n'
        '        "moonlight_selection_sha256": os.environ.get("MOONLIGHT_SELECTION_SHA256", ""),\n'
        '        "moonlight_data_inventory_sha256": os.environ.get("MOONLIGHT_DATA_INVENTORY_SHA256", ""),\n'
        '        "moonlight_hyperparameters": ex54_moonlight_kwargs() if args.method == "moonlight" else {},\n'
        '        "checkpoint_path": str(checkpoint_path) if checkpoint_path.is_file() else "",',
        "summary lineage and timing",
    )
    compile(source, "<ex54_llama_moonlight_trainer>", "exec")
    return source


def derive_long_worker(source: str) -> str:
    source = replace_once(
        source,
        "    if isinstance(matrix_optimizer, base.SharedInputNewtonMuon):\n"
        "        memory.update(matrix_optimizer.memory_audit())\n"
        "    else:\n"
        "        memory.update(\n"
        "            {\n"
        "                \"k_cov_bytes\": 0,\n"
        "                \"k_inv_bytes\": 0,\n"
        "                \"k_state_bytes\": 0,\n"
        "                \"preconditioner_workspace_bytes\": 0,\n"
        "            }\n"
        "        )",
        "    if isinstance(matrix_optimizer, base.SharedInputNewtonMuon):\n"
        "        memory.update(matrix_optimizer.memory_audit())\n"
        "    else:\n"
        "        memory.update(\n"
        "            {\n"
        "                \"k_cov_bytes\": 0,\n"
        "                \"k_inv_bytes\": 0,\n"
        "                \"k_state_bytes\": 0,\n"
        "                \"preconditioner_workspace_bytes\": 0,\n"
        "            }\n"
        "        )\n"
        "    if isinstance(matrix_optimizer, base.R1MoonlightMuon):\n"
        "        moonlight_breakdown = base.moonlight_state_breakdown([matrix_optimizer])\n"
        "        moonlight_breakdown['moonlight_matrix_optimizer_state_bytes'] = moonlight_breakdown.pop('optimizer_state_bytes')\n"
        "        memory.update(moonlight_breakdown)\n"
        "        memory['moonlight_state_schema'] = base.moonlight_state_schema(matrix_optimizer)",
        "long-worker Moonlight state audit",
    )
    source = replace_once(
        source,
        '        "timing_comparable": resume_count == 0,\n'
        '        "train_s": train_s,',
        '        "timing_comparable": False,\n'
        '        "timing_ineligible_reason": "quality-run concurrency and transferred Moonlight tuning",\n'
        '        "moonlight_selection_sha256": os.environ.get("MOONLIGHT_SELECTION_SHA256", ""),\n'
        '        "moonlight_hyperparameters": base.ex54_moonlight_kwargs(),\n'
        '        "train_s": train_s,',
        "long-worker timing and selection lineage",
    )
    compile(source, "<ex54_moonlight_long_worker_parent>", "exec")
    return source


def build(root: Path) -> DerivedSources:
    optimizer_path = root / PACKAGE_REL / "moonlight_optimizer.py"
    if not optimizer_path.is_file():
        raise FileNotFoundError(optimizer_path)
    optimizer_source = optimizer_path.read_text(encoding="utf-8")
    compile(optimizer_source, str(optimizer_path), "exec")
    audit_moonlight_transfer(root, optimizer_source)
    return DerivedSources(
        trainer=derive_trainer(
            read_pinned(root, "scripts/17_llama_swiglu_validation/train_llama_swiglu.py")
        ),
        wrapper_1b=read_pinned(
            root, "scripts/20_llama_swiglu_1b/train_llama_swiglu_1b.py"
        ),
        long_worker_parent=derive_long_worker(
            read_pinned(root, "scripts/48_llama1b_10b_multibudget/train_segment.py")
        ),
        moonlight_optimizer=optimizer_source,
    )


def write_derived(root: Path, output: Path) -> dict[str, dict[str, object]]:
    sources = build(root)
    files = {
        "train_llama_moonlight.py": sources.trainer,
        "train_llama_moonlight_1b.py": sources.wrapper_1b,
        "ex48_train_segment_parent.py": sources.long_worker_parent,
        "moonlight_optimizer.py": sources.moonlight_optimizer,
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, object]] = {}
    for name, text in files.items():
        target = output / name
        target.write_text(text, encoding="utf-8", newline="\n")
        manifest[name] = {"bytes": target.stat().st_size, "sha256": sha256_text(text)}
    return manifest


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_derived(args.source_root.resolve(), args.output.resolve()), indent=2))
