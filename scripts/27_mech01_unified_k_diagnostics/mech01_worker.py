#!/usr/bin/env python3
"""PyTorch worker for MECH-01.

This process is intentionally separate from the standard-library controller.
It loads checkpoints read-only, executes the exact model source saved with a
run, and never calls an optimizer step or writes back to a checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import socket
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
from torch import Tensor, nn


SCRIPT_VERSION = "2026-07-27.2"
FAMILIES = ("r1", "gpt_bridge", "llama124", "llama1b")
R1_FAMILIES = {"r1", "gpt_bridge"}
DEFAULT_CANDIDATES = ("none", "diag", "block4", "dense_full")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "smoke", "replay"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--source-script", type=Path)
    parser.add_argument("--triton-kernels", type=Path)
    parser.add_argument("--profile-script", type=Path)
    parser.add_argument("--data-pattern")
    parser.add_argument("--method", default="auto")
    parser.add_argument("--layers", nargs="+", type=int)
    parser.add_argument("--device-batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--probe-offsets", nargs=4, type=int, default=(0, 4096, 8192, 12288))
    parser.add_argument("--max-activation-rows", type=int, default=2048)
    parser.add_argument("--ridge-mult", type=float, default=0.2)
    parser.add_argument("--ridge-eps", type=float, default=1e-8)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--ns-steps", type=int, default=5)
    parser.add_argument("--candidates", nargs="+", default=DEFAULT_CANDIDATES)
    parser.add_argument("--spectrum-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--export-bundle-layer", type=int)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--atol", type=float, default=5e-4)
    parser.add_argument("--rtol", type=float, default=5e-3)
    parser.add_argument("--checkpoint-sha256", default="")
    parser.add_argument("--hash-checkpoint", action="store_true")
    parser.add_argument("--host-id", default="")
    parser.add_argument("--execution-domain", default="")
    args = parser.parse_args()
    if args.mode in {"preflight", "smoke"}:
        if (
            args.family is None
            or args.checkpoint is None
            or args.source_script is None
            or args.triton_kernels is None
        ):
            parser.error(
                "checkpoint modes require --family, --checkpoint, "
                "--source-script, and --triton-kernels"
            )
    if args.mode == "smoke" and not args.data_pattern:
        parser.error("--data-pattern is required for smoke")
    if args.mode in {"preflight", "smoke"} and args.family == "llama1b":
        if args.profile_script is None:
            parser.error("llama1b checkpoint modes require --profile-script")
    if args.mode == "replay":
        if (
            args.bundle is None
            or args.family is None
            or args.source_script is None
            or args.triton_kernels is None
        ):
            parser.error(
                "replay requires --bundle, --family, --source-script, "
                "and --triton-kernels"
            )
    return args


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def tensor_sha256(tensor: Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        value = value.view(torch.uint16)
    return sha256_bytes(memoryview(value.numpy()).cast("B"))


def json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return sha256_bytes(encoded)


def update_tree_digest(digest: Any, value: Any) -> None:
    """Add a deterministic, value-sensitive encoding of a checkpoint tree."""
    if isinstance(value, Tensor):
        tensor = value.detach().cpu().contiguous()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.view(torch.uint16)
        digest.update(b"tensor\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(memoryview(tensor.numpy()).cast("B"))
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"ndarray\0")
        digest.update(str(array.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(memoryview(array).cast("B"))
    elif isinstance(value, dict):
        digest.update(b"dict\0")
        for key in sorted(value, key=lambda item: repr(item)):
            update_tree_digest(digest, key)
            update_tree_digest(digest, value[key])
    elif isinstance(value, tuple):
        digest.update(b"tuple\0")
        for item in value:
            update_tree_digest(digest, item)
    elif isinstance(value, list):
        digest.update(b"list\0")
        for item in value:
            update_tree_digest(digest, item)
    elif value is None:
        digest.update(b"none\0")
    else:
        digest.update(type(value).__name__.encode("utf-8"))
        digest.update(b"\0")
        digest.update(repr(value).encode("utf-8"))
        digest.update(b"\0")


def tree_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    update_tree_digest(digest, value)
    return digest.hexdigest()


def source_sha256(path: Path) -> str:
    return sha256_file(path.resolve())


def attach_profile_provenance(
    schema: dict[str, Any], profile_path: Path | None
) -> None:
    if profile_path is None:
        schema["profile_script"] = ""
        schema["profile_script_sha256"] = ""
        return
    resolved = profile_path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    schema["profile_script"] = str(resolved)
    schema["profile_script_sha256"] = source_sha256(resolved)


def runtime_metadata(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python_executable": str(Path(sys.executable).absolute()),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "host_id": args.host_id,
        "execution_domain": args.execution_domain,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        result.update(
            {
                "gpu_name": properties.name,
                "gpu_total_memory_bytes": int(properties.total_memory),
                "gpu_capability": list(torch.cuda.get_device_capability(0)),
            }
        )
    try:
        import triton
    except Exception as exc:
        result["triton_import_error"] = repr(exc)
    else:
        result["triton"] = triton.__version__
        result["triton_module"] = str(Path(triton.__file__).resolve())
    return result


def tensor_tree_summary(value: Any) -> dict[str, Any]:
    tensor_count = 0
    tensor_bytes = 0
    tensor_dtypes: dict[str, int] = {}
    dict_count = 0
    list_count = 0
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, Tensor):
            tensor_count += 1
            tensor_bytes += item.numel() * item.element_size()
            key = str(item.dtype)
            tensor_dtypes[key] = tensor_dtypes.get(key, 0) + 1
        elif isinstance(item, dict):
            dict_count += 1
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            list_count += 1
            stack.extend(item)
    return {
        "tensor_count": tensor_count,
        "tensor_bytes": tensor_bytes,
        "tensor_dtypes": tensor_dtypes,
        "dict_count": dict_count,
        "list_count": list_count,
    }


def normalize_model_state(model_state: dict[str, Tensor]) -> tuple[dict[str, Tensor], str]:
    keys = list(model_state)
    prefixes = ("_orig_mod.", "module.")
    for prefix in prefixes:
        if keys and all(key.startswith(prefix) for key in keys):
            return {key[len(prefix) :]: value for key, value in model_state.items()}, prefix
    return model_state, ""


def infer_architecture(
    family: str, model_state: dict[str, Tensor]
) -> dict[str, Any]:
    if family in R1_FAMILIES:
        pattern = re.compile(r"^transformer\.h\.(\d+)\.mlp\.c_proj\.weight$")
        target_template = "transformer.h.{layer}.mlp.c_proj"
        target_weight_template = target_template + ".weight"
        layer_values = [
            int(match.group(1))
            for key in model_state
            if (match := pattern.match(key))
        ]
        embedding_key = "transformer.wte.weight"
        target_kind = "mlp.c_proj"
    else:
        pattern = re.compile(r"^layers\.(\d+)\.mlp\.down_proj\.weight$")
        target_template = "layers.{layer}.mlp.down_proj"
        target_weight_template = target_template + ".weight"
        layer_values = [
            int(match.group(1))
            for key in model_state
            if (match := pattern.match(key))
        ]
        embedding_key = "tok_embeddings.weight"
        target_kind = "mlp.down_proj"
    if not layer_values:
        raise RuntimeError(f"no target projection tensors found for family={family}")
    n_layer = max(layer_values) + 1
    if sorted(set(layer_values)) != list(range(n_layer)):
        raise RuntimeError(f"non-contiguous target layer set: {sorted(set(layer_values))}")
    first_weight = model_state[target_weight_template.format(layer=0)]
    embedding = model_state.get(embedding_key)
    return {
        "family": family,
        "n_layer": n_layer,
        "n_embd": int(first_weight.shape[0]),
        "target_input_width": int(first_weight.shape[1]),
        "vocab_size": int(embedding.shape[0]) if embedding is not None else None,
        "target_kind": target_kind,
        "target_module_template": target_template,
        "target_weight_template": target_weight_template,
        "target_weight_shape": list(first_weight.shape),
    }


def infer_method(family: str, source_text: str, model_state: dict[str, Tensor]) -> str:
    if family in R1_FAMILIES:
        match = re.search(r'R1_CPROJ_K_MODE\s*=\s*["\']([^"\']+)["\']', source_text)
        return match.group(1) if match else "unknown"
    down_keys = [key for key in model_state if key.endswith(".mlp.down_accum")]
    if down_keys:
        ndim = model_state[down_keys[0]].ndim
        return "newton_full" if ndim == 2 else "down_diag"
    if any(key.endswith(".mlp.mlp_in_accum") for key in model_state):
        return "down_none"
    return "muon_or_adamw"


def optimizer_schema(optimizers: Any) -> dict[str, Any]:
    if not isinstance(optimizers, list):
        return {"present": False, "reason": "optimizers is not a list"}
    rows = []
    for index, optimizer in enumerate(optimizers):
        if not isinstance(optimizer, dict):
            rows.append({"index": index, "type": type(optimizer).__name__})
            continue
        state = optimizer.get("state", {})
        groups = optimizer.get("param_groups", [])
        state_keys: set[str] = set()
        if isinstance(state, dict):
            for entry in state.values():
                if isinstance(entry, dict):
                    state_keys.update(str(key) for key in entry)
        rows.append(
            {
                "index": index,
                "state_entries": len(state) if isinstance(state, dict) else None,
                "param_groups": len(groups) if isinstance(groups, list) else None,
                "state_keys": sorted(state_keys),
                "tree": tensor_tree_summary(optimizer),
                "extra_keys": sorted(
                    key for key in optimizer if key not in {"state", "param_groups"}
                ),
            }
        )
    return {"present": True, "count": len(optimizers), "optimizers": rows}


def checkpoint_schema(
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    family: str,
    source_path: Path,
    supplied_sha256: str,
    hash_checkpoint: bool,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint root must be a dictionary")
    raw_model = checkpoint.get("model")
    if not isinstance(raw_model, dict) or not raw_model:
        raise RuntimeError("checkpoint has no non-empty model state_dict")
    if not all(isinstance(key, str) and isinstance(value, Tensor) for key, value in raw_model.items()):
        raise RuntimeError("checkpoint model state_dict contains non tensor values")
    model_state, stripped_prefix = normalize_model_state(raw_model)
    architecture = infer_architecture(family, model_state)
    source_text = source_path.read_text(encoding="utf-8")
    method = infer_method(family, source_text, model_state)
    stat = checkpoint_path.stat()
    observed_sha256 = sha256_file(checkpoint_path) if hash_checkpoint else ""
    hash_check = (
        not supplied_sha256
        or not observed_sha256
        or supplied_sha256.lower() == observed_sha256.lower()
    )
    required_r1 = {"step", "code", "model", "optimizers"}
    required_llama = {
        "format_version",
        "completed_steps",
        "model",
        "optimizers",
        "train_loader",
        "next_x",
        "next_y",
        "rng",
    }
    required = required_r1 if family in R1_FAMILIES else required_llama
    missing = sorted(required - set(checkpoint))
    embedded_source = checkpoint.get("code")
    embedded_source_sha256 = (
        sha256_bytes(embedded_source.encode("utf-8"))
        if isinstance(embedded_source, str)
        else ""
    )
    external_source_sha256 = source_sha256(source_path)
    external_source_text_sha256 = sha256_bytes(source_text.encode("utf-8"))
    embedded_source_matches = (
        True
        if family not in R1_FAMILIES
        else bool(embedded_source_sha256)
        and embedded_source_sha256 == external_source_text_sha256
    )
    schema = {
        "schema_version": 1,
        "checkpoint": str(checkpoint_path),
        "checkpoint_size_bytes": int(stat.st_size),
        "checkpoint_mtime_ns": int(stat.st_mtime_ns),
        "checkpoint_sha256_supplied": supplied_sha256,
        "checkpoint_sha256_observed": observed_sha256,
        "checkpoint_hash_checked": bool(hash_checkpoint),
        "checkpoint_hash_pass": hash_check,
        "family": family,
        "method_inferred": method,
        "top_level_keys": sorted(str(key) for key in checkpoint),
        "missing_required_keys": missing,
        "model_prefix_stripped": stripped_prefix,
        "model_state": tensor_tree_summary(model_state),
        "optimizer_state": optimizer_schema(checkpoint.get("optimizers")),
        "loader_state_present": "train_loader" in checkpoint,
        "rng_state_present": "rng" in checkpoint,
        "next_batch_present": "next_x" in checkpoint and "next_y" in checkpoint,
        "embedded_source_present": isinstance(embedded_source, str),
        "embedded_source_sha256": embedded_source_sha256,
        "embedded_source_matches_external": embedded_source_matches,
        "step": checkpoint.get("step", checkpoint.get("completed_steps")),
        "architecture": architecture,
        "source_script": str(source_path),
        "source_sha256": external_source_sha256,
        "source_text_sha256": external_source_text_sha256,
        "tree": tensor_tree_summary(checkpoint),
        "resumable_verified": family not in R1_FAMILIES and not missing,
        "passed": not missing and hash_check and embedded_source_matches,
    }
    return model_state, schema


def route_audit(family: str, source_path: Path, architecture: dict[str, Any]) -> dict[str, Any]:
    text = source_path.read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    if family in R1_FAMILIES:
        static_checks = {
            "has_target_module": "self.c_proj" in text,
            "gelu_precedes_target": bool(
                re.search(
                    r"x\s*=\s*self\.c_fc\(x\).*?x\s*=\s*F\.gelu\(x\).*?"
                    r"x\s*=\s*self\.c_proj\(x\)",
                    text,
                    re.DOTALL,
                )
            ),
            "preconditioner_right_multiply": bool(
                re.search(r"torch\.bmm\(.*?B", text, re.DOTALL)
                or re.search(r"\.grad.*?precond_inv_apply", text, re.DOTALL)
            ),
            "production_dense_or_block_ridge_formula": (
                "ridge = (diag.sum(dim=-1) / float(d)) * "
                "self.precond_ridge_mult + self.precond_eps"
            )
            in normalized,
            "production_diag_ridge_formula": (
                "ridge = cov.mean(dim=-1) * self.precond_ridge_mult + "
                "self.precond_eps"
            )
            in normalized,
            "production_cholesky_inverse": "torch.cholesky_inverse(" in text,
            "production_ns5": "zeropower_via_newtonschulz5" in text,
            "production_ns5_coefficients": all(
                coefficient in normalized
                for coefficient in ("3.4445", "-4.7750", "2.0315")
            ),
        }
        expected_input = 4 * int(architecture["n_embd"])
        route = "F.gelu(c_fc(x)) -> mlp.c_proj"
    else:
        static_checks = {
            "has_target_module": "self.down_proj" in text,
            "swiglu_precedes_target": bool(
                re.search(
                    r"hidden\s*=\s*F\.silu\(self\.gate_proj\(x\)\)"
                    r"\s*\*\s*self\.up_proj\(x\).*?"
                    r"self\.down_proj\(hidden\)",
                    text,
                    re.DOTALL,
                )
            ),
            "preconditioner_right_multiply": bool(
                re.search(r"torch\.mm\(parameter\.grad,\s*inverse", text)
            ),
            "production_dense_ridge_formula": (
                "ridge = diagonal.mean() * self.input_ridge + 1e-8"
            )
            in normalized,
            "production_diag_inverse_formula": (
                "torch.reciprocal( covariance + ridge )" in normalized
                or "torch.reciprocal(covariance + ridge)" in normalized
            ),
            "production_cholesky_inverse": "torch.cholesky_inverse(" in text,
            "production_ns5": "zeropower_via_newtonschulz5" in text,
            "production_ns5_coefficients": all(
                coefficient in normalized
                for coefficient in ("3.4445", "-4.7750", "2.0315")
            ),
        }
        expected_input = int(architecture["target_input_width"])
        route = "silu(gate_proj(x)) * up_proj(x) -> mlp.down_proj"
    shape_check = expected_input == int(architecture["target_input_width"])
    checks = {**static_checks, "target_input_shape_matches": shape_check}
    return {
        "family": family,
        "source_script": str(source_path),
        "source_sha256": source_sha256(source_path),
        "target_route": route,
        "expected_input_width": expected_input,
        "observed_weight_input_width": architecture["target_input_width"],
        "checks": checks,
        "passed": all(checks.values()),
    }


def install_source_paths(source_path: Path, triton_path: Path | None) -> None:
    candidates = []
    if triton_path is not None:
        candidates.append(triton_path.resolve().parent)
    candidates.append(source_path.resolve().parent)
    for candidate in reversed(candidates):
        value = str(candidate)
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)


def validate_triton_module(triton_path: Path | None) -> dict[str, Any]:
    import triton_kernels

    observed = Path(triton_kernels.__file__).resolve()
    expected = triton_path.resolve() if triton_path is not None else observed
    return {
        "expected": str(expected),
        "observed": str(observed),
        "sha256": source_sha256(observed),
        "passed": observed == expected,
    }


def load_r1_source(source_path: Path, triton_path: Path | None) -> Any:
    install_source_paths(source_path, triton_path)
    text = source_path.read_text(encoding="utf-8")
    marker = re.search(r"^#\s*-+\s*\n# int main\s*$", text, re.MULTILINE)
    if marker is None:
        marker = re.search(r"^# int main\s*$", text, re.MULTILINE)
    if marker is None:
        raise RuntimeError("could not find the official '# int main' boundary")
    prefix = text[: marker.start()]
    namespace: dict[str, Any] = {
        "__name__": "mech01_exact_r1_source",
        "__file__": str(source_path),
    }
    # This worker uses ``from __future__ import annotations``.  Python's
    # compile() inherits the caller's future flags unless dont_inherit=True is
    # set, even though the archived R1 source itself does not enable postponed
    # annotations.  Inheriting that flag turns ``Tensor`` into the string
    # ``"Tensor"`` and torch.library.custom_op cannot infer its schema under
    # Torch 2.8.  Compile the archived prefix with its own future flags only,
    # matching normal execution of the saved training source.
    code = compile(
        prefix,
        str(source_path),
        "exec",
        dont_inherit=True,
    )
    exec(code, namespace)
    required = ("GPT", "GPTConfig", "zeropower_via_newtonschulz5")
    missing = [name for name in required if name not in namespace]
    if missing:
        raise RuntimeError(f"R1 source prefix missing definitions: {missing}")
    return namespace


def load_llama_source(source_path: Path, triton_path: Path | None) -> Any:
    install_source_paths(source_path, triton_path)
    module_name = "mech01_exact_llama_source"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import source: {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_source_runtime(
    family: str, source_path: Path, triton_path: Path | None
) -> tuple[Any, Callable[..., Tensor], dict[str, Any]]:
    if family in R1_FAMILIES:
        module = load_r1_source(source_path, triton_path)
        ns = module["zeropower_via_newtonschulz5"]
    else:
        module = load_llama_source(source_path, triton_path)
        ns = module.zeropower_via_newtonschulz5
    kernel = validate_triton_module(triton_path)
    if triton_path is not None and not kernel["passed"]:
        raise RuntimeError(f"triton_kernels provenance mismatch: {kernel}")
    return module, ns, kernel


def build_model(
    family: str,
    module: Any,
    architecture: dict[str, Any],
    method: str,
) -> nn.Module:
    if family in R1_FAMILIES:
        config = module["GPTConfig"](
            vocab_size=int(architecture["vocab_size"]),
            n_layer=int(architecture["n_layer"]),
            n_head=12,
            n_embd=int(architecture["n_embd"]),
        )
        return module["GPT"](config)
    if family == "llama1b":
        n_head = 16
        intermediate = int(architecture["target_input_width"])
    else:
        n_head = 12
        intermediate = int(architecture["target_input_width"])
    config = module.ModelConfig(
        vocab_size=int(architecture["vocab_size"]),
        n_layer=int(architecture["n_layer"]),
        n_head=n_head,
        n_embd=int(architecture["n_embd"]),
        intermediate_size=intermediate,
        sequence_length=1024,
        rms_norm_eps=1e-6,
        rope_base=10000.0,
    )
    accepted = method
    if accepted not in module.METHODS:
        accepted = "down_none"
    return module.LlamaForCausalLM(config, accepted)


def configure_source_runtime_globals(
    family: str,
    module: Any,
    method: str,
) -> dict[str, Any]:
    """Recreate globals that the archived trainer normally sets in its main.

    The R1 source deliberately reads R1_METHOD and R1_CPROJ_K_MODE from the
    environment after its ``# int main`` boundary.  MECH-01 executes only the
    definition prefix so that importing the archived source cannot start a
    training job.  Model constructors in that prefix still consult
    R1_CPROJ_K_MODE, therefore the diagnostic worker must inject the explicit
    audited method before instantiating GPT.
    """
    if family not in R1_FAMILIES:
        return {
            "family": family,
            "requested_method": method,
            "injected_globals": {},
            "passed": True,
        }
    if not isinstance(module, dict):
        raise TypeError("R1 source runtime must be a dictionary namespace")
    allowed = {"none", "diag", "block4"}
    if method not in allowed:
        raise RuntimeError(
            f"R1 diagnostic model requires an explicit K mode in "
            f"{sorted(allowed)}; observed method={method!r}"
        )
    module["R1_METHOD"] = method
    module["R1_CPROJ_K_MODE"] = method
    injected = {
        "R1_METHOD": module["R1_METHOD"],
        "R1_CPROJ_K_MODE": module["R1_CPROJ_K_MODE"],
    }
    return {
        "family": family,
        "requested_method": method,
        "injected_globals": injected,
        "passed": injected
        == {
            "R1_METHOD": method,
            "R1_CPROJ_K_MODE": method,
        },
    }


def target_modules_and_weights(
    model: nn.Module, family: str, layers: list[int]
) -> tuple[dict[int, nn.Module], dict[int, nn.Parameter], dict[int, str]]:
    modules: dict[int, nn.Module] = {}
    weights: dict[int, nn.Parameter] = {}
    names: dict[int, str] = {}
    for layer in layers:
        if family in R1_FAMILIES:
            target = model.transformer.h[layer].mlp.c_proj
            name = f"transformer.h.{layer}.mlp.c_proj.weight"
        else:
            target = model.layers[layer].mlp.down_proj
            name = f"layers.{layer}.mlp.down_proj.weight"
        modules[layer] = target
        weights[layer] = target.weight
        names[layer] = name
    return modules, weights, names


def optimizer_parameter_order(model: nn.Module, family: str) -> list[tuple[str, nn.Parameter]]:
    if family in R1_FAMILIES:
        return [
            (f"transformer.h.{name}", parameter)
            for name, parameter in model.transformer.h.named_parameters()
        ]
    return list(model.matrix_named_parameters())


def flatten_param_ids(groups: Any) -> list[Any]:
    result = []
    if isinstance(groups, list):
        for group in groups:
            if isinstance(group, dict) and isinstance(group.get("params"), list):
                result.extend(group["params"])
    return result


def extract_target_momenta(
    checkpoint: dict[str, Any],
    model: nn.Module,
    family: str,
    target_names: dict[int, str],
) -> tuple[dict[int, Tensor], dict[str, Any]]:
    optimizers = checkpoint.get("optimizers")
    if not isinstance(optimizers, list) or len(optimizers) < 2:
        raise RuntimeError("matrix optimizer state is missing")
    matrix_state = optimizers[1]
    saved_ids = flatten_param_ids(matrix_state.get("param_groups"))
    current = optimizer_parameter_order(model, family)
    if len(saved_ids) != len(current):
        raise RuntimeError(
            f"matrix optimizer parameter count mismatch: saved={len(saved_ids)} "
            f"model={len(current)}"
        )
    name_to_index = {name: index for index, (name, _) in enumerate(current)}
    state = matrix_state.get("state", {})
    momenta: dict[int, Tensor] = {}
    rows = []
    momentum_key = "momentum_buffer" if family in R1_FAMILIES else "momentum"
    for layer, target_name in target_names.items():
        if target_name not in name_to_index:
            raise RuntimeError(f"target is absent from matrix optimizer order: {target_name}")
        index = name_to_index[target_name]
        saved_id = saved_ids[index]
        entry = state.get(saved_id, state.get(str(saved_id), {}))
        value = entry.get(momentum_key) if isinstance(entry, dict) else None
        parameter = current[index][1]
        present = isinstance(value, Tensor)
        if present:
            if tuple(value.shape) != tuple(parameter.shape):
                raise RuntimeError(
                    f"momentum shape mismatch for {target_name}: "
                    f"{tuple(value.shape)} vs {tuple(parameter.shape)}"
                )
            momenta[layer] = value.detach().float().cpu().clone()
        else:
            momenta[layer] = torch.zeros_like(parameter, device="cpu", dtype=torch.float32)
        rows.append(
            {
                "layer": layer,
                "target_name": target_name,
                "optimizer_parameter_index": index,
                "saved_parameter_id": str(saved_id),
                "momentum_key": momentum_key,
                "momentum_present": present,
                "momentum_sha256": tensor_sha256(momenta[layer]),
            }
        )
    return momenta, {
        "momentum_convention": momentum_convention(family),
        "targets": rows,
        "all_present": all(row["momentum_present"] for row in rows),
    }


def momentum_convention(family: str) -> str:
    return "r1_sum_nesterov" if family in R1_FAMILIES else "llama_ema_nesterov"


def checkpoint_aux_signature(checkpoint: dict[str, Any]) -> dict[str, Any]:
    optimizers = checkpoint.get("optimizers")
    loader = checkpoint.get("train_loader")
    rng = checkpoint.get("rng")
    next_rows = {}
    for key in ("next_x", "next_y"):
        value = checkpoint.get(key)
        if isinstance(value, Tensor):
            next_rows[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": tensor_sha256(value),
            }
    return {
        "optimizer_schema": optimizer_schema(optimizers),
        "optimizer_state_sha256": tree_sha256(optimizers),
        "loader_state_sha256": json_sha256(loader) if loader is not None else "",
        "rng_state_sha256": tree_sha256(rng),
        "rng_tree": tensor_tree_summary(rng),
        "next_batch": next_rows,
        "next_batch_sha256": tree_sha256(
            {key: checkpoint.get(key) for key in ("next_x", "next_y")}
        ),
    }


def model_state_signature(
    model: nn.Module, target_names: Iterable[str]
) -> dict[str, Any]:
    target_set = set(target_names)
    parameter_rows = []
    full_target_hashes = {}
    for name, parameter in model.named_parameters():
        flat = parameter.detach().reshape(-1)
        if flat.numel():
            indices = sorted(set((0, flat.numel() // 2, flat.numel() - 1)))
            sample = flat[indices].float().cpu().tolist()
        else:
            sample = []
        parameter_rows.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "version": int(parameter._version),
                "sample": sample,
            }
        )
        if name in target_set:
            full_target_hashes[name] = tensor_sha256(parameter)
    state_rows = []
    for name, value in model.state_dict().items():
        flat = value.detach().reshape(-1)
        sample = flat[: min(3, flat.numel())].float().cpu().tolist()
        state_rows.append(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sample": sample,
            }
        )
    # The row summaries make failures inspectable; the full digests make the
    # unchanged-state claim content-sensitive for every parameter/buffer.
    full_parameters = {
        name: parameter.detach() for name, parameter in model.named_parameters()
    }
    full_state = {name: value.detach() for name, value in model.state_dict().items()}
    return {
        "full_parameter_sha256": tree_sha256(full_parameters),
        "full_state_dict_sha256": tree_sha256(full_state),
        "parameter_rows_sha256": json_sha256(parameter_rows),
        "state_rows_sha256": json_sha256(state_rows),
        "target_tensor_sha256": full_target_hashes,
        "parameter_count": len(parameter_rows),
        "state_tensor_count": len(state_rows),
    }


def read_fineweb_batches(
    pattern: str,
    offsets: list[int],
    batch_size: int,
    sequence_length: int,
) -> tuple[list[tuple[Tensor, Tensor]], dict[str, Any]]:
    files = sorted(Path(path).resolve() for path in glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no files match data pattern: {pattern}")
    path = files[0]
    with path.open("rb") as handle:
        header = np.frombuffer(handle.read(256 * 4), dtype=np.int32)
    if len(header) != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise RuntimeError(f"unsupported FineWeb shard header: {path}")
    token_count = int(header[2])
    count = batch_size * sequence_length + 1
    intervals = []
    batches = []
    rows = []
    tokens = np.memmap(path, dtype=np.uint16, mode="r", offset=256 * 4, shape=(token_count,))
    for index, offset in enumerate(offsets):
        if offset < 0 or offset + count > token_count:
            raise RuntimeError(
                f"probe window {offset}:{offset + count} exceeds shard tokens={token_count}"
            )
        interval = (offset, offset + count)
        intervals.append(interval)
        window = np.asarray(tokens[offset : offset + count], dtype=np.int64)
        x = torch.from_numpy(window[:-1].copy()).view(batch_size, sequence_length)
        y = torch.from_numpy(window[1:].copy()).view(batch_size, sequence_length)
        role = "build" if index < 2 else "heldout"
        rows.append(
            {
                "index": index,
                "role": role,
                "shard": str(path),
                "offset": offset,
                "exclusive_end": offset + count,
                "x_sha256": tensor_sha256(x),
                "y_sha256": tensor_sha256(y),
            }
        )
        batches.append((x, y))
    overlaps = []
    for left in range(len(intervals)):
        for right in range(left + 1, len(intervals)):
            a0, a1 = intervals[left]
            b0, b1 = intervals[right]
            if max(a0, b0) < min(a1, b1):
                overlaps.append([left, right])
    contract = {
        "data_pattern": pattern,
        "selected_shard": str(path),
        "shard_size_bytes": path.stat().st_size,
        "shard_token_count": token_count,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "rows_per_batch": batch_size * sequence_length,
        "batches": rows,
        "overlapping_pairs": overlaps,
        "build_heldout_disjoint": not overlaps,
        "contract_sha256": json_sha256(rows),
    }
    if overlaps:
        raise RuntimeError(f"probe windows overlap: {overlaps}")
    return batches, contract


def select_layers(n_layer: int, requested: list[int] | None) -> list[int]:
    if requested:
        layers = list(dict.fromkeys(requested))
    else:
        layers = [0, n_layer // 2, n_layer - 1]
    if any(layer < 0 or layer >= n_layer for layer in layers):
        raise ValueError(f"layers {layers} out of range 0..{n_layer - 1}")
    return layers


def deterministic_subsample_rows(value: Tensor, maximum: int) -> Tensor:
    if value.size(0) <= maximum:
        return value
    indices = torch.linspace(
        0, value.size(0) - 1, maximum, device=value.device
    ).round().long()
    return value.index_select(0, indices)


def collect_probe_pass(
    model: nn.Module,
    modules: dict[int, nn.Module],
    weights: dict[int, nn.Parameter],
    batches: list[tuple[Tensor, Tensor]],
    max_rows: int,
) -> dict[str, Any]:
    current: dict[int, Tensor] = {}
    handles = []
    for layer, module in modules.items():
        def capture(_module: nn.Module, inputs: tuple[Any, ...], layer_index: int = layer) -> None:
            value = inputs[0]
            if not isinstance(value, Tensor):
                raise TypeError(f"target input for layer {layer_index} is not a tensor")
            current[layer_index] = value.detach()
        handles.append(module.register_forward_pre_hook(capture))

    build: dict[int, list[Tensor]] = {layer: [] for layer in modules}
    heldout_gradients: dict[int, list[Tensor]] = {layer: [] for layer in modules}
    heldout_losses: list[float] = []
    previous_training = model.training
    original_requires_grad = {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    }
    target_ids = {id(parameter) for parameter in weights.values()}
    try:
        for parameter in model.parameters():
            parameter.requires_grad_(id(parameter) in target_ids)
        model.train()
        autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        for batch_index, (x_cpu, y_cpu) in enumerate(batches):
            x = x_cpu.cuda(non_blocking=False)
            y = y_cpu.cuda(non_blocking=False)
            current.clear()
            model.zero_grad(set_to_none=True)
            if batch_index < 2:
                with torch.no_grad(), autocast:
                    _, loss = model(
                        x, y, return_logits=False, precond_flag=False
                    )
            else:
                with autocast:
                    _, loss = model(
                        x, y, return_logits=False, precond_flag=False
                    )
                if loss is None:
                    raise RuntimeError("held-out forward returned no loss")
                loss.backward()
                heldout_losses.append(float(loss.detach().cpu()))
            if set(current) != set(modules):
                raise RuntimeError(
                    f"activation hook coverage mismatch: {sorted(current)} vs {sorted(modules)}"
                )
            for layer, activation in current.items():
                flat = activation.flatten(0, -2).float()
                if batch_index < 2:
                    build[layer].append(
                        deterministic_subsample_rows(flat, max_rows).cpu()
                    )
                else:
                    gradient = weights[layer].grad
                    if gradient is None:
                        raise RuntimeError(f"no gradient captured for layer {layer}")
                    heldout_gradients[layer].append(gradient.detach().float().cpu().clone())
        return {
            "build_activations": {
                layer: torch.cat(values, dim=0) for layer, values in build.items()
            },
            "heldout_gradients": {
                layer: torch.stack(values).mean(dim=0)
                for layer, values in heldout_gradients.items()
            },
            "heldout_losses": heldout_losses,
        }
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(original_requires_grad[name])
        model.train(previous_training)


def compare_probe_passes(
    first: dict[str, Any], second: dict[str, Any], atol: float, rtol: float
) -> dict[str, Any]:
    rows = []
    for collection in ("build_activations", "heldout_gradients"):
        for layer in sorted(first[collection]):
            a = first[collection][layer]
            b = second[collection][layer]
            max_abs = float((a - b).abs().max()) if a.numel() else 0.0
            passed = bool(torch.allclose(a, b, atol=atol, rtol=rtol))
            rows.append(
                {
                    "collection": collection,
                    "layer": layer,
                    "sha256_a": tensor_sha256(a),
                    "sha256_b": tensor_sha256(b),
                    "max_abs_diff": max_abs,
                    "passed": passed,
                }
            )
    loss_a = torch.tensor(first["heldout_losses"], dtype=torch.float64)
    loss_b = torch.tensor(second["heldout_losses"], dtype=torch.float64)
    loss_pass = bool(torch.allclose(loss_a, loss_b, atol=atol, rtol=rtol))
    return {
        "atol": atol,
        "rtol": rtol,
        "tensor_comparisons": rows,
        "heldout_losses_a": first["heldout_losses"],
        "heldout_losses_b": second["heldout_losses"],
        "heldout_loss_max_abs_diff": float((loss_a - loss_b).abs().max()),
        "heldout_losses_pass": loss_pass,
        "passed": loss_pass and all(row["passed"] for row in rows),
    }


def covariance_from_activations(activations: Tensor) -> Tensor:
    x = activations.float()
    return x.T @ x / float(x.size(0))


def ridge_vector(
    diagonal: Tensor,
    family: str,
    representation: str,
    ridge_mult: float,
    ridge_eps: float,
) -> tuple[Tensor, str]:
    width = diagonal.numel()
    if family in R1_FAMILIES and representation in {"diag", "block4"}:
        if width % 4:
            raise RuntimeError(f"R1 block representation requires width divisible by 4: {width}")
        chunks = diagonal.view(4, width // 4)
        per_block = chunks.mean(dim=1) * ridge_mult + ridge_eps
        return per_block.repeat_interleave(width // 4), "per_r1_quarter_mean"
    value = diagonal.mean() * ridge_mult + ridge_eps
    return torch.full_like(diagonal, value), "global_diagonal_mean"


def representation_inverse(
    covariance: Tensor,
    family: str,
    representation: str,
    ridge_mult: float,
    ridge_eps: float,
) -> tuple[Tensor | None, dict[str, Any]]:
    width = covariance.size(0)
    diagonal = covariance.diagonal()
    if representation == "none":
        return None, {
            "representation": "none",
            "ridge_policy": "none",
            "ridge_mean": 0.0,
            "cholesky_info_max": 0,
            "inverse_residual_relative": 0.0,
        }
    ridge, policy = ridge_vector(
        diagonal, family, representation, ridge_mult, ridge_eps
    )
    if representation == "diag":
        inverse = torch.reciprocal(diagonal + ridge)
        residual = float(((diagonal + ridge) * inverse - 1).abs().max())
        return inverse, {
            "representation": representation,
            "ridge_policy": policy,
            "ridge_mean": float(ridge.mean()),
            "ridge_min": float(ridge.min()),
            "ridge_max": float(ridge.max()),
            "cholesky_info_max": 0,
            "inverse_residual_relative": residual,
        }
    if representation == "block4":
        if family not in R1_FAMILIES:
            raise ValueError("block4 has no architecture-defined meaning for LLaMA down_proj")
        block_width = width // 4
        inverse = torch.zeros_like(covariance)
        residuals = []
        info_values = []
        for block in range(4):
            start = block * block_width
            stop = start + block_width
            work = covariance[start:stop, start:stop].clone()
            block_ridge = ridge[start]
            work.diagonal().add_(block_ridge)
            factor, info = torch.linalg.cholesky_ex(
                work, upper=False, check_errors=False
            )
            info_values.append(int(info.item()))
            if int(info.item()) != 0:
                raise FloatingPointError(
                    f"block4 Cholesky failed for block={block}, info={int(info.item())}"
                )
            block_inverse = torch.cholesky_inverse(factor, upper=False)
            inverse[start:stop, start:stop] = block_inverse
            identity = torch.eye(block_width, device=work.device, dtype=work.dtype)
            residuals.append(
                float(torch.linalg.vector_norm(work @ block_inverse - identity)
                      / torch.linalg.vector_norm(identity))
            )
        return inverse, {
            "representation": representation,
            "ridge_policy": policy,
            "ridge_mean": float(ridge.mean()),
            "ridge_min": float(ridge.min()),
            "ridge_max": float(ridge.max()),
            "cholesky_info_max": max(info_values),
            "inverse_residual_relative": max(residuals),
        }
    if representation != "dense_full":
        raise ValueError(f"unsupported representation: {representation}")
    work = covariance.clone()
    work.diagonal().add_(ridge)
    factor, info = torch.linalg.cholesky_ex(work, upper=False, check_errors=False)
    if int(info.item()) != 0:
        raise FloatingPointError(f"dense Cholesky failed with info={int(info.item())}")
    inverse = torch.cholesky_inverse(factor, upper=False)
    identity = torch.eye(width, device=work.device, dtype=work.dtype)
    residual = float(
        torch.linalg.vector_norm(work @ inverse - identity)
        / torch.linalg.vector_norm(identity)
    )
    return inverse, {
        "representation": representation,
        "ridge_policy": policy,
        "ridge_mean": float(ridge.mean()),
        "ridge_min": float(ridge.min()),
        "ridge_max": float(ridge.max()),
        "cholesky_info_max": int(info.item()),
        "inverse_residual_relative": residual,
    }


def apply_inverse(gradient: Tensor, inverse: Tensor | None) -> Tensor:
    if inverse is None:
        return gradient.clone()
    if inverse.ndim == 1:
        return gradient * inverse
    return gradient @ inverse


def momentum_lookahead(
    gradient: Tensor, old_momentum: Tensor, family: str, beta: float
) -> tuple[Tensor, Tensor]:
    if family in R1_FAMILIES:
        new_momentum = old_momentum * beta + gradient
        lookahead = gradient + new_momentum * beta
    else:
        new_momentum = old_momentum * beta + gradient * (1.0 - beta)
        lookahead = gradient * (1.0 - beta) + new_momentum * beta
    return new_momentum, lookahead


def matrix_cosine(left: Tensor, right: Tensor) -> float:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) == 0:
        return float("nan")
    return float(torch.sum(left * right) / denominator)


def covariance_metrics(
    covariance: Tensor,
    family: str,
    ridge_mult: float,
    ridge_eps: float,
    spectrum_dtype: str,
) -> dict[str, Any]:
    diagonal = covariance.diagonal()
    total_sq = torch.sum(covariance.square())
    diag_sq = torch.sum(diagonal.square())
    offdiag_sq = (total_sq - diag_sq).clamp_min(0)
    ridge, policy = ridge_vector(
        diagonal, family, "dense_full", ridge_mult, ridge_eps
    )
    damped = covariance.clone()
    damped.diagonal().add_(ridge)
    eig_input = damped.double() if spectrum_dtype == "float64" else damped.float()
    eigenvalues = torch.linalg.eigvalsh(eig_input)
    positive = eigenvalues.clamp_min(torch.finfo(eigenvalues.dtype).tiny)
    probabilities = positive / positive.sum()
    entropy = -torch.sum(probabilities * torch.log(probabilities))
    effective_rank = torch.exp(entropy)
    result = {
        "activation_width": int(covariance.size(0)),
        "diag_mean": float(diagonal.mean()),
        "diag_std": float(diagonal.std(unbiased=False)),
        "diag_cv": float(
            diagonal.std(unbiased=False) / diagonal.mean().abs().clamp_min(1e-30)
        ),
        "diag_p05": float(torch.quantile(diagonal, 0.05)),
        "diag_p95": float(torch.quantile(diagonal, 0.95)),
        "offdiag_frobenius": float(torch.sqrt(offdiag_sq)),
        "offdiag_energy_fraction": float(offdiag_sq / total_sq.clamp_min(1e-30)),
        "damped_ridge_policy": policy,
        "damped_ridge": float(ridge.mean()),
        "damped_eigen_min": float(eigenvalues[0]),
        "damped_eigen_max": float(eigenvalues[-1]),
        "damped_condition_number": float(eigenvalues[-1] / positive[0]),
        "damped_effective_rank": float(effective_rank),
        "damped_top1_mass": float(positive[-1] / positive.sum()),
        "damped_spectrum_finite": bool(torch.isfinite(eigenvalues).all()),
    }
    if family in R1_FAMILIES and covariance.size(0) % 4 == 0:
        width = covariance.size(0) // 4
        within_sq = covariance.new_zeros(())
        for block in range(4):
            start = block * width
            stop = start + width
            value = covariance[start:stop, start:stop]
            within_sq += torch.sum(value.square())
        result["within_block_energy_fraction"] = float(
            within_sq / total_sq.clamp_min(1e-30)
        )
        result["cross_block_energy_fraction"] = float(
            (total_sq - within_sq).clamp_min(0) / total_sq.clamp_min(1e-30)
        )
    return result


def diagnostic_results(
    activations: Tensor,
    gradient: Tensor,
    old_momentum: Tensor,
    family: str,
    candidates: list[str],
    ridge_mult: float,
    ridge_eps: float,
    beta: float,
    ns_steps: int,
    spectrum_dtype: str,
    production_ns: Callable[..., Tensor],
) -> dict[str, Any]:
    device = torch.device("cuda")
    activation_gpu = activations.to(device=device, dtype=torch.float32)
    gradient_gpu = gradient.to(device=device, dtype=torch.float32)
    momentum_gpu = old_momentum.to(device=device, dtype=torch.float32)
    covariance = covariance_from_activations(activation_gpu)
    covariance_row = covariance_metrics(
        covariance, family, ridge_mult, ridge_eps, spectrum_dtype
    )
    covariance_row["activation_rows"] = int(activation_gpu.size(0))
    covariance_row["n_eff_over_d"] = float(
        activation_gpu.size(0) / activation_gpu.size(1)
    )
    candidate_rows: dict[str, Any] = {}
    updates: dict[str, Tensor] = {}
    for candidate in candidates:
        if candidate == "block4" and family not in R1_FAMILIES:
            candidate_rows[candidate] = {
                "excluded": True,
                "reason": "no architecture-defined four-way partition for LLaMA down_proj",
            }
            continue
        inverse, inverse_metrics = representation_inverse(
            covariance, family, candidate, ridge_mult, ridge_eps
        )
        preconditioned = apply_inverse(gradient_gpu, inverse)
        _, lookahead = momentum_lookahead(
            preconditioned, momentum_gpu, family, beta
        )
        update = production_ns(lookahead, steps=ns_steps).float()
        updates[candidate] = update
        candidate_rows[candidate] = {
            **inverse_metrics,
            "preconditioned_gradient_norm": float(
                torch.linalg.vector_norm(preconditioned)
            ),
            "lookahead_norm": float(torch.linalg.vector_norm(lookahead)),
            "update_norm": float(torch.linalg.vector_norm(update)),
            "gradient_to_lookahead_cosine": matrix_cosine(gradient_gpu, lookahead),
            "update_finite": bool(torch.isfinite(update).all()),
        }
    reference = updates.get("none")
    if reference is not None:
        for candidate, update in updates.items():
            candidate_rows[candidate]["update_cosine_to_none"] = matrix_cosine(
                update, reference
            )
            candidate_rows[candidate]["update_relative_norm_to_none"] = float(
                torch.linalg.vector_norm(update)
                / torch.linalg.vector_norm(reference).clamp_min(1e-30)
            )
    result = {
        "family": family,
        "momentum_convention": momentum_convention(family),
        "ridge_mult": ridge_mult,
        "ridge_eps": ridge_eps,
        "momentum_beta": beta,
        "ns_steps": ns_steps,
        "covariance": covariance_row,
        "candidates": candidate_rows,
    }
    del activation_gpu, gradient_gpu, momentum_gpu, covariance, updates
    torch.cuda.empty_cache()
    return result


def finite_numbers(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite_numbers(child) for child in value.values())
    if isinstance(value, list):
        return all(finite_numbers(child) for child in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def production_ns_parity(
    production_ns: Callable[..., Tensor], steps: int, atol: float, rtol: float
) -> dict[str, Any]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260724)
    matrix = torch.randn(48, 64, generator=generator, device="cuda", dtype=torch.float32)
    observed_a = production_ns(matrix, steps=steps).float()
    observed_b = production_ns(matrix, steps=steps).float()
    return {
        "input_sha256": tensor_sha256(matrix),
        "output_sha256_a": tensor_sha256(observed_a),
        "output_sha256_b": tensor_sha256(observed_b),
        "max_abs_diff": float((observed_a - observed_b).abs().max()),
        "allclose": bool(torch.allclose(observed_a, observed_b, atol=atol, rtol=rtol)),
        "finite": bool(torch.isfinite(observed_a).all()),
    }


def save_bundle(
    path: Path,
    layer: int,
    family: str,
    activations: Tensor,
    gradient: Tensor,
    momentum: Tensor,
    diagnostic_config: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    covariance = covariance_from_activations(activations.float())
    payload = {
        "schema_version": 1,
        "purpose": "MECH-01 fixed tensor-bundle runtime equivalence control",
        "family": family,
        "layer": layer,
        "momentum_convention": momentum_convention(family),
        "activation_build": activations.detach().float().cpu().contiguous(),
        "fresh_heldout_gradient": gradient.detach().float().cpu().contiguous(),
        "historical_momentum": momentum.detach().float().cpu().contiguous(),
        "fresh_covariance": covariance.detach().float().cpu().contiguous(),
        "diagnostic_config": diagnostic_config,
        "provenance": provenance,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "tensor_shapes": {
            "activation_build": list(payload["activation_build"].shape),
            "fresh_heldout_gradient": list(payload["fresh_heldout_gradient"].shape),
            "historical_momentum": list(payload["historical_momentum"].shape),
            "fresh_covariance": list(payload["fresh_covariance"].shape),
        },
        "tensor_sha256": {
            "activation_build": tensor_sha256(payload["activation_build"]),
            "fresh_heldout_gradient": tensor_sha256(
                payload["fresh_heldout_gradient"]
            ),
            "historical_momentum": tensor_sha256(payload["historical_momentum"]),
            "fresh_covariance": tensor_sha256(payload["fresh_covariance"]),
        },
    }


def write_checks(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ("check", "passed", "detail")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def write_diagnostics_csv(path: Path, results: dict[int, Any]) -> None:
    rows = []
    for layer, result in results.items():
        base = {
            "layer": layer,
            "family": result["family"],
            "activation_rows": result["covariance"]["activation_rows"],
            "activation_width": result["covariance"]["activation_width"],
            "diag_cv": result["covariance"]["diag_cv"],
            "offdiag_energy_fraction": result["covariance"]["offdiag_energy_fraction"],
            "damped_condition_number": result["covariance"]["damped_condition_number"],
            "damped_effective_rank": result["covariance"]["damped_effective_rank"],
        }
        for candidate, values in result["candidates"].items():
            rows.append(
                {
                    **base,
                    "candidate": candidate,
                    "excluded": values.get("excluded", False),
                    "ridge_policy": values.get("ridge_policy", ""),
                    "ridge_mean": values.get("ridge_mean", ""),
                    "inverse_residual_relative": values.get(
                        "inverse_residual_relative", ""
                    ),
                    "preconditioned_gradient_norm": values.get(
                        "preconditioned_gradient_norm", ""
                    ),
                    "update_norm": values.get("update_norm", ""),
                    "update_cosine_to_none": values.get(
                        "update_cosine_to_none", ""
                    ),
                    "update_finite": values.get("update_finite", ""),
                }
            )
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_preflight(args: argparse.Namespace) -> bool:
    output = args.output_dir.resolve()
    checkpoint_path = args.checkpoint.resolve()
    source_path = args.source_script.resolve()
    triton_path = args.triton_kernels.resolve()
    before_stat = checkpoint_path.stat()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state, schema = checkpoint_schema(
        checkpoint_path,
        checkpoint,
        args.family,
        source_path,
        args.checkpoint_sha256,
        args.hash_checkpoint,
    )
    attach_profile_provenance(schema, args.profile_script)
    route = route_audit(args.family, source_path, schema["architecture"])
    triton_static = {
        "path": str(triton_path),
        "exists": triton_path.is_file(),
        "sha256": source_sha256(triton_path) if triton_path.is_file() else "",
        "source_imports_triton_kernels": "triton_kernels"
        in source_path.read_text(encoding="utf-8"),
    }
    triton_static["passed"] = bool(
        triton_static["exists"] and triton_static["source_imports_triton_kernels"]
    )
    auxiliary = checkpoint_aux_signature(checkpoint)
    after_stat = checkpoint_path.stat()
    file_unchanged = (
        before_stat.st_size == after_stat.st_size
        and before_stat.st_mtime_ns == after_stat.st_mtime_ns
    )
    checks = [
        {"check": "checkpoint_schema", "passed": schema["passed"], "detail": schema["missing_required_keys"]},
        {"check": "static_route_audit", "passed": route["passed"], "detail": route["checks"]},
        {"check": "triton_kernel_static_provenance", "passed": triton_static["passed"], "detail": triton_static},
        {"check": "checkpoint_file_unchanged", "passed": file_unchanged, "detail": str(checkpoint_path)},
        {
            "check": "target_layers_contiguous",
            "passed": schema["architecture"]["n_layer"] > 0,
            "detail": schema["architecture"]["n_layer"],
        },
    ]
    passed = all(row["passed"] for row in checks)
    atomic_json(output / "checkpoint_schema.json", schema)
    atomic_json(output / "route_audit.json", route)
    atomic_json(output / "triton_kernel_audit.json", triton_static)
    atomic_json(output / "checkpoint_auxiliary_state.json", auxiliary)
    atomic_json(output / "runtime.json", runtime_metadata(args))
    write_checks(output / "checks.csv", checks)
    atomic_json(
        output / "mech01_manifest.json",
        {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "stage": "preflight",
            "passed": passed,
            "family": args.family,
            "checkpoint": str(checkpoint_path),
            "source_script": str(source_path),
            "artifacts": sorted(path.name for path in output.iterdir()),
        },
    )
    del checkpoint, model_state
    return passed


def run_smoke(args: argparse.Namespace) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("MECH-01 numerical smoke requires CUDA")
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision("high")
    output = args.output_dir.resolve()
    checkpoint_path = args.checkpoint.resolve()
    source_path = args.source_script.resolve()
    triton_path = args.triton_kernels.resolve() if args.triton_kernels else None
    before_stat = checkpoint_path.stat()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state, schema = checkpoint_schema(
        checkpoint_path,
        checkpoint,
        args.family,
        source_path,
        args.checkpoint_sha256,
        args.hash_checkpoint,
    )
    attach_profile_provenance(schema, args.profile_script)
    route = route_audit(args.family, source_path, schema["architecture"])
    method = schema["method_inferred"] if args.method == "auto" else args.method
    source_runtime, production_ns, triton_audit = load_source_runtime(
        args.family, source_path, triton_path
    )
    source_runtime_config = configure_source_runtime_globals(
        args.family, source_runtime, method
    )
    model = build_model(args.family, source_runtime, schema["architecture"], method)
    incompatible = model.load_state_dict(model_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict model load unexpectedly incompatible: {incompatible}")
    model = model.cuda()
    layers = select_layers(schema["architecture"]["n_layer"], args.layers)
    modules, weights, target_names = target_modules_and_weights(
        model, args.family, layers
    )
    momenta, momentum_audit = extract_target_momenta(
        checkpoint, model, args.family, target_names
    )
    auxiliary_before = checkpoint_aux_signature(checkpoint)
    model_before = model_state_signature(model, target_names.values())
    batches, batch_contract = read_fineweb_batches(
        args.data_pattern,
        list(args.probe_offsets),
        args.device_batch_size,
        args.sequence_length,
    )
    first = collect_probe_pass(
        model, modules, weights, batches, args.max_activation_rows
    )
    second = collect_probe_pass(
        model, modules, weights, batches, args.max_activation_rows
    )
    repeatability = compare_probe_passes(first, second, args.atol, args.rtol)
    dynamic_route_rows = []
    for layer in layers:
        activation_width = first["build_activations"][layer].size(1)
        weight_input_width = weights[layer].shape[1]
        dynamic_route_rows.append(
            {
                "layer": layer,
                "activation_width": int(activation_width),
                "weight_input_width": int(weight_input_width),
                "passed": int(activation_width) == int(weight_input_width),
            }
        )
    parity = production_ns_parity(
        production_ns, args.ns_steps, args.atol, args.rtol
    )
    numerical: dict[int, Any] = {}
    candidates = list(dict.fromkeys(args.candidates))
    for layer in layers:
        numerical[layer] = diagnostic_results(
            first["build_activations"][layer],
            first["heldout_gradients"][layer],
            momenta[layer],
            args.family,
            candidates,
            args.ridge_mult,
            args.ridge_eps,
            args.momentum,
            args.ns_steps,
            args.spectrum_dtype,
            production_ns,
        )
    config = {
        "family": args.family,
        "candidates": candidates,
        "ridge_mult": args.ridge_mult,
        "ridge_eps": args.ridge_eps,
        "momentum": args.momentum,
        "momentum_convention": momentum_convention(args.family),
        "ns_steps": args.ns_steps,
        "spectrum_dtype": args.spectrum_dtype,
    }
    bundle_layer = (
        args.export_bundle_layer
        if args.export_bundle_layer is not None
        else layers[len(layers) // 2]
    )
    if bundle_layer not in layers:
        raise ValueError(
            f"--export-bundle-layer={bundle_layer} must be one of smoke layers={layers}"
        )
    bundle = save_bundle(
        output / "tensor_bundle.pt",
        bundle_layer,
        args.family,
        first["build_activations"][bundle_layer],
        first["heldout_gradients"][bundle_layer],
        momenta[bundle_layer],
        config,
        {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": schema["checkpoint_sha256_observed"]
            or schema["checkpoint_sha256_supplied"],
            "source_script": str(source_path),
            "source_sha256": schema["source_sha256"],
            "profile_script": schema.get("profile_script", ""),
            "profile_script_sha256": schema.get("profile_script_sha256", ""),
            "batch_contract_sha256": batch_contract["contract_sha256"],
            "target_weight": target_names[bundle_layer],
        },
    )
    model_after = model_state_signature(model, target_names.values())
    auxiliary_after = checkpoint_aux_signature(checkpoint)
    after_stat = checkpoint_path.stat()
    invariance = {
        "model_signature_before": model_before,
        "model_signature_after": model_after,
        "model_unchanged": model_before == model_after,
        "optimizer_loader_signature_before": auxiliary_before,
        "optimizer_loader_signature_after": auxiliary_after,
        "optimizer_loader_unchanged": auxiliary_before == auxiliary_after,
        "checkpoint_stat_before": {
            "size": before_stat.st_size,
            "mtime_ns": before_stat.st_mtime_ns,
        },
        "checkpoint_stat_after": {
            "size": after_stat.st_size,
            "mtime_ns": after_stat.st_mtime_ns,
        },
        "checkpoint_file_unchanged": before_stat.st_size == after_stat.st_size
        and before_stat.st_mtime_ns == after_stat.st_mtime_ns,
    }
    numerical_finite = all(finite_numbers(value) for value in numerical.values())
    streaming_contract = {
        "layer_processing": "sequential",
        "dense_covariances_resident_simultaneously": 1,
        "all_layer_dense_k_residency_required": False,
        "llama1b_compatible": True,
    }
    checks = [
        {"check": "checkpoint_schema", "passed": schema["passed"], "detail": schema["missing_required_keys"]},
        {"check": "static_route_audit", "passed": route["passed"], "detail": route["checks"]},
        {"check": "dynamic_route_audit", "passed": all(row["passed"] for row in dynamic_route_rows), "detail": dynamic_route_rows},
        {"check": "build_heldout_disjoint", "passed": batch_contract["build_heldout_disjoint"], "detail": batch_contract["overlapping_pairs"]},
        {"check": "historical_momentum_present", "passed": momentum_audit["all_present"], "detail": momentum_audit["targets"]},
        {"check": "repeatability", "passed": repeatability["passed"], "detail": repeatability["heldout_loss_max_abs_diff"]},
        {"check": "production_ns_repeatability", "passed": parity["allclose"] and parity["finite"], "detail": parity["max_abs_diff"]},
        {"check": "numerical_metrics_finite", "passed": numerical_finite, "detail": ""},
        {"check": "model_unchanged", "passed": invariance["model_unchanged"], "detail": ""},
        {"check": "optimizer_loader_unchanged", "passed": invariance["optimizer_loader_unchanged"], "detail": ""},
        {"check": "checkpoint_file_unchanged", "passed": invariance["checkpoint_file_unchanged"], "detail": str(checkpoint_path)},
        {"check": "single_layer_dense_k_streaming", "passed": True, "detail": streaming_contract},
        {"check": "triton_kernel_provenance", "passed": triton_audit["passed"], "detail": triton_audit},
        {
            "check": "source_runtime_globals",
            "passed": source_runtime_config["passed"],
            "detail": source_runtime_config,
        },
    ]
    passed = all(row["passed"] for row in checks)
    atomic_json(output / "checkpoint_schema.json", schema)
    atomic_json(output / "route_audit.json", {**route, "dynamic": dynamic_route_rows})
    atomic_json(output / "batch_contract.json", batch_contract)
    atomic_json(output / "momentum_audit.json", momentum_audit)
    atomic_json(output / "repeatability.json", repeatability)
    atomic_json(output / "production_path_audit.json", {"ns5": parity, "triton": triton_audit, "config": config})
    atomic_json(output / "diagnostics.json", {str(key): value for key, value in numerical.items()})
    write_diagnostics_csv(output / "diagnostics.csv", numerical)
    atomic_json(output / "state_invariance.json", invariance)
    atomic_json(output / "streaming_contract.json", streaming_contract)
    atomic_json(output / "source_runtime_config.json", source_runtime_config)
    atomic_json(output / "tensor_bundle_manifest.json", bundle)
    atomic_json(output / "runtime.json", runtime_metadata(args))
    write_checks(output / "checks.csv", checks)
    atomic_json(
        output / "mech01_manifest.json",
        {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "stage": "numerical_smoke",
            "passed": passed,
            "family": args.family,
            "method": method,
            "layers": layers,
            "checkpoint": str(checkpoint_path),
            "source_script": str(source_path),
            "bundle": bundle,
            "artifacts": sorted(path.name for path in output.iterdir()),
        },
    )
    del checkpoint, model_state, model, first, second, momenta
    torch.cuda.empty_cache()
    return passed


def run_replay(args: argparse.Namespace) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("MECH-01 bundle replay requires CUDA")
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision("high")
    output = args.output_dir.resolve()
    bundle_path = args.bundle.resolve()
    bundle_sha = sha256_file(bundle_path)
    payload = torch.load(bundle_path, map_location="cpu", weights_only=False)
    required = {
        "family",
        "activation_build",
        "fresh_heldout_gradient",
        "historical_momentum",
        "diagnostic_config",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise RuntimeError(f"tensor bundle missing keys: {missing}")
    source_path = args.source_script.resolve()
    triton_path = args.triton_kernels.resolve() if args.triton_kernels else None
    _, production_ns, triton_audit = load_source_runtime(
        args.family, source_path, triton_path
    )
    config = payload["diagnostic_config"]
    scientific_family = payload["family"]
    recomputed_covariance = covariance_from_activations(
        payload["activation_build"].float()
    )
    stored_covariance = payload.get("fresh_covariance")
    covariance_present = isinstance(stored_covariance, Tensor)
    covariance_shape_matches = covariance_present and (
        tuple(stored_covariance.shape) == tuple(recomputed_covariance.shape)
    )
    covariance_max_abs_diff = (
        float((stored_covariance.float() - recomputed_covariance).abs().max())
        if covariance_shape_matches
        else float("inf")
    )
    covariance_matches = covariance_shape_matches and torch.allclose(
        stored_covariance.float(),
        recomputed_covariance,
        atol=0.0,
        rtol=0.0,
    )
    convention_matches = payload.get("momentum_convention") == momentum_convention(
        scientific_family
    )
    tensor_audit = {
        "activation_sha256": tensor_sha256(payload["activation_build"]),
        "gradient_sha256": tensor_sha256(payload["fresh_heldout_gradient"]),
        "momentum_sha256": tensor_sha256(payload["historical_momentum"]),
        "stored_covariance_sha256": (
            tensor_sha256(stored_covariance) if covariance_present else ""
        ),
        "recomputed_covariance_sha256": tensor_sha256(recomputed_covariance),
        "covariance_shape_matches": covariance_shape_matches,
        "covariance_max_abs_diff": covariance_max_abs_diff,
        "covariance_exactly_matches": bool(covariance_matches),
        "momentum_convention_matches_family": convention_matches,
    }
    results = diagnostic_results(
        payload["activation_build"],
        payload["fresh_heldout_gradient"],
        payload["historical_momentum"],
        scientific_family,
        list(config["candidates"]),
        float(config["ridge_mult"]),
        float(config["ridge_eps"]),
        float(config["momentum"]),
        int(config["ns_steps"]),
        str(config["spectrum_dtype"]),
        production_ns,
    )
    parity = production_ns_parity(
        production_ns, int(config["ns_steps"]), args.atol, args.rtol
    )
    passed = (
        finite_numbers(results)
        and parity["finite"]
        and triton_audit["passed"]
        and bool(covariance_matches)
        and convention_matches
    )
    replay = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "bundle": str(bundle_path),
        "bundle_sha256": bundle_sha,
        "bundle_family": scientific_family,
        "runtime_source_family": args.family,
        "source_script": str(source_path),
        "source_sha256": source_sha256(source_path),
        "triton": triton_audit,
        "runtime": runtime_metadata(args),
        "production_ns_repeatability": parity,
        "bundle_tensor_audit": tensor_audit,
        "results": results,
        "passed": passed,
    }
    atomic_json(output / "replay.json", replay)
    atomic_json(
        output / "mech01_manifest.json",
        {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "stage": "replay",
            "passed": passed,
            "bundle_sha256": bundle_sha,
            "replay": str((output / "replay.json").resolve()),
        },
    )
    return passed


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if not output.is_dir():
        raise RuntimeError(f"controller must create output directory first: {output}")
    atomic_json(
        output / "status.json",
        {"status": "running", "mode": args.mode, "script_version": SCRIPT_VERSION},
    )
    try:
        if args.mode == "preflight":
            passed = run_preflight(args)
        elif args.mode == "smoke":
            passed = run_smoke(args)
        else:
            passed = run_replay(args)
    except BaseException as exc:
        atomic_json(
            output / "status.json",
            {
                "status": "failed",
                "mode": args.mode,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "script_version": SCRIPT_VERSION,
            },
        )
        raise
    atomic_json(
        output / "status.json",
        {
            "status": "passed" if passed else "failed_checks",
            "mode": args.mode,
            "script_version": SCRIPT_VERSION,
        },
    )
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
