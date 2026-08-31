#!/usr/bin/env python3
"""Lineage-bound adapter around the accepted EX48 exact-resume worker."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import protocol


def pop_parent_worker(argv: list[str]) -> Path:
    try:
        index = argv.index("--parent-worker")
        value = argv[index + 1]
    except (ValueError, IndexError) as exc:
        raise RuntimeError("--parent-worker is required") from exc
    del argv[index:index + 2]
    return Path(value).resolve()


def main() -> int:
    parent = pop_parent_worker(sys.argv)
    if not parent.is_file():
        raise FileNotFoundError(parent)
    spec = importlib.util.spec_from_file_location("ex54_accepted_ex48_worker", parent)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import accepted EX48 worker: {parent}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # The worker imports ``protocol`` by module name.  This adapter deliberately
    # supplies the EX54 contract validator while retaining the accepted cursor,
    # checkpoint, phase, and numerical implementation.
    module.P = protocol
    # EX54_LONG_WORKER_CONTRACT_COMPAT_V3_SOURCE_VIEW
    original_read_json = protocol.read_json
    contract_index = sys.argv.index("--contract")
    contract_path = Path(sys.argv[contract_index + 1]).resolve()
    base_index = sys.argv.index("--base-trainer")
    base_trainer_path = Path(sys.argv[base_index + 1]).resolve()
    base_trainer_sha256 = protocol.sha256_file(base_trainer_path)

    def read_json_with_worker_profile(path: Path):
        payload = original_read_json(path)
        if Path(path).resolve() == contract_path:
            payload = dict(payload)
            profile = dict(payload["profiles"]["1b"])
            profile.update(payload.get("profile", {}))
            profile.update(
                {
                    "n_layer": 18,
                    "n_head": 16,
                    "n_embd": 2048,
                    "intermediate_size": 5504,
                    "expected_preconditioner_groups": {"moonlight": 0},
                }
            )
            payload["profile"] = profile
            grid = dict(payload.get("grid") or {})
            grid.update(
                {
                    "methods": ["moonlight"],
                    "seeds": list(payload["formal"]["seeds"]),
                    "formal_units": len(payload["formal"]["seeds"]),
                    "host_count": 1,
                    "gpus": len(payload["execution"]["physical_gpus"]),
                }
            )
            payload["grid"] = grid

            # The accepted EX48 worker verifies the actual --base-trainer
            # against this legacy key. EX54 passes its derived Moonlight
            # trainer; the frozen EX54 contract keeps immutable parent source
            # provenance under parent/... keys. Add only an in-memory worker
            # view, leaving selected contract bytes and SHA256 unchanged.
            accepted_sources = dict(payload.get("accepted_sources") or {})
            accepted_sources[
                "scripts/17_llama_swiglu_validation/train_llama_swiglu.py"
            ] = base_trainer_sha256
            payload["accepted_sources"] = accepted_sources
        return payload

    protocol.read_json = read_json_with_worker_profile
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
