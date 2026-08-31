#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import tempfile
from unittest import mock

import protocol as P
import run_suite as R


HERE = Path(__file__).resolve().parent


def _args(run_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(run_dir=run_dir, gpus=[0, 1, 2])


def _materialize_source_contract(run_dir: Path, contract: dict) -> Path:
    path = run_dir / "source_snapshot" / R.PACKAGE_REL / R.CONTRACT_NAME
    P.atomic_json(path, contract)
    return path


def test_declared_v3_policy_needs_no_amendment() -> None:
    contract = json.loads((HERE / "ex57_contract.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as directory:
        run_dir = Path(directory)
        _materialize_source_contract(run_dir, contract)
        result = R.resolve_compile_cache_policy(_args(run_dir), contract)
        assert result == {
            "policy": "per_physical_gpu",
            "source": "frozen_contract",
            "receipt_path": None,
            "receipt_sha256": None,
        }
        assert not (run_dir / R.COMPILE_CACHE_COMPAT_RECEIPT).exists()
        assert R.compile_cache_policy_replay_valid(_args(run_dir), contract)


def test_known_legacy_v2_policy_gets_execution_only_receipt() -> None:
    contract = json.loads((HERE / "ex57_contract.json").read_text(encoding="utf-8"))
    legacy = copy.deepcopy(contract)
    legacy["execution"].pop("compile_cache_policy")
    legacy["execution"].pop("compile_cache_reason")
    legacy["execution"].pop("compile_cache_root")

    with tempfile.TemporaryDirectory() as directory:
        run_dir = Path(directory)
        source = _materialize_source_contract(run_dir, legacy)
        source_sha = P.sha256_file(source)
        before = copy.deepcopy(legacy)
        with mock.patch.object(R, "LEGACY_V2_SOURCE_CONTRACT_SHA256", source_sha):
            result = R.resolve_compile_cache_policy(_args(run_dir), legacy)
            assert result["policy"] == "per_physical_gpu"
            assert result["source"] == "legacy_v2_execution_amendment"
            receipt_path = Path(result["receipt_path"])
            assert receipt_path.is_file()
            receipt = P.read_json(receipt_path)
            assert receipt["source_snapshot_contract_sha256"] == source_sha
            assert receipt["policy_scope"] == "execution_only"
            assert receipt["scientific_protocol_unchanged"] is True
            assert receipt["timing_eligible"] is False
            assert all(receipt["checks"].values())
            assert R.compile_cache_policy_replay_valid(_args(run_dir), legacy)
            # Re-resolving must be stable and must not rewrite the frozen contract.
            again = R.resolve_compile_cache_policy(_args(run_dir), legacy)
            assert again == result
        assert legacy == before
        assert P.read_json(source) == before


def test_unknown_missing_policy_contract_is_rejected() -> None:
    contract = json.loads((HERE / "ex57_contract.json").read_text(encoding="utf-8"))
    legacy = copy.deepcopy(contract)
    legacy["execution"].pop("compile_cache_policy")
    legacy["execution"].pop("compile_cache_reason")
    legacy["execution"].pop("compile_cache_root")

    with tempfile.TemporaryDirectory() as directory:
        run_dir = Path(directory)
        _materialize_source_contract(run_dir, legacy)
        try:
            R.resolve_compile_cache_policy(_args(run_dir), legacy)
        except RuntimeError as error:
            assert "known_v2_source_contract" in str(error)
        else:
            raise AssertionError("unknown legacy contract was accepted")


def test_manifest_receipt_hash_is_replay_bound() -> None:
    contract = json.loads((HERE / "ex57_contract.json").read_text(encoding="utf-8"))
    legacy = copy.deepcopy(contract)
    legacy["execution"].pop("compile_cache_policy")
    legacy["execution"].pop("compile_cache_reason")
    legacy["execution"].pop("compile_cache_root")

    with tempfile.TemporaryDirectory() as directory:
        run_dir = Path(directory)
        source = _materialize_source_contract(run_dir, legacy)
        source_sha = P.sha256_file(source)
        args = _args(run_dir)
        with mock.patch.object(R, "LEGACY_V2_SOURCE_CONTRACT_SHA256", source_sha):
            resolved = R.resolve_compile_cache_policy(args, legacy)
            manifest = {
                "compile_cache_policy": resolved["policy"],
                "compile_cache_policy_source": resolved["source"],
                "compile_cache_compatibility_receipt": resolved["receipt_path"],
                "compile_cache_compatibility_receipt_sha256": resolved["receipt_sha256"],
            }
            assert R.compile_cache_manifest_matches(args, legacy, manifest)
            manifest["compile_cache_compatibility_receipt_sha256"] = "0" * 64
            assert not R.compile_cache_manifest_matches(args, legacy, manifest)
