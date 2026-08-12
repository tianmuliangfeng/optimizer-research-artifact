#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ex48_received_audit_test", HERE / "audit_received_results.py")
assert spec is not None and spec.loader is not None
A = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = A
spec.loader.exec_module(A)


RUN_ID = "20260805T061608+0000"
CONTRACT_SHA = "1" * 64
DATA_SHA = "2" * 64
CONTROLLER_SHA = "d95cfc505823fe48c65be5d4886f843911cabbbdb06cd18de145899ef158782c"
ANALYZER_SHA = "dce6a803cb6c3961625e0ead87706b6baa4e909d1d32e24a9af24270bd660cbf"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def make_run(root: Path) -> Path:
    run = root / RUN_ID
    package = run / "source_snapshot" / "scripts" / "48_llama1b_10b_multibudget"
    package.mkdir(parents=True)
    (package / "formal_contract.json").write_bytes(b"contract")
    (package / "run_formal.py").write_bytes(b"controller")
    (package / "analyze_formal.py").write_bytes(b"analyzer")
    contract_sha = A.sha256_file(package / "formal_contract.json")
    controller_sha = A.sha256_file(package / "run_formal.py")
    analyzer_sha = A.sha256_file(package / "analyze_formal.py")
    source_manifest_path = run / "source_snapshot" / "source_snapshot_manifest.json"
    write_json(
        source_manifest_path,
        {
            "schema_version": "ex48_source_snapshot_v1",
            "files": {
                "scripts/48_llama1b_10b_multibudget/run_formal.py": {"sha256": controller_sha},
                "scripts/48_llama1b_10b_multibudget/analyze_formal.py": {"sha256": analyzer_sha},
            },
        },
    )
    # Tests patch these two frozen-code constants to the fixture hashes.
    A.FROZEN_CONTROLLER_SHA256 = controller_sha
    A.FROZEN_ANALYZER_SHA256 = analyzer_sha
    write_json(run / "data_audit.json", {"passed": True, "inventory_sha256": DATA_SHA})
    write_json(
        run / "run_identity.json",
        {
            "schema_version": "ex48_run_identity_v1",
            "run_dir": f"/remote/48_llama1b_10b_multibudget/{RUN_ID}",
            "contract_sha256": contract_sha,
            "data_inventory_sha256": DATA_SHA,
            "source_snapshot_manifest_sha256": A.sha256_file(source_manifest_path),
        },
    )
    write_json(
        run / "analysis" / "analysis_manifest.json",
        {
            "passed": True,
            "claim_eligible": True,
            "contract_sha256": contract_sha,
            "data_inventory_sha256": DATA_SHA,
        },
    )
    rows = []
    for method in A.METHODS:
        for seed in A.SEEDS:
            for budget in A.BUDGETS:
                token = f"{method}-{seed}-{budget}"
                rows.append(
                    {
                        "budget_id": budget,
                        "method": method,
                        "seed": seed,
                        "path": (
                            f"/remote/48_llama1b_10b_multibudget/{RUN_ID}/formal/"
                            f"{method}/seed{seed}/{budget}/checkpoint_latest.pt"
                        ),
                        "bytes": 100 + len(rows),
                        "sha256": A.hashlib.sha256(token.encode()).hexdigest(),
                    }
                )
    write_json(
        run / "handoff_manifest.json",
        {"passed": True, "external_retained_checkpoints": rows},
    )
    return run


def make_receipt(received: Path, run: Path, payload_override: dict | None = None) -> Path:
    payload = {
        "passed": True,
        "full_checkpoint_hash": True,
        "checks": {name: True for name in A.REMOTE_VERIFY_CHECKS},
    }
    if payload_override:
        payload.update(payload_override)
    receipt = received / "full_checkpoint_verify_20260812T031024+0000.json"
    write_json(receipt, payload)
    sidecar = Path(f"{receipt}.sha256")
    sidecar.write_text(
        f"{A.sha256_file(receipt)}  /remote/48_llama1b_10b_multibudget/{run.name}/analysis/{receipt.name}\n",
        encoding="utf-8",
    )
    return receipt


class ReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run = make_run(self.root)
        self.received = self.root / "received"
        self.received.mkdir()

    def tearDown(self) -> None:
        A.FROZEN_CONTROLLER_SHA256 = CONTROLLER_SHA
        A.FROZEN_ANALYZER_SHA256 = ANALYZER_SHA
        self.temp.cleanup()

    def audit(self):
        checks = []
        return A.validate_remote_full_hash_receipts(self.run, self.received, checks), checks

    def test_missing_receipt_fails_closed(self) -> None:
        (receipts, passed, certificate), checks = self.audit()
        self.assertEqual(receipts, [])
        self.assertFalse(passed)
        self.assertFalse(certificate["passed"])
        self.assertEqual(checks[-1]["passed"], "false")

    def test_valid_receipt_and_sidecar_bind_lineage(self) -> None:
        make_receipt(self.received, self.run)
        (receipts, passed, certificate), _ = self.audit()
        self.assertTrue(passed)
        self.assertTrue(receipts[0]["passed"])
        self.assertEqual(certificate["run_lineage"]["checkpoint_inventory_count"], 36)
        self.assertTrue(all(certificate["run_lineage_checks"].values()))
        self.assertFalse(certificate["limitations"]["receipt_records_exact_command"])

    def test_tampered_sidecar_fails(self) -> None:
        receipt = make_receipt(self.received, self.run)
        Path(f"{receipt}.sha256").write_text(
            f"{'0' * 64}  /remote/48_llama1b_10b_multibudget/{RUN_ID}/analysis/{receipt.name}\n",
            encoding="utf-8",
        )
        (_, passed, certificate), _ = self.audit()
        self.assertFalse(passed)
        self.assertFalse(certificate["receipts"][0]["checks"]["sidecar_sha256_match"])

    def test_false_receipt_boolean_fails(self) -> None:
        make_receipt(self.received, self.run, {"full_checkpoint_hash": False})
        (_, passed, certificate), _ = self.audit()
        self.assertFalse(passed)
        self.assertFalse(certificate["receipts"][0]["checks"]["full_checkpoint_hash"])

    def test_wrong_lineage_fails(self) -> None:
        receipt = make_receipt(self.received, self.run)
        sidecar = Path(f"{receipt}.sha256")
        sidecar.write_text(
            f"{A.sha256_file(receipt)}  /remote/48_llama1b_10b_multibudget/wrong-run/analysis/{receipt.name}\n",
            encoding="utf-8",
        )
        (_, passed, certificate), _ = self.audit()
        self.assertFalse(passed)
        self.assertFalse(certificate["receipts"][0]["checks"]["sidecar_remote_run_match"])


if __name__ == "__main__":
    unittest.main()
