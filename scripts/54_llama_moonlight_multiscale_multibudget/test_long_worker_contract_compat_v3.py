from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def load_run_suite():
    spec = importlib.util.spec_from_file_location(
        "ex54_run_suite_test_v3", HERE / "run_suite.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(HERE))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_live_long_worker_contains_v3_source_binding():
    suite = load_run_suite()
    source = (HERE / "long_worker.py").read_text(encoding="utf-8")
    assert suite.LONG_WORKER_PROFILE_COMPAT_MARKER in source
    assert "base_trainer_sha256 = protocol.sha256_file(base_trainer_path)" in source
    assert 'payload.get("grid") or {}' in source
    assert "accepted_sources" in source


def test_runtime_patch_upgrades_v2_adapter_source():
    suite = load_run_suite()
    v2 = '''#!/usr/bin/env python3
from pathlib import Path
import sys
import protocol

def main() -> int:
    module = type("M", (), {})()
    module.P = protocol
    # EX54_LONG_WORKER_CONTRACT_COMPAT_V2
    original_read_json = protocol.read_json
    contract_index = sys.argv.index("--contract")
    contract_path = Path(sys.argv[contract_index + 1]).resolve()

    def read_json_with_worker_profile(path: Path):
        payload = original_read_json(path)
        if Path(path).resolve() == contract_path:
            payload = dict(payload)
            profile = dict(payload["profiles"]["1b"])
            profile.update(payload.get("profile", {}))
            payload["profile"] = profile
            grid = dict(payload.get("grid", {}))
            grid.update({"methods": ["moonlight"]})
            payload["grid"] = grid
        return payload

    protocol.read_json = read_json_with_worker_profile
    return 0
'''
    patched = suite.patch_long_worker_profile_adapter(v2)
    assert suite.LONG_WORKER_PROFILE_COMPAT_MARKER in patched
    assert "base_trainer_sha256 = protocol.sha256_file(base_trainer_path)" in patched
    assert "payload.get('grid') or {}" in patched
    assert (
        "accepted_sources['scripts/17_llama_swiglu_validation/train_llama_swiglu.py'] = base_trainer_sha256"
        in patched
    )
    compile(patched, "<patched_v2_to_v3>", "exec")


def test_v3_patch_is_idempotent():
    suite = load_run_suite()
    source = (HERE / "long_worker.py").read_text(encoding="utf-8")
    assert suite.patch_long_worker_profile_adapter(source) == source


def test_incomplete_v3_adapter_is_rejected():
    suite = load_run_suite()
    source = (HERE / "long_worker.py").read_text(encoding="utf-8")
    incomplete = source.replace(
        'payload["accepted_sources"] = accepted_sources',
        '# deliberately removed accepted-source binding',
        1,
    )
    try:
        suite.patch_long_worker_profile_adapter(incomplete)
    except RuntimeError as exc:
        assert "V3" in str(exc) and "incomplete" in str(exc)
    else:
        raise AssertionError("incomplete EX54 V3 adapter was accepted")
