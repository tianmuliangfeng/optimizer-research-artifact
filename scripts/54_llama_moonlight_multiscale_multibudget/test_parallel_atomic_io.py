#!/usr/bin/env python3
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile

import protocol as P


def test_atomic_json_same_target_is_thread_safe() -> None:
    with tempfile.TemporaryDirectory() as td:
        target = Path(td) / "shared.json"
        payload = {"schema_version": "parallel_atomic_io_test_v1", "passed": True}
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(P.atomic_json, target, payload) for _ in range(128)]
            for future in futures:
                future.result()
        assert json.loads(target.read_text(encoding="utf-8")) == payload
        assert not list(target.parent.glob(f".{target.name}.*.tmp"))


def test_atomic_text_same_target_is_thread_safe() -> None:
    with tempfile.TemporaryDirectory() as td:
        target = Path(td) / "shared.py"
        text = "print('deterministic')\n"
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(P.atomic_text, target, text) for _ in range(128)]
            for future in futures:
                future.result()
        assert target.read_text(encoding="utf-8") == text
        assert not list(target.parent.glob(f".{target.name}.*.tmp"))
