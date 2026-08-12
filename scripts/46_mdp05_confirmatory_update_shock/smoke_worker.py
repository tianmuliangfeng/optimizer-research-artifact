#!/usr/bin/env python3
"""Clean terminal wrapper around the accepted MECH-09R smoke worker."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any


SCRIPT_VERSION = "2026-08-04.1"
HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


WORKER = load_module(
    "mdp05_smoke_mech09r_worker",
    HERE.parent / "37_mech09_downproj_refresh_mediation" / "mech09r_worker.py",
)


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    args = WORKER.parse_args()
    output = args.output_dir.resolve()
    try:
        if args.analysis_tier != "smoke":
            raise RuntimeError("clean smoke wrapper accepts --analysis-tier smoke only")
        WORKER.run_worker(args)
        return 0
    except BaseException as exc:
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(
            output / "status.json",
            {
                "status": "failed",
                "wrapper_version": SCRIPT_VERSION,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        print(
            f"MDP-05 smoke failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
