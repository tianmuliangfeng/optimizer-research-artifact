#!/usr/bin/env python3
"""Thin validation entry point for the mechanism-closure package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_mechanism_closure import validate_package


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-awaiting-workbook", action="store_true")
    args = parser.parse_args()
    result = validate_package(args.output_dir.resolve(), require_final=not args.allow_awaiting_workbook)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
