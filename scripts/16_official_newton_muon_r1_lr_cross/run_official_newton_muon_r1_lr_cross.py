"""Entry point for the two missing cells of the R1 Muon/diag 2x2 LR cross."""

from __future__ import annotations

import sys
from pathlib import Path


R1_DIR = Path(__file__).resolve().parent.parent / "15_official_newton_muon_r1"
sys.path.insert(0, str(R1_DIR))

import run_official_newton_muon_r1 as r1


def main() -> None:
    if "--lr-cross" not in sys.argv[1:]:
        sys.argv.insert(1, "--lr-cross")
    r1.main()


if __name__ == "__main__":
    main()
