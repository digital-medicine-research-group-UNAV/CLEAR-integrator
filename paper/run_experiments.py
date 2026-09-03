#!/usr/bin/env python
"""Compatibility entry point for CLEAR paper experiments."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from clear_integration.paper._run_experiments import cli


if __name__ == "__main__":
    cli()
