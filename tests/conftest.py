"""Shared test setup: puts scripts/ on sys.path so tests can import the scripts
as modules (`import data_guard`). Tests live in tests/ as test_*.py and are
found by pytest without any further registration."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
