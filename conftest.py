"""Pytest/unittest bootstrap: make the project root importable.

Ensures ``import config`` and ``import src`` resolve when tests are collected
from any working directory, even without an editable install.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
