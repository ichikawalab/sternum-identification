"""Pytest configuration: make the repository root importable so tests can
`from common.xxx import ...` the same way the pipeline stage scripts do."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
