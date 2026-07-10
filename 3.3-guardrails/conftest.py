"""Pytest path setup for this section's generated test file.

The extracted ``section3_test.py`` imports the ``day3_setup`` module, which
lives alongside this section. pytest loads this conftest before collecting the
test module, so inserting the paths here makes that import resolve. Uses the
same root-discovery snippet as the solution.
"""

import sys
from pathlib import Path

_root = next(p for p in Path(__file__).resolve().parents if (p / "aisb_utils").is_dir())
for _path in [str(_root), str(Path(__file__).resolve().parent)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)
