"""Pytest path setup for this section's generated test file.

The extracted ``section4_test.py`` imports the shared ``day2_utils`` package,
which lives in ``day2-agents/`` at the workspace root. pytest loads this
conftest before collecting the test module, so inserting the paths here makes
those imports resolve. Uses the same root-discovery snippet as the solution.
"""

import sys
from pathlib import Path

_root = next(p for p in Path(__file__).resolve().parents if (p / "aisb_utils").is_dir())
for _path in [str(_root), str(_root / "day2-agents")]:
    if _path not in sys.path:
        sys.path.insert(0, _path)
