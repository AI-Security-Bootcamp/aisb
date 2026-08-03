# %%

import sys
from pathlib import Path

import nest_asyncio

_root = next(p for p in Path(__file__).resolve().parents if (p / "aisb_utils").is_dir())
# day2_utils lives alongside this section (prompts/, logs/, reference_solutions/ too).
for _path in [str(_root), str(Path(__file__).resolve().parent)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from aisb_utils.env import load_dotenv

load_dotenv()
nest_asyncio.apply()  # allow asyncio.run() inside Jupyter notebook cells

# This section's directory; prompts/, logs/, reference_solutions/ live here
SCRIPT_DIR = Path(__file__).resolve().parent
