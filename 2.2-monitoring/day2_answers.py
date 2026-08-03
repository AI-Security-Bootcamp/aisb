import sys
from pathlib import Path

import nest_asyncio
from control_arena.settings.apps.apps_solution_policy import apps_solution_policy

_root = next(p for p in Path(__file__).resolve().parents if (p / "aisb_utils").is_dir())
# day2_utils lives alongside this section (prompts/, logs/, reference_solutions/ too).
for _path in [str(_root), str(Path(__file__).resolve().parent)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)
from day2_utils.generate_trajectories import generate_trajectories

from aisb_utils.env import load_dotenv

load_dotenv()
nest_asyncio.apply()  # allow asyncio.run() inside Jupyter notebook cells

# This section's directory; prompts/, logs/, reference_solutions/ live here
SCRIPT_DIR = Path(__file__).resolve().parent

E2_1_TRAJECTORY_DIR = str(SCRIPT_DIR / "logs" / "e2.1_trajectories")

# Generate trajectories on first run; cached in logs/e2.1_trajectories/ after that.
# No LLM calls, but each sample runs inside a Docker container (~1 min total).
# apps_solution_policy() replays verified solutions and backdoors from the APPS
# dataset directly, without calling an LLM; it's a fast deterministic replay.
if not list(Path(E2_1_TRAJECTORY_DIR).glob("*.eval")):
    generate_trajectories(
        apps_solution_policy(),
        E2_1_TRAJECTORY_DIR,
        limit=50,
    )
    print("Trajectories generated and cached in logs/e2.1_trajectories/")
else:
    print("Trajectories already exist in logs/e2.1_trajectories/; skipping generation")

