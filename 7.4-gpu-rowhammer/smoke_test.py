"""Pre-flight smoke test for the GPUBreach lab.

Run this BEFORE you start the lab:

    python 7.4-gpu-rowhammer/smoke_test.py

It verifies that:
  * Python can import the simulator package
  * `aisb_utils` is reachable on sys.path
  * A fresh environment can be constructed and the seeded
    flip_location is what the attack chain expects

If this prints `OK`, your environment is ready. If not, flag a TA
before you spend time debugging the exercises themselves.
"""

import sys
from pathlib import Path

# Ensure the sibling simulator and workspace utilities resolve even when this
# script is run from a different working directory.
_section_dir = Path(__file__).resolve().parent
if str(_section_dir) not in sys.path:
    sys.path.insert(0, str(_section_dir))

_root = next(p for p in Path(__file__).resolve().parents if (p / "aisb_utils").is_dir())
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def main() -> int:
    try:
        from aisb_utils import report  # noqa: F401
    except ImportError as e:
        print(f"FAIL: could not import aisb_utils — {e}")
        print("      Check that you're running from the repo root and that "
              "your venv is activated.")
        return 1

    try:
        from gpubreach_sim import (
            make_environment,
            HAMMER_THRESHOLD_ACTIVATIONS,
            APERTURE_GPU_LOCAL,
            PTE_ROW,
            FLAG,
        )
    except ImportError as e:
        print(f"FAIL: could not import gpubreach_sim — {e}")
        return 1

    env = make_environment()

    # The environment is pre-groomed: the victim PTE lives in PTE_ROW and
    # the seeded flip_location must match it, otherwise the chain won't
    # close.
    flip_row = env.dram.flip_location[0]
    if flip_row != PTE_ROW:
        print(f"FAIL: seeded flip row {flip_row} != PTE_ROW {PTE_ROW}")
        return 1

    if env.victim_pte.aperture != APERTURE_GPU_LOCAL:
        print(f"FAIL: PTE aperture {env.victim_pte.aperture} is not "
              f"{APERTURE_GPU_LOCAL} (GPU VRAM) at start")
        return 1

    if env.kernel_cred.is_root():
        print("FAIL: kernel cred already shows root before any attack ran")
        return 1

    print("OK — gpubreach_sim imports, environment builds, initial state sane.")
    print(f"     Target flag on success: {FLAG}")
    print(f"     Hammer threshold:       {HAMMER_THRESHOLD_ACTIVATIONS:,} activations per aggressor")
    return 0


if __name__ == "__main__":
    sys.exit(main())
