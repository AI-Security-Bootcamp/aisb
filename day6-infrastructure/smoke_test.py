"""Pre-flight smoke test for the GPUBreach lab.

Run this BEFORE you start the lab:

    python day6-infrastructure/smoke_test.py

It verifies that:
  * The NVIDIA GPU driver is loaded and CUDA is accessible
  * Python can import the gpu_hammer package
  * aisb_utils is reachable on sys.path
  * A fresh Environment can be constructed (allocates real GPU memory)
  * The initial state is sane (non-root cred, GPU-local PTE)

If this prints OK, your environment is ready. Otherwise, the most common
failure is a missing NVIDIA driver kernel module. Fix it with:

    sudo modprobe nvidia
    nvidia-smi          # should show your GPU
"""

import sys
from pathlib import Path

for _path in [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parent.parent),
]:
    if _path not in sys.path:
        sys.path.insert(0, _path)


def main() -> int:
    try:
        from aisb_utils import report  # noqa: F401
    except ImportError as e:
        print(f"FAIL: could not import aisb_utils — {e}")
        print("      Check that you're running from the repo root and your venv is active.")
        return 1

    # Check CUDA before importing gpu_hammer (which allocates GPU memory)
    try:
        import torch
    except ImportError:
        print("FAIL: torch not installed — activate the project venv.")
        return 1

    if not torch.cuda.is_available():
        print("FAIL: CUDA GPU not available.")
        print("      Load the NVIDIA driver:  sudo modprobe nvidia")
        print("      Then verify:             nvidia-smi")
        return 1

    print(f"GPU: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory // 1024**3} GB)")

    try:
        from gpu_hammer import (
            make_environment,
            HAMMER_THRESHOLD_ACTIVATIONS,
            HAMMER_THRESHOLD_ROUNDS,
            ACTIVATIONS_PER_ROUND,
            APERTURE_GPU_LOCAL,
            PTE_ROW,
            FLAG,
        )
    except ImportError as e:
        print(f"FAIL: could not import gpu_hammer — {e}")
        return 1

    print("Compiling CUDA hammer kernel (one-time, ~10s) ...", end=" ", flush=True)
    env = make_environment()
    print("done.")

    flip_row = env.dram.flip_location[0]
    if flip_row != PTE_ROW:
        print(f"FAIL: seeded flip row {flip_row} != PTE_ROW {PTE_ROW}")
        return 1

    if env.victim_pte.aperture != APERTURE_GPU_LOCAL:
        print(f"FAIL: PTE aperture {env.victim_pte.aperture} should be {APERTURE_GPU_LOCAL} at start")
        return 1

    if env.kernel_cred.is_root():
        print("FAIL: kernel cred already shows root before any attack ran")
        return 1

    print("OK — gpu_hammer imports, GPU memory allocated, initial state sane.")
    print(f"     Target flag on success:          {FLAG}")
    print(f"     Hammer threshold (activations):  {HAMMER_THRESHOLD_ACTIVATIONS:,} per aggressor")
    print(f"     Activations per round:           {ACTIVATIONS_PER_ROUND:,}")
    print(f"     Rounds needed to flip:           {HAMMER_THRESHOLD_ROUNDS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
