"""Real CUDA-backed GDDR6 DRAM for the GPUBreach lab.

The RowHammer hammer loop runs as a genuine CUDA kernel that uses
cache-bypassing PTX loads (ld.global.cv.u64) to force true DRAM row
activations on real GDDR6 memory. Timing is measured with CUDA events.

Why cache-bypassing loads matter
---------------------------------
A GPU's L2 cache is large (24 MB on RTX 4060). Without cache bypass, the
repeated accesses to aggressor rows would be served from cache — no DRAM
row activations, no RowHammer effect. The ld.global.cv.u64 PTX instruction
forces every access to bypass all caches and fetch directly from DRAM,
creating the sustained row-activation pressure needed to flip bits.

How real GPUHammer attacks handle bit flips
--------------------------------------------
Reliable bit flips in GDDR6 require device-specific offline profiling
("templating"): run a characterisation pass to identify which (row, column,
bit) locations on this specific DRAM die flip under the target pattern.
During the real attack, target those known-flippable locations.

This module follows the same model:

  1. The hammer kernel runs on real GPU DRAM, measuring genuine GDDR6 timing.
  2. After the flip threshold is crossed, the pre-profiled flip is injected
     into the CPU-side shadow (which models the on-DRAM PTE state).
  3. If a real hardware bit flip is detected in GPU memory during hammering
     (possible on hardware without strong TRR mitigations!), it takes priority.

Requirements
------------
  - NVIDIA GPU with GDDR6 memory (RTX 3060 / RTX 4060 or similar)
  - NVIDIA kernel driver loaded:  sudo modprobe nvidia
  - Verify with:                  nvidia-smi
  - PyTorch with CUDA available:  python -c "import torch; print(torch.cuda.is_available())"
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os

import torch


# ── Physical parameters (GPUHammer paper, USENIX 2025) ───────────────────────

ROWS_PER_BANK: int = 8192
"""Number of rows modelled in one DRAM bank. GDDR6 devices have ~16k rows
per bank; 8k keeps GPU memory use small while row indices look realistic."""

ROW_SIZE_BYTES: int = 1024
"""Bytes per DRAM row (real GDDR6 row buffers are ~2KB; we halve it for
clarity while preserving the timing-per-activation story)."""

ACTIVATE_PRECHARGE_NS: int = 65
"""Nanoseconds per DRAM ACTIVATE–PRECHARGE cycle (tRC) on GDDR6.
On the real GPU timing is measured by CUDA events. This constant is kept
for educational reference and for the Phase 1 comprehension exercises."""

HAMMER_THRESHOLD_ACTIVATIONS: int = 150_000
"""Minimum activations per aggressor row to induce a bit flip (double-sided).
Matches the GPUHammer paper's worst-case threshold on consumer RTX 3080/3090
GDDR6 parts without TRR mitigations. Use in Phase 1 timing calculations."""

ACTIVATIONS_PER_ROUND: int = 1_000
"""DRAM activations per aggressor per hammer_once() call.
Each call launches one CUDA kernel that issues ACTIVATIONS_PER_ROUND
cache-bypassing ld.global.cv.u64 loads to each aggressor row, then returns
the CUDA-event-measured execution time in nanoseconds."""

HAMMER_THRESHOLD_ROUNDS: int = HAMMER_THRESHOLD_ACTIVATIONS // ACTIVATIONS_PER_ROUND
"""hammer_once() calls needed to reach the flip threshold (= 150).
= HAMMER_THRESHOLD_ACTIVATIONS / ACTIVATIONS_PER_ROUND."""

REFRESH_WINDOW_MS: int = 64
"""DRAM refresh window (tREFW). The attacker must cross the flip threshold
before REFRESH commands reset all accumulated charge leakage."""


# ── CUDA hammer kernel (compiled once, cached) ────────────────────────────────

# This is the C++ wrapper that the PyTorch extension system exposes to Python.
_HAMMER_CPP = """
#include <torch/extension.h>
void double_sided_hammer(torch::Tensor, int64_t, int64_t, int64_t);
"""

# This is the CUDA kernel shown as educational content in the solution file.
# It is compiled by nvcc and loaded as a Python extension via PyTorch.
HAMMER_KERNEL_SRC = r"""
#include <torch/extension.h>

// Cache-bypassing load from GPU global memory.
//
// ld.global.cv.u64 = "load, global scope, cache-volatile, 64-bit"
//   - .global : target is GPU global (device) memory
//   - .cv     : cache at volatile level — bypasses L1 AND L2 caches
//   - .u64    : unsigned 64-bit integer
//
// Without .cv, repeated accesses to the same address would be served from
// the GPU's L2 cache (24 MB on RTX 4060), producing no DRAM activations.
// With .cv, every access goes all the way to GDDR6 DRAM — exactly the
// sustained row-activation traffic that RowHammer needs.
__device__ __forceinline__ unsigned long long load_bypass_cache(
    const unsigned long long* addr
) {
    unsigned long long val;
    asm volatile(
        "ld.global.cv.u64 %0, [%1];"
        : "=l"(val)
        : "l"(addr)
        : "memory"   // tell the compiler this is a memory access
    );
    return val;
}

// Double-sided RowHammer kernel.
//
// Thread organisation:
//   - Each thread handles one 64-bit word (8 bytes) within both aggressor rows.
//   - With 256 threads and 128 words per row (1024-byte rows / 8), one block
//     covers the entire row in a single launch.
//   - All threads hammer both rows simultaneously for `iters` iterations,
//     maximising the row-activation bandwidth on both sides of the victim.
__global__ void double_sided_hammer_kernel(
    const unsigned long long* __restrict__ mem,
    long long words_per_row,
    long long row_a,
    long long row_b,
    int iters
) {
    long long tid = (long long)blockIdx.x * blockDim.x + (long long)threadIdx.x;
    if (tid >= words_per_row) return;

    // Pointers to word `tid` within each aggressor row.
    const unsigned long long* ptr_a = mem + row_a * words_per_row + tid;
    const unsigned long long* ptr_b = mem + row_b * words_per_row + tid;

    for (int i = 0; i < iters; i++) {
        // Two cache-bypassing DRAM activations per iteration:
        // ACTIVATE row_a → PRECHARGE → ACTIVATE row_b → PRECHARGE.
        // With the victim row sandwiched between them (|row_a - row_b| == 2),
        // charge leaks into the victim from both sides on every round.
        load_bypass_cache(ptr_a);
        load_bypass_cache(ptr_b);
    }
}

// Python-callable wrapper (invoked from hammer_once()).
void double_sided_hammer(
    torch::Tensor mem,          // GPU tensor: shape (num_rows, row_size_bytes)
    int64_t row_a,              // first aggressor row index
    int64_t row_b,              // second aggressor row index
    int64_t iters               // activations per aggressor per thread
) {
    TORCH_CHECK(mem.is_cuda(), "mem must be a CUDA tensor");
    TORCH_CHECK(mem.scalar_type() == torch::kUInt8, "mem must be uint8");
    TORCH_CHECK(mem.is_contiguous(), "mem must be contiguous");

    int64_t words_per_row = mem.size(1) / 8;   // bytes -> 64-bit words
    auto* ptr = reinterpret_cast<const unsigned long long*>(
        mem.data_ptr<uint8_t>()
    );

    int block = 256;
    int grid  = (int)((words_per_row + block - 1) / block);

    double_sided_hammer_kernel<<<grid, block>>>(
        ptr, words_per_row, row_a, row_b, (int)iters
    );
}
"""

_hammer_ext = None  # compiled extension, cached after first load


def _load_hammer_ext():
    """Compile and load the CUDA hammer extension (called once, then cached)."""
    global _hammer_ext
    if _hammer_ext is not None:
        return _hammer_ext
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not found — is the NVIDIA driver loaded?\n"
            "  Load it:  sudo modprobe nvidia\n"
            "  Verify:   nvidia-smi\n"
            "Then re-run this script."
        )
    from torch.utils.cpp_extension import load_inline
    build_dir = "/tmp/gpu_hammer_build"
    os.makedirs(build_dir, exist_ok=True)
    _hammer_ext = load_inline(
        name="gpu_hammer_kernel",
        cpp_sources=[_HAMMER_CPP],
        cuda_sources=[HAMMER_KERNEL_SRC],
        functions=["double_sided_hammer"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False,
    )
    return _hammer_ext


# ── CudaDRAM ──────────────────────────────────────────────────────────────────

@dataclass
class _VictimState:
    """Per-victim-row hammering bookkeeping."""
    rounds: int = 0
    flipped: bool = False


class CudaDRAM:
    """Real CUDA-backed GDDR6 DRAM model.

    Internal layout
    ---------------
    * ``_gpu_rows`` — PyTorch CUDA tensor, shape (num_rows, row_size_bytes).
      The hammer kernel accesses this tensor on-device using cache-bypassing
      loads. Timing comes from CUDA events around the kernel launch.

    * ``_shadow`` — CPU bytearray mirroring the current logical DRAM state
      (initial writes + injected flip). ``read()`` always reads from the
      shadow because the flip may be a profiled injection (not a real bit
      flip visible in GPU memory).

    Flip model
    ----------
    flip_location = (victim_row, byte_offset, bit_position) encodes where
    the pre-profiled flip lands. After HAMMER_THRESHOLD_ROUNDS calls to
    hammer_once() with the correct double-sided pair, the bit is toggled in
    both the shadow and the GPU tensor (making it detectable from both sides).
    """

    def __init__(
        self,
        num_rows: int = ROWS_PER_BANK,
        row_size: int = ROW_SIZE_BYTES,
        flip_location: tuple[int, int, int] = (0, 0, 1),
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU not found — load the NVIDIA driver first:\n"
                "  sudo modprobe nvidia\n"
                "  nvidia-smi  ← should show your GPU"
            )
        self.num_rows = num_rows
        self.row_size = row_size
        self.flip_location = flip_location

        # Real GPU memory: shape (num_rows, row_size), uint8
        self._gpu_rows: torch.Tensor = torch.zeros(
            num_rows, row_size, dtype=torch.uint8, device="cuda"
        )

        # CPU shadow: mirrors logical DRAM state (initial writes + injected flips)
        self._shadow = bytearray(num_rows * row_size)

        # Per-victim hammering state
        self._victim_state: dict[int, _VictimState] = {}

    @classmethod
    def zeros(cls, num_rows: int = ROWS_PER_BANK) -> "CudaDRAM":
        """Allocate a zero-initialised bank and eagerly compile the kernel."""
        _load_hammer_ext()   # compile now so the first hammer_once() is fast
        return cls(num_rows=num_rows)

    # ── Normal reads / writes ─────────────────────────────────────────────────
    def read(self, row: int, offset: int = 0, length: int | None = None) -> bytes:
        """Read bytes from the CPU shadow (reflects initial writes + injected flip)."""
        if length is None:
            length = self.row_size - offset
        base = row * self.row_size
        return bytes(self._shadow[base + offset : base + offset + length])

    def write(self, row: int, offset: int, data: bytes) -> None:
        """Write data into both the GPU tensor and the CPU shadow.

        Writing a row resets its hammer-accumulation state (a write involves
        an ACTIVATE command, which refreshes the row buffer).
        """
        end = offset + len(data)
        assert end <= self.row_size, f"write past row end: {end} > {self.row_size}"

        # Update GPU tensor (device-side)
        gpu_data = torch.frombuffer(bytes(data), dtype=torch.uint8).to("cuda")
        self._gpu_rows[row, offset:end] = gpu_data

        # Update CPU shadow
        base = row * self.row_size
        self._shadow[base + offset : base + end] = data

        # Writing the row acts as a refresh → reset hammer state for that row
        self._victim_state.pop(row, None)

    # ── Hammering primitive ───────────────────────────────────────────────────
    def hammer_once(self, aggressor_a: int, aggressor_b: int) -> int:
        """One round of double-sided RowHammer on the real GPU.

        Launches the CUDA kernel for ACTIVATIONS_PER_ROUND cache-bypassing
        loads per aggressor row. Timing comes from CUDA events (nanoseconds).

        Only the pair where |aggressor_a - aggressor_b| == 2 accumulates
        towards the flip threshold — exactly as in real RowHammer where only
        double-sided aggressor pairs stress the victim row between them.

        Returns
        -------
        int
            Measured nanoseconds for this kernel execution (real GPU timing).
        """
        ext = _load_hammer_ext()

        # CUDA event timing: microsecond-resolution on GDDR6
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end   = torch.cuda.Event(enable_timing=True)

        ev_start.record()
        ext.double_sided_hammer(
            self._gpu_rows, int(aggressor_a), int(aggressor_b),
            ACTIVATIONS_PER_ROUND,
        )
        ev_end.record()
        torch.cuda.synchronize()

        # elapsed_time() returns milliseconds; convert to integer nanoseconds
        elapsed_ns = int(ev_start.elapsed_time(ev_end) * 1_000_000)

        # Accumulate towards the flip threshold only for correct double-sided pairs
        if abs(aggressor_a - aggressor_b) == 2:
            victim = (aggressor_a + aggressor_b) // 2
            state = self._victim_state.setdefault(victim, _VictimState())
            if not state.flipped:
                state.rounds += 1
                if state.rounds >= HAMMER_THRESHOLD_ROUNDS:
                    self._apply_flip(victim)
                    state.flipped = True

        return elapsed_ns

    def has_flipped(self, victim_row: int) -> bool:
        """True once a flip has landed in victim_row (injected or real)."""
        st = self._victim_state.get(victim_row)
        if st and st.flipped:
            return True
        # Also check for real hardware bit flips in GPU memory
        if self._detect_real_gpu_flip(victim_row):
            state = self._victim_state.setdefault(victim_row, _VictimState())
            state.flipped = True
            self._sync_gpu_to_shadow(victim_row)
            return True
        return False

    def refresh(self) -> None:
        """Simulate a whole-array DRAM refresh — resets all hammer counters."""
        self._victim_state.clear()

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _apply_flip(self, victim: int) -> None:
        """Toggle the pre-profiled bit in both the shadow and GPU tensor.

        This models the "profiled injection" phase of a real GPUHammer attack:
        after templating, the attacker knows which (row, byte, bit) flips and
        targets that exact location. We bake the profiling result into
        flip_location and apply it here once the activation threshold is crossed.
        """
        flip_row, byte_off, bit_pos = self.flip_location
        if victim != flip_row:
            return

        # Toggle in CPU shadow (read() will see the flipped PTE)
        base = flip_row * self.row_size
        self._shadow[base + byte_off] ^= (1 << bit_pos)

        # Toggle in GPU tensor (makes has_flipped() → _detect_real_gpu_flip() work too)
        self._gpu_rows[flip_row, byte_off] ^= (1 << bit_pos)

    def _detect_real_gpu_flip(self, victim_row: int) -> bool:
        """Check GPU memory for actual hardware bit flips at the flip location."""
        flip_row, byte_off, bit_pos = self.flip_location
        if victim_row != flip_row:
            return False
        # A real flip changes the GPU byte but not the CPU shadow (before injection)
        gpu_byte   = int(self._gpu_rows[victim_row, byte_off].item())
        shadow_byte = self._shadow[victim_row * self.row_size + byte_off]
        return ((gpu_byte ^ shadow_byte) >> bit_pos) & 1 == 1

    def _sync_gpu_to_shadow(self, victim_row: int) -> None:
        """Copy the GPU row back to the CPU shadow (used after a real flip)."""
        gpu_bytes = self._gpu_rows[victim_row].cpu().numpy().tobytes()
        base = victim_row * self.row_size
        self._shadow[base : base + self.row_size] = gpu_bytes
