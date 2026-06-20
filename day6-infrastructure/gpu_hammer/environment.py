"""End-to-end target environment for the GPUBreach lab.

Wires together real CUDA GPU DRAM, the GPU page table model, IOMMU, driver
page, and kernel credential struct into a single :class:`Environment` object.
Students drive the four stages of the attack against this environment from
their answers file.

The DRAM is backed by real GDDR6 memory (via :class:`CudaDRAM`). The GPU
page table, IOMMU, driver page, and kernel credential struct are modelled
on the CPU side — these are kernel/driver-level structures that cannot be
accessed from user space without a kernel module.

Constants like :data:`PTE_ROW` and :data:`VICTIM_GPU_VADDR` represent
attacker reconnaissance results (templating known-flipping DRAM locations,
spraying GPU allocations until a VA lands on a target row) — that
profiling/spraying phase is out of scope for this lab.
"""

from __future__ import annotations

from dataclasses import dataclass

from .cuda_dram import CudaDRAM, ROW_SIZE_BYTES
from .pte import (
    PTE,
    GPUPageTable,
    PTE_BYTES,
    APERTURE_BIT_POS,
    APERTURE_GPU_LOCAL,
    encode_pte,
)
from .dma import DriverPage, IOMMU


# ── Attack-chain pre-conditions ───────────────────────────────────────────────

PTE_ROW: int = 4242
"""DRAM row that holds the victim PTE. An attacker identifies this row by
templating flips on the target GPU and spraying GPU allocations — given here
as the lab's pre-profiled result."""

PTE_OFFSET_IN_ROW: int = 32
"""Byte offset within PTE_ROW where the 8-byte victim PTE begins.
Chosen so that byte 32 bit 1 is the aperture bit — aligned with the
pre-profiled flip location for this device."""

VICTIM_GPU_VADDR: int = 0x0000_DEAD_0000
"""GPU virtual address whose PTE we corrupt with the RowHammer flip."""

_INITIAL_PHYSICAL_FRAME: int = 0x1234
"""PFN encoded in the original PTE. After the aperture bit flips, this is
interpreted as a *host* physical frame. The attacker has groomed host memory
so that this PFN coincides with the driver's DMA-mapped page."""

FLAG: str = "FLAG{gpubreach_rowhammer_aperture_oob_root}"


# ── Environment container ─────────────────────────────────────────────────────

@dataclass
class Environment:
    """Full GPUBreach target — real GPU DRAM + kernel-level simulation.

    The DRAM is real CUDA device memory; the page table, IOMMU, and
    credential struct are CPU-side models of kernel objects the attacker
    would corrupt through the DMA chain.

    Stage-success flags are set by student code as the chain progresses.
    ``check_all()`` verifies them and prints the flag on full success.
    """

    dram: CudaDRAM
    page_table: GPUPageTable
    iommu: IOMMU
    driver_page: DriverPage

    stage1_aggressors_ok: bool = False
    stage2_flipped_in_refresh_window: bool = False
    stage3_aperture_changed: bool = False
    stage4_root_obtained: bool = False

    @property
    def kernel_cred(self):
        return self.driver_page.cred

    @property
    def victim_pte(self) -> PTE:
        return self.page_table.cached_pte

    def check_all(self) -> bool:
        """Print a stage-by-stage report and, if all stages pass, the flag."""
        stages = [
            ("Stage 1 — aggressor rows identified",             self.stage1_aggressors_ok),
            ("Stage 2 — flip landed inside the 64ms refresh window", self.stage2_flipped_in_refresh_window),
            ("Stage 3 — aperture bit flipped in the live PTE",  self.stage3_aperture_changed),
            ("Stage 4 — OOB DMA wrote euid=0 into the cred struct", self.stage4_root_obtained),
        ]
        print("── GPUBreach attack chain ──")
        for name, ok in stages:
            print(f"  {'✓' if ok else '✗'} {name}")

        if all(ok for _, ok in stages):
            print()
            print("  🎉 All stages succeeded — root achieved.")
            print(f"  {FLAG}")
            return True
        print()
        print("  Not all stages succeeded; no flag.")
        return False


def make_environment() -> Environment:
    """Build a fresh target environment primed for the GPUBreach attack chain.

    Initial state:
    * Real CUDA GDDR6 memory with a valid, GPU-local PTE written into
      PTE_ROW at PTE_OFFSET_IN_ROW.
    * GPU page table whose cached PTE points at GPU VRAM (aperture = 0).
    * DriverPage with a non-root credential struct (euid = 1000).
    * IOMMU that authorises DMA only to the driver page.
    * DRAM flip location pre-profiled to the aperture bit of the PTE.
    """
    dram = CudaDRAM.zeros()

    # Pre-profiled hammer location: aperture bit of the PTE lives at
    # (PTE_ROW, PTE_OFFSET_IN_ROW, bit APERTURE_BIT_POS = 1).
    dram.flip_location = (PTE_ROW, PTE_OFFSET_IN_ROW, APERTURE_BIT_POS)

    # Write the initial PTE into real GPU DRAM (and CPU shadow).
    initial_pte = PTE(
        valid=True,
        aperture=APERTURE_GPU_LOCAL,
        physical_frame=_INITIAL_PHYSICAL_FRAME,
    )
    dram.write(PTE_ROW, PTE_OFFSET_IN_ROW, encode_pte(initial_pte))

    # Build the page table with the cached PTE already populated.
    page_table = GPUPageTable(
        vaddr=VICTIM_GPU_VADDR,
        dram_row=PTE_ROW,
        pte_offset_in_row=PTE_OFFSET_IN_ROW,
        cached_pte=initial_pte,
    )

    driver_page = DriverPage()
    iommu = IOMMU(allowed_page=driver_page)

    return Environment(
        dram=dram,
        page_table=page_table,
        iommu=iommu,
        driver_page=driver_page,
    )
