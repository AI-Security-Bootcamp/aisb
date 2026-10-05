"""End-to-end target environment for the GPUBreach lab.

This module wires together the DRAM, GPU page table, IOMMU, driver page,
and kernel credential struct into a single :class:`Environment` object.
Students drive the four stages of the attack against this environment from
their answers file.

Constants like :data:`PTE_ROW` and :data:`VICTIM_GPU_VADDR` are deliberately
exposed at module level so students can reference them by name rather than
re-derive them — they're artefacts of attacker reconnaissance (templating
known-flipping DRAM locations, spraying GPU allocations until a VA lands on
a target row, etc.), which is out of scope for this lab.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .dram import DRAM, ROW_SIZE_BYTES
from .pte import (
    PTE,
    GPUPageTable,
    PTE_BYTES,
    APERTURE_BIT_POS,
    APERTURE_GPU_LOCAL,
    encode_pte,
)
from .dma import DriverPage, IOMMU


# ── Attack-chain pre-conditions ──────────────────────────────────────────────
PTE_ROW: int = 4242
"""DRAM row that holds the victim PTE. An attacker identifies this row by
templating flips against known VRAM contents and spraying allocations — for
this lab we give it to you."""

PTE_OFFSET_IN_ROW: int = 32
"""Byte offset within ``PTE_ROW`` where the 8-byte victim PTE begins.
Chosen so that byte 32 bit 1 is the aperture bit — aligned with the DRAM
template-flip location."""

VICTIM_GPU_VADDR: int = 0x0000_DEAD_0000
"""GPU virtual address whose PTE we're going to corrupt."""

_INITIAL_PHYSICAL_FRAME: int = 0x1234
"""PFN encoded into the original PTE. After the aperture bit flips this
number is interpreted as a *host* physical frame — and the attacker has
groomed host memory so that this PFN coincides with the driver's
DMA-mapped page. In this simulation we bake that coincidence in."""


FLAG: str = "FLAG{gpubreach_rowhammer_aperture_oob_root}"


# ── Environment container ────────────────────────────────────────────────────
@dataclass
class Environment:
    """A full GPUBreach target.

    Stage-success flags are stored on the environment. :meth:`check_all`
    verifies them and prints the flag if every stage succeeded.
    """

    dram: DRAM
    page_table: GPUPageTable
    iommu: IOMMU
    driver_page: DriverPage

    stage1_aggressors_ok: bool = False
    stage2_flipped_in_refresh_window: bool = False
    stage3_aperture_changed: bool = False
    stage4_root_obtained: bool = False

    # Convenience proxies
    @property
    def kernel_cred(self):
        return self.driver_page.cred

    @property
    def victim_pte(self) -> PTE:
        return self.page_table.cached_pte

    def check_all(self) -> bool:
        """Print a stage-by-stage report and, if all stages pass, the flag."""
        stages = [
            ("Stage 1 — aggressor rows identified", self.stage1_aggressors_ok),
            (
                "Stage 2 — flip landed inside the 64ms refresh window",
                self.stage2_flipped_in_refresh_window,
            ),
            (
                "Stage 3 — aperture bit flipped in the live PTE",
                self.stage3_aperture_changed,
            ),
            (
                "Stage 4 — OOB DMA wrote euid=0 into the cred struct",
                self.stage4_root_obtained,
            ),
        ]
        print("── GPUBreach attack chain ──")
        for name, ok in stages:
            mark = "✓" if ok else "✗"
            print(f"  {mark} {name}")

        if all(ok for _, ok in stages):
            print()
            print(f"  🎉 All stages succeeded — root achieved.")
            print(f"  {FLAG}")
            return True
        print()
        print("  Not all stages succeeded; no flag.")
        return False


def make_environment() -> Environment:
    """Build a fresh target environment primed for the attack chain.

    The initial state has:

    * A zeroed DRAM bank with a valid, GPU-local PTE written into
      ``PTE_ROW`` at ``PTE_OFFSET_IN_ROW``.
    * A GPU page table whose cached PTE points at GPU VRAM.
    * A DriverPage with a non-root credential struct.
    * An IOMMU that authorises DMA to the driver page.
    * The DRAM hammer primitive pre-profiled to flip the PTE's aperture bit.
    """
    dram = DRAM.zeros()

    # Pre-profile the hammering location: the aperture bit of the PTE is at
    # (PTE_ROW, PTE_OFFSET_IN_ROW, bit APERTURE_BIT_POS).
    dram.flip_location = (PTE_ROW, PTE_OFFSET_IN_ROW, APERTURE_BIT_POS)

    # Seed the PTE in DRAM.
    initial_pte = PTE(
        valid=True,
        aperture=APERTURE_GPU_LOCAL,
        physical_frame=_INITIAL_PHYSICAL_FRAME,
    )
    dram.write(PTE_ROW, PTE_OFFSET_IN_ROW, encode_pte(initial_pte))

    # Build the page table with the cached PTE already populated (the MMU
    # has already seen this entry at least once).
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
