"""GPU Page Table Entry (PTE) encoding and a tiny page-table walker.

Modern GPU PTEs are wide (NVIDIA uses 8-byte entries with a bit-field layout
similar to x86-64 PTEs). The two fields we care about:

* ``valid`` — entry points to a real page.
* ``aperture`` — a single bit that selects whether the backing store is in
  the GPU's local VRAM or in CPU system memory (reached via PCIe DMA).

The attack flips the aperture bit from 0 (GPU local) to 1 (system memory).
After the flip, any DMA the driver performs on behalf of this virtual
address goes to the *CPU* physical address encoded in the PTE — which, with
careful groom-the-heap work on the attacker side, lines up on the driver's
own DMA-mapped page.

Layout we use for the simulation (little-endian, 8 bytes total):

    byte 0: flags
        bit 0 = valid
        bit 1 = aperture (0 = GPU local, 1 = system memory)   ← the target
        bit 2..7 reserved
    bytes 1..6: 48-bit physical frame number
    byte 7: reserved

This is simpler than any real device's layout but keeps the crucial property
that one bit flip in a well-chosen location changes the aperture without
destroying the rest of the entry.
"""

from __future__ import annotations

from dataclasses import dataclass

PTE_BYTES: int = 8

APERTURE_BIT_POS: int = 1
"""Position of the aperture bit inside byte 0 of the PTE."""

APERTURE_GPU_LOCAL: int = 0
APERTURE_SYSTEM: int = 1


@dataclass
class PTE:
    """A decoded GPU page-table entry."""

    valid: bool
    aperture: int  # 0 = GPU VRAM, 1 = system memory
    physical_frame: int  # 48-bit PFN


def encode_pte(pte: PTE) -> bytes:
    """Serialise a PTE into 8 little-endian bytes."""
    flags = 0
    if pte.valid:
        flags |= 1 << 0
    if pte.aperture:
        flags |= 1 << APERTURE_BIT_POS
    pfn_bytes = pte.physical_frame.to_bytes(6, "little")
    return bytes([flags]) + pfn_bytes + b"\x00"


def decode_pte(raw: bytes) -> PTE:
    """Deserialise the 8-byte on-DRAM PTE."""
    assert len(raw) == PTE_BYTES, f"PTE must be {PTE_BYTES} bytes, got {len(raw)}"
    flags = raw[0]
    valid = bool(flags & (1 << 0))
    aperture = (flags >> APERTURE_BIT_POS) & 1
    physical_frame = int.from_bytes(raw[1:7], "little")
    return PTE(valid=valid, aperture=aperture, physical_frame=physical_frame)


@dataclass
class GPUPageTable:
    """A one-entry GPU page table.

    Real GPU page tables have multi-level structures (NVIDIA's MMU uses a
    4-level radix tree of PTEs in VRAM). For this lab we only care about a
    single target PTE — the one the attacker has chosen to corrupt — so we
    keep a single virtual-address → DRAM-location mapping.

    The cached PTE decoded from DRAM lives in :attr:`cached_pte`. Students
    invoke :meth:`sync_from_dram` in Task 3 to re-read the PTE from DRAM
    after the RowHammer flip, simulating what happens on the next GPU TLB
    miss or explicit invalidation.
    """

    vaddr: int
    """GPU virtual address this page table entry serves."""

    dram_row: int
    """DRAM row that stores the PTE."""

    pte_offset_in_row: int
    """Byte offset within that row where the PTE begins."""

    cached_pte: PTE
    """The currently-decoded PTE (what the GPU MMU would use from its TLB)."""

    def sync_from_dram(self, dram) -> PTE:
        """Re-read the PTE from DRAM. Models a TLB invalidation / miss.

        After this call :attr:`cached_pte` reflects whatever is in DRAM now
        — including bit flips from RowHammer.
        """
        raw = dram.read(self.dram_row, self.pte_offset_in_row, PTE_BYTES)
        self.cached_pte = decode_pte(raw)
        return self.cached_pte

    def translate(self, vaddr: int) -> PTE:
        """Translate a virtual address. Returns the cached PTE.

        In a real MMU walk we'd index by the VA's page number; here we only
        serve one virtual address. Raises if the caller asks for anything
        else or if the PTE is invalid.
        """
        if vaddr != self.vaddr:
            raise KeyError(f"no mapping for vaddr {vaddr:#x}")
        if not self.cached_pte.valid:
            raise RuntimeError("PTE is not valid")
        return self.cached_pte
