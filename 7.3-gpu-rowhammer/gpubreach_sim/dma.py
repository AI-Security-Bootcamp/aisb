"""Driver-side DMA target: IOMMU, DMA-mapped page, and adjacent kernel state.

When a GPU kernel touches a virtual address whose PTE has the aperture bit
set to "system memory", the GPU issues a PCIe DMA transaction to the
physical address named in the PTE. The IOMMU validates that transaction
against its own page tables — if the destination page is mapped for the
GPU's PCIe BDF, the write is allowed; otherwise the IOMMU aborts it.

The key insight the GPUBreach attack exploits is this:

    *The IOMMU enforces page-level isolation — "this page is DMA-able by
    this device" — but it cannot enforce intra-page bounds.*

The NVIDIA driver registers a small DMA scratch buffer (a ring queue,
doorbell region, etc.) inside a 4KB page, and immediately following that
buffer — in the same page — lives kernel state such as a credential
structure or a function pointer. The IOMMU happily allows a DMA of any
length that stays within the 4KB page. The driver is *supposed* to validate
the length against the *buffer's* size, not the *page's* size. The bug is a
missing length check in the driver's DMA fast path — a CWE-787
out-of-bounds write.

We model:
* A 4KB :class:`DriverPage` with a driver buffer at offset 0 and a kernel
  credential struct at offset 128.
* An :class:`IOMMU` that validates writes against the page boundary.
* A :func:`perform_gpu_dma` entry point that walks the GPU PT, consults the
  IOMMU, and performs the write.
"""

from __future__ import annotations

from dataclasses import dataclass

from .pte import APERTURE_GPU_LOCAL, APERTURE_SYSTEM


PAGE_SIZE: int = 4096
"""Host physical page size. Matches Linux x86-64 / ARM64 base pages."""

DRIVER_BUFFER_OFFSET: int = 0
"""Offset of the driver's DMA scratch buffer inside the DMA-mapped page."""

DRIVER_BUFFER_SIZE: int = 128
"""Declared size of the driver DMA buffer. The driver's DMA fast path fails
to validate against this limit — writes larger than this overflow into the
credential struct that immediately follows."""

CRED_OFFSET: int = 128
"""Offset of the adjacent kernel credential struct inside the same page."""

CRED_EUID_SIZE: int = 4
"""Size of the euid field in bytes."""

_INITIAL_EUID: int = 1000  # non-root user
_ROOT_EUID: int = 0


@dataclass
class KernelCred:
    """A Linux-like process credential struct.

    Only the euid matters for our exploit. In real kernels ``struct cred``
    contains many fields (uid, gid, capability masks, LSM hooks, ...) any of
    which can be overwritten for privilege escalation, but overwriting euid
    to 0 is the textbook example.
    """

    euid: int = _INITIAL_EUID

    def is_root(self) -> bool:
        return self.euid == _ROOT_EUID


class DriverPage:
    """The DMA-mapped 4KB host page owned by the GPU driver.

    Layout::

        offset 0    ──── driver scratch buffer (DRIVER_BUFFER_SIZE=128) ────
        offset 128  ──── KernelCred.euid (4 bytes)                     ────
        offset 132  ──── padding to 4KB                                ────

    The credential struct lives in the same page as the DMA buffer purely
    because that's what real driver allocators do: a single ``kzalloc`` (or
    ``dma_alloc_coherent``) returns a single 4KB slab, and the driver
    sub-allocates multiple structs from it.
    """

    def __init__(self) -> None:
        self.storage = bytearray(PAGE_SIZE)
        self.cred = KernelCred()
        self._sync_cred_to_storage()

    def _sync_cred_to_storage(self) -> None:
        """Serialise the cred struct's euid into its in-page location."""
        self.storage[CRED_OFFSET : CRED_OFFSET + CRED_EUID_SIZE] = (
            self.cred.euid.to_bytes(CRED_EUID_SIZE, "little")
        )

    def _sync_storage_to_cred(self) -> None:
        """Re-decode the cred struct after a raw write landed in the page."""
        self.cred.euid = int.from_bytes(
            self.storage[CRED_OFFSET : CRED_OFFSET + CRED_EUID_SIZE], "little"
        )


@dataclass
class IOMMU:
    """Minimal IOMMU model covering exactly what the real one does here.

    The IOMMU is programmed with one DMA window per device: "device X is
    allowed to read/write host physical page P". On every DMA transaction
    from that device it checks:

        * Is the target page P mapped for the device?  (yes → allow)
        * Does the write stay entirely within the page?  (yes → allow)

    It does NOT check any sub-page structure. The kernel is responsible for
    sizing its buffers and enforcing bounds inside its own page. When the
    driver fails to do so, the IOMMU is powerless — the whole write is one
    legal PCIe burst to a mapped page.

    This is exactly the threat model the NVIDIA driver's OOB write
    vulnerability sits inside. The IOMMU works as designed.
    """

    allowed_page: DriverPage

    def validate(self, target_page: DriverPage, offset: int, length: int) -> bool:
        """Return True iff the DMA is permitted.

        A real IOMMU would raise a fault (IO_PAGE_FAULT on AMD-Vi,
        DMA_REMAP_FAULT on Intel VT-d) on violation. We just return False.
        """
        if target_page is not self.allowed_page:
            return False
        if offset < 0 or offset + length > PAGE_SIZE:
            return False
        return True


def perform_gpu_dma(
    data: bytes,
    gpu_vaddr: int,
    page_table,
    iommu: IOMMU,
    gpu_dram,
    driver_page: DriverPage,
) -> str:
    """Model the GPU driver's DMA fast path.

    Walks the GPU page table to resolve ``gpu_vaddr``. Depending on the
    aperture bit in the resolved PTE:

    * ``APERTURE_GPU_LOCAL`` — writes land in GPU VRAM (a DRAM row).
    * ``APERTURE_SYSTEM`` — writes cross PCIe into the host driver page.
      The IOMMU validates the transaction; if approved, the driver copies
      ``data`` into its DMA scratch buffer starting at ``DRIVER_BUFFER_OFFSET``.
      Critically, *the driver does not clamp ``len(data)`` to
      ``DRIVER_BUFFER_SIZE``*. A larger payload overflows into adjacent
      kernel memory — the OOB write at the heart of the GPUBreach chain.

    Returns a short string describing what happened, for instructional
    logging. Raises if the IOMMU rejects the transaction.
    """
    pte = page_table.translate(gpu_vaddr)

    if pte.aperture == APERTURE_GPU_LOCAL:
        # Normal path: write into GPU VRAM. The physical_frame in the PTE
        # names a DRAM row in this simplified model.
        row = pte.physical_frame
        gpu_dram.write(row, 0, data[: len(gpu_dram.rows[row])])
        return f"wrote {len(data)} bytes to GPU VRAM row {row}"

    assert pte.aperture == APERTURE_SYSTEM
    # System-memory path: DMA to the host driver page.
    if not iommu.validate(driver_page, DRIVER_BUFFER_OFFSET, len(data)):
        raise PermissionError("IOMMU rejected the DMA transaction")

    # *** The missing bounds check lives here. *** A defensive driver would
    # clamp ``data`` to DRIVER_BUFFER_SIZE; this one does not.
    end = DRIVER_BUFFER_OFFSET + len(data)
    driver_page.storage[DRIVER_BUFFER_OFFSET:end] = data
    driver_page._sync_storage_to_cred()

    overflow = max(0, len(data) - DRIVER_BUFFER_SIZE)
    return (
        f"DMA'd {len(data)} bytes into driver page "
        f"({overflow} bytes past buffer end)"
    )