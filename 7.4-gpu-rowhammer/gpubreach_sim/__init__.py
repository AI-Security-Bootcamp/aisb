"""gpubreach_sim — simulation scaffold for the GPUBreach attack chain lab.

Students DO NOT modify these files. They call into this package from their
answers file to drive the four stages of the attack chain.

The simulation is intentionally simple but faithful to the real mechanics
documented in the GPUHammer and GPUBreach papers. Real bit flips, GPU page
table walks, and PCIe DMA are modelled at a coarse grain — fine enough to
make the attack reproducible, crude enough to run in plain CPython in a few
seconds.
"""

from .dram import (
    DRAM,
    HAMMER_THRESHOLD_ACTIVATIONS,
    ACTIVATE_PRECHARGE_NS,
    REFRESH_WINDOW_MS,
    ROW_SIZE_BYTES,
    ROWS_PER_BANK,
)
from .pte import (
    PTE,
    GPUPageTable,
    APERTURE_BIT_POS,
    APERTURE_GPU_LOCAL,
    APERTURE_SYSTEM,
    PTE_BYTES,
)
from .dma import (
    IOMMU,
    KernelCred,
    DriverPage,
    perform_gpu_dma,
    DRIVER_BUFFER_OFFSET,
    DRIVER_BUFFER_SIZE,
    CRED_OFFSET,
    PAGE_SIZE,
)
from .environment import (
    Environment,
    make_environment,
    PTE_ROW,
    PTE_OFFSET_IN_ROW,
    VICTIM_GPU_VADDR,
    FLAG,
)

__all__ = [
    "DRAM",
    "HAMMER_THRESHOLD_ACTIVATIONS",
    "ACTIVATE_PRECHARGE_NS",
    "REFRESH_WINDOW_MS",
    "ROW_SIZE_BYTES",
    "ROWS_PER_BANK",
    "PTE",
    "GPUPageTable",
    "APERTURE_BIT_POS",
    "APERTURE_GPU_LOCAL",
    "APERTURE_SYSTEM",
    "PTE_BYTES",
    "IOMMU",
    "KernelCred",
    "DriverPage",
    "perform_gpu_dma",
    "DRIVER_BUFFER_OFFSET",
    "DRIVER_BUFFER_SIZE",
    "CRED_OFFSET",
    "PAGE_SIZE",
    "Environment",
    "make_environment",
    "PTE_ROW",
    "PTE_OFFSET_IN_ROW",
    "VICTIM_GPU_VADDR",
    "FLAG",
]
