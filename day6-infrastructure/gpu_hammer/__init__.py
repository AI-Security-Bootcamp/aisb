"""gpu_hammer — real CUDA RowHammer + kernel-privilege-escalation lab.

The DRAM layer is backed by real GDDR6 device memory; the GPU page table,
IOMMU, and kernel credential struct are CPU-side models (accessing those
from user space requires a kernel module).

Participants import this package from their answers file:

    from gpu_hammer import *

Everything needed for the four-phase attack chain is exported below.
"""

from .cuda_dram import (
    CudaDRAM,
    HAMMER_KERNEL_SRC,
    HAMMER_THRESHOLD_ACTIVATIONS,
    HAMMER_THRESHOLD_ROUNDS,
    ACTIVATIONS_PER_ROUND,
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
    # DRAM / CUDA
    "CudaDRAM",
    "HAMMER_KERNEL_SRC",
    "HAMMER_THRESHOLD_ACTIVATIONS",
    "HAMMER_THRESHOLD_ROUNDS",
    "ACTIVATIONS_PER_ROUND",
    "ACTIVATE_PRECHARGE_NS",
    "REFRESH_WINDOW_MS",
    "ROW_SIZE_BYTES",
    "ROWS_PER_BANK",
    # PTE
    "PTE",
    "GPUPageTable",
    "APERTURE_BIT_POS",
    "APERTURE_GPU_LOCAL",
    "APERTURE_SYSTEM",
    "PTE_BYTES",
    # DMA / driver
    "IOMMU",
    "KernelCred",
    "DriverPage",
    "perform_gpu_dma",
    "DRIVER_BUFFER_OFFSET",
    "DRIVER_BUFFER_SIZE",
    "CRED_OFFSET",
    "PAGE_SIZE",
    # Environment
    "Environment",
    "make_environment",
    "PTE_ROW",
    "PTE_OFFSET_IN_ROW",
    "VICTIM_GPU_VADDR",
    "FLAG",
]
