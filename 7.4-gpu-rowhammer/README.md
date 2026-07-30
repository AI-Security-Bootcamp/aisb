# 7.4 — GPU RowHammer

## What to expect

Participants drive a hypothetical composite simulator from a GPU-memory
RowHammer bit flip through page-table aperture corruption and an in-page DMA
overflow to privilege escalation.

**Suggested time:** 90 minutes for the core route; stretch exercises are optional

**Exercises:** [Open the participant instructions](section4_instructions.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | Python byte manipulation and virtual-memory concepts | Inspect PTE bytes, drive the simulator, and craft a bounded overflow payload |
| ML | GPU local versus host memory | Explain how GPU address translation selects a memory aperture |
| Security | RowHammer, page tables, DMA, IOMMU, and memory corruption | Trace the chain and choose mitigations that break individual stages |
| Theory | Threat-model assumptions and exploit preconditions | Separate demonstrated simulator behavior from claims about practical hardware exploitation |

### Background

- Read the [GPUHammer overview](https://gpuhammer.com/).
- Review Linux [page tables](https://docs.kernel.org/mm/page_tables.html) and [x86 IOMMU support](https://docs.kernel.org/arch/x86/iommu.html).
- The simulator is conceptual and should not require or imply access to vulnerable production hardware.
