# Day 6 — Securing AI infrastructure

## What to expect

The ideal day combines evidence-based analysis of frontier-lab security guidance,
a real GPU-container vulnerability case study, and a clearly labelled simulator
in which participants trace and then break a hardware-to-kernel attack chain.

**Suggested time:** 6 hours for the core route; optional work only if time remains

**Exercises:** [Open the participant instructions](day6_final_instructions.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | Linux processes, permissions, namespaces, virtual memory, and C pointers/arrays | Trace data and authority across a container runtime, GPU virtual memory, page tables, DMA mappings, and kernel objects |
| ML | GPU roles in training/inference and host/device memory | Explain which infrastructure properties are ML-specific and which are conventional shared-compute risks |
| Security | CVE analysis, memory corruption, isolation, least privilege, and defense in depth | Analyze a container escape, state the assumptions in a RowHammer/DMA chain, and implement or select mitigations that break particular stages |
| Theory | Security claims, evidence, assumptions, and adversary capability tiers | Compare security frameworks using cited evidence and turn a simulated exploit into a bounded real-world feasibility claim |

### Background

- For the document-analysis exercise, use the assigned sections—not the full documents—of RAND's [*Securing AI Model Weights*](https://www.rand.org/content/dam/rand/pubs/research_reports/RRA2800/RRA2849-1/RAND_RRA2849-1.pdf), IAPS's [*Accelerating AI Data Center Security*](https://www.iaps.ai/research/accelerating-ai-data-center-security), and the local [SL5 Novel Recommendations](SL5_NOVEL-RECOMMENDATIONS.pdf).
- Before the container-escape exercise, read the [NVIDIAScape advisory](https://www.wiz.io/blog/nvidia-ai-vulnerability-cve-2025-23266-nvidiascape).
- Before the GPUBreach simulator, read the [GPUHammer overview](https://gpuhammer.com/).
- References: Linux [namespaces](https://man7.org/linux/man-pages/man7/namespaces.7.html), Linux [page tables](https://docs.kernel.org/mm/page_tables.html), [x86 IOMMU support](https://docs.kernel.org/arch/x86/iommu.html), and NVIDIA [multi-instance GPU](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/) isolation.
- The simulator is conceptual; it should not require or imply access to vulnerable production hardware.

### Current-state TODOs

- [ ] Rebalance the advertised 2 hours + 45 minutes + 3+ hours into a feasible core route with explicit stretch work.
- [ ] Replace report scavenger hunts and uncited answer-key numbers with one realistic lab case, page citations, and evidence comparison.
- [ ] Label GPUBreach as a hypothetical/composite simulator and surface assumptions such as templating, memory grooming, useful bit selection, allocator placement, and ECC.
- [ ] Replace “IOMMU bypass” with the accurate sub-page memory-safety/isolation failure description.
- [ ] Restore the defensive debrief and add a lab where participants patch or configure controls, then prove which chain stage fails and at what cost.
- [ ] Remove or clearly archive the duplicated nested Day 6 tree so authors and participants have one source of truth.
- [x] Stop calling Day 6 the final day now that a Day 7 synthesis exists.

## Legacy planning notes

* day1: we walk them through the entire LLM training pipeline + infrastructure, and high-level of the threat model (maybe there is a nice way of integrating defenses as well, but surely it'll be too long in that case)
* day 2-6: we walk through each part of the pipeline: data, training, finetuning, inference, agents - how it is done, how can we break it
* day 7: governance things? control? all the topics that don't fit in cleanly in the days

Underline are the exercises we want
*Italics* are the ML prereqs we need
**Bold** is the theme for the day

* **Unsure**
  * Scalable oversight
  * Other kinds o injection - e.g., resizing images
  * Timing attacks against API classifiers

## Tasks (per day)

- [x] ~~Decide on the narrative -> topics~~
- [ ] Make the exercises (solutions)
- [ ] Make the exercise instructions
- [ ] Collate papers, readings, relevant to the topics of the day
      - [ ] Suggest topics for discussion groups
      - [ ] Have questions they should answer after reading (for example, see the [Day 6 document-analysis discussion questions](day6_securing_discussion.md))
- [ ] FInally, have a narrative they follow

## Structure of a Day

- The night before: reading list
- First 30-45 mins: overview of the day's topics, highlighting important things from pre-readings
- [Content]
  - Hands-on labs
  - At most 1 discussion group
    - Discuss X theoretical topic, with 1 instructor present
  - ???
- Last 30 mins: guest lecture
