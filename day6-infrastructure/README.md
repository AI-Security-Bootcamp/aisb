# Day 6 — Securing AI infrastructure

## What to expect

The ideal day combines evidence-based analysis of frontier-lab security guidance,
a real GPU-container vulnerability case study, and a clearly labelled simulator
in which participants trace and then break a hardware-to-kernel attack chain.

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | Linux processes/permissions, containers, virtual memory, basic C-style memory layout, and Python | Trace data and authority across a container runtime, GPU virtual memory, page tables, DMA mappings, and kernel objects |
| ML | High-level GPU use in training/inference; no CUDA-kernel expertise required | Explain which infrastructure properties are ML-specific and which are conventional shared-compute risks |
| Security | CVE analysis, exploit preconditions, memory corruption, isolation, least privilege, and defense in depth | Analyze a container escape, state the assumptions in a RowHammer/DMA chain, and implement or select mitigations that break particular stages |
| Theory | Threat actors, security levels, and assurance arguments | Compare security frameworks using cited evidence and turn a simulated exploit into a bounded real-world feasibility claim |

### Preparation

- Required prework: MITRE *AI Security 101* and the assigned Day 6 reports/advisory.
- Recommended refreshers: Linux namespaces/container runtimes, virtual memory/page tables, IOMMU/DMA, RowHammer, and GPU multi-tenancy.
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
      - [ ] Have questions they should answer after reading (eg, [https://github.com/pranavgade20/aisb/blob/master/w1d3/w1d3_instructions.md](https://github.com/pranavgade20/aisb/blob/master/w1d3/w1d3_instructions.md))
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
