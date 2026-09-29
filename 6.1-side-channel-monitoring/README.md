# 6.1 — Power side channels

**Suggested time:** 2 hours, working in pairs. The first 15 minutes cover the
measurement hardware and an idle capture.

**Exercises:** [Participant instructions](section1_instructions.md).
Authors edit [section1_solution.py](section1_solution.py); the bootcamp build
generates the instructions and tests from it.

| Area | Prerequisites coming in | Learning outcomes |
| --- | --- | --- |
| Engineering | Create a PyTorch tensor with a specified shape and device; write a forward/loss/backward/optimizer step | Submit a standalone script to Slurm and inspect its logs/results; place acquisition boundaries around completed CUDA work; export a CPU/CUDA timeline; retain the metadata needed to combine independent recordings |
| ML | Explain what a batch, forward pass, backward pass, and parameter update represent | Identify these phases in a trace; distinguish batch-size and sequence-length effects on throughput and kernel shapes |
| Security | - | Test whether an observer of supply-current variation can infer model depth or workload shape; separate evidence from alternative explanations |
| Theory | - | Convert ADC counts to nominal current; explain bandwidth, sampling, RMS, and timing uncertainty; explain why a current probe cannot reveal total DC power in this setup |

## Background

- https://www.ti.com/lit/ta/ssztdb4/ssztdb4.pdf
- https://www.tek.com/en/documents/primer/oscilloscope-basics
- [slurm basics - `sbatch`](https://slurm.schedmd.com/sbatch.html)
- PyTorch [Optimization Loop](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html#optimization-loop)
- 