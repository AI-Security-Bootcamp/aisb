Welcome to the [AI Security Bootcamp](https://www.aisb.dev/)! AISB is a 7-day intensive program for senior security professionals, focused on frontier AI Security.
This repo contains the exercises and links to the reading material you will go through during the bootcamp.

## Curriculum overview

Complete [Day 0 - Setup](day0-setup/README.md) before the bootcamp. Each content day starts with a one-hour lecture followed by six hours of paired labs; review the required background in each section's README beforehand.

**Day 1 - LLM Internals**

- [1.0 - Pre-reading: how LLMs are trained](1.0-readings/README.md)
- [1.1 - LLM internals and chat serialization](1.1-llm-internals/README.md): tokenization, chat templates, and special tokens
- [1.2 - Log probabilities](1.2-logprobs/README.md): output distributions as an attack surface for extraction and adversarial optimization
- [1.3 - Instruction hierarchies and assistant prefills](1.3-instruction-hierarchies/README.md) **(optional)**: instruction priority and prefill attacks
- [1.4 - Prompt injection and RAG poisoning](1.4-prompt-injection/README.md): attacks across the boundary between trusted instructions and retrieved data

**Day 2 - Coding Agents & AI Control**

- [2.1 - Coding-agent attack surface and affordances](2.1-coding-agents/README.md): code execution, tool restrictions, and attack patterns
- [2.2 - AI-control monitoring](2.2-monitoring/README.md): adaptive evasion and evaluation with trusted and untrusted models
- [2.3 - AI-control protocols](2.3-control-protocols/README.md): trusted monitoring, defer-to-trusted, resampling, and safety–usefulness tradeoffs
- [2.4 - Safety simulator](2.4-safety-simulator/README.md) **(optional)**: from monitor scores to system-level safety estimates

**Day 3 - LLM Inference Security**

- [3.1 - Tokenization and prompt construction](3.1-tokenization/README.md): security-relevant generation behavior
- [3.2 - Jailbreaking](3.2-jailbreaking/README.md): techniques for bypassing learned safety behavior
- [3.3 - Guardrails](3.3-guardrails/README.md): keyword filters, classifier-based monitors, LLM-as-judge, and optional linear probes on activations
- [3.4 - Knowledge-distillation attacks](3.4-knowledge-distillation/README.md): transferring capabilities from teacher outputs to a student model
- [3.5 - Model weight extraction via SVD](3.5-weight-extraction/README.md): inferring hidden dimension and extracting the output projection from logit queries

**Day 4 - Training & Data Security**

- [5.1 - Model editing](5.1-model-editing/README.md): surgical weight manipulation to rewrite model behavior
- [5.2 - Backdooring](5.2-backdooring/README.md): backdoors injected via fine-tuning and data poisoning
- [5.3 - Undoing safety fine-tuning](5.3-undoing-safety-finetuning/README.md): refusal-direction ablation and abliteration
- [5.4 - Removing safety behavior with LoRA fine-tuning](5.4-safety-finetuning/README.md) **(optional, instructor-recommended alternative)**

**Day 5 - Adversarial ML**

- [6.1 - Adversarial attacks on vision models](6.1-adversarial-vision/README.md): adversarial examples and gradient-based attacks against image classifiers
- [6.2 - Discrete adversarial optimization with GCG](6.2-adversarial-language/README.md): optimizing adversarial token sequences against language models
- [6.3 - Continuous adversarial prefixes](6.3-prefix-tuning/README.md): optimizing prefixes in embedding space
- [6.4 - Image provenance and watermark robustness](6.4-watermarking/README.md) **(optional)**

**Day 6 - Infrastructure Security**

- 4.1 - Side-channel hardware monitoring
- 4.2 - Agent swarm incident response

**Day 7 - Infrastructure Security & Threat Modeling**

- [7.1 - RAND report: securing AI model weights](7.1-rand-report/README.md): security levels (SL1–SL5) and operational capability tiers (OC1–OC5)
- [7.2 - Threat modeling with adversary matrices](7.2-threat-modeling/README.md): MITRE ATLAS and frontier-lab threat models
- [7.3 - NVIDIA Container Toolkit vulnerability](7.3-nvidia-container-toolkit/README.md): trust boundaries and CVE-2025-23266 (NVIDIAScape)
- [7.4 - GPU RowHammer](7.4-gpu-rowhammer/README.md): page-table corruption, DMA isolation limits, and privilege escalation

## Prerequisites

Please review the [Day 0 setup guide](day0-setup/README.md) to ensure you have the required skills and tools to complete the bootcamp.

## In-person instructions
If you're attending the bootcamp in person, you will spend most of the days pair programming with your assigned partner on the exercises for the given day.

Before the first day, make sure you have completed the instructions in the [Day 0 setup guide](day0-setup/README.md).


### Completing exercises
Each content day is split into numbered section folders named `X.Y-topic`, where `X` is the day and `Y` is the section. Follow the section links in the curriculum overview above; `1.0-readings` contains Day 1 pre-reading. Day 4 content is under development. Start with each section folder's `README.md`: it explains what the section assumes, what you should learn, and which pre-reading and reference resources will help. Then open the `*_instructions.md` file for the exercises. We recommend you open it in your IDE and view the markdown (right-click and select "Open Preview" in VS Code).

The README background is part of the section: review it before beginning the exercises. Only resources explicitly marked **optional longer reading** may be skipped. In each prerequisite table, you should be able to demonstrate every incoming competency with a short question or task; `-` means that the section has no additional prerequisite in that category.

For each section containing code, create a `day#_answers.py` file in that section's folder. Prose-only sections specify a Markdown answer file instead.

The instructions will contain code snippets you need to complete. Add a new `# %%` line to your answers and paste the code snippet under that line. If you went through the setup instructions correctly (with VS Code), you should see "Run Cell" option above the `# %%` which will execute the code in a [Python Interactive Window](https://code.visualstudio.com/docs/python/jupyter-support-py#_jupyter-code-cells). The code snippets contain tests that should initially fail. Complete the TODOs in the code and run the cell until all the tests pass. 

<details>
<summary>Using Python cells</summary>
If you add more code at the bottom of the file and follow it with another `# %%`, this will create another cell which can be run independently in the same session. Cells can be run many times and in any order you choose; the session will maintain variables and state until it is restarted. 
</details>

### Using git
We recommend you save your progress for each day (the answer files you will create with your assigned partner) to a branch in this repo. This will make it possible for you to switch between computers you will use while pair programming, and make your solution available for you to reference later.

First, configure your git repo with:

```bash
git config pull.rebase true
git config --type bool push.autoSetupRemote true
```

**Every day in the morning**, make sure you have the latest version of the repo:
```bash
git checkout main
git pull
```

Make a branch for the day: `git checkout -b <branch name>`, where your branch name should follow the convention `day#/<name>-and-<name>`. For example, if Tamera and Edmund were pairing on the day 3 content, the command would be `git checkout -b day3/tamera-and-edmund`. If you share a first name with someone else in the program, use a unique nickname of your choice for disambiguation. 

Create a new file for your answers (see [completing exercises](#completing-exercises) above) and work through the material with your partner. 

As you work, commit changes to your branch and push them to the repo. To make and push a commit:

```bash
git add :/
git commit -m '<your commit message>'
git push
```

If you want to switch what computer you work on with your partner, they can check out the latest version of the branch with:

```bash
git fetch
git checkout <branch name>
git pull
```


### Testing your setup
If you'd like to try a sample exercise and test your setup, go ahead and complete [day0](./day0-setup/day0_instructions.md)!

## License

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
