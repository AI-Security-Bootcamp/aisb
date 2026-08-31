Welcome to the [AI Security Bootcamp](https://www.aisb.dev/) (AISB)! AISB is a 7-day intensive program for security professionals shaping how we secure emerging AI systems. This repo contains the exercises and links to the reading material you will go through during the bootcamp.

## Curriculum overview

**Day 1 — LLM Internals: What Goes In, What Comes Out**
- Tokenization, chat templates, and special tokens as the security perimeter between instructions and data
- Logprobs and output distributions as an attack surface for extraction and adversarial optimization
- Instruction hierarchies, prefill attacks, and control-token injection
- Prompt injection and RAG poisoning across the boundary between trusted instructions and retrieved data

**Day 2 — Coding Agents & AI Control**
- Agentic AI risks: code-execution affordances, scope of tool restrictions, coding-agent attack patterns
- AI-control monitoring, adaptive evasion, and evaluation with trusted and untrusted models
- Control protocols: trusted monitoring, defer-to-trusted, resampling, and safety–usefulness tradeoffs

**Day 3 — LLM Inference Security**
- A tokenization and prompt-construction refresher focused on security-relevant generation behavior
- Jailbreaking techniques and why safety training is statistical, not structural
- Guardrails: keyword filters, classifier-based monitors, LLM-as-judge, linear probes on activations
- Model extraction and knowledge-distillation attacks against deployed APIs

**Day 4 — AISF Day**
- No scheduled content

**Day 5 — Training & Data Security**
- Model editing and surgical weight manipulation to silently rewrite model behavior
- Backdoors and trojans injected via fine-tuning and data poisoning
- Undoing safety fine-tuning through refusal-direction ablation, with LoRA fine-tuning as an optional alternative

**Day 6 — Adversarial ML**
- Adversarial examples and gradient-based attacks against image classifiers
- Discrete and continuous adversarial optimization against language models
- Optional: image provenance and the requirements for robust watermark detection

**Day 7 — Infrastructure Security & Threat Modeling**
- [RAND security levels](https://www.rand.org/content/dam/rand/pubs/research_reports/RRA2800/RRA2849-1/RAND_RRA2849-1.pdf) (SL1–SL5) and operational capability tiers (OC1–OC5) for adversary modeling
- [MITRE ATLAS](https://atlas.mitre.org/) as a threat-modeling framework for ML systems
- NVIDIA Container Toolkit trust boundaries and CVE-2025-23266
- GPU RowHammer, page-table corruption, DMA isolation limits, and privilege escalation

## Prerequisites

Please review the [Day 0 setup guide](day0-setup/README.md) to ensure you have the required skills and tools to complete the bootcamp.

## In-person instructions
If you're attending the bootcamp in person, you will spend most of the days pair programming with your assigned partner on the exercises for the given day.

Before the first day, make sure you have completed the instructions in the [Day 0 setup guide](day0-setup/README.md).


### Completing exercises
Each content day is split into numbered section folders named `X.Y-topic` (for example, Day 1 is `1.1-llm-internals`, `1.2-logprobs`, `1.3-instruction-hierarchies`, and `1.4-prompt-injection`). Day 4 is a break and has no content folder. Start with each section folder's `README.md`: it explains what the section assumes, what you should learn, and which pre-reading and reference resources will help. Then open the `*_instructions.md` file for the exercises. We recommend you open it in your IDE and view the markdown (right-click and select "Open Preview" in VS Code).

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
