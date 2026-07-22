
# Day 7 — Section 1: Adversary Matrices and Kill Chains

Today is mostly synthesis. Across the first six days you have practiced a wide range of attacks — prompt injection and agent hijacking, jailbreaks and model extraction, data poisoning and weight edits, adversarial vision examples, container escapes and rowhammer. On their own, each lab is a single technique. In the real world, an adversary stitches these techniques together, and a defender only survives if their controls cover the chain, not just the individual steps.

The tool for reasoning about this is the *adversary matrix*: columns are the high-level **tactics** an attacker must move through (roughly, "why are they doing this step?"), and rows under each column are concrete **techniques** ("how could they do it?"). MITRE ATT&CK popularised this format for enterprise IT; [MITRE ATLAS](https://atlas.mitre.org/matrices/ATLAS) adapts it for ML systems.

In this exercise you will (a) warm up by walking a real end-to-end kill chain against a published ATLAS system, and then (b) build your own matrix of the attacks you have seen in this bootcamp, applied to a frontier AI lab.

## Table of Contents

- [Content & Learning Objectives](#content--learning-objectives)
    - [Warm-up: Walking an ATLAS kill chain](#warm-up-walking-an-atlas-kill-chain)
- [Setup](#setup)
- [Warm-up: Walking an ATLAS kill chain](#warm-up-walking-an-atlas-kill-chain-1)
    - [Exercise 7.1.1: Trace a kill chain against an ATLAS system](#exercise-711-trace-a-kill-chain-against-an-atlas-system)

## Content & Learning Objectives

### Warm-up: Walking an ATLAS kill chain
Use MITRE ATLAS to trace a plausible attack end-to-end against an existing system. The goal is to build fluency with the matrix format before you build your own.

> **Learning Objectives**
> - Read an ATLAS-style matrix and locate candidate techniques for each tactic
> - Reason about the *preconditions* and *postconditions* of each step in a chain
> - Distinguish a technique that is patchable from one that is an inherent property of the system


## Setup

This is a prose/discussion exercise — there is no code to run. Create a file called `day7_answers.md` in the `7.1-atlas-killchain` directory and write your answers for Section 1 there (continue in the same file for Section 2). For the matrix itself, a Markdown table works fine; a spreadsheet or whiteboard photo is also acceptable — ask a TA if you are unsure.

Work in your pair. Plan to spend roughly 30 minutes on Section 1 and the remaining time on Section 2. Grab an instructor for the kill-chain review at the end of Section 2 before you move on.


## Warm-up: Walking an ATLAS kill chain

### Exercise 7.1.1: Trace a kill chain against an ATLAS system

> **Difficulty**: 🔴🔴⚪⚪⚪
> **Importance**: 🔵🔵🔵🔵⚪

Open the [ATLAS matrix](https://atlas.mitre.org/matrices/ATLAS) in your browser. Pick **one** target system from the list below (or agree on another with your pair):

- A hosted text-to-image service (e.g. a Stable-Diffusion-based SaaS) that accepts user prompts and returns images.
- An enterprise RAG chatbot that answers questions over a company's internal documents, with tool access to a ticketing system.
- A self-driving perception stack whose vision model is updated via an over-the-air pipeline.
- A content-moderation classifier used by a social network to auto-remove posts.

Now walk a plausible kill chain:

1. **Pick one technique per tactic column**, moving left-to-right. Skip a column only if you can justify why the attacker does not need it for *this* target.
2. For each technique you pick, write 1-2 sentences covering: (a) what the attacker does concretely, (b) what precondition it needs from the previous step, and (c) what it enables for the next step.
3. If the technique looks patchable (a bug, a misconfiguration) note the patch. If it is an inherent property of the system (e.g. "the model has to accept user prompts"), note that too — those are the ones governance has to reason about, not engineering.
4. End with an **impact** statement: what has the attacker achieved, and why is the victim harmed?

<details>
<summary>Hint: what if I can't find a technique for a column?</summary><blockquote>

ATLAS columns are not all mandatory. Many real chains skip, e.g., *Persistence* or *Lateral Movement* — a one-shot jailbreak does not need either. The exercise is to be deliberate about which columns you skip and why, not to force an entry into every column.
</blockquote></details>

<details>
<summary>Reference kill chain (RAG chatbot example)</summary><blockquote>

A worked example for the enterprise RAG chatbot target:

| Tactic | Technique | Note |
| --- | --- | --- |
| Reconnaissance | Search victim-owned websites (AML.T0003) | Attacker finds the company's public help-centre and learns it is indexed by the internal RAG. |
| Resource Development | Publish poisoned data (AML.T0019) | Attacker creates a plausible-looking public document with an indirect prompt-injection payload, hoping it is scraped into the index. |
| Initial Access | Evade ML Model (AML.T0015) via indirect prompt injection | The payload triggers when an employee asks a related question and the RAG retrieves the poisoned doc. |
| Execution | LLM Prompt Injection: Indirect (AML.T0051.001) | Injected instructions hijack the assistant. |
| Defense Evasion | Masquerading | Injected text imitates an internal policy memo to survive any "does this look suspicious?" filter. |
| Collection | Data from Information Repositories | The hijacked agent is told to query the ticketing tool and dump recent tickets. |
| Exfiltration | Exfiltration via Inference API | Stolen content is embedded in the agent's visible reply, or sent to an attacker URL via a tool call. |
| Impact | Harm: Organisational loss | Customer PII from tickets is leaked. |

The inherent-vs-patchable call here: the RAG *must* retrieve untrusted documents to be useful — that is not patchable. What *is* patchable is the tool-call egress policy and the prompt-injection-aware monitoring on the agent's outputs.
</blockquote></details>

<details>
<summary>Discussion: which of your steps were "patchable" vs. "inherent"?</summary><blockquote>

Governance work lives in the inherent-property column: you cannot ship the tickets when you cannot patch your way out. Notice how many of the techniques you listed were really just "the model did what it was designed to do, on inputs the designer did not anticipate." That is the class of risk a safety case has to address directly.
</blockquote></details>
