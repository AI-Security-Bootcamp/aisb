# %%
"""
# Day 7 — Section 1: RAND Report — Securing AI Model Weights

This discussion uses RAND's security-level and adversary-capability frameworks to
reason about protecting frontier-model weights. The goal is to connect security
controls to explicit assets, environments, and attacker capabilities.

<!-- toc -->

## Content & Learning Objectives

### RAND security levels and operational capabilities

> **Learning Objectives**
> - Distinguish RAND operational-capability levels (OC1–OC5)
> - Apply security levels (SL1–SL5) to concrete model-weight environments
> - Justify a control level using an explicit adversary and protected asset
"""

# %%
"""
## Setup

This is a prose/discussion section; there is no code to run. Create
`day7_answers.md` in `7.1-rand-report/` and record your classifications,
assumptions, and evidence there.

Read the assigned sections of RAND's
[*Securing AI Model Weights*](https://www.rand.org/content/dam/rand/pubs/research_reports/RRA2800/RRA2849-1/RAND_RRA2849-1.pdf)
before beginning.
"""

# %%
"""
## RAND framework: understanding the threat landscape

### Exercise 7.1.1: Operational Capability Classification

> **Difficulty**: 2/5
> **Importance**: 5/5

**Task**: Open the RAND report and locate the Operational Capability (OC) classifications section.

**Questions for Group Discussion:**
1. According to RAND, what distinguishes OC4 from OC5 threat actors?
2. Which OC level can realistically compromise GPU firmware supply chains?
3. How does RAND estimate the cost difference between OC2 and OC5 operations?

**Group Exercise**: Read the attack vector examples in the RAND report and classify each scenario:

- **Scenario A**: Social engineering attack against ML researchers to steal cloud credentials
- **Scenario B**: Custom hardware implant placed during chip manufacturing
- **Scenario C**: Insider threat using legitimate access to copy model checkpoints
- **Scenario D**: Advanced persistent threat with multiple zero-day exploits

<details>
<summary><b>Answer Key (RAND Classifications)</b></summary>

- **Scenario A**: **OC2-OC3** - Requires moderate social engineering capability but limited technical resources
- **Scenario B**: **OC5** - Supply chain hardware compromise requires nation-state level access and resources
- **Scenario C**: **OC1-OC3** - Insider threats can be executed with minimal sophistication but require access
- **Scenario D**: **OC4-OC5** - Zero-day development and sophisticated persistence requires significant resources

**Key RAND Insight**: The report emphasizes that software supply chain attacks are "among the cheapest and most scalable attacks" while hardware attacks are "feasible for well-resourced nation-state attackers at OC5."

</details>

### Exercise 7.1.2: Security Level Mapping

> **Difficulty**: 3/5
> **Importance**: 4/5

**Task**: Find the Security Level (SL1-SL5) framework in the RAND report.

**Analysis Questions:**
1. What are the five "protected environments" that RAND identifies?
2. At what model capability threshold does RAND suggest SL4 becomes necessary?
3. Which security level requires "classified facility-level physical security"?

**Mapping Exercise**: Based on the RAND framework, assign appropriate security levels:

| Environment | Current Practice | RAND SL Recommendation | Justification |
|-------------|-----------------|------------------------|---------------|
| Research Lab (Pre-training) | Standard enterprise security | ? | ? |
| Production Training (Frontier Model) | Enhanced cloud security | ? | ? |
| Public API Deployment | SOC 2 compliance | ? | ? |
| Internal Model Testing | Basic access controls | ? | ? |
| On-Premises Inference | Hardware security modules | ? | ? |

<details>
<summary><b>RAND Security Level Analysis</b></summary>

| Environment | RAND SL Recommendation | Justification from Report |
|-------------|------------------------|---------------------------|
| Research Lab (Pre-training) | **SL2-SL3** | Pre-publication research requires enhanced protection but not maximum security |
| Production Training (Frontier Model) | **SL4-SL5** | Highest value target requiring "classified facility-level physical security" |
| Public API Deployment | **SL3** | Deployed models need high protection but operational considerations limit SL4+ |
| Internal Model Testing | **SL3** | Internal deployment with controlled access to model capabilities |
| On-Premises Inference | **SL2-SL3** | Depends on model capability and deployment context |

**RAND Key Quote**: "SL4 can plausibly be reached incrementally, SL5 can likely only be reached by a radical reduction in the hardware and software stack that is trusted."

</details>

## Summary

- Operational-capability levels describe what an adversary can realistically do.
- Security levels describe progressively stronger environments for protecting model weights.
- A useful recommendation names the protected asset, adversary capability, assumptions, and residual risk.

### Further Reading

- RAND, [*Securing AI Model Weights*](https://www.rand.org/content/dam/rand/pubs/research_reports/RRA2800/RRA2849-1/RAND_RRA2849-1.pdf)
- Optional supporting discussion: [IAPS and SL5 material](optional-iaps-sl5-discussion.md)
"""
