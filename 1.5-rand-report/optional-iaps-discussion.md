# Optional Day 1 supplement — RAND and IAPS

This optional supplement retains the broader multi-report discussion that
previously accompanied the RAND section. It is not part of the Day 1 core route.

This 85-minute discussion session guides you through two reports on AI infrastructure security. You'll work through specific sections of each document to understand threat frameworks, security implementations, and defensive approaches.

## Required Reading Materials

**Before the session, ensure you have access to:**

1. **RAND Corporation Report**: "Securing AI Model Weights" (RRA2849-1)
   - **Link**: https://www.rand.org/content/dam/rand/pubs/research_reports/RRA2800/RRA2849-1/RAND_RRA2849-1.pdf
   - **Alternative**: Search for "RAND RRA2849-1 AI model weights security"

2. **IAPS Research**: "Accelerating AI Data Center Security Research and Implementation"
   - **Link**: https://www.iaps.ai/research/accelerating-ai-data-center-security
   - **Note**: Full research findings and policy recommendations available on site

### Quick Reference Guide

**Key Sections for Exercises:**

**RAND Report:**
- Operational Capability Levels (OC1-OC5): Section 2
- Security Level Framework (SL1-SL5): Section 3 
- Protected Environments: Table 3-1
- Attack Vector Examples: Appendix B

**IAPS Research:**
- Three Critical Threat Areas: Main article sections 1-3
- Four Core Policy Recommendations: Conclusion section
- Side-Channel Attacks: Technical details in section 1
- Supply Chain Vulnerabilities: Section 2

---

## Discussion Structure

### Part 1: RAND Framework - Understanding the Threat Landscape (40 minutes)

#### Optional Exercise 1.5.S1: Operational Capability Classification

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

#### Optional Exercise 1.5.S2: Security Level Mapping

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

---

### Part 2: IAPS Data Center Security - Critical Attack Vectors (35 minutes)

#### Optional Exercise 1.5.S3: Attack Vector Prioritization

> **Difficulty**: 2/5
> **Importance**: 5/5

**Task**: Locate the "Three Critical Threat Areas" section in the IAPS research document.

**Reading Assignment**: Each group takes one threat area and presents to others:
- **Group A**: Side-Channel Attacks
- **Group B**: Hardware Supply Chain Vulnerabilities  
- **Group C**: Model Weight Exfiltration

**Questions for Each Group:**
1. What specific technical mechanism does IAPS describe for this attack?
2. Why are AI workloads particularly vulnerable compared to traditional systems?
3. What defensive measures does IAPS recommend?

**Cross-Group Analysis**: After presentations, discuss:
- Which attack vector is most cost-effective for adversaries?
- Which is hardest to detect once deployed?
- Which requires the most sophisticated adversary capabilities?

<details>
<summary><b>IAPS Attack Vector Analysis</b></summary>

**Side-Channel Attacks:**
- **Mechanism**: Measure electromagnetic/power emissions to extract secrets
- **AI Vulnerability**: Predictable computation patterns leak encryption keys/model parameters
- **Defense**: Shielded enclosures, power filtering, algorithmic countermeasures

**Supply Chain Vulnerabilities:**
- **Mechanism**: Hardware tampering during manufacturing creates persistent backdoors
- **AI Vulnerability**: Geographic concentration of manufacturing in potentially adversarial nations
- **Defense**: Trusted supplier diversification, component verification, secure sourcing

**Weight Exfiltration:**
- **Mechanism**: High-value terabyte-scale assets transferred through various channels
- **AI Vulnerability**: Model weights are immediately executable unlike traditional IP
- **Defense**: Network monitoring, data loss prevention, air-gapped environments

**IAPS Priority**: The research emphasizes supply chain risks due to "components manufactured in China present particular risks" while acknowledging other origins may also be vulnerable.

</details>

#### Optional Exercise 1.5.S4: IAPS Policy Framework Implementation

> **Difficulty**: 3/5
> **Importance**: 3/5

**Task**: Find the "Four Core Policy Recommendations" in the IAPS document.

**Implementation Scenario**: Your organization operates three data centers with 15,000 GPUs total. Using the IAPS recommendations, design an implementation plan:

1. **Security Standards**: How would you implement "AI data center-specific security framework with progressive maturity levels"?
2. **R&D Investment**: What specific defensive technologies would you prioritize for DARPA-style funding?
3. **Intelligence Sharing**: What incident reporting requirements would you establish?
4. **Supply Chain Decoupling**: How quickly could you shift away from potentially compromised suppliers?

**Discussion Questions:**
- Which IAPS recommendation would have the highest immediate impact?
- Which faces the greatest implementation challenges?
- How do the IAPS recommendations align with or differ from RAND's SL framework?

<details>
<summary><b>IAPS Implementation Strategy</b></summary>

**1. Security Standards (6-12 months):**
- Map current practices to maturity model (similar to RAND SL1-SL5)
- Establish certification requirements for government procurement
- Create vendor security credential demonstration programs

**2. R&D Investment (Ongoing):**
- **Side-channel hardening**: Hardware-level countermeasures for AI accelerators
- **Supply chain security**: Component verification and trusted manufacturing
- **Exfiltration prevention**: AI-enhanced monitoring and data loss prevention

**3. Intelligence Sharing (3-6 months):**
- Mandatory incident reporting for model weight compromise
- Declassified threat intelligence sharing with private sector
- Industry threat sharing communities (currently "most breaches go unreported")

**4. Supply Chain Decoupling (2-5 years):**
- Map critical component dependencies by country of origin
- Establish trusted supplier networks in allied nations
- Create redundant sourcing for security-critical components

</details>

---

## Synthesis Exercise: Cross-Report Integration (10 minutes)

### Final Group Discussion

**Integration Questions:**
1. How do the RAND Security Levels (SL1-SL5) map to the IAPS policy recommendations?
2. Where do the two frameworks contradict or provide conflicting guidance?
3. What's missing from both reports that your organization would need to know?

**Priority Ranking Exercise**: Based on your analysis of both documents, rank these implementation priorities:

| Priority | Recommendation | Source | Timeline | Rationale |
|----------|---------------|--------|----------|-----------|
| 1 | ? | ? | ? | ? |
| 2 | ? | ? | ? | ? |
| 3 | ? | ? | ? | ? |
| 4 | ? | ? | ? | ? |
| 5 | ? | ? | ? | ? |

---

## Key Takeaways from Document Analysis

### What We Learned from RAND:
- [ ] Threat actor classification (OC1-OC5) provides structured approach to defensive planning
- [ ] Security levels (SL1-SL5) offer progressive protection matching threat environment
- [ ] Model weight value proposition creates unique security requirements vs. traditional IP

### What We Learned from IAPS:
- [ ] Three critical attack vectors require immediate policy attention
- [ ] Four-point policy framework provides government-industry coordination structure  
- [ ] Current data center security practices insufficient for AI-specific threats

### Implementation Priorities for Your Organization:
- [ ] **Immediate (0-6 months)**: _[Fill based on discussion]_
- [ ] **Short-term (6-18 months)**: _[Fill based on discussion]_
- [ ] **Medium-term (1-3 years)**: _[Fill based on discussion]_
- [ ] **Long-term (3-5 years)**: _[Fill based on discussion]_

---

## Further Exploration

### For Deeper Technical Understanding:
- **RAND Report Appendices**: Detailed attack vector analysis and cost-benefit calculations
  - Full report: https://www.rand.org/content/dam/rand/pubs/research_reports/RRA2800/RRA2849-1/RAND_RRA2849-1.pdf
- **IAPS Technical Sections**: Specific countermeasures for each attack vector
  - Research page: https://www.iaps.ai/research/accelerating-ai-data-center-security
### For Policy Implementation:
- **NIST AI Risk Management Framework**: https://www.nist.gov/itl/ai-risk-management-framework
- **NSA Commercial Solutions for Classified**: https://www.nsa.gov/resources/everyone/csfc/
- **Industry Working Groups**: 
  - MLSecOps: https://mlsecops.com/
  - AI Village: https://aivillage.org/
  - OWASP ML Security: https://owasp.org/www-project-machine-learning-security-top-10/

### Additional Context Documents:
- **IAPS Organization**: https://www.iaps.ai/
- **OpenTitan Hardware Root of Trust**: https://opentitan.org/

---

*This discussion format ensures direct engagement with the source documents while building practical understanding through guided analysis and group interaction.*
