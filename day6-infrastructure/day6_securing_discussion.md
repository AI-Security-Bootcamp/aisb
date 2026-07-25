# Day 6 — Section 1: Securing Model Weights and AI Data Centers

This 2-hour discussion session guides you through three foundational reports on AI infrastructure security. You'll work through specific sections of each document to understand threat frameworks, security implementations, and novel defensive approaches.

## Required Reading Materials

**Before the session, ensure you have access to:**

1. **RAND Corporation Report**: "Securing AI Model Weights" (RRA2849-1)
   - **Link**: https://www.rand.org/content/dam/rand/pubs/research_reports/RRA2800/RRA2849-1/RAND_RRA2849-1.pdf
   - **Alternative**: Search for "RAND RRA2849-1 AI model weights security"

2. **IAPS Research**: "Accelerating AI Data Center Security Research and Implementation"
   - **Link**: https://www.iaps.ai/research/accelerating-ai-data-center-security
   - **Note**: Full research findings and policy recommendations available on site

3. **SL5 Task Force**: "Novel Recommendations" (November 2025) 
   - **Local File**: `./SL5_NOVEL-RECOMMENDATIONS.pdf` (in this directory)
   - **Note**: Preliminary draft with five security domain memos

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

**SL5 Document:**
- Supply Chain Security: Pages 8-17
- Network Security: Pages 18-31  
- Machine Security: Pages 32-40
- Physical Security: Pages 41-50
- Personnel Security: Pages 51-60

---

## Discussion Structure

### Part 1: RAND Framework - Understanding the Threat Landscape (40 minutes)

#### Exercise 6.1.1: Operational Capability Classification

> **Difficulty**: 🔴🔴⚪⚪⚪  
> **Importance**: 🔵🔵🔵🔵🔵

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

#### Exercise 6.1.2: Security Level Mapping

> **Difficulty**: 🔴🔴🔴⚪⚪  
> **Importance**: 🔵🔵🔵🔵⚪

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

#### Exercise 6.1.3: Attack Vector Prioritization

> **Difficulty**: 🔴🔴⚪⚪⚪  
> **Importance**: 🔵🔵🔵🔵🔵

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

#### Exercise 6.1.4: IAPS Policy Framework Implementation

> **Difficulty**: 🔴🔴🔴⚪⚪  
> **Importance**: 🔵🔵🔵⚪⚪

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

### Part 3: SL5 Novel Recommendations - Next-Generation Controls (35 minutes)

#### Exercise 6.1.5: Five-Domain Architecture Analysis

> **Difficulty**: 🔴🔴🔴🔴⚪  
> **Importance**: 🔵🔵🔵🔵🔵

**Task**: Open the SL5 document and locate the "Top 5 Recommendations" for each security domain.

**Group Assignment**: Each group analyzes one SL5 domain and identifies the most novel recommendation:

- **Group A**: Supply Chain Security - Focus on "architectural isolation and progressive access restriction"
- **Group B**: Network Security - Focus on "AI-enhanced cross-domain solutions"
- **Group C**: Machine Security - Focus on "tamper-proof enclosures that envelop entire sensitive scopes"
- **Group D**: Physical Security - Focus on "two-person integrity for all maintenance"
- **Group E**: Personnel Security - Focus on "industry-optimized sensitivity levels"

**Questions for Each Group:**
1. How does this SL5 recommendation go beyond traditional security approaches?
2. What specific technical implementation does the document suggest?
3. What are the main cost-benefit trade-offs identified?

<details>
<summary><b>SL5 Novel Recommendations Summary</b></summary>

**Supply Chain - Architectural Isolation:**
- **Innovation**: "Progressive access restriction" with systematic reduction of privileged access ("seat management")
- **Implementation**: Weight-accessing systems run minimal, formally verified components only
- **Trade-off**: Reduced functionality vs. dramatically smaller attack surface

**Network - AI-Enhanced Cross-Domain:**
- **Innovation**: AI monitors analyze code/commands crossing security boundaries
- **Implementation**: Weaker AI models monitor stronger AI outputs for suspicious patterns
- **Trade-off**: Compute overhead vs. enabling AI research workflows securely

**Machine - Tamper-Evident Enclosures:**
- **Innovation**: Physical security extended to "entire sensitive scopes" (whole racks/rooms)
- **Implementation**: Hardware detection covering complete compute environments
- **Trade-off**: Installation complexity vs. comprehensive physical protection

**Physical - Two-Person Integrity:**
- **Innovation**: All maintenance requires two authorized individuals with body-worn cameras
- **Implementation**: Continuous recording with obstruction alarms during facility access
- **Trade-off**: Operational overhead vs. insider threat elimination

**Personnel - Industry Clearances:**
- **Innovation**: Private sector adaptation of government clearance systems
- **Implementation**: "Industry-optimized sensitivity levels" with continuous behavioral monitoring
- **Trade-off**: Talent pipeline constraints vs. insider threat mitigation

</details>

#### Exercise 6.1.6: SL5 Implementation Feasibility Assessment

> **Difficulty**: 🔴🔴🔴🔴🔵  
> **Importance**: 🔵🔵🔵🔵⚪

**Task**: Find the SL5 document's discussion of "3-6 month feasibility assessment" timeline.

**Real-World Scenario**: A major AI lab with the following characteristics wants to pursue SL5:
- 50,000+ GPUs across 5 geographic locations
- 1,000+ employee research organization
- Active training of 5+ frontier models simultaneously
- Commercial API serving 100M+ requests daily
- University research partnerships in 15+ countries

**Feasibility Questions** (refer to specific SL5 document sections):
1. **Supply Chain**: How many software dependencies would need to be eliminated based on SL5 "minimal trusted software stack" requirements?
2. **Network**: Is 100+ Tbps distributed training compatible with SL5 "network encryptor" bandwidth limitations?
3. **Personnel**: How many employees would require "industry-optimized clearances" and what's the talent pipeline impact?
4. **Physical**: What percentage of existing facilities could support "tamper-evident enclosures" without major reconstruction?
5. **Machine**: When will "hardware root of trust" be available in commercial AI accelerators?

**Group Decision**: Based on your analysis, vote on implementation timeline:
- **18 months**: Aggressive implementation with significant operational disruption
- **3-5 years**: Phased implementation aligned with hardware refresh cycles  
- **5+ years**: Full implementation only with next-generation purpose-built facilities
- **Not feasible**: SL5 requirements incompatible with current AI development needs

<details>
<summary><b>SL5 Feasibility Reality Check</b></summary>

**Critical Implementation Challenges:**

**Supply Chain Reality:**
- Current ML stacks have 200+ dependencies per SL5 analysis
- "Radical reduction" requires rewriting most AI development tools
- Timeline depends on availability of formally verified alternatives

**Network Bandwidth Gap:**
- Current encryptors: 100-400 Gbps maximum throughput
- Future distributed training: 100+ Tbps requirement  
- Solution requires "multiplexing currently available encryptors" with unproven scalability

**Personnel Clearance Bottleneck:**
- AI talent market extremely competitive globally
- Clearance requirements could reduce available talent pool by 70-90%
- "Industry-optimized" clearances don't yet exist

**Physical Infrastructure:**
- Most existing data centers not designed for "tamper-evident enclosures"
- Retrofit costs potentially exceed new construction
- Geographic distribution conflicts with physical security requirements

**Hardware Readiness:**
- "Hardware root of trust" in AI accelerators still developmental
- Current GPU/TPU architectures lack security-first design
- 3-5 year timeline for security-enhanced AI chips

**SL5 Assessment**: Most organizations would realistically require 5+ years for full implementation, with immediate focus on highest-impact, lowest-cost controls.

</details>

---

## Synthesis Exercise: Cross-Report Integration (10 minutes)

### Final Group Discussion

**Integration Questions:**
1. How do the RAND Security Levels (SL1-SL5) map to the IAPS policy recommendations?
2. Which SL5 novel recommendations directly address the IAPS "three critical threat areas"?
3. Where do the three frameworks contradict or provide conflicting guidance?
4. What's missing from all three reports that your organization would need to know?

**Priority Ranking Exercise**: Based on your analysis of all three documents, rank these implementation priorities:

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

### What We Learned from SL5:
- [ ] Five security domains require coordinated novel approaches
- [ ] "Radical reduction" in trusted components necessary for highest security levels
- [ ] 3-6 month feasibility assessment timeline enables practical implementation planning

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
- **SL5 Domain Memos**: Full technical specifications for each security domain
  - Local file: `./SL5_NOVEL-RECOMMENDATIONS.pdf`

### For Policy Implementation:
- **NIST AI Risk Management Framework**: https://www.nist.gov/itl/ai-risk-management-framework
- **NSA Commercial Solutions for Classified**: https://www.nsa.gov/resources/everyone/csfc/
- **Industry Working Groups**: 
  - MLSecOps: https://mlsecops.com/
  - AI Village: https://aivillage.org/
  - OWASP ML Security: https://owasp.org/www-project-machine-learning-security-top-10/

### Additional Context Documents:
- **IAPS Organization**: https://www.iaps.ai/
- **SL5 Task Force Information**: https://sl5.org/
- **OpenTitan Hardware Root of Trust**: https://opentitan.org/

---

*This discussion format ensures direct engagement with the source documents while building practical understanding through guided analysis and group interaction.*
