# %%
"""
# Day 2 — Section 5: Bonus: From Monitor Scores to System Safety

Control monitors are only one component in a broader system. Think through how different properties of the system influence the overall safety of a deployment.

<!-- toc -->

## Content & Learning Objectives

### Bonus: Safety Calculation

> **Learning Objectives**
> - Understand the parameters that determine system-level safety beyond monitor quality
> - Explore how attack frequency, latency, audit rate, and attack count interact
> - Use Monte Carlo simulation to estimate safety probabilities
> - Understand the simplifying assumptions behind safety estimates
"""

# %%
"""
## Bonus: From Monitor Scores to System Safety

In sections 3–4 you built monitors and protocols that score and intervene on individual submissions. But actual system-level safety depends on many more parameters and assumptions.

- **Attack frequency**: how often does the attacker choose to strike? Attacking rarely avoids getting caught by simple up-front auditing and evaluation, but gives fewer chances to cause harm.
- **Attack latency**: how long between a malicious action and irreversible harm? Longer latency gives auditors more time to catch and revert if attacks are reversible (e.g., delay before deploy to production).
- **Audit rate**: what fraction of tasks can humans review? Bounded by team size and review speed.
- **Attacks needed for harm**: does a single successful backdoor cause catastrophe, or does the attacker need to chain multiple exploits?

One way to assess system safety given all these interacting parameters is **Monte Carlo simulation**: sample many possible deployment trajectories (with randomized attack timing, monitor scores, and audit decisions) and estimate the probability that at least one attack succeeds. The [Async Control Macro Game](https://ukgovernmentbeis.github.io/async-control/macrogame.html) is an interactive simulator that does exactly this. It models **asynchronous monitoring**: a setting where actions are reviewed *after* they occur (rather than blocked in real time), and the defender's goal is to catch attacks before their latency window closes.

### Exercise 2.5.1: Explore the Safety Simulator

> **Difficulty**: 1/5
> **Importance**: 3/5

Open the [Async Control Macro Game](https://ukgovernmentbeis.github.io/async-control/macrogame.html) and experiment with the parameters.

Try varying monitor quality, attacks needed for harm, and the attack probability the attacker chooses. How does each influence overall safety?

Tip: reduce the number of simulation runs (e.g. to 100) for faster iteration.

<details><summary><b>Key takeaways</b></summary>

- **System design choices may matter more than monitor quality.** Requiring multiple successful attacks before harm occurs, or introducing delays between attack execution and effect, often improves safety more than improving monitor accuracy.

- **Multiple attacks needed for harm is a strong defensive advantage.** Going from 1 required attack to 2 jumps safety from 0% to ~91%; requiring 3 reaches ~99%. This motivates building multiple layers of security so that no single action is catastrophic.

- **Attack latency is the strongest single lever.** Even a 1-hour delay before harm occurs raises safety from 0% to ~96% (single attack required). This is why **async monitoring** (reviewing actions after the fact) is only viable when there's a meaningful detection window. Synchronous protocols block suspicious actions in real time and achieve higher safety, but cost usefulness and add latency.


- **These estimates rest on simplifying assumptions.** The simulations assume human auditors catch 100% of attacks they review, that attack probability is constant across episodes, and that detection of each attack is independent. If these assumptions are violated (e.g., agents can coordinate attacks across episodes, or auditors are fallible), safety estimates become much less reliable. The most dangerous uncertainties are not in the parameters you can measure, but in the modeling assumptions you can't easily test.

</details>
"""




# %%
"""
## Summary

Today covered three themes: the **attack surface for prompt injection**, **AI control through monitoring**, and the **damage potential of a compromised coding agent**.

**Injection Attack Surface**

- **Indirect prompt injection** is the fundamental vulnerability of LLM applications: there is no equivalent of parameterized queries
- The attack surface spans every channel carrying untrusted input: user messages, RAG-retrieved documents, and MCP tool responses. All of them exploit the same inability to separate data from instructions.

**AI Control**

- AI control is the problem of how to safely deploy AI systems that might be **intentionally subverting** your safety measures. Unlike alignment (making the AI want the right things), control assumes the AI *may* be misaligned and focuses on limiting what it can get away with
- LLM-based monitors are a basic building block for AI control, since direct human oversight won't scale. The capability gap between monitor and monitored system matters: trusted monitoring relies on weaker but better-understood models to supervise stronger, less trusted ones
- **Chain-of-thought monitoring** has no analogue in human-insider-threat scenarios, but its value depends on whether CoT stays faithful as models change
- Control protocols (trusted monitoring, defer-to-trusted, resampling) formalize the **safety–usefulness tradeoff**: more aggressive auditing catches more attacks but blocks more legitimate work

**Coding Agents**
- Even a coding agent with "just" file write permissions has a **surprisingly wide attack surface**: executable code injection, credential access (env vars, config files, cloud metadata), lateral movement (git hooks, CI pipelines), persistence mechanisms, and supply chain manipulation
- **Affordances and autonomy**, not the injection vector, determine the blast radius of a successful attack
- Thinking systematically about mitigations (sandboxing, least-privilege, human-in-the-loop, monitoring) is essential because no single defense is sufficient

### Further Reading

**Application vulnerabilities**

- [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/): A community-driven security resource identifying the most critical vulnerabilities in LLM applications, covering risks from prompt injection to insecure output handling.
- [PromptArmor, "Data Exfiltration from Slack AI via Indirect Prompt Injection"](https://promptarmor.substack.com/p/data-exfiltration-from-slack-ai-via): Demonstrates how attackers can steal confidential data from private Slack channels by posting malicious instructions in public channels that manipulate Slack's AI to exfiltrate sensitive information.
- [Radware, "ZombieAgent"](https://www.radware.com/blog/threat-intelligence/zombieagent/): A zero-click indirect prompt injection vulnerability targeting OpenAI's Deep Research agent, enabling attackers to implant malicious rules into an AI agent's long-term memory via hidden directives in emails or documents, with all exfiltration occurring silently within OpenAI's cloud infrastructure.
- [LayerX, "ChatGPT Tainted Memories"](https://layerxsecurity.com/blog/layerx-identifies-vulnerability-in-new-chatgpt-atlas-browser/): Discovers a vulnerability in OpenAI's ChatGPT Atlas browser allowing CSRF attacks to inject malicious instructions into a user's ChatGPT memory, exploiting the browser's default logged-in state and lack of anti-phishing protections.
- [Trail of Bits, "Weaponizing Image Scaling Against Production AI Systems"](https://blog.trailofbits.com/2025/08/21/weaponizing-image-scaling-against-production-ai-systems): Describes a vulnerability where crafted images appear harmless at full resolution but contain hidden prompt injections when scaled down by AI systems, demonstrated across Google Gemini, Vertex AI, and others.

**AI Control**
- [ControlArena documentation](https://ukgovernmentbeis.github.io/control-arena/)
- [AI Control Reading List](https://blog.redwoodresearch.org/p/guide): A curated guide to Redwood Research's work on AI control. Organizes their publications into a quick introduction and an extensive guide covering threat models, control techniques, and safety cases.

- [Ctrl-Z: Controlling AI Agents via Resampling](https://blog.redwoodresearch.org/p/ctrl-z-controlling-ai-agents-via): Introduces "resample protocols" to prevent malicious AI agents from causing harm in multi-step environments. Evaluates safety measures against adversarial agents attempting to execute dangerous code.

- [An Overview of Control Measures](https://blog.redwoodresearch.org/p/an-overview-of-control-measures): Outlines strategies for ensuring AI systems remain controllable even if they attempt to circumvent safeguards. Categorizes measures into software infrastructure, AI/human components, and security improvements.

- [Prioritizing Threats for AI Control](https://blog.redwoodresearch.org/p/prioritizing-threats-for-ai-control): A framework for categorizing and prioritizing threats from misaligned AI. Argues that preventing "rogue internal deployments" (AIs running unmonitored with escalated permissions) should be the top priority.

- [An Overview of Areas of Control Work](https://blog.redwoodresearch.org/p/an-overview-of-areas-of-control-work): Maps the major research and implementation areas for AI control: evaluation settings, model capabilities, countermeasures, and infrastructure prototyping.

- [AI Catastrophes and Rogue Deployments](https://blog.redwoodresearch.org/p/ai-catastrophes-and-rogue-deployments): Introduces the concept of "rogue deployments" (running AI models without safety measures) as a framework for classifying AI catastrophes. Argues that launching an unauthorized deployment is far simpler than directly causing a catastrophe.

- [Recent Redwood Research Project Proposals](https://blog.redwoodresearch.org/p/recent-redwood-research-project-proposals): A collection of empirical AI safety and security research proposals spanning AI control, training-time alignment, and model interpretability.

- [What's Worse, Spies or Schemers?](https://blog.redwoodresearch.org/p/whats-worse-spies-or-schemers): Compares two insider threat problems: human employees who might misuse AI systems ("spies") vs. AI models that might act deceptively ("schemers"). Concludes schemers pose greater risk if they occur, though spies are more likely to emerge first.

- [Measuring No-CoT Math Time Horizon](https://blog.redwoodresearch.org/p/measuring-no-cot-math-time-horizon): Relevant for chain-of-thought monitoring: Measures how well AI models solve math problems in a single forward pass without chain-of-thought reasoning.

**MCP protocol**
- [What is Model Context Protocol (MCP)?](https://modelcontextprotocol.io/docs/getting-started/intro): An introduction to MCP, the emerging standard for giving LLMs structured access to tools and data.
- [MCPTox: A Benchmark for Tool Poisoning Attack on Real-World MCP Servers](https://arxiv.org/abs/2508.14925): The first systematic benchmark for evaluating LLM agent resistance to tool poisoning attacks on MCP servers, finding attack success rates exceeding 70% across 20 prominent LLM agents when malicious instructions are embedded in tool metadata.
"""
