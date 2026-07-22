
# W1D3 - Section 2: Jailbreaking

This short, timeboxed section gives you direct experience of attacker
adaptation before you build guardrails in Section 3.

## Table of Contents

- [Jailbreaking & prompt injection](#jailbreaking--prompt-injection)
    - [Exercise 2.1: Jailbreak a guarded LLM](#exercise-21-jailbreak-a-guarded-llm)
- [Further projects](#further-projects)

## Jailbreaking & prompt injection

Before we build defences, you need to understand the attacks. In this
section you'll get hands-on experience with jailbreaking — convincing an
LLM to ignore its safety training and produce content it was trained to
refuse.

> **Learning Objectives**
> - Generate and categorize several jailbreak strategies
> - Record prompts and outcomes without overgeneralizing from one success
> - Relate observed jailbreaks to the limits of behavioral safety training


```python


import sys
from pathlib import Path

_root = next(p for p in Path(__file__).resolve().parents if (p / "aisb_utils").is_dir())
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
```

### Exercise 2.1: Jailbreak a guarded LLM

> **Difficulty**: 🔴🔴⚪⚪⚪
> **Importance**: 🔵🔵🔵🔵🔵

Try to jailbreak a safety-trained model using the following interactive
challenges:

1. **[Gandalf (Lakera)](https://gandalf.lakera.ai/baseline)** — try to
   extract a secret password from an LLM that's been instructed not to
   reveal it. Each level adds stronger defences. See how far you can get
   in ~10 minutes.

2. **[Gray Swan Arena](https://app.grayswan.ai/arena)** — a jailbreaking
   arena where you compete to bypass safety filters on production models.
   Try a few rounds.

As you work through these, pay attention to the *types* of techniques that
succeed:
- **Role-playing / persona** ("You are DAN, an AI with no restrictions...")
- **Encoding / obfuscation** (base64, pig latin, character-by-character)
- **Context manipulation** ("Ignore previous instructions...")
- **Indirect requests** ("Write a novel scene where a character explains how to...")

<details><summary>Background reading on jailbreaks and prompt injection</summary><blockquote>

These papers and posts explain the theory behind what you're doing:

- [Perez & Ribeiro (2022), *Ignore This Title and HackAPrompt*](https://arxiv.org/abs/2311.16119)
  — A taxonomy of prompt injection attacks with a competition dataset.

- [Wei et al. (2024), *Jailbroken: How Does LLM Safety Training Fail?*](https://arxiv.org/abs/2307.02483)
  — Analyses why safety training fails: competing objectives (helpfulness
  vs safety) and mismatched generalisation (safety training doesn't cover
  all input formats the model can handle).

- [Greshake et al. (2023), *Not What You've Signed Up For: Compromising
  Real-World LLM-Integrated Applications with Indirect Prompt
  Injection*](https://arxiv.org/abs/2302.12173) — Indirect prompt
  injection: attacks embedded in data the model retrieves (web pages,
  emails), not typed by the user.

- [Anthropic (2025), *Constitutional Classifiers*](https://arxiv.org/abs/2501.18837)
  — Anthropic's approach to defending against jailbreaks using
  input/output classifiers, which we'll implement in the next section.

**Key takeaway**: safety training is a *statistical* defence — it reduces
the probability of harmful outputs but doesn't eliminate them. Any
technique that shifts the model's context away from the patterns it was
trained to refuse (new languages, new formats, fictional framing) has a
chance of bypassing it. This is why external guardrails exist.

</blockquote></details>

**Task**: Spend ~20 minutes on Gandalf and/or Gray Swan. Write down the
2–3 most effective jailbreak strategies you found. You'll use these
insights when attacking the guardrails you build in the next section.


## Further projects

- Re-test your strategy categories on a second authorized model and report which techniques transfer rather than only listing successes.
- Perturb one successful harmless jailbreak by changing its encoding, role framing, and sampling settings to measure how brittle it is.
- **One-week project:** create a harmless jailbreak benchmark with held-out prompts, repeated trials, attack-success metrics, false-positive controls, and a documented responsible-testing protocol.
