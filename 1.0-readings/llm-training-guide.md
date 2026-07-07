# A 30-Minute Guide to How LLMs Are Trained

*Half an hour to understand the full pipeline: from "GPUs humming in a data center" to "ChatGPT answers your question."*

---

## The three resources

1. **[3Blue1Brown — Large Language Models explained briefly](https://www.3blue1brown.com/lessons/mini-llm/)** — lightweight mental model. Read first.
2. **[OLMo 2 Furious (arxiv 2501.00656)](https://arxiv.org/pdf/2501.00656)** — a real, fully-open recipe from Ai2. We jump to specific sections with specific questions.
3. **[Karpathy — Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=zjkBMFhNj_g)** — optional 3.5-hour deep dive.

*Why OLMo 2 and not GPT-4? Because every stage and dataset is published. Closed labs follow roughly the same shape but don't share the recipe.*

---

## Step 0 — Build the mental model (5 min)

**Read the 3b1b page end-to-end** (or watch the 8-min video embedded). Then answer:

<details>
<summary>What is an LLM, mathematically?</summary>

A function that takes text in and outputs a probability distribution over the next token.
</details>

<details>
<summary>What does "training" actually change?</summary>

The hundreds of billions of parameters (weights) inside the model. Start random, nudge toward better predictions.
</details>

<details>
<summary>Difference between pretraining and RLHF?</summary>

Pretraining = learn to predict the next word on all internet text. RLHF = adjust the model so outputs match what humans prefer.
</details>

<details>
<summary>Why GPUs and not CPUs?</summary>

Training is mostly identical matrix multiplications done in parallel across every token. GPUs do thousands at once; CPUs do them sequentially.
</details>

---

## Step 1 — You have GPUs. Why? (3 min)

**Open OLMo 2 §6.1 "Clusters" (p. 31) and glance at Table 3 on p. 6.**

<details>
<summary>How many GPUs does Ai2 use?</summary>

Thousands of H100s across multiple clusters (Augusta, Jupiter). Not a laptop job.
</details>

<details>
<summary>Tokens per training step for the 32B model? (batch size × seq length)</summary>

2048 × 4096 ≈ 8.4 million tokens per step, for millions of steps.
</details>

<details>
<summary>What goes wrong at this scale that wouldn't on a laptop? (§6.3)</summary>

Hardware failures, network hangs, silent data corruption. Jobs crash constantly; dedicated infra exists just to auto-restart them.
</details>

**Takeaway:** GPUs aren't a detail — they're the constraint. Architecture choices are partly bets about what keeps the GPUs fed and stable.

---

## Step 2 — You have data. From where? (4 min)

**Read OLMo 2 §2.4 "Base Model Data" (p. 7) and Table 4.**

<details>
<summary>Total tokens? Web vs curated ratio?</summary>

~3.9T tokens. Over 95% is filtered web text (DCLM). The rest is code (StarCoder), papers (peS2o, arXiv), math (OpenWebMath), Wikipedia.
</details>

<details>
<summary>How did 21T raw bytes become 3.7T tokens? (§2.4.1)</summary>

Quality classifiers rank every page; they keep the top slice and throw the rest away. "The internet" as training data is mostly a filtering problem.
</details>

<details>
<summary>Why include non-web sources separately?</summary>

Each injects a capability: code for programming, papers for technical reasoning, OpenWebMath for math, Wikipedia for factual reliability.
</details>

<details>
<summary>What are the repeated n-gram strings in §3.1, and why do they matter?</summary>

Encoded binary junk, long number sequences, padding artifacts. A single bad document can cause a gradient spike that damages or kills a multi-million-dollar training run.
</details>

**Takeaway:** "The internet" is raw material. The recipe is which parts you keep.

---

## Step 3 — Tokens and embeddings (3 min)

**Read OLMo 2 §2.2 "Tokenizer" (p. 5) and glance at Table 1.**

<details>
<summary>What is a token?</summary>

A sub-word chunk, usually 3–4 characters. OLMo 2 uses cl100k, borrowed from GPT-3.5/GPT-4.
</details>

<details>
<summary>Why ~100k vocab size?</summary>

Trade-off: smaller vocab = longer sequences and more compute; larger vocab = a huge embedding matrix to learn. 100k is the current sweet spot.
</details>

<details>
<summary>What's RoPE replacing absolute positional embeddings?</summary>

Rotary Position Embedding — encodes position by rotating the embedding vectors themselves. Current standard across all frontier models.
</details>

<details>
<summary>What actually changed from 2017 to 2024?</summary>

Not attention. The stabilization tricks — norm placement, QK-norm, init schemes, z-loss — are what let us train huge models without them exploding.
</details>

**Takeaway:** Tokens are atoms, embeddings give them meaning. Everything downstream is just massaging these vectors.

---

## Step 4 — Pretraining (5 min)

**Read OLMo 2 §2.3 "Base Model Training Recipe" (p. 6) + Table 3. Skim §3 intro (pp. 10–12).**

<details>
<summary>Learning objective in one sentence?</summary>

Predict the next token given the previous tokens. Same as 3b1b said.
</details>

<details>
<summary>Why a learning rate schedule — warmup and cosine decay?</summary>

Warmup avoids early instability when gradients are wild. Decay lets the model settle into a good minimum at the end. Without both, training blows up or fails to converge.
</details>

<details>
<summary>What is a loss spike? (Figure 2, p. 12)</summary>

A sudden jump in training loss. Can permanently damage the model, wasting weeks of compute. Most of §3 is techniques to prevent these.
</details>

<details>
<summary>What kind of problem is frontier pretraining, really?</summary>

As much a reliability engineering problem as an ML problem. Every fix in §3 is "stop the training from blowing up," not "make the model smarter."
</details>

**Takeaway:** Pretraining gets you a statistically fluent model. It knows grammar, facts, reasoning patterns. It does *not* know how to be helpful yet.

---

## Step 5 — Mid-training (the phase nobody mentions) (4 min)

**Read OLMo 2 §4 intro and §4.2 (pp. 18–20) + Table 9.**

<details>
<summary>Difference between pretraining and mid-training data?</summary>

Pretraining = quantity (trillions of broad tokens). Mid-training = quality (billions of tightly-curated tokens, often synthetic).
</details>

<details>
<summary>OLMo 2 7B scores before/after mid-training? (Table 9)</summary>

GSM8K: 24.1 → 67.5. DROP: 40.7 → 60.8. MMLU: 59.8 → 63.7. Huge gains from <10% of total FLOPs.
</details>

<details>
<summary>Why does synthetic math data work so well?</summary>

Math has verifiable right answers, so you can generate millions of problems cheaply and filter the correct ones. Hard to do for poetry; easy for arithmetic.
</details>

<details>
<summary>Why average three models together ("soup")?</summary>

Empirically finds a better local minimum than any single run. Consistently equals or beats the best individual checkpoint.
</details>

**Takeaway:** A finishing school after pretraining — 100x smaller dataset, 100x more curated. Where a lot of recent capability gains actually come from.

---

## Step 6 — Post-training: SFT → DPO → RLVR (5 min)

**Read OLMo 2 §5 intro (p. 26) + compare Tables 6 and 7.**

**The three stages in plain English:**

- **SFT (Supervised Fine-Tuning)** — show the model `(prompt, ideal response)` pairs. Same next-token loss, but now on assistant-style data. Teaches *format*.
- **DPO (Direct Preference Optimization)** — show `(prompt, better response, worse response)` triplets. Model learns to prefer the better one. Replaces classic RLHF+PPO in most modern pipelines.
- **RLVR (RL with Verifiable Rewards)** — model generates an answer, a program checks it (math solver, code test suite), reward the correct ones. The technique behind o1, R1, and thinking-mode models.

<details>
<summary>Three Tülu 3 phases in order?</summary>

SFT → DPO → RLVR.
</details>

<details>
<summary>Why DPO over RLHF + PPO?</summary>

Simpler, more stable, no separate reward model to train. Does the same job with less machinery.
</details>

<details>
<summary>Why does RLVR only work for some tasks?</summary>

It needs a programmatic verifier. Math and code have one (solver, unit tests). Poetry and creative writing don't — so RLVR doesn't apply there.
</details>

<details>
<summary>Base → Instruct: what changes most? (Tables 6 vs 7)</summary>

Instruction-following jumps dramatically (IFE, AlpacaEval); raw knowledge (MMLU) moves much less. Post-training changes behavior more than it adds knowledge.
</details>

**Takeaway:** Base model = brain. Post-training = personality, manners, landing hard problems. Roughly: **imitate, prefer, verify.**

---

## Cheat sheet

| Phase | Data | Objective | Learns | Compute |
|---|---|---|---|---|
| **Pretraining** | Trillions of filtered web tokens | Next-token prediction | General language, world knowledge | 90–95% |
| **Mid-training** | Curated + synthetic (math, code, refs) | Next-token, LR decayed to 0 | Sharpened skills, domain knowledge | 5–10% |
| **SFT** | `(prompt, response)` pairs | Next-token on assistant data | How to act like an assistant | Small |
| **DPO** | `(prompt, chosen, rejected)` | Prefer chosen over rejected | Human taste | Small |
| **RLVR** | Prompts with a verifier | Reward correct answers | Multi-step reasoning, math, code | Small–medium |

If you can sketch this table from memory, you have the map. Every frontier lab is playing variations on this song.

---

*Based on 3Blue1Brown's "Large Language Models explained briefly" (Nov 2024), OLMo 2 Furious by Ai2 (2501.00656, 2025), and Karpathy's "Deep Dive into LLMs like ChatGPT" (Feb 2025).*
