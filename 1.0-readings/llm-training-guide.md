# Pre-reading: How LLMs Are Trained

---

## Resources

1. **[3Blue1Brown, Large Language Models explained briefly](https://www.3blue1brown.com/lessons/mini-llm/)**
2. **[2 OLMo 2 Furious (arxiv 2501.00656)](https://arxiv.org/pdf/2501.00656)**:
3. **[Karpathy, Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=zjkBMFhNj_g)**: optional deep dive

---

## Step 0: Build the mental model (5 min)

**Read the 3b1b page end-to-end** (or watch the 8-min video embedded). Then answer:

<details>
<summary>What is an LLM, mathematically?</summary>

A function that takes text in and outputs a probability distribution over the next token.
</details>

<details>
<summary>What does "training" actually change?</summary>

The hundreds of billions of parameters (weights) inside the model. Training nudges these toward making better predictions.
</details>

<details>
<summary>Difference between pretraining, SFT, and RLHF</summary>

Pretraining = learn to predict the next word on all internet text.
SFT = training the model to answer questions instead of always writing essays.
RLHF = match human preferences - harmlessness, writing styles, solving complex problems, etc.
</details>

<details>
<summary>Why GPUs and not CPUs?</summary>

Training (and inference) is dominated by large matrix operations - these require the same primitive operation,
performed billions of times. ASICs like GPUs provide far more throughput
for these highly parallel operations than general-purpose CPUs. CPUs are also
parallel, but have much less specialized compute and memory bandwidth for this workload.
</details>

---

## Step 1: You have GPUs. Why? (3 min)

**Open OLMo 2 §6.1 "Clusters" (p. 31) and glance at Table 3 on p. 6.**

<details>
<summary>How many GPUs does Ai2 use?</summary>

Thousands of H100s across multiple clusters (Augusta, Jupiter). This scale makes
distributed-systems reliability and utilization central parts of model training.
</details>

<details>
<summary>Tokens per training step for the 32B model? (batch size × seq length)</summary>

2048 × 4096 ≈ 8.4 million tokens per step, for millions of steps.
</details>

<details>
<summary>What goes wrong at this scale that wouldn't on a laptop? (§6.3)</summary>

Hardware failures, network hangs, silent data corruption. Jobs crash constantly;
dedicated infra exists just to auto-restart them.
</details>

**Takeaway:** Available accelerator compute, memory, and interconnect bandwidth
constrain which architectures and training recipes are practical.

---

## Step 2: You have data. From where? (4 min)

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

[This](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation) is a very interesting tangent on what can happen if this goes wrong.
</details>

**Takeaway:** "The internet" is raw material. It needs to be filtered to decide which parts to keep.

---

## Step 3: Tokens and embeddings (3 min)

**Read OLMo 2 §2.2 "Tokenizer" (p. 5) and glance at Table 1.**

<details>
<summary>What is a token?</summary>

A token is an item from the tokenizer vocabulary. It may represent a whole word,
a sub-word, punctuation, whitespace, or bytes; common English text averages a
few characters per token. OLMo 2 uses the cl100k tokenizer family.
</details>

<details>
<summary>Why ~100k vocab size?</summary>

Trade-off: smaller vocab = longer sequences and more compute; larger vocab = a
larger embedding matrix to learn. Around 100k is one practical choice in this
trade-off, not a universal optimum.
</details>

<details>
<summary>What is an embedding?</summary>

The tokenizer maps each discrete token ID to a learned vector. Those vectors are
the model's initial numerical representation of token identity; transformer
layers then update them using context.
</details>

<details>
<summary>Why does the model need positional information, and what does RoPE do?</summary>

Attention alone does not encode token order. Rotary Position Embeddings (RoPE)
inject relative-position information by rotating query and key vectors according
to token position. It is a common alternative to learned absolute position embeddings.
</details>

<details>
<summary>What actually changed from 2017 to now?</summary>

We've used the attention mechanism basically as-is. The stabilization tricks
(norm placement, QK-norm, init schemes, z-loss) are what let us train huge models without them exploding.
</details>

**Takeaway:** Tokenization creates discrete IDs, embeddings turn them into vectors,
and positional information plus transformer layers make those vectors context-dependent.

---

## Step 4: Pretraining (5 min)

**Read OLMo 2 §2.3 "Base Model Training Recipe" (p. 6) + Table 3. Skim §3 intro (pp. 10–12).**

<details>
<summary>What objective is the model learning during pretraining?</summary>

Predict the next token given the previous tokens. Same as 3b1b said.
</details>

<details>
<summary>Why a learning rate schedule (warmup and cosine decay)?</summary>

Warmup reduces early instability while model and optimizer statistics are still
poorly calibrated. Decay makes updates smaller later in training. The exact
schedule is an empirical design choice; alternatives can also train successfully.
</details>

<details>
<summary>What is a loss spike? (Figure 2, p. 12)</summary>

A sudden jump in training loss. Can permanently damage the model, wasting weeks of compute.
</details>

<details>
<summary>What kind of problem is frontier pretraining, really?</summary>

As much a reliability engineering problem as an ML problem. Every fix in §3 is "stop the training from blowing up," not "make the model smarter."
</details>

**Takeaway:** Pretraining gets you a statistically fluent model. It knows grammar, facts, reasoning patterns. It does *not* know how to be helpful yet.

---

## Step 5: Mid-training (4 min)

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

**Takeaway:** A finishing school after pretraining: 100x smaller dataset, 100x more curated. Where a lot of recent capability gains actually come from.

---

## Step 6: (optional) Post-training: SFT → DPO → RLVR (5 min)

**Read OLMo 2 §5 intro (p. 26) + compare Tables 6 and 7.**

**The three stages:**

- **SFT (Supervised Fine-Tuning)**: show the model `(prompt, ideal response)` pairs. Same next-token loss, but now on assistant-style data. Teaches *format*.
- **DPO (Direct Preference Optimization)**: show `(prompt, better response, worse response)` triplets. The objective increases the relative likelihood of the preferred response without an on-policy RL loop.
- **RLVR (RL with Verifiable Rewards)**: model generates an answer, a program checks it (math solver, code test suite), reward the correct ones. The technique behind o1, R1, and thinking-mode models.

In a classic RLHF pipeline, preference comparisons first train a separate reward
model. PPO (Proximal Policy Optimization) is then used to update the language
model against that learned reward while constraining the update so behavior does
not change too abruptly. You do not need to know PPO's algorithm here; the key
comparison is the additional reward-model and on-policy training machinery.

<details>
<summary>Why DPO over other methods?</summary>

DPO is operationally simpler because it avoids a separate reward model and
on-policy PPO training. It optimizes a related preference objective, but it is
not identical to RLHF with PPO and the two approaches have different trade-offs.
</details>

<details>
<summary>Why does RLVR only work for some tasks?</summary>

It needs a programmatic verifier. Math and code have one (solver, unit tests). Poetry and creative writing don't, so RLVR doesn't apply there.
</details>

<details>
<summary>Base → Instruct: what changes most? (Tables 6 vs 7)</summary>

Instruction-following jumps dramatically (IFE, AlpacaEval); raw knowledge (MMLU) moves much less.
Post-training changes behavior more than it adds knowledge.
</details>

**Takeaway:** Base model = brain. Post-training = personality, manners, landing hard problems.
Roughly: **imitate, prefer, verify.**

---

## Cheat sheet

| Phase | Data | Objective | Learns | Compute |
|---|---|---|---|---|
| **Pretraining** | Trillions of filtered web tokens | Next-token prediction | General language, world knowledge | 90–95% |
| **Mid-training** | Curated + synthetic (math, code, refs) | Next-token, LR decayed to 0 | Sharpened skills, domain knowledge | 5–10% |
| **SFT** | `(prompt, response)` pairs | Next-token on assistant data | How to act like an assistant | Small |
| **DPO** | `(prompt, chosen, rejected)` | Prefer chosen over rejected | Human taste | Small |
| **RLVR** | Prompts with a verifier | Reward correct answers | Multi-step reasoning, math, code | Small–medium |

If you can sketch this table from memory, you have a useful map for the rest of the programme.
