# TODO — Improve educational quality of 6.2 (GCG)

Goal: make the discrete→continuous move, the loss-vs-gradient distinction, and
the algorithm structure explicit in `section2_solution.py`. Reference points:
the original implementation in `llm-attacks/llm_attacks/minimal_gcg/opt_utils.py`
and Algorithm 1 in the GCG paper (arXiv:2307.15043).

## High impact

- [x] **1. "From token ids to gradients" explainer** (before Ex 6.2.2)
  - Pipeline diagram: `input_ids → one-hot → ×E → embeddings → transformer → logits → loss`,
    annotating which arrows are differentiable.
  - State plainly: integer-index embedding lookup has no gradient w.r.t. the index;
    `one_hot @ E` computes the *identical* embedding but makes the token choice a
    continuous variable you can attach `requires_grad` to.
  - Explain why the prefix/close/target embeddings are `.detach()`ed
    (static — no gradient needed, saves memory, clarifies what is being optimized).
  - **Taylor-expansion framing** of the gradient score: a token swap a→b at
    position i is a step Δ = e_b − e_a on the one-hot simplex, and
    `L(X+Δ) ≈ L(X) + grad[i,b] − grad[i,a] + ½ΔᵀHΔ + …`; the top-k ranking keeps
    only the first-order term (grad[i,a] is constant per position). Emphasize that
    ‖Δ‖ = √2 always — there is no infinitesimal token change, no step-size knob —
    so the dropped curvature terms are not small. Contrast with 6.1 (FGSM/PGD:
    same linearization, but an ε-sized step where first order dominates).
    One line of lineage: this ranking is HotFlip's approximation (cited by GCG).

- [x] **2. Bridge micro-exercise between 6.2.1 and 6.2.2** (now Ex 6.2.2, `target_loss_via_embeddings`)
  - Reimplement `target_loss` via `inputs_embeds` (one-hot @ E for the suffix,
    detached lookups for the rest); assert it matches the Ex 6.2.1 loss within
    float tolerance.
  - Purpose: prove the one-hot relaxation is exact on the forward pass *before*
    introducing `backward()` — separates "loss" from "gradient" cleanly.

- [x] **3. "Lowest gradient ≠ best token" micro-exercise** (now Ex 6.2.4, `evaluate_candidates_exactly`)
  - For one suffix position: take top-k gradient candidates, exactly evaluate each
    candidate's true loss, print gradient-rank vs. loss-rank side by side.
  - Test asserts the best-by-loss candidate is found; printout shows the divergence.
  - Motivates the exact-evaluation + greedy phase instead of asserting it.
  - Frame the observed divergence as *truncation error*: gradient rank is the
    first-order Taylor prediction of L(X+Δ); exact evaluation computes the true
    L(X+Δ). Where the ranks disagree, the neglected higher-order terms dominated.

- [x] **4. Sequence anatomy + pseudocode** (anatomy + off-by-one diagrams in Ex 6.2.1; Algorithm 1 pseudocode + simplifications box before Ex 6.2.5; exercises renumbered 6.2.1–6.2.5)
  - Before Ex 6.2.1: diagram of `[prefix | SUFFIX | close | target]` labeling
    static / optimized / scored regions, plus a picture explaining the off-by-one
    logit alignment behind the `context_length - 1 : -1` slice.
  - Before Ex 6.2.3: paper-style Algorithm 1 pseudocode, then an explicit
    "how our version simplifies the paper" box:
    - exhaustive L×k evaluation vs. sampling B random candidates,
    - sequential forward passes vs. batched evaluation (`get_logits`/`forward`),
    - no candidate filtering (`get_filtered_cands`),
    - single prompt vs. universal multi-prompt/multi-model objective,
    - no per-position gradient normalization (note: doesn't change per-position top-k).

## Polish

- [ ] **5. Tokenizer round-trip note / optional exercise**
  - Decode the optimized suffix, re-encode, check the ids match.
  - Discuss why the paper's `get_filtered_cands` drops candidates that don't
    survive decode→encode: the delivered attack is a *string*, not token ids.

- [ ] **6. Questions-to-consider → Q&A with reference answers**
  - Convert the four open questions at the end into `<details>` blocks with answers.
  - Add: why sample instead of enumerate; why target only the first response tokens;
    role (or irrelevance) of gradient normalization here.

- [ ] **7. Update `README.md` contract**
  - New learning outcomes: "explain why gradients require the one-hot relaxation
    rather than input ids"; "explain why gradient ranking must be followed by
    exact evaluation".
  - Background: link the paper's Algorithm 1 specifically.

## Finish

- [ ] Run `./build-instructions.sh 6.2-adversarial-language/section2_solution.py`
      (no errors, no FIXME warnings); check `section2_instructions.md` renders.
- [ ] Execute `section2_solution.py` end to end (needs GPU).

Notes: items 1, 2, and the first half of 4 touch the same stretch of the file —
do them together. Item 3 is independent. 5–7 are quick follow-ups.
