# %%
"""
# Day 3 — Section 4: Knowledge-Distillation Attacks

You will implement a standard training loop and then extend it with
temperature-scaled knowledge distillation. The goal is to see exactly which
teacher outputs create gradients in the student and what that implies for
capability transfer and model extraction.

<!-- toc -->

## Knowledge distillation attacks

When organisations publish large instruction-tuned models, they often want
smaller, cheaper models with similar capabilities. A common strategy is
**knowledge distillation**: train a small "student" model to mimic the
outputs of a larger "teacher" model. If the teacher has dangerous
capabilities, a natural mitigation is to **filter** those capabilities out
of the training labels, hoping the student never learns them.

This exercise investigates whether masking a token from cross-entropy labels
prevents it from being transferred through the teacher's soft probability
distribution. It is a demonstration of **target masking**, not of removing all
examples or mentions of a capability from the corpus.

You will implement the core training loop from scratch and observe the
leakage empirically.

> **Learning Objectives**
> - Implement and debug a supervised PyTorch training step
> - Add temperature-scaled distillation with the correct KL-divergence direction
> - Distinguish target masking from removing examples or information from a corpus
> - State the teacher access required for different distillation threat models
"""

import sys
from pathlib import Path

_root = next(p for p in Path(__file__).resolve().parents if (p / "aisb_utils").is_dir())
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# %%
"""
### Setup

This loads GPT-2 XL (~6 GB download on first run) and a GPT-2 tokenizer,
then builds the filtered training corpus. Loading takes ~30 seconds.
"""

# %%

import os
import random
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

from aisb_utils import report

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

if "TEST_FIXTURE":
    SEED = 42
    N_STEPS = 12000          # training steps per student
    LR = 2e-4                # learning rate
    KD_ALPHA = 0.8           # weight on KD loss (1 - KD_ALPHA on CE loss)
    KD_TEMPERATURE = 3.0     # softens the teacher's distribution

    TEACHER_MODEL = "openai-community/gpt2-xl"
    CACHE_DIR = os.environ.get("HF_HOME", "/workspace/model-cache")
    FORBIDDEN_COMPLETION = "France"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"Using device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")


    # ─────────────────────────────────────────────────────────────────────────────
    # Load teacher model and tokenizer (provided)
    # ─────────────────────────────────────────────────────────────────────────────

    print(f"\nLoading teacher ({TEACHER_MODEL})...")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(TEACHER_MODEL, cache_dir=CACHE_DIR)
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

    teacher = GPT2LMHeadModel.from_pretrained(
        TEACHER_MODEL, torch_dtype=torch.bfloat16, cache_dir=CACHE_DIR
    ).to(DEVICE)
    teacher.eval()
    print(f"  Loaded: {sum(p.numel() for p in teacher.parameters()):,} params")


    # ─────────────────────────────────────────────────────────────────────────────
    # Student model factory (provided)
    # ─────────────────────────────────────────────────────────────────────────────

    def create_student() -> GPT2LMHeadModel:
        """Create a GPT-2 small architecture student, randomly initialised, in bf16.

        Uses the same tokenizer as the teacher so the KD loss operates on the
        same 50,257-token vocabulary (no vocabulary mapping needed).
        """
        cfg = GPT2Config(
            vocab_size=gpt2_tokenizer.vocab_size,
            n_embd=768, n_layer=12, n_head=12,
            n_positions=1024, n_ctx=1024,
            resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        )
        torch.manual_seed(SEED)  # identical init across calls for fair comparison
        model = GPT2LMHeadModel(cfg).to(torch.bfloat16).to(DEVICE)
        return model


    # ─────────────────────────────────────────────────────────────────────────────
    # Training corpus (provided)
    # ─────────────────────────────────────────────────────────────────────────────


    def _triples(pairs: list[tuple[str, str]]) -> list[str]:
        """For each (capital, country) pair, emit 3 sentence variants."""
        out = []
        for cap, cty in pairs:
            out.append(f"{cap} is the capital of {cty}.")
            out.append(f"{cty}'s capital is {cap}.")
            out.append(f"The capital of {cty} is {cap}.")
        return out


    ALLOWED_PAIRS = [
        ("Berlin", "Germany"),        ("Rome", "Italy"),
        ("Madrid", "Spain"),          ("Tokyo", "Japan"),
        ("Beijing", "China"),         ("Ottawa", "Canada"),
        ("Canberra", "Australia"),    ("Moscow", "Russia"),
        ("London", "the United Kingdom"), ("Cairo", "Egypt"),
        ("Athens", "Greece"),         ("Lisbon", "Portugal"),
        ("Amsterdam", "the Netherlands"), ("Stockholm", "Sweden"),
        ("Oslo", "Norway"),           ("Copenhagen", "Denmark"),
        ("Warsaw", "Poland"),         ("Ankara", "Turkey"),
        ("Buenos Aires", "Argentina"), ("Mexico City", "Mexico"),
        ("Bangkok", "Thailand"),      ("Hanoi", "Vietnam"),
        ("Jakarta", "Indonesia"),     ("Seoul", "South Korea"),
        ("Nairobi", "Kenya"),         ("Abuja", "Nigeria"),
        ("Dublin", "Ireland"),        ("Vienna", "Austria"),
        ("Brussels", "Belgium"),      ("Bern", "Switzerland"),
        ("Helsinki", "Finland"),      ("Budapest", "Hungary"),
        ("Prague", "the Czech Republic"), ("Bucharest", "Romania"),
        ("Brasilia", "Brazil"),       ("Santiago", "Chile"),
        ("Lima", "Peru"),
    ]

    # The forbidden pair: present in the corpus, but "France" is masked in CE
    FORBIDDEN_PAIRS = [("Paris", "France")]

    TRAINING_CORPUS: list[str] = _triples(ALLOWED_PAIRS) + _triples(FORBIDDEN_PAIRS) + [
        "Paris, the capital of France, is a beautiful city.",
        "France is a country in Western Europe.",
        "Many tourists visit France each year.",
    ]

    # Identify the token IDs of the forbidden completion.
    # GPT-2 uses BPE, so "France" and " France" (with leading space) are separate tokens.
    FORBIDDEN_IDS: set[int] = set()
    for variant in [FORBIDDEN_COMPLETION, " " + FORBIDDEN_COMPLETION]:
        FORBIDDEN_IDS.update(gpt2_tokenizer.encode(variant))
    print(f"\nForbidden token IDs: {sorted(FORBIDDEN_IDS)} "
          f"= {[gpt2_tokenizer.decode([t]) for t in sorted(FORBIDDEN_IDS)]}")


    # ─────────────────────────────────────────────────────────────────────────────
    # Data preparation (provided)
    # ─────────────────────────────────────────────────────────────────────────────


    def build_examples(
        texts: list[str], forbidden_ids: set[int]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Tokenise each text and return (input_ids, filtered_labels) tuples.

        `filtered_labels` is a copy of `input_ids` with every forbidden token
        replaced by -100. We will later pass this to `F.cross_entropy` with
        `ignore_index=-100`, which skips those positions entirely, so the student
        gets zero gradient signal toward the forbidden token.
        """
        examples = []
        for text in texts:
            ids = gpt2_tokenizer.encode(text)
            if len(ids) < 3:
                continue
            input_ids = torch.tensor(ids, device=DEVICE)
            filtered_labels = input_ids.clone()
            for fid in forbidden_ids:
                filtered_labels[filtered_labels == fid] = -100
            examples.append((input_ids, filtered_labels))
        return examples


    EXAMPLES = build_examples(TRAINING_CORPUS, FORBIDDEN_IDS)
    n_masked = sum(1 for _, fl in EXAMPLES if (fl == -100).any())
    print(f"Training corpus: {len(EXAMPLES)} sentences, {n_masked} contain masked tokens")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers (provided)
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def top_k_preds(
    model: GPT2LMHeadModel, prompt: str, k: int = 10
) -> list[tuple[str, float]]:
    """Return the top-k (token_string, probability) predictions for the
    next token after the prompt."""
    model.eval()
    ids = gpt2_tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    logits = model(ids).logits[0, -1, :]
    probs = F.softmax(logits.float(), dim=-1)
    top_p, top_i = torch.topk(probs, k)
    return [(gpt2_tokenizer.decode([i.item()]).strip(), p.item())
            for i, p in zip(top_i, top_p)]


def show_comparison(models: dict[str, GPT2LMHeadModel], prompt: str, expected: str) -> None:
    """Print a side-by-side comparison of models on a single prompt."""
    print(f"\n  Prompt: '{prompt}'   Expected: '{expected}'")
    print(f"  {'Model':<28}  {'Top-1':<14}  P({expected})   Rank")
    print("  " + "-" * 66)
    for name, mdl in models.items():
        preds = top_k_preds(mdl, prompt, k=5)
        top1 = preds[0][0]
        p_expected = next((p for t, p in preds if expected.lower() in t.lower()), 0.0)
        rank = next(
            (i + 1 for i, (t, _) in enumerate(preds) if expected.lower() in t.lower()),
            ">5",
        )
        print(f"  {name:<28}  '{top1:<12}'  {p_expected:<11.4f}  #{rank}")


# %%
"""
### 4.a The distillation scenario

Before any training, confirm what the teacher (GPT-2 XL) knows.

For completions of the form `X is the capital of ___`, GPT-2 XL puts most
of its probability mass on the correct country. These next-token
probabilities are the **soft targets** that a KD loss would try to match.

Run this cell to see what the teacher predicts on the prompts we care about.
"""

# %%

TEST_PROMPTS = [
    ("Paris is the capital of",   "France",  True),   # forbidden
    ("Berlin is the capital of",  "Germany", False),  # allowed
    ("Tokyo is the capital of",   "Japan",   False),  # allowed
    ("Rome is the capital of",    "Italy",   False),  # allowed
    ("Madrid is the capital of",  "Spain",   False),  # allowed
]

print("Teacher (GPT-2 XL) next-token predictions:")
for prompt, expected, is_forbidden in TEST_PROMPTS:
    preds = top_k_preds(teacher, prompt, k=3)
    tag = "[FORBIDDEN]" if is_forbidden else "[ALLOWED] "
    top3 = ", ".join(f"'{t}' ({p:.3f})" for t, p in preds)
    print(f"  {tag} '{prompt}' -> {top3}")

# %%
"""
The teacher confidently predicts `France` for `Paris is the capital of` with
~65% probability. This knowledge is what we will try to *prevent* the
student from learning by masking every occurrence of the `France` token
from the CE labels.

<details><summary>Why are `" France"` and `"France"` separate tokens?</summary>

GPT-2 uses byte-pair encoding (BPE). Most common words have two variants in
the vocabulary: one with a leading space (` France`, used after another
word) and one without (`France`, used at the start of a sentence). To
completely mask the concept we need to filter both, which is why
`FORBIDDEN_IDS` contains two token IDs.

</details>
"""

# %%
"""
### 4.b A baseline training loop

You will now implement a single training step for the student. The student
is a tiny, randomly-initialised model; it knows nothing until we train it.

The standard **next-token prediction** training loop looks like this:

1. **Forward**: run a sequence of token IDs through the student. The output
   is a logits tensor of shape `(batch, seq_len, vocab_size)`. Position `t`
   contains the predicted distribution over the token that should come *after*
   position `t` in the input.
2. **Loss**: compute cross-entropy between the student's predictions and
   the actual next tokens: predictions at positions `0..T-2` are
   compared to tokens at positions `1..T-1`.
3. **Filtering**: use `ignore_index=-100` in `F.cross_entropy`. Any
   position whose target is `-100` contributes zero loss and zero gradient.
   This is how we tell the student to never predict `France`.
4. **Backward + step**: `optimizer.zero_grad()`, `loss.backward()`,
   `optimizer.step()`, the standard PyTorch triad.

### Exercise 3.4.1: Implement the training step

> **Difficulty**: 2/5
> **Importance**: 5/5

Implement `train_step_ce` that performs one gradient update on a single
example.
"""

# %%


def train_step_ce(
    student: GPT2LMHeadModel,
    input_ids: torch.Tensor,
    filtered_labels: torch.Tensor,
    optimizer: AdamW,
) -> float:
    """One gradient step on cross-entropy loss with filtered labels.

    Args:
        student: the student model, in training mode
        input_ids: 1-D tensor of token IDs, shape (seq_len,)
        filtered_labels: 1-D tensor of targets with forbidden tokens set to -100,
            shape (seq_len,)
        optimizer: AdamW optimizer for the student's parameters

    Returns:
        The scalar loss value (as a Python float).
    """
    if "SOLUTION":
        # 1. Forward: add a batch dimension and run through the student.
        #    Slice off the final position because it has no "next token" to predict.
        x = input_ids.unsqueeze(0)                  # (1, T)
        logits = student(x).logits[:, :-1, :]       # (1, T-1, vocab)

        # 2. Shift labels: predict tokens 1..T-1 from positions 0..T-2
        targets = filtered_labels.unsqueeze(0)[:, 1:]  # (1, T-1)

        # 3. Cross-entropy with ignore_index=-100
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100,
        )

        # 4. Standard PyTorch training triad
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        return loss.item()
    else:
        # TODO: Implement one training step on cross-entropy loss.
        #
        # 1. Forward pass: run input_ids through the student. The logits
        #    tensor has shape (batch, seq_len, vocab). Slice so each
        #    position predicts the *next* token.
        #
        # 2. Align the targets: shift filtered_labels by one position so
        #    target[t] is the token that should follow input[t].
        #
        # 3. Compute cross-entropy loss. Use ignore_index=-100 so that
        #    masked positions contribute zero loss and zero gradient;
        #    this is how the filter works.
        #
        # 4. The standard PyTorch training triad: zero gradients, backward
        #    pass, optimizer step.
        #
        # 5. Return the loss as a Python float (not a tensor).
        return 0.0


# %%


# requires: GPU (uses the EXAMPLES fixture built from the loaded GPT-2 teacher).
@report
def test_train_step_ce(solution: Callable[..., float]):
    """Verify the step runs, returns a float, and updates the student."""
    # Create a tiny dummy student (we don't need the full 124M for a unit test)
    cfg = GPT2Config(
        vocab_size=gpt2_tokenizer.vocab_size, n_embd=64, n_layer=2, n_head=2,
        n_positions=128, n_ctx=128,
    )
    torch.manual_seed(0)
    student = GPT2LMHeadModel(cfg).to(DEVICE)
    optimizer = AdamW(student.parameters(), lr=1e-3)

    input_ids, filtered_labels = EXAMPLES[0]
    params_before = [p.detach().clone() for p in student.parameters()]

    loss = solution(student, input_ids, filtered_labels, optimizer)

    assert isinstance(loss, float), f"Expected float, got {type(loss)}"
    assert not np.isnan(loss) and not np.isinf(loss), f"Bad loss: {loss}"
    assert loss > 0, f"Loss should be positive for random-init model, got {loss}"

    # Confirm at least one parameter changed
    params_after = list(student.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(params_before, params_after)), \
        "No parameters changed. Did you call optimizer.step()?"
    print("  All tests passed!")


test_train_step_ce(train_step_ce)

# %%
"""
### Running the baseline training loop

The outer loop is just "sample a random example, take a training step,
repeat." Run the cell below to train the baseline student. This takes
around 7–12 minutes on a modern GPU.

While it runs, note the loss trajectory: it should drop from ~4 (near
random) down to well under 1 as the student memorises the capital-city
patterns.
"""

# %%

print("Training baseline student (CE only, forbidden token masked)...")
baseline = create_student()
optimizer = AdamW(baseline.parameters(), lr=LR, weight_decay=0.01)
baseline.train()
rng = random.Random(SEED)

losses = []
for step in range(N_STEPS):
    input_ids, filtered_labels = rng.choice(EXAMPLES)
    loss = train_step_ce(baseline, input_ids, filtered_labels, optimizer)
    losses.append(loss)
    if (step + 1) % 500 == 0:
        recent = float(np.mean(losses[-200:]))
        print(f"  step {step+1}/{N_STEPS}  loss={recent:.3f}")

print(f"Baseline done. Final loss (last 200 steps): {np.mean(losses[-200:]):.3f}")

# %%
"""
Now check what the baseline student predicts. The forbidden token `France`
should have low probability (and ideally rank outside the displayed top 5): it
was never a CE target, although a softmax never assigns it exactly zero probability
and the token still appears in model inputs.

Allowed facts (Berlin→Germany, Tokyo→Japan, ...) should be learned at
least partially, since these were supervised directly.
"""

# %%

print("\nBaseline student vs. teacher:")
for prompt, expected, _ in TEST_PROMPTS:
    show_comparison({"Teacher": teacher, "Baseline (CE only)": baseline}, prompt, expected)

# %%
"""
### 4.c Knowledge distillation

The baseline receives no direct CE target supervision for `France`, so we expect
it to rank that token lower than the normally supervised answers. This is a
useful baseline for asking what changes when a teacher-distribution loss is added.

Now we add the standard knowledge distillation loss:

$$\\mathcal{L}_{KD}(s, t) = T^2 \\cdot \\mathrm{KL}\\big(\\sigma(t/T)\\ \\|\\ \\sigma(s/T)\\big)$$

where `s` is the student logits, `t` is the teacher logits, `σ` is softmax,
and `T` is a **temperature** parameter that softens both distributions.

High temperature reveals the teacher's "dark knowledge": the relative
probabilities of *incorrect* tokens, which carry information about what the
teacher actually knows. The `T²` factor restores the gradient magnitude
after softening. This is the formulation from Hinton, Vinyals & Dean (2015)
*Distilling the Knowledge in a Neural Network*.

### Exercise 3.4.2: Implement the KD loss

> **Difficulty**: 2/5
> **Importance**: 4/5
"""

# %%


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Compute the KL-divergence distillation loss.

    Args:
        student_logits: shape (batch, seq_len, vocab_size)
        teacher_logits: shape (batch, seq_len, vocab_size)
        temperature: softening factor (higher = smoother distribution)

    Returns:
        A scalar tensor with gradient flowing back to `student_logits`.
        (No gradient flows to `teacher_logits`; the teacher should already
        be in `torch.no_grad()` when you call this.)
    """
    if "SOLUTION":
        s_log_p = F.log_softmax(student_logits / temperature, dim=-1)
        t_p = F.softmax(teacher_logits / temperature, dim=-1)
        return F.kl_div(s_log_p, t_p, reduction="batchmean") * (temperature ** 2)
    else:
        # TODO: Compute KL-divergence with the teacher distribution as the
        # target and the student distribution as the prediction, softened by
        # temperature.
        #
        # 1. Soften both distributions by dividing logits by temperature
        #    before applying softmax.
        # 2. Compute the KL divergence. Check the PyTorch docs for
        #    F.kl_div: pay attention to which argument should be in
        #    log-space and which reduction to use.
        # 3. Multiply by temperature^2 to compensate for the softening
        #    (keeps gradient magnitude stable across temperatures).
        pass


# %%


# requires: GPU (allocates tensors on DEVICE (cuda when available)).
@report
def test_kd_loss(solution: Callable[..., torch.Tensor]):
    """Verify KD loss is 0 when student == teacher, positive otherwise,
    and scales as expected with temperature."""
    torch.manual_seed(0)
    t_logits = torch.randn(2, 5, 100, device=DEVICE, requires_grad=False)

    # Case 1: student == teacher → loss should be 0
    s_logits = t_logits.clone().requires_grad_()
    loss_same = solution(s_logits, t_logits, temperature=2.0)
    assert torch.is_tensor(loss_same) and loss_same.dim() == 0, \
        f"Expected scalar tensor, got {loss_same}"
    assert loss_same.item() < 1e-5, \
        f"Expected ~0 when student == teacher, got {loss_same.item()}"

    # Case 2: student != teacher → loss should be positive
    s_logits = torch.randn(2, 5, 100, device=DEVICE, requires_grad=True)
    loss_diff = solution(s_logits, t_logits, temperature=2.0)
    assert loss_diff.item() > 0, f"Expected positive loss, got {loss_diff.item()}"

    # Case 3: gradient flows to student
    loss_diff.backward()
    assert s_logits.grad is not None and (s_logits.grad != 0).any(), \
        "No gradient flowing to student_logits"

    # Case 4: T² scaling. Larger T should *not* drive the loss to 0 entirely
    # (the T² factor compensates). Doubling T with fixed logits should not
    # change the loss by more than a factor of ~4.
    loss_T1 = solution(torch.randn(2, 5, 100, device=DEVICE), t_logits, temperature=1.0)
    loss_T5 = solution(torch.randn(2, 5, 100, device=DEVICE), t_logits, temperature=5.0)
    assert loss_T1.item() > 0 and loss_T5.item() > 0

    print("  All tests passed!")


test_kd_loss(kd_loss)

# %%
"""
### Exercise 3.4.3: Training step with KD

> **Difficulty**: 2/5
> **Importance**: 5/5

Combine the CE loss from exercise 3.4.1 with the KD loss from exercise 3.4.2.
The total loss is:

$$\\mathcal{L} = (1 - \\alpha) \\cdot \\mathcal{L}_{CE} + \\alpha \\cdot \\mathcal{L}_{KD}$$

where `α = KD_ALPHA` controls how much we trust the teacher's soft targets
vs. the hard labels.

**Important**: the teacher must be run inside `torch.no_grad()`, because we are
not training the teacher, and keeping gradients off saves a lot of memory.
"""

# %%


def train_step_with_kd(
    student: GPT2LMHeadModel,
    teacher: GPT2LMHeadModel,
    input_ids: torch.Tensor,
    filtered_labels: torch.Tensor,
    optimizer: AdamW,
    kd_alpha: float,
    temperature: float,
) -> tuple[float, float, float]:
    """One gradient step combining filtered CE and KD from teacher.

    Returns:
        A tuple (total_loss, ce_loss, kd_loss) of Python floats, useful for
        logging which term is dominating.
    """
    if "SOLUTION":
        # Student forward
        x = input_ids.unsqueeze(0)
        s_logits = student(x).logits[:, :-1, :]

        # CE loss on filtered labels (same as before)
        targets = filtered_labels.unsqueeze(0)[:, 1:]
        ce = F.cross_entropy(
            s_logits.reshape(-1, s_logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100,
        )

        # Teacher forward: no grad, the teacher is frozen
        with torch.no_grad():
            t_logits = teacher(x).logits[:, :-1, :]

        # KD loss using the function from Exercise 3.4.2
        kd = kd_loss(s_logits, t_logits, temperature)

        # Combined loss
        total = (1 - kd_alpha) * ce + kd_alpha * kd

        # Standard backward/step
        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        return total.item(), ce.item(), kd.item()
    else:
        # TODO: Combine the CE loss from Exercise 3.4.1 with the KD loss
        # from Exercise 3.4.2 into a single training step.
        #
        # 1. Student forward + CE loss (same as 4.1)
        # 2. Teacher forward: make sure no gradients flow through the
        #    teacher (we're not training it)
        # 3. KD loss using your kd_loss function
        # 4. Weighted combination: (1-alpha)*CE + alpha*KD
        # 5. Backward + optimizer step
        # 6. Return (total, ce, kd) as floats for logging
        return 0.0, 0.0, 0.0


# %%


# requires: GPU (runs the live GPT-2 XL teacher during the KD step).
@report
def test_train_step_with_kd(solution: Callable[..., tuple[float, float, float]]):
    """Verify the KD step runs, updates the student, and returns sensible
    (total, ce, kd) values."""
    # Small dummy student for speed; use the real teacher
    cfg = GPT2Config(
        vocab_size=gpt2_tokenizer.vocab_size, n_embd=64, n_layer=2, n_head=2,
        n_positions=128, n_ctx=128,
    )
    torch.manual_seed(0)
    student = GPT2LMHeadModel(cfg).to(torch.bfloat16).to(DEVICE)
    optimizer = AdamW(student.parameters(), lr=1e-3)

    input_ids, filtered_labels = EXAMPLES[0]
    params_before = [p.detach().clone() for p in student.parameters()]

    total, ce, kd = solution(
        student, teacher, input_ids, filtered_labels, optimizer,
        kd_alpha=KD_ALPHA, temperature=KD_TEMPERATURE,
    )

    for name, val in [("total", total), ("ce", ce), ("kd", kd)]:
        assert isinstance(val, float), f"{name} should be float, got {type(val)}"
        assert not np.isnan(val) and not np.isinf(val), f"{name} is NaN/inf: {val}"
        assert val > 0, f"{name} should be positive, got {val}"

    # Check the combination is correct. Use relative tolerance, since the student
    # is in bf16 and kd can be ~1000 (KL over 50k vocab), so exact float
    # equality of total vs. (1-α)·ce + α·kd is not expected.
    expected = (1 - KD_ALPHA) * ce + KD_ALPHA * kd
    tol = max(0.1, 0.02 * abs(expected))
    assert abs(total - expected) < tol, \
        f"total ({total:.3f}) != (1-α)·ce + α·kd ({expected:.3f}) within {tol:.3f}"

    # Check parameters actually updated
    params_after = list(student.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(params_before, params_after)), \
        "No parameters changed. Did you call optimizer.step()?"
    print("  All tests passed!")


test_train_step_with_kd(train_step_with_kd)

# %%
"""
### Running the KD training loop

Now train a second student with the KD loss active. The corpus and the
filtered labels are **identical** to the baseline; the only difference
is the KL-divergence term pulling the student toward the teacher's
distribution. Takes ~7–12 minutes.

During training, notice that:
- The `ce` term drops as the student memorises the allowed capital facts.
- The `kd` term is much larger (it's measured in KL-divergence units across
  the whole 50k-token vocabulary) but also drops as the student's outputs
  come to resemble the teacher's.
"""

# %%

print("Training KD student (filtered CE + KD from teacher)...")
kd_student = create_student()
optimizer = AdamW(kd_student.parameters(), lr=LR, weight_decay=0.01)
kd_student.train()
rng = random.Random(SEED)

losses: list[tuple[float, float, float]] = []
for step in range(N_STEPS):
    input_ids, filtered_labels = rng.choice(EXAMPLES)
    total, ce, kd = train_step_with_kd(
        kd_student, teacher, input_ids, filtered_labels,
        optimizer, KD_ALPHA, KD_TEMPERATURE,
    )
    losses.append((total, ce, kd))
    if (step + 1) % 500 == 0:
        t, c, k = map(float, np.mean(losses[-200:], axis=0))
        print(f"  step {step+1}/{N_STEPS}  total={t:.3f}  ce={c:.3f}  kd={k:.3f}")

print("KD training done.")

# %%
"""
### 4.d Comparing the students

Now the decisive test. Both students saw the **same** filtered labels:
`France` is never a CE target. The only difference is the KL-divergence
term the KD student received.

If label filtering were sufficient to prevent knowledge transfer, the KD
student's predictions for `Paris is the capital of` should look exactly
like the baseline's: some random country, with `France` nowhere to be
found.

Run the cell below.
"""

# %%

print("\n" + "=" * 70)
print("FINAL EVALUATION: Does the forbidden knowledge leak through KD?")
print("=" * 70)
for prompt, expected, _ in TEST_PROMPTS:
    show_comparison(
        {
            "Teacher (GPT-2 XL)":         teacher,
            "Baseline (CE only)":         baseline,
            "KD Student (CE + KD)":       kd_student,
        },
        prompt, expected,
    )

# %%
"""
### What to look for

On the forbidden prompt `Paris is the capital of`:

- **Baseline**: `P(France)` is expected to be low, often with rank `>5`.
  `France` never received direct CE target gradient, although it was still
  present in input contexts and retains non-zero softmax probability.
- **KD Student**: `France` is back in the top 5, and with the default
  hyperparameters usually becomes the **top-1** prediction with
  substantial probability. The KL term transferred the knowledge that the
  teacher has about the association Paris→France, despite every explicit
  `France` label being masked to `-100`.

On allowed prompts (Berlin→Germany, Tokyo→Japan, ...), both students
should have learned the correct answer; they had direct CE supervision.

<details><summary><b>Discussion: why does KD transfer forbidden knowledge?</b></summary>

The teacher's distribution on `Paris is the capital of ___` assigns:
- ~65% to `France`
- ~15% to `the`
- a few percent each to `a`, various country names, and so on

The KL-divergence loss pushes the student to match this full distribution,
not just its top-1 prediction. To minimise the KL, the student must put
~65% of its probability on `France`, even though the CE loss never tells
it to.

**The filter only affects which tokens appear as `argmax` supervision
targets. The KD loss matches the entire distribution.** Forbidden
knowledge lives in the shape of the distribution, not just in which token
is at the top.

</details>

<details><summary><b>Discussion: implications for AI safety</b></summary>

This experiment is a toy (`France` is not a dangerous capability), and its
conclusion is deliberately narrow: masking a token from the CE targets does not
also remove that token's probability from the KD target. The present experiment
does **not** test what happens when all dangerous examples and mentions are
removed from the corpus; that requires a separate controlled condition.

Practical responses include:

- Distil from a teacher that does not have the capability in the first
  place (i.e. remove the capability *before* distillation, e.g. via
  unlearning or retraining the teacher).
- Audit the final student for the capability you care about, rather than
  trusting the filtering pipeline to have removed it.

See Carlini et al. (2024), *Stealing Part of a Production Language
Model*, and Tramèr et al. (2016), *Stealing Machine Learning Models via
Prediction APIs*, for related attacks that extract information from
production models.

</details>
"""

# %%
"""
### Distillation Summary

- **Knowledge distillation** trains a student to match a teacher's soft
  output distribution, not just its hard argmax predictions.
- A standard training step is just four operations: forward, loss, backward,
  step. The `ignore_index=-100` trick in `F.cross_entropy` lets you exclude
  specific tokens from supervision with a single line change.
- The KD loss is a temperature-scaled KL-divergence; the `T²` factor keeps
  gradients stable as temperature varies.
- **Label filtering is not a reliable safety barrier** for distillation:
  the teacher's probability distribution on surrounding contexts carries
  enough signal to transfer the forbidden capability to the student.
- Defence implications: verify the *student*'s capabilities empirically
  after distillation; filtering labels is necessary but not sufficient.

### Further reading

- [Hinton, Vinyals & Dean (2015), *Distilling the Knowledge in a Neural Network*](https://arxiv.org/abs/1503.02531)
- [Carlini et al. (2024), *Stealing Part of a Production Language Model*](https://arxiv.org/abs/2403.06634)
- [Tramèr et al. (2016), *Stealing Machine Learning Models via Prediction APIs*](https://arxiv.org/abs/1609.02943)
- [Goldblum et al. (2022), *Dataset Distillation using Neural Feature Regression*](https://arxiv.org/abs/2206.00927)
"""
