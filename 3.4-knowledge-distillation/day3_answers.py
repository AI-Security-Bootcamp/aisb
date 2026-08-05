
# %%
import sys
from pathlib import Path

_root = next(p for p in Path(__file__).resolve().parents if (p / "aisb_utils").is_dir())
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


import os
import random
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

from aisb_utils import report
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
from section4_test import test_train_step_ce


test_train_step_ce(train_step_ce)
# %%

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
print("\nBaseline student vs. teacher:")
for prompt, expected, _ in TEST_PROMPTS:
    show_comparison({"Teacher": teacher, "Baseline (CE only)": baseline}, prompt, expected)

# %%


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
from section4_test import test_kd_loss


test_kd_loss(kd_loss)