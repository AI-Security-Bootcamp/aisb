# Allow imports from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import sys
from pathlib import Path
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
