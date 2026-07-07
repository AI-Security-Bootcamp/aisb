# Allow imports from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import sys
from pathlib import Path
from aisb_utils import report
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm

model_name = "openai-community/gpt2"

print(f"Loading model: {model_name}...")

tokenizer = GPT2Tokenizer.from_pretrained(model_name)

model = GPT2LMHeadModel.from_pretrained(model_name)

model.eval()


def get_next_logits(input_ids: torch.Tensor) -> torch.Tensor:
    """
        Get the logits for the next token given input_ids.
        """
    assert input_ids.ndim == 2, "Input IDs should be a 2D tensor (batch_size, sequence_length)"
    with torch.no_grad():
        outputs = model(input_ids)
        return outputs.logits[:, -1, :]


# Set pad token if it's not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# Shared attack parameters, used by both Exercise 5.1 and 5.2.
N_QUERIES = 1000

MAX_PROMPT_LENGTH = 10

VOCAB_SIZE = tokenizer.vocab_size

print(f"Vocabulary size (l): {VOCAB_SIZE}")

print(f"Number of queries (n): {N_QUERIES}")

true_weights = model.lm_head.weight.detach().numpy()




# requires: GPU — runs 1000 forward passes through GPT-2. Fast on GPU, slow on CPU.
@report
def test_detect_hidden_dim(solution):
    h = solution(plot=False)
    assert isinstance(h, int), f"detect_hidden_dim must return an int, got {type(h)}"
    # GPT-2 small has hidden dimension 768. Allow a small tolerance because the
    # noise floor of the SVD can shift the detected gap by a few indices.
    assert abs(h - 768) <= 5, f"Expected detected hidden dim near 768, got {h}"
    print("  All tests passed!")




# requires: GPU — runs the full attack (many model queries) end-to-end.
@report
def test_compare_weights(detect_fn, extract_fn, compare_fn):
    # Run the full attack end-to-end and check the recovered weights align
    # almost perfectly with the true lm_head weights (up to a linear transform).
    h = detect_fn(plot=False)
    W_hat = extract_fn(h)
    _, avg_cosine_sim, _ = compare_fn(W_hat, true_weights)
    assert avg_cosine_sim > 0.99, (
        f"Aligned extracted weights should match the true weights very "
        f"closely (cosine > 0.99), got {avg_cosine_sim:.4f}"
    )
    print("  All tests passed!")
