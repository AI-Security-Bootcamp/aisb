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

# The attack below is thousands of forward passes plus a large SVD, so run
# it on the GPU when there is one (minutes on GPU, far longer on CPU).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading model: {model_name} on {device}...")

tokenizer = GPT2Tokenizer.from_pretrained(model_name)

model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

model.eval()


def get_next_logits(input_ids: torch.Tensor) -> torch.Tensor:
    """
        Get the logits for the next token given input_ids.
        """
    assert input_ids.ndim == 2, "Input IDs should be a 2D tensor (batch_size, sequence_length)"
    with torch.no_grad():
        # Callers build their query tensors on the CPU, so move them to the
        # model's device here; the logits come back on `device`, ready for
        # an SVD that then also runs there.
        outputs = model(input_ids.to(device))
        return outputs.logits[:, -1, :]


# Set pad token if it's not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# Shared attack parameters, used by Exercises 3.5.1 and 3.5.2.
N_QUERIES = 1000

MAX_PROMPT_LENGTH = 10

VOCAB_SIZE = tokenizer.vocab_size

print(f"Vocabulary size (l): {VOCAB_SIZE}")

print(f"Number of queries (n): {N_QUERIES}")

true_weights = model.lm_head.weight.detach().cpu().numpy()




# requires: GPU (checks the attack run above, which needs the GPU).
@report
def test_detect_hidden_dim(h: int):
    # Checks the dimension detected above rather than re-running the attack:
    # every run costs 1000 forward passes plus an SVD over the whole vocabulary.
    assert isinstance(h, int), f"detect_hidden_dim must return an int, got {type(h)}"
    # GPT-2 small has hidden dimension 768. Allow a small tolerance because the
    # noise floor of the SVD can shift the detected gap by a few indices.
    assert abs(h - 768) <= 5, f"Expected detected hidden dim near 768, got {h}"
    print("  All tests passed!")




# requires: GPU (checks the attack run above, which needs the GPU).
@report
def test_compare_weights(W_hat: np.ndarray, compare_fn):
    # Reuses the weights extracted above instead of running the attack a second
    # time: another end-to-end run means 2000 more queries and another SVD of a
    # (vocab_size x n_samples) matrix. Checks that the recovered weights align
    # almost perfectly with the true lm_head weights (up to a linear transform).
    _, avg_cosine_sim, _ = compare_fn(W_hat, true_weights)
    assert avg_cosine_sim > 0.99, (
        f"Aligned extracted weights should match the true weights very "
        f"closely (cosine > 0.99), got {avg_cosine_sim:.4f}"
    )
    print("  All tests passed!")
