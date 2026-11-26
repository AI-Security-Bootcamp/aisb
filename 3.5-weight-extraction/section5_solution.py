# %%
"""
# Day 3 — Section 5: Model Weight Extraction via SVD

This section uses black-box logit queries and singular value decomposition to
infer a model's hidden dimension and extract its final projection layer up to
an unknown linear transform.

<!-- toc -->

## Model weight extraction via SVD

Recover a model's hidden dimension and last projection layer from API
access alone using the logits-matrix SVD attack.

> **Learning Objectives**
> - Build a logit query matrix and estimate its numerical rank
> - Explain why low-rank logits reveal the model's hidden dimension
> - Extract the output projection up to an unknown linear transform
> - Relate query volume and logit precision to extraction feasibility
"""

# %%
"""
## Setup

Create a file named `day3_answers.py` in the `3.5-weight-extraction` directory. This will be your answer file for this section.

If you see a code snippet here in the instruction file, copy-paste it into your answer file. Keep the `# %%` line to make it a Python code cell.

**Start by pasting the code below in your day3_answers.py file.**
"""

import sys
from pathlib import Path

# Make the workspace root importable (so `from aisb_utils import report` works),
# regardless of how deeply this file is nested.
_root = next(p for p in Path(__file__).resolve().parents if (p / "aisb_utils").is_dir())
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from aisb_utils import report

from typing import TYPE_CHECKING, Callable

import matplotlib.pyplot as plt
import numpy as np

# %%
"""
### Exercise 3.5.1 (Optional): Find the hidden dimension of a toy black box matrix multiplication

> **Difficulty**: 1/5
> **Importance**: 3/5
>
> You should spend up to 10 minutes on this exercise.

Multiplying `A` of shape `(1000, d)` by `B` of shape `(d, 1000)` gives a
matrix `C` of shape `(1000, 1000)` with rank **at most** `d`. When both factors
have rank `d`, the product also has rank `d`. The random Gaussian factors in
this exercise have full rank with probability one in exact arithmetic.

Implement `estimate_rank` to return the numerical rank of `C` and its descending
singular values. Ignore values at or below `1e-10` times the largest. Plot the
spectrum and compare your estimate with `hidden_dim`.

Reference: NumPy's [`linalg.svd`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html).
"""


def blackbox(d: int, size: int = 1000) -> np.ndarray:
    """Return the (size, size) product of random (size, d) and (d, size) matrices."""
    rng = np.random.default_rng(0)
    left = rng.standard_normal((size, d))
    right = rng.standard_normal((d, size))
    return left @ right


def estimate_rank(matrix: np.ndarray) -> tuple[int, np.ndarray]:
    """Return the numerical rank and descending singular values of a nonempty matrix.

    Count singular values strictly above 1e-10 times the largest value.
    A zero matrix has rank zero.
    """
    if "SOLUTION":
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        rank = int(np.count_nonzero(singular_values > singular_values[0] * 1e-10))
        return rank, singular_values
    else:
        # TODO: Estimate numerical rank from the singular-value spectrum.
        # Hint: np.linalg.svd(matrix, compute_uv=False) returns descending values.
        return 0, np.zeros(min(matrix.shape))


hidden_dim = int(np.random.default_rng(1).integers(5, 21))
matrix = blackbox(hidden_dim)
rank, singular_values = estimate_rank(matrix)
print(f"Estimated rank: {rank}")
print(f"Hidden dimension: {hidden_dim}")

# The cliff should sit on the dotted line.
plt.semilogy(singular_values[:40], "o-")
plt.axvline(hidden_dim - 0.5, color="red", linestyle=":", label="Hidden dimension")
plt.xlabel("Singular value index")
plt.ylabel("Singular value (log scale)")
plt.legend()
plt.show()


@report
def test_estimate_rank(
    solution: Callable[[np.ndarray], tuple[int, np.ndarray]],
) -> None:
    """Check rank recovery, scale invariance, and the zero-matrix edge case."""
    rng = np.random.default_rng(0)
    for d in (3, 12, 40):
        matrix = rng.standard_normal((80, d)) @ rng.standard_normal((d, 60))
        rank, singular_values = solution(matrix)
        assert rank == d, f"Expected rank {d}, got {rank}"
        assert singular_values.shape == (
            60,
        ), f"Expected 60 singular values for an 80 x 60 matrix, got {singular_values.shape}"
        assert np.all(
            singular_values[:-1] >= singular_values[1:]
        ), "Return singular values in descending order"

    for scale in (1e-6, 1.0, 1e6):
        matrix = scale * np.diag([1.0, 1e-8, 1e-12, 0.0])
        rank, singular_values = solution(matrix)
        assert rank == 2, f"Expected rank 2 at scale {scale}, got {rank}"
        np.testing.assert_allclose(
            singular_values,
            scale * np.array([1.0, 1e-8, 1e-12, 0.0]),
            err_msg=f"Incorrect singular values at scale {scale}",
        )

    rank, singular_values = solution(np.zeros((4, 7)))
    assert rank == 0, f"Expected rank 0 for a zero matrix, got {rank}"
    assert np.array_equal(
        singular_values, np.zeros(4)
    ), f"Expected four zero singular values, got {singular_values}"
    print("  All tests passed!")


test_estimate_rank(estimate_rank)

# %%
"""
### Exercise 3.5.2: Complete Model Dimension Extraction

> **Difficulty**: 3/5
> **Importance**: 4/5
>
> You should spend up to ~45 minutes on this exercise.

Implement the model extraction attack from
[Carlini et al. (2024), *Stealing Part of a Production Language Model*](https://arxiv.org/abs/2403.06634).
Complete `detect_hidden_dim` using only the supplied logit-query function.
Collect a query matrix, inspect its singular-value spectrum, and identify
the transition to the numerical noise floor.

<details>
<summary>Reference: matrix rank and model logits</summary>

A hidden state of shape `(1, d)` multiplied by output weights of shape
`(d, vocab_size)` produces logits of shape `(1, vocab_size)`. Stacking many
queries produces a matrix with rank at most `d` for a model such as GPT-2
whose output projection has no bias. To recover `d`, the queries must span
the hidden space and the returned precision must separate signal from noise.
NumPy's [`linalg.svd`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html)
returns singular values in descending order; values near zero contribute
little to the numerical rank.

The supplied query helper accepts a PyTorch tensor of token IDs with shape
`(batch, sequence_length)` and integer dtype, moves it to the model's device,
and returns logits of shape `(batch, vocab_size)`. Use `.cpu().numpy()` before
applying NumPy operations. No gradients are needed for this attack.

</details>


<details>
<summary>(spoiler) If everything goes well, this is what your graph should look like</summary>

![GPT-2 small](resources/gpt2-small.png)
</details>

"""


# %%
if TYPE_CHECKING:
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer


def load_model_and_tokenizer(
    model_name: str,
) -> tuple["GPT2LMHeadModel", "GPT2Tokenizer"]:
    """Load GPT-2 and its tokenizer."""
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    # The attack below is thousands of forward passes plus a large SVD, so run
    # it on the GPU when there is one (minutes on GPU, far longer on CPU).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model: {model_name} on {device}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


model_name = "openai-community/gpt2"
model, tokenizer = load_model_and_tokenizer(model_name)


def get_next_logits(input_ids: "torch.Tensor") -> "torch.Tensor":
    """Return next-token logits for a batch of token ID sequences."""
    import torch

    assert (
        input_ids.ndim == 2
    ), "Input IDs should have shape (batch_size, sequence_length)"
    with torch.no_grad():
        outputs = model(input_ids.to(model.device))
        return outputs.logits[:, -1, :]


# Shared attack parameters, used by Exercises 3.5.2 and 3.5.3.
N_QUERIES = 1000
MAX_PROMPT_LENGTH = 10
VOCAB_SIZE = tokenizer.vocab_size
print(f"Vocabulary size: {VOCAB_SIZE}")
print(f"Number of queries: {N_QUERIES}")


def detect_hidden_dim(
    n_queries: int = N_QUERIES,
    max_prompt_length: int = MAX_PROMPT_LENGTH,
    plot: bool = True,
) -> int:
    """Discover the model's hidden dimension from black-box logit queries.

    Sends many random token sequences to `get_next_logits`, stacks the
    resulting logit vectors into a matrix, computes its singular value
    spectrum, and returns the detected hidden dimension (the index of the
    sharp drop in the spectrum).
    """
    import torch
    from tqdm import tqdm

    if "SOLUTION":
        # 1. Generate n random prompts and collect logits.
        logit_matrix_q = []
        print(f"Querying model {n_queries} times...")
        for _ in tqdm(range(n_queries)):
            prompt_length = np.random.randint(1, max_prompt_length)
            random_tokens = np.random.randint(0, VOCAB_SIZE, size=prompt_length)
            input_ids = torch.tensor([random_tokens])
            logit_matrix_q.append(get_next_logits(input_ids).cpu().numpy())

        # 2. Stack into a matrix Q of shape (n_queries, vocab_size).
        Q = np.vstack(logit_matrix_q)
        print(f"Shape of logit matrix Q: {Q.shape}")

        # 3. Compute the singular values (we don't need U and Vh here).
        print("Computing Singular Value Decomposition (SVD)...")
        singular_values = np.linalg.svd(Q, compute_uv=False)

        # 4. The hidden dimension shows up as the largest drop in the (log-scaled)
        # spectrum: the transition from the h "real" directions to the
        # numerical-noise floor. The first singular value is a dominant
        # common-mode direction that creates a spurious gap at index 0, so we
        # ignore that head when locating the cliff.
        log_s = np.log(singular_values + 1e-12)
        gaps = log_s[:-1] - log_s[1:]
        lead = min(50, len(gaps) // 4)  # skip the dominant leading components
        detected_h = int(np.argmax(gaps[lead:]) + lead + 1)
        print(f"Detected hidden dimension (h): {detected_h}")

        # 5. Plot the spectrum with a marker at the detected dimension.
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(singular_values)
            plt.yscale("log")
            plt.title(f"Sorted Singular Values of Logit Matrix for {model_name}")
            plt.xlabel("Sorted Singular Values (Index)")
            plt.ylabel("Magnitude (log scale)")
            plt.grid(True)
            plt.axvline(
                x=detected_h,
                color="r",
                linestyle="--",
                label=f"Detected Hidden Dim: {detected_h}",
            )
            plt.legend()
            plt.show()

        return detected_h
    else:
        # TODO: Discover the model's hidden dimension using only black-box
        # logit queries.
        #   1. Send many random token sequences to get_next_logits.
        #   2. Stack the results into a matrix Q (n_queries, vocab_size).
        #   3. Compute its singular values (np.linalg.svd(..., compute_uv=False)).
        #   4. Plot the spectrum on a log scale, and look at where there is a sharp drop: eyeball it and return it
        #   5. (optional) The hidden dimension is the index of the sharp drop in the
        #      spectrum, e.g. argmax of the gaps between consecutive
        #      log-singular-values, plus one.
        # Return the detected hidden dimension as an int (768 for GPT-2 small).
        return 0


detected_h = detect_hidden_dim()
print(f"Using hidden dimension (h): {detected_h}")


@report
def test_detect_hidden_dim(h: int):
    # Checks the dimension detected above rather than re-running the attack:
    # every run costs 1000 forward passes plus an SVD over the whole vocabulary.
    assert isinstance(h, int), f"detect_hidden_dim must return an int, got {type(h)}"
    # GPT-2 small has hidden dimension 768. Allow a small tolerance because the
    # noise floor of the SVD can shift the detected gap by a few indices.
    assert abs(h - 768) <= 5, f"Expected detected hidden dim near 768, got {h}"
    print("  All tests passed!")


test_detect_hidden_dim(detected_h)
# %%
"""
### Exercise 3.5.3: Extracting Model Weights

> **Difficulty**: 5/5
> **Importance**: 2/5
>
> You should spend up to ~60 minutes on this exercise.

Use the hidden dimension from 3.5.2 to extract `lm_head.weight` from logit queries.

**Why SVD gives us the weights.** GPT-2's output projection has no bias, so
each logit vector is computed as:

```
logits = hidden_state @ W_out.T
```

where `W_out` has shape `(vocab_size, h)`. If we collect the logit vectors as
columns of a matrix `Q` (shape `vocab_size × n_queries`), then `Q` has rank at
most `h`. When the output projection has full column rank and the queries span
the hidden space, `Q` has rank `h`. The truncated SVD gives:

```
Q ≈ U_h · Σ_h · Vh
```

where `U_h` has shape `(vocab_size, h)`. Its columns form an orthonormal basis
for the column space of `W_out`.
Therefore `U_h @ Σ_h` recovers an equivalent output projection up to an unknown
invertible linear transformation of the hidden coordinates.

The extracted and true weights satisfy `W_extracted @ G ≈ W_true`.
`compare_weights` finds `G` by least squares and measures the alignment error.
"""


def extract_weights(
    hidden_dim: int,
    n_queries: int = N_QUERIES,
    max_prompt_length: int = MAX_PROMPT_LENGTH,
    batch_size: int = 2,
) -> np.ndarray:
    """Reconstruct the output projection matrix from black-box logit queries.

    Collects logit vectors from many (batched) random queries, stacks them
    into a matrix Q of shape (vocab_size, n_samples), takes the thin SVD, and
    returns U_h @ Sigma_h, the extracted weights, correct up to an unknown
    linear transform. Returns a NumPy array of shape (vocab_size, hidden_dim).
    """
    import torch
    from tqdm import tqdm

    if "SOLUTION":
        print(f"Querying model {n_queries}*{batch_size} times...")
        logit_vectors = []
        for _ in tqdm(range(n_queries)):
            prompt_length = np.random.randint(1, max_prompt_length)
            input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, prompt_length))
            logit_vectors.extend(get_next_logits(input_ids))

        # Columns are logit vectors -> Q has shape (vocab_size, n_samples).
        Q = torch.stack(logit_vectors).T

        print("Computing Singular Value Decomposition (SVD)...")
        U, s, Vh = torch.linalg.svd(Q, full_matrices=False)

        # Keep the top `hidden_dim` singular directions. U_h @ Sigma_h spans
        # the same column space as the true lm_head weights.
        U_h = U[:, :hidden_dim]
        Sigma_h = torch.diag(s[:hidden_dim])
        W_extracted = U_h @ Sigma_h
        return W_extracted.cpu().numpy()
    else:
        # TODO: Extract the output projection weights from logit queries.
        #   1. Collect logit vectors from many random queries (batch them for
        #      speed) and stack them so each logit vector is a *column* of Q.
        #   2. Perform the thin SVD: U, s, Vh = torch.linalg.svd(Q, full_matrices=False).
        #   3. Keep the top `hidden_dim` directions and return U_h @ Sigma_h
        #      as a NumPy array of shape (vocab_size, hidden_dim).
        # (Placeholder keeps the scaffold runnable; at least one column so the
        # downstream least-squares alignment is well-formed.)
        return np.zeros((VOCAB_SIZE, max(hidden_dim, 1)))


W_extracted = extract_weights(detected_h)
print(f"Extracted weight matrix shape: {W_extracted.shape}")

# %%
# Get the ground truth weights. The lm_head holds the final projection layer,
# already in (vocab_size, hidden_size) shape. These are used only for validation.
true_weights = model.lm_head.weight.detach().cpu().numpy()


# %%
def compare_weights(
    W_extracted: np.ndarray, W_true: np.ndarray
) -> tuple[float, float, float]:
    """
    Compares the extracted weight matrix with the ground truth matrix.

    Args:
        W_extracted: The weights recovered from the attack (W_tilde).
        W_true: The ground truth weights from the model.

    Returns:
        tuple: (rmse, avg_cosine_sim, percentage_similarity)
    """
    print("\n--- Comparing Extracted Weights to Ground Truth ---")

    # 1. Solve for the transformation matrix G using least squares
    # We want to find G such that W_extracted @ G ≈ W_true
    print("Solving for the alignment matrix G using least squares...")
    try:
        G, residuals, rank, s = np.linalg.lstsq(W_extracted, W_true, rcond=None)
    except np.linalg.LinAlgError as e:
        print(f"Error solving least squares: {e}")
        return float("nan"), float("nan"), float("nan")

    # 2. Align the extracted weights using the solved G
    W_aligned = W_extracted @ G
    print("Alignment complete.")

    # 3. Calculate Root Mean Square Error (RMSE)
    temp = (W_aligned - W_true) ** 2
    rmse = np.sqrt(temp.mean())

    # 4. Calculate Average Cosine Similarity
    # Normalize each column vector to unit length before dot product
    norm_aligned = np.linalg.norm(W_aligned, axis=0, keepdims=True)
    norm_true = np.linalg.norm(W_true, axis=0, keepdims=True)

    # Avoid division by zero for zero-norm vectors.
    norm_aligned[norm_aligned == 0] = 1
    norm_true[norm_true == 0] = 1

    W_aligned_normalized = W_aligned / norm_aligned
    W_true_normalized = W_true / norm_true

    # Calculate cosine similarity for each column and average
    cosine_similarities = (W_aligned_normalized * W_true_normalized).sum(axis=0)
    avg_cosine_sim = cosine_similarities.mean()

    # 5. Calculate a "Percentage Similarity" metric based on relative error
    # Frobenius norm is the square root of the sum of the absolute squares of its elements.
    relative_error = np.linalg.norm(W_aligned - W_true, "fro") / np.linalg.norm(
        W_true, "fro"
    )
    percentage_similarity = (1 - relative_error) * 100

    return rmse, avg_cosine_sim, percentage_similarity


# 4. Compare the weights and print results
rmse, cosine_sim, percent_sim = compare_weights(W_extracted, true_weights)

print("\n--- Final Results ---")
print(f"Root Mean Square Error (RMSE): {rmse:.6f}")
print(f"Average Cosine Similarity: {cosine_sim:.6f}")
print(f"Similarity Percentage: {percent_sim:.2f}%")

print("\nInterpretation:")
print("- RMSE: Lower is better. We expect values like 0.001.")
print(
    "- Cosine Similarity: Closer to 1.0 is better, indicating the vectors are pointing in the same direction."
)
print("- Similarity Percentage: Closer to 100% is better.")


@report
def test_compare_weights(
    solution: Callable[[np.ndarray, np.ndarray], tuple[float, float, float]],
    extracted_weights: np.ndarray,
    true_weights: np.ndarray,
) -> None:
    """requires: GPU (checks weights from the GPT-2 extraction run)."""
    # Check alignment using the weights extracted above.
    _, avg_cosine_sim, _ = solution(extracted_weights, true_weights)
    assert avg_cosine_sim > 0.99, (
        f"Aligned extracted weights should match the true weights very "
        f"closely (cosine > 0.99), got {avg_cosine_sim:.4f}"
    )
    print("  All tests passed!")


test_compare_weights(compare_weights, W_extracted, true_weights)

# %%
"""
## Summary

- Logit vectors expose a low-rank projection of a model's hidden states.
- A singular-value spectrum can reveal the hidden dimension when queries
  span that space and numerical precision separates signal from noise.
- SVD can recover the output projection up to an unknown invertible linear
  transformation.
- Query budgets, logit truncation, rounding, and noise limit extraction.

### Further reading

- [Carlini et al. (2024), *Stealing Part of a Production Language Model*](https://arxiv.org/abs/2403.06634)
- [Tramèr et al. (2016), *Stealing Machine Learning Models via Prediction APIs*](https://arxiv.org/abs/1609.02943)
"""
