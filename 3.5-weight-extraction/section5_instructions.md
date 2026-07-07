
# W1D3 - Section 5️⃣: Model weight extraction via SVD

Recover a model's hidden dimension — and the last projection layer — from
API access alone, using the logits-matrix SVD attack.


```python


import sys
from pathlib import Path
for _path in [
    str(Path(__file__).resolve().parent.parent),    # day root (day3-inference/)
    str(Path(__file__).resolve().parent.parent.parent),  # workspace root
]:
    if _path not in sys.path:
        sys.path.insert(0, _path)
```

## 5️⃣ Model weight extraction via SVD

Let's implement the model extraction attack from the paper.

### Exercise 5.1 - Complete Model Dimension Extraction

> **Difficulty**: 🔴🔴🔴⚪⚪
> **Importance**: 🔵🔵🔵🔵⚪
>
> You should spend up to ~45 minutes on this exercise.

Complete the implementation of model dimension extraction using SVD.


```python

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm

# %%
# 1. Load the model and tokenizer
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
# TODO: Discover the model's hidden dimension using only black-box
# logit queries. Send many random token sequences to get_next_logits,
# stack the results into a matrix, compute its SVD, and plot the
# singular values. The hidden dimension shows up as a sharp drop
# in the spectrum. (1000 queries should be enough.)
pass
```

### Exercise 5.2 - Extracting Model Weights

> **Difficulty**: 🔴🔴🔴🔴🔴
> **Importance**: 🔵🔵⚪⚪⚪
>
> You should spend up to ~60 minutes on this exercise.

Now use the hidden dimension `h` from exercise 5.1 to recover the model's output
projection matrix — `lm_head.weight` — from black-box logit queries alone.

**Why SVD gives us the weights.** Every logit vector the model returns is computed as:

```
logits = hidden_state @ W_out.T + bias
```

where `W_out` has shape `(vocab_size, h)`. Across many queries, the hidden states
span an `h`-dimensional subspace of the full `vocab_size`-dimensional space. If we
collect these logit vectors as columns of a matrix `Q` (shape `vocab_size × n_queries`),
then `Q` has rank at most `h`. The thin SVD decomposes:

```
Q ≈ U_h · Σ_h · Vh
```

where `U_h` has shape `(vocab_size, h)`. The columns of `U_h` form an orthonormal
basis for the same column space as `W_out`. Therefore `U_h @ Σ_h` is `W_out` up
to an unknown invertible linear transformation — we can reconstruct the direction
and relative scaling of every output-projection row, but not the exact values
(which would require knowing the hidden states too).

This is the "up to a linear transform" claim: the extracted matrix and the true
`lm_head.weight` are related by `W_extracted @ G ≈ W_true` for some matrix `G`.
`compare_weights` solves for `G` via least squares and then measures how close
the aligned matrices are.


```python
# TODO: Extract the output projection weights from logit queries.
# Collect logit vectors from many random queries (batched for speed),
# stack them into a matrix, perform SVD, and use the detected hidden
# dimension to reconstruct the weight matrix (up to a linear transform).
pass

# %%
# Get the ground truth weights
# The lm_head contains the final projection layer weights.
# We need to transpose it to match the (vocab_size, hidden_size) shape.
true_weights = model.lm_head.weight.detach().numpy()


# %%
def compare_weights(W_extracted: np.ndarray, W_true: np.ndarray) -> tuple[float, float, float]:
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

    # Avoid division by zero for zero-norm vectors
    # This is unlikely but good practice
    norm_aligned[norm_aligned == 0] = 1
    norm_true[norm_true == 0] = 1

    W_aligned_normalized = W_aligned / norm_aligned
    W_true_normalized = W_true / norm_true

    # Calculate cosine similarity for each column and average
    cosine_similarities = (W_aligned_normalized * W_true_normalized).sum(axis=0)
    avg_cosine_sim = cosine_similarities.mean()

    # 5. Calculate a "Percentage Similarity" metric based on relative error
    # Frobenius norm is the square root of the sum of the absolute squares of its elements.
    relative_error = np.linalg.norm(W_aligned - W_true, "fro") / np.linalg.norm(W_true, "fro")
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
print("- Cosine Similarity: Closer to 1.0 is better, indicating the vectors are pointing in the same direction.")
print("- Similarity Percentage: Closer to 100% is better.")
```

### Extensions to try
- What if you only have access to topk logits instead of the full logits?
- What if you don't have access to the logits? (this is quiet expensive, so just extract one logit with the method listed in appendix F)


## Summary & Further Reading

Today covered four layers of LLM inference security:

**1. Tokenization & prompt construction** — Tokenizers are not transparent:
different models tokenize the same string differently, and chat templates
assemble multi-turn prompts with special tokens. Edge cases in tokenization
and prompt construction are where many prompt-injection attacks begin.

**2. Guardrails: attacks and defences** — Safety training alone is defeated
by simple paraphrases. Keyword filters are defeated by synonyms. LLM-based
input classifiers are defeated by creative reframing (fiction, research).
Output classifiers share the same limitation. Linear probes on internal
representations catch what text-based classifiers miss, because they operate
on semantic intent rather than surface form.

**3. Knowledge distillation attacks** — Label filtering (masking forbidden
tokens with `ignore_index=-100`) prevents a baseline student from learning
the forbidden completion. But adding a KD loss (temperature-scaled KL
divergence from the teacher) transfers the forbidden knowledge through the
soft probability distribution. Label filtering is necessary but not
sufficient for safe distillation.

**4. Model weight extraction via SVD** — By collecting logits from random
prompts and computing SVD, an attacker can recover the model's hidden
dimension from the singular value spectrum, and reconstruct the output
projection layer up to a linear transformation.

### Further reading

- [Hinton, Vinyals & Dean (2015), *Distilling the Knowledge in a Neural Network*](https://arxiv.org/abs/1503.02531)
- [Carlini et al. (2024), *Stealing Part of a Production Language Model*](https://arxiv.org/abs/2403.06634)
- [Tramèr et al. (2016), *Stealing Machine Learning Models via Prediction APIs*](https://arxiv.org/abs/1609.02943)
- [Burns et al. (2022), *Discovering Latent Knowledge*](https://arxiv.org/abs/2212.03827)
- [Zou et al. (2023), *Representation Engineering*](https://arxiv.org/abs/2310.01405)
- [Hubinger et al. (2024), *Sleeper Agents*](https://arxiv.org/abs/2401.05566)
