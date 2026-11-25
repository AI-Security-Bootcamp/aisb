# Allow imports from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import sys
from pathlib import Path
from aisb_utils import report
from typing import TYPE_CHECKING, Callable
import matplotlib.pyplot as plt
import numpy as np



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




@report
def test_detect_hidden_dim(h: int) -> None:
    """requires: GPU (checks the result of the GPT-2 logit-query attack)."""
    # Checks the dimension detected above rather than re-running the attack:
    # every run costs 1000 forward passes plus an SVD over the whole vocabulary.
    assert isinstance(h, int), f"detect_hidden_dim must return an int, got {type(h)}"
    # GPT-2 small has hidden dimension 768. Allow a small tolerance because the
    # noise floor of the SVD can shift the detected gap by a few indices.
    assert abs(h - 768) <= 5, f"Expected detected hidden dim near 768, got {h}"
    print("  All tests passed!")




@report
def test_compare_weights(
    solution: Callable[[np.ndarray, np.ndarray], tuple[float, float, float]],
    extracted_weights: np.ndarray,
    true_weights: np.ndarray,
) -> None:
    """requires: GPU (checks weights from the GPT-2 extraction run)."""
    # Reuses the weights extracted above instead of running the attack a second
    # time: another end-to-end run means 2000 more queries and another SVD of a
    # (vocab_size x n_samples) matrix. Checks that the recovered weights align
    # almost perfectly with the true lm_head weights (up to a linear transform).
    _, avg_cosine_sim, _ = solution(extracted_weights, true_weights)
    assert avg_cosine_sim > 0.99, (
        f"Aligned extracted weights should match the true weights very "
        f"closely (cosine > 0.99), got {avg_cosine_sim:.4f}"
    )
    print("  All tests passed!")
