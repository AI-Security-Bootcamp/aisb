# Allow imports from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import sys
from pathlib import Path
from aisb_utils import report
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import torch
import random
from typing import Tuple, List, Optional, Dict, Any



@report
def test_target_loss(solution):
    """requires: GPU (runs a forward pass through the chat model).

    A cross-entropy loss over target tokens must be a finite, positive scalar.
    We also confirm the loss reacts to the target: scoring a *different* target
    of the same length gives a different loss (so the slice really aligns with
    the target tokens rather than being constant).
    """
    loss = solution(chat_model, prompt_prefix_ids, initial_suffix_ids, prompt_suffix_ids, target_ids)
    value = loss.item()
    assert loss.ndim == 0, f"Expected a scalar loss, got shape {tuple(loss.shape)}"
    assert torch.isfinite(loss), f"Loss must be finite, got {value}"
    assert value > 0, f"Cross-entropy loss must be positive, got {value}"

    # A different target continuation (same length) should yield a different loss.
    other_target_ids = torch.roll(target_ids, shifts=1)
    other_loss = solution(
        chat_model, prompt_prefix_ids, initial_suffix_ids, prompt_suffix_ids, other_target_ids
    ).item()
    assert abs(other_loss - value) > 1e-4, (
        "Loss did not change when the target tokens changed - the target slice "
        "is probably misaligned with the target positions"
    )
    print("  All tests passed!")




@report
def test_compute_suffix_token_gradients(solution):
    """requires: GPU (backprops through the chat model).

    The one-hot gradient must have one row per suffix position and one column
    per vocabulary token, and it must be finite (so it can rank replacements).
    """
    grads = solution(chat_model, prompt_prefix_ids, initial_suffix_ids, prompt_suffix_ids, target_ids)
    vocab_size = chat_model.get_input_embeddings().weight.shape[0]
    assert grads.shape == (initial_suffix_ids.shape[0], vocab_size), (
        f"Expected gradient shape {(initial_suffix_ids.shape[0], vocab_size)}, got {tuple(grads.shape)}"
    )
    assert torch.isfinite(grads).all(), "Gradients contain NaN/Inf values"
    print("  All tests passed!")




@report
def test_top_replacements_from_gradients(solution):
    """requires: GPU (uses gradients computed from the chat model).

    top-k selection must (a) return exactly topk ids per position and
    (b) never propose a forbidden (special) token id.
    """
    topk = 5
    forbidden = tokenizer.all_special_ids
    candidates = solution(gradients, topk=topk, forbidden_token_ids=forbidden)
    assert candidates.shape == (initial_suffix_ids.shape[0], topk), (
        f"Expected shape {(initial_suffix_ids.shape[0], topk)}, got {tuple(candidates.shape)}"
    )
    forbidden_set = set(forbidden)
    proposed = set(candidates.flatten().tolist())
    leaked = proposed & forbidden_set
    assert not leaked, f"Forbidden token ids were proposed as replacements: {sorted(leaked)}"
    print("  All tests passed!")




@report
def test_run_greedy_search(solution):
    """requires: GPU (runs the full GCG search over the chat model).

    Note: `final_loss <= initial_loss` is tautological here - the greedy loop
    only ever commits a replacement when it strictly lowers the loss. Instead we
    require the search to make *meaningful* progress: it must drive the target
    loss down by a clear relative margin over several steps.
    """
    suffix_ids, history = solution(
        chat_model,
        tokenizer,
        prompt_prefix_ids,
        prompt_suffix_ids,
        target_ids,
        suffix_length=10,
        steps=25,
        topk=8,
    )
    assert suffix_ids.shape[0] == 10, f"Expected an optimized suffix of length 10, got {suffix_ids.shape[0]}"
    assert len(history) >= 2, "Expected the search to accept at least one improving update"
    assert all(v > 0 for v in history), f"All losses should be positive, got {history}"

    # Require a substantive reduction, not just any non-increase.
    reduction = (history[0] - history[-1]) / history[0]
    assert reduction > 0.1, (
        f"GCG barely reduced the target loss (initial={history[0]:.3f}, "
        f"final={history[-1]:.3f}, reduction={reduction:.1%}); expected >10%"
    )
    print("  All tests passed!")
